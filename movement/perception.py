"""
Perception system for autonomous driving.
Processes sensor data to understand the environment.
"""

import numpy as np
import cv2
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import sys

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow" 
    GREEN = "green"
    UNKNOWN = "unknown"

class ObjectType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_LIGHT = "traffic_light"
    TRAFFIC_SIGN = "traffic_sign"
    OBSTACLE = "obstacle"

class LaneType(Enum):
    SOLID_WHITE = "solid_white"
    DASHED_WHITE = "dashed_white"
    SOLID_YELLOW = "solid_yellow"
    DASHED_YELLOW = "dashed_yellow"
    UNKNOWN = "unknown"

@dataclass
class DetectedObject:
    """Object detection with comprehensive information"""
    object_type: ObjectType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    distance: float
    relative_position: Tuple[float, float]  # (lateral, longitudinal)
    relative_speed: float = 0.0
    lane_assignment: int = 0  # -1=left, 0=same, 1=right
    size_2d: Tuple[float, float] = (0.0, 0.0)  # (width, height) in pixels
    is_moving: bool = False
    track_id: Optional[int] = None  # For temporal tracking

@dataclass
class LaneInfo:
    """Comprehensive lane information with polynomial fitting"""
    left_boundary: Optional[np.ndarray] = None  # Polynomial coefficients
    right_boundary: Optional[np.ndarray] = None  # Polynomial coefficients
    center_line: Optional[np.ndarray] = None  # Polynomial coefficients
    left_lane_pixels: Optional[np.ndarray] = None  # Raw pixel coordinates
    right_lane_pixels: Optional[np.ndarray] = None  # Raw pixel coordinates
    lane_width: float = 3.5  # meters
    curvature: float = 0.0  # 1/radius
    heading_angle: float = 0.0  # radians relative to lane
    lane_type_left: LaneType = LaneType.UNKNOWN
    lane_type_right: LaneType = LaneType.UNKNOWN
    confidence: float = 0.0
    lane_departure_left: bool = False
    lane_departure_right: bool = False
    lane_center_offset: float = 0.0  # meters from lane center

@dataclass
class TrafficLightInfo:
    """Traffic light information"""
    state: TrafficLightState
    distance: float
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    time_to_change: Optional[float] = None  # Estimated seconds until state change
    relevant_for_ego: bool = True  # Whether this light affects our lane

@dataclass
class SafetyMetrics:
    """Safety-related metrics for rule-based decisions"""
    collision_risk_front: float = 0.0  # 0-1 scale
    collision_risk_left: float = 0.0
    collision_risk_right: float = 0.0
    emergency_brake_needed: bool = False
    following_distance: float = float('inf')
    time_to_collision: float = float('inf')
    safe_to_change_left: bool = True
    safe_to_change_right: bool = True

@dataclass
class PerceptionOutput:
    """Comprehensive perception system output"""
    # Objects
    detected_objects: List[DetectedObject] = field(default_factory=list)
    vehicles: List[DetectedObject] = field(default_factory=list)
    pedestrians: List[DetectedObject] = field(default_factory=list)
    obstacles: List[DetectedObject] = field(default_factory=list)
    
    # Lane information
    lane_info: Optional[LaneInfo] = None
    
    # Traffic elements
    traffic_light: Optional[TrafficLightInfo] = None
    traffic_lights: List[TrafficLightInfo] = field(default_factory=list)  # All detected lights
    
    # Road analysis
    drivable_area: Optional[np.ndarray] = None
    road_ahead_clear: bool = True
    intersection_ahead: bool = False
    closest_vehicle_distance: float = float('inf')
    
    # Safety metrics
    safety_metrics: SafetyMetrics = field(default_factory=SafetyMetrics)
    
    # Speed and limits
    speed_limit: Optional[float] = None
    
    # Metadata
    frame_id: int = 0
    timestamp: float = 0.0
    processing_time: float = 0.0
    sensor_quality: Dict[str, float] = field(default_factory=dict)

class PerceptionSystem:
    """
    Perception system with improved algorithms and safety focus
    """
    
    # CARLA semantic segmentation classes
    SEMANTIC_CLASSES = {
        0: 'unlabeled',
        1: 'building', 
        2: 'fence',
        3: 'other',
        4: 'pedestrian',
        5: 'pole',
        6: 'roadline',
        7: 'road',
        8: 'sidewalk', 
        9: 'vegetation',
        10: 'vehicles',
        11: 'wall',
        12: 'traffic_sign',
        13: 'sky',
        14: 'ground',
        15: 'bridge',
        16: 'railtrack',
        17: 'guardrail',
        18: 'traffic_light',
        19: 'static',
        20: 'dynamic',
        21: 'water',
        22: 'terrain'
    }
    
    # Reverse mapping for easier lookup
    CLASS_TO_ID = {v: k for k, v in SEMANTIC_CLASSES.items()}
    
    def __init__(self, image_width: int = 800, image_height: int = 600):
        self.image_width = image_width
        self.image_height = image_height
        
        # Camera calibration parameters (for distance estimation)
        self.camera_matrix = self._get_default_camera_matrix()
        self.focal_length = 400.0  # Approximate focal length in pixels
        self.camera_height = 2.4   # Camera height in meters
        
        # Traffic light color detection parameters (improved HSV ranges)
        self.color_ranges = {
            'red': [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                   (np.array([170, 120, 70]), np.array([180, 255, 255]))],
            'yellow': [(np.array([15, 120, 70]), np.array([35, 255, 255]))],
            'green': [(np.array([40, 120, 70]), np.array([80, 255, 255]))]
        }
        
        # Object detection parameters
        self.min_object_area = 100
        self.min_vehicle_area = 500  # Increased for better filtering
        self.min_pedestrian_area = 200  # Increased for better filtering
        self.max_detection_distance = 100.0  # meters
        
        # Lane detection parameters
        self.min_lane_pixels = 50
        self.roi_height_ratio = 0.4  # Focus on relevant area
        self.lane_width_pixels = 100  # Approximate lane width in pixels
        
        # Safety parameters
        self.safe_following_distance = 15.0  # meters
        self.emergency_brake_distance = 8.0  # meters
        self.lane_change_clearance = 20.0   # meters
        
        # Temporal tracking (simple implementation)
        self.object_history = {}
        self.next_track_id = 0
        
    def _get_default_camera_matrix(self) -> np.ndarray:
        """Default camera matrix for CARLA setup"""
        fx = fy = 400.0  # Focal length approximation
        cx, cy = self.image_width / 2, self.image_height / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def process_sensors(self, sensor_data: Dict[str, Any], frame_id: int = 0, 
                       timestamp: float = 0.0) -> PerceptionOutput:
        """
        Main processing function with safety focus
        """
        start_time = time.time()
        
        # Extract sensor data
        rgb_image = sensor_data.get('rgb')
        semantic_image = sensor_data.get('semantic') 
        depth_image = sensor_data.get('depth')
        vehicle_speed = sensor_data.get('speed', 0.0)
        gps_data = sensor_data.get('gps', {})
        imu_data = sensor_data.get('imu', {})
        
        if rgb_image is None or semantic_image is None:
            return self._empty_perception(frame_id, timestamp)
        
        # Assess sensor quality
        sensor_quality = self._assess_sensor_quality(rgb_image, semantic_image, depth_image)
        
        perception = PerceptionOutput(
            frame_id=frame_id,
            timestamp=timestamp,
            sensor_quality=sensor_quality
        )
        
        # Object detection with algorithms
        all_objects = self._detect_objects(semantic_image, depth_image, rgb_image)
        
        # Temporal tracking
        all_objects = self._update_object_tracking(all_objects)
        
        # Categorize objects
        perception.detected_objects = all_objects
        perception.vehicles = [obj for obj in all_objects if obj.object_type == ObjectType.VEHICLE]
        perception.pedestrians = [obj for obj in all_objects if obj.object_type == ObjectType.PEDESTRIAN]
        perception.obstacles = [obj for obj in all_objects 
                              if obj.distance < 50.0 and obj.confidence > 0.5]
        
        # Lane detection
        perception.lane_info = self._detect_lanes(semantic_image, rgb_image, depth_image)
        
        # Traffic light detection with multiple lights
        perception.traffic_lights = self._detect_traffic_lights(
            rgb_image, semantic_image, depth_image
        )
        # Set primary traffic light (most relevant)
        perception.traffic_light = self._get_most_relevant_traffic_light(perception.traffic_lights)
        
        # Road analysis
        perception.drivable_area = self._get_drivable_area(semantic_image)
        perception.road_ahead_clear = self._is_road_clear(
            semantic_image, depth_image, perception.obstacles
        )
        perception.intersection_ahead = self._detect_intersection(
            semantic_image, perception.traffic_lights
        )
        
        # Calculate closest vehicle distance
        if perception.vehicles:
            perception.closest_vehicle_distance = min(v.distance for v in perception.vehicles)
        
        # Safety metrics calculation
        perception.safety_metrics = self._calculate_safety_metrics(
            perception.vehicles, perception.pedestrians, perception.lane_info, vehicle_speed
        )
        
        # Speed limit detection
        perception.speed_limit = self._detect_speed_limit(rgb_image, semantic_image)
        
        # Record processing time
        perception.processing_time = time.time() - start_time
        
        return perception
    
    def _detect_objects(self, semantic_image: np.ndarray, depth_image: Optional[np.ndarray],
                                rgb_image: np.ndarray) -> List[DetectedObject]:
        """Object detection with better filtering and distance estimation"""
        detected_objects = []
        
        # Define object class mappings
        object_mappings = {
            ObjectType.VEHICLE: [self.CLASS_TO_ID.get('vehicles', 10)],
            ObjectType.PEDESTRIAN: [self.CLASS_TO_ID.get('pedestrian', 4)],
        }
        
        for obj_type, class_ids in object_mappings.items():
            for class_id in class_ids:
                if class_id is None:
                    continue
                    
                # Create mask for this object type
                object_mask = (semantic_image == class_id)
                
                if not np.any(object_mask):
                    continue
                
                # Extract objects using method
                objects = self._extract_objects(
                    object_mask, obj_type, depth_image, rgb_image
                )
                detected_objects.extend(objects)
        
        # Filter by distance and confidence
        detected_objects = [
            obj for obj in detected_objects 
            if obj.distance <= self.max_detection_distance and obj.confidence > 0.3
        ]
        
        return detected_objects
    
    def _extract_objects(self, mask: np.ndarray, object_type: ObjectType,
                                 depth_image: Optional[np.ndarray], 
                                 rgb_image: np.ndarray) -> List[DetectedObject]:
        """Object extraction with morphological operations"""
        objects = []
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Use connected components for better object separation
        num_labels, labels = cv2.connectedComponents(mask_cleaned)
        
        for label in range(1, num_labels):  # Skip background
            component_mask = (labels == label)
            area = np.sum(component_mask)
            
            # Filter by minimum area based on object type
            min_area = self.min_vehicle_area if object_type == ObjectType.VEHICLE else self.min_pedestrian_area
            if area < min_area:
                continue
            
            # Get bounding box
            rows, cols = np.where(component_mask)
            x1, y1 = cols.min(), rows.min()
            x2, y2 = cols.max(), rows.max()
            bbox = (x1, y1, x2, y2)
            
            # Filter by aspect ratio (basic sanity check)
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / max(height, 1)
            if object_type == ObjectType.VEHICLE and (aspect_ratio < 0.3 or aspect_ratio > 5.0):
                continue
            if object_type == ObjectType.PEDESTRIAN and (aspect_ratio < 0.2 or aspect_ratio > 2.0):
                continue
            
            # Distance calculation
            distance = self._calculate_object_distance(
                component_mask, depth_image, bbox, object_type
            )
            
            # Skip very distant or very close (likely noise) objects
            if distance < 2.0 or distance > self.max_detection_distance:
                continue
            
            # Calculate relative position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            lateral_pos = (center_x - self.image_width / 2) / (self.image_width / 2)
            
            # Convert to real-world coordinates
            lateral_distance = self._pixel_to_lateral_distance(center_x, distance)
            
            # Assign lane based on lateral position
            lane_assignment = self._assign_lane(lateral_distance)
            
            # Confidence calculation
            confidence = self._calculate_object_confidence(
                component_mask, depth_image, object_type, bbox
            )
            
            # Motion estimation (placeholder for tracking)
            is_moving = self._estimate_object_motion(bbox, object_type)
            
            detected_object = DetectedObject(
                object_type=object_type,
                bbox=bbox,
                confidence=confidence,
                distance=distance,
                relative_position=(lateral_distance, distance),
                lane_assignment=lane_assignment,
                size_2d=(width, height),
                is_moving=is_moving
            )
            
            objects.append(detected_object)
        
        return objects
    
    def _detect_lanes(self, semantic_image: np.ndarray, 
                              rgb_image: np.ndarray,
                              depth_image: Optional[np.ndarray]) -> Optional[LaneInfo]:
        """Lane detection with better polynomial fitting and validation"""
        # Get road and roadline masks
        road_mask = (semantic_image == self.CLASS_TO_ID.get('road', 7))
        roadline_mask = (semantic_image == self.CLASS_TO_ID.get('roadline', 6))
        
        if not np.any(roadline_mask):
            return None
        
        # ROI selection - perspective-aware
        roi_mask = self._create_lane_roi_mask()
        roadline_roi = roadline_mask & roi_mask
        
        if np.sum(roadline_roi) < self.min_lane_pixels:
            return None
        
        lane_info = LaneInfo()
        
        # Lane boundary extraction
        left_boundary, right_boundary, left_pixels, right_pixels = self._extract_lane_boundaries(
            roadline_roi
        )
        
        lane_info.left_boundary = left_boundary
        lane_info.right_boundary = right_boundary
        lane_info.left_lane_pixels = left_pixels
        lane_info.right_lane_pixels = right_pixels
        
        # Calculate metrics
        if left_boundary is not None or right_boundary is not None:
            lane_info.lane_width = self._calculate_lane_width(
                left_boundary, right_boundary, depth_image
            )
            lane_info.curvature = self._calculate_curvature(left_boundary, right_boundary)
            lane_info.center_line = self._calculate_center_line(
                left_boundary, right_boundary, lane_info.lane_width
            )
            lane_info.heading_angle = self._calculate_heading_angle(lane_info.center_line)
            
            # Lane departure detection
            lane_info.lane_departure_left, lane_info.lane_departure_right, lane_info.lane_center_offset = \
                self._detect_lane_departure(lane_info.center_line, lane_info.lane_width)
            
            # Lane type classification
            lane_info.lane_type_left, lane_info.lane_type_right = self._classify_lane_types(
                rgb_image, left_boundary, right_boundary, left_pixels, right_pixels
            )
            
            # Confidence calculation
            lane_info.confidence = self._calculate_lane_confidence(
                roadline_roi, left_boundary, right_boundary, left_pixels, right_pixels
            )
        
        return lane_info
    
    def _detect_traffic_lights(self, rgb_image: np.ndarray, semantic_image: np.ndarray,
                                       depth_image: Optional[np.ndarray]) -> List[TrafficLightInfo]:
        """Traffic light detection with multiple lights and relevance scoring"""
        traffic_lights = []
        
        # Find traffic light regions
        traffic_light_mask = (semantic_image == self.CLASS_TO_ID.get('traffic_light', 18))
        
        if not np.any(traffic_light_mask):
            return traffic_lights
        
        # Get connected components for multiple traffic lights
        mask_uint8 = traffic_light_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum size threshold
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            
            # ROI extraction with padding
            padding = 5
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(rgb_image.shape[1], x + w + padding)
            y_end = min(rgb_image.shape[0], y + h + padding)
            
            tl_roi = rgb_image[y_start:y_end, x_start:x_end]
            
            if tl_roi.size == 0:
                continue
            
            # Color classification
            state, color_confidence = self._classify_traffic_light_color(tl_roi)
            
            if color_confidence < 0.3:  # Skip low confidence detections
                continue
            
            # Distance calculation
            distance = self._calculate_traffic_light_distance(
                x, y, w, h, depth_image
            )
            
            # Determine relevance for ego vehicle
            relevant_for_ego = self._is_traffic_light_relevant(x, y, w, h, distance)
            
            traffic_light_info = TrafficLightInfo(
                state=state,
                distance=distance,
                confidence=color_confidence,
                bbox=bbox,
                relevant_for_ego=relevant_for_ego
            )
            
            traffic_lights.append(traffic_light_info)
        
        # Sort by distance (closest first)
        traffic_lights.sort(key=lambda tl: tl.distance)
        
        return traffic_lights
    
    def _calculate_safety_metrics(self, vehicles: List[DetectedObject], 
                                 pedestrians: List[DetectedObject],
                                 lane_info: Optional[LaneInfo],
                                 ego_speed: float) -> SafetyMetrics:
        """Calculate comprehensive safety metrics for rule-based decisions"""
        safety = SafetyMetrics()
        
        # Collision risk assessment
        for vehicle in vehicles:
            if vehicle.distance < self.emergency_brake_distance:
                safety.emergency_brake_needed = True
                safety.collision_risk_front = 1.0
            elif vehicle.distance < self.safe_following_distance:
                risk = 1.0 - (vehicle.distance / self.safe_following_distance)
                safety.collision_risk_front = max(safety.collision_risk_front, risk)
            
            # Update following distance
            if vehicle.lane_assignment == 0:  # Same lane
                safety.following_distance = min(safety.following_distance, vehicle.distance)
        
        # Pedestrian risk
        for pedestrian in pedestrians:
            if pedestrian.distance < 10.0:  # Close pedestrian
                if pedestrian.relative_position[0] < 0:  # Left side
                    safety.collision_risk_left = max(safety.collision_risk_left, 0.8)
                else:  # Right side
                    safety.collision_risk_right = max(safety.collision_risk_right, 0.8)
        
        # Time to collision calculation
        front_vehicles = [v for v in vehicles if v.lane_assignment == 0 and v.distance < 50.0]
        if front_vehicles and ego_speed > 1.0:
            closest_vehicle = min(front_vehicles, key=lambda v: v.distance)
            relative_speed = ego_speed - closest_vehicle.relative_speed
            if relative_speed > 0:
                safety.time_to_collision = closest_vehicle.distance / relative_speed
        
        # Lane change safety
        left_vehicles = [v for v in vehicles if v.lane_assignment == -1]
        right_vehicles = [v for v in vehicles if v.lane_assignment == 1]
        
        safety.safe_to_change_left = all(
            v.distance > self.lane_change_clearance for v in left_vehicles
        )
        safety.safe_to_change_right = all(
            v.distance > self.lane_change_clearance for v in right_vehicles
        )
        
        return safety
    
    # Helper methods
    def _calculate_object_distance(self, mask: np.ndarray, depth_image: Optional[np.ndarray],
                                          bbox: Tuple[int, int, int, int], 
                                          object_type: ObjectType) -> float:
        """Distance calculation with fallback methods"""
        if depth_image is not None:
            object_depths = depth_image[mask]
            valid_depths = object_depths[(object_depths > 1.0) & (object_depths < 200.0)]
            
            if len(valid_depths) > 10:
                # Use median for robustness
                distance = float(np.median(valid_depths))
                if 2.0 <= distance <= self.max_detection_distance:
                    return distance
        
        # Fallback: estimate from bounding box size
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        
        # Rough approximation based on typical object sizes
        if object_type == ObjectType.VEHICLE:
            # Assume vehicle height ~1.8m
            distance = (1.8 * self.focal_length) / max(height, 1)
        else:  # Pedestrian
            # Assume pedestrian height ~1.7m
            distance = (1.7 * self.focal_length) / max(height, 1)
        
        return max(5.0, min(distance, 100.0))  # Reasonable bounds
    
    def _pixel_to_lateral_distance(self, pixel_x: float, distance: float) -> float:
        """Convert pixel coordinate to lateral distance in meters"""
        # Simple pinhole camera model
        lateral_angle = (pixel_x - self.image_width / 2) / self.focal_length
        lateral_distance = distance * np.tan(lateral_angle)
        return float(lateral_distance)
    
    def _assign_lane(self, lateral_distance: float) -> int:
        """Lane assignment based on real-world distance"""
        # Assume ~3.7m lane width
        if lateral_distance < -1.85:
            return -1  # Left lane
        elif lateral_distance > 1.85:
            return 1   # Right lane
        else:
            return 0   # Same lane
    
    def _create_lane_roi_mask(self) -> np.ndarray:
        """Create perspective-aware ROI mask for lane detection"""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        
        # Trapezoidal ROI considering perspective
        bottom_width = self.image_width
        top_width = int(self.image_width * 0.6)
        height_start = int(self.image_height * 0.4)
        
        for y in range(height_start, self.image_height):
            progress = (y - height_start) / (self.image_height - height_start)
            width = int(top_width + (bottom_width - top_width) * progress)
            x_start = (self.image_width - width) // 2
            x_end = x_start + width
            mask[y, x_start:x_end] = True
        
        return mask
    
    def _extract_lane_boundaries(self, roadline_roi: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], 
        Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Lane boundary extraction with better separation"""
        rows, cols = np.where(roadline_roi)
        
        if len(cols) < self.min_lane_pixels:
            return None, None, None, None
        
        # Use image center for initial separation
        img_center = roadline_roi.shape[1] // 2
        
        # Separate left and right with some overlap handling
        left_mask = cols < img_center + 50  # Allow some overlap
        right_mask = cols > img_center - 50
        
        left_boundary = None
        right_boundary = None
        left_pixels = None
        right_pixels = None
        
        # Fit polynomials to lane boundaries
        if np.sum(left_mask) > 20:
            left_points = np.column_stack((cols[left_mask], rows[left_mask]))
            left_pixels = left_points
            left_boundary = self._fit_lane_polynomial(left_points)
        
        if np.sum(right_mask) > 20:
            right_points = np.column_stack((cols[right_mask], rows[right_mask]))
            right_pixels = right_points
            right_boundary = self._fit_lane_polynomial(right_points)
        
        return left_boundary, right_boundary, left_pixels, right_pixels
    
    def _fit_lane_polynomial(self, points: np.ndarray, degree: int = 2) -> Optional[np.ndarray]:
        """Polynomial fitting with RANSAC-like robustness"""
        if len(points) < degree + 1:
            return None
        
        try:
            best_poly = None
            best_score = 0
            
            # Multiple attempts for robust fitting
            for attempt in range(10):
                # Sample subset of points
                n_sample = min(len(points), max(50, len(points) // 3))
                sample_indices = np.random.choice(len(points), n_sample, replace=False)
                sample_points = points[sample_indices]
                
                # Fit polynomial
                poly = np.polyfit(sample_points[:, 1], sample_points[:, 0], degree)
                
                # Evaluate quality
                predicted_x = np.polyval(poly, points[:, 1])
                errors = np.abs(predicted_x - points[:, 0])
                inliers = np.sum(errors < 10.0)  # 10 pixel threshold
                avg_error = np.mean(errors[errors < 10.0]) if inliers > 0 else float('inf')
                
                # Score combines inlier count and accuracy
                score = inliers - avg_error * 0.1
                
                if score > best_score:
                    best_score = score
                    best_poly = poly
            
            return best_poly if best_score > 0 else None
            
        except Exception:
            return None
    
    def _calculate_lane_width(self, left_boundary: Optional[np.ndarray], 
                                     right_boundary: Optional[np.ndarray],
                                     depth_image: Optional[np.ndarray]) -> float:
        """Lane width calculation using depth information"""
        if left_boundary is None or right_boundary is None:
            return 3.7  # Standard lane width
        
        # Sample multiple y positions for robust measurement
        y_positions = np.linspace(self.image_height * 0.6, self.image_height - 20, 10)
        widths = []
        
        for y in y_positions:
            try:
                left_x = np.polyval(left_boundary, y)
                right_x = np.polyval(right_boundary, y)
                width_pixels = abs(right_x - left_x)
                
                # Convert to meters using depth information if available
                if depth_image is not None and 0 <= int(y) < depth_image.shape[0]:
                    # Sample depth at lane boundaries
                    left_depth = depth_image[int(y), int(max(0, min(left_x, self.image_width-1)))]
                    right_depth = depth_image[int(y), int(max(0, min(right_x, self.image_width-1)))]
                    avg_depth = (left_depth + right_depth) / 2
                    
                    if avg_depth > 0:
                        # Convert pixel width to meters using depth
                        meters_per_pixel = avg_depth / self.focal_length
                        width_meters = width_pixels * meters_per_pixel
                        widths.append(width_meters)
                else:
                    # Fallback: rough approximation
                    # Assume perspective scaling - closer = wider in pixels
                    estimated_depth = 50.0 - (y - self.image_height * 0.6) * 0.5
                    meters_per_pixel = estimated_depth / self.focal_length
                    width_meters = width_pixels * meters_per_pixel * 0.01  # Calibration factor
                    widths.append(width_meters)
                    
            except:
                continue
        
        if widths:
            median_width = np.median(widths)
            # Sanity check - typical lane width 3.0-4.0m
            return max(2.5, min(median_width, 5.0))
        
        return 3.7
    
    def _detect_lane_departure(self, center_line: Optional[np.ndarray], 
                              lane_width: float) -> Tuple[bool, bool, float]:
        """Detect lane departure and calculate offset from center"""
        if center_line is None:
            return False, False, 0.0
        
        # Check vehicle position relative to lane center at bottom of image
        y_check = self.image_height - 50
        try:
            lane_center_x = np.polyval(center_line, y_check)
            vehicle_center_x = self.image_width / 2
            
            # Convert pixel offset to meters
            pixel_offset = vehicle_center_x - lane_center_x
            # Rough conversion (needs calibration)
            meter_offset = pixel_offset * 0.01  # Approximate
            
            # Lane departure thresholds (half lane width minus vehicle width margin)
            departure_threshold = (lane_width / 2) - 0.5  # 0.5m margin
            
            left_departure = meter_offset < -departure_threshold
            right_departure = meter_offset > departure_threshold
            
            return left_departure, right_departure, meter_offset
            
        except:
            return False, False, 0.0
    
    def _classify_traffic_light_color(self, tl_roi: np.ndarray) -> Tuple[TrafficLightState, float]:
        """Traffic light color classification with better filtering"""
        if tl_roi.size == 0 or tl_roi.shape[0] < 10 or tl_roi.shape[1] < 10:
            return TrafficLightState.UNKNOWN, 0.0
        
        # Preprocessing for better color detection
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(tl_roi, (3, 3), 0)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        
        # Test each color with improved scoring
        color_scores = {}
        color_pixels = {}
        
        for color_name, ranges in self.color_ranges.items():
            total_pixels = 0
            total_intensity = 0
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                pixel_count = np.sum(mask > 0)
                total_pixels += pixel_count
                
                # Weight by intensity (brighter = more confident)
                if pixel_count > 0:
                    intensity = np.mean(hsv[mask > 0, 2])  # V channel
                    total_intensity += intensity * pixel_count
            
            color_pixels[color_name] = total_pixels
            if total_pixels > 0:
                color_scores[color_name] = total_intensity / total_pixels
            else:
                color_scores[color_name] = 0
        
        # Find dominant color considering both pixel count and intensity
        best_color = None
        best_score = 0
        
        for color_name in color_scores:
            # Combined score: pixel count weighted by intensity
            combined_score = color_pixels[color_name] * (color_scores[color_name] / 255.0)
            if combined_score > best_score and color_pixels[color_name] > 20:  # Minimum pixels
                best_score = combined_score
                best_color = color_name
        
        if best_color is None:
            return TrafficLightState.UNKNOWN, 0.0
        
        # Calculate confidence
        total_colored_pixels = sum(color_pixels.values())
        if total_colored_pixels == 0:
            return TrafficLightState.UNKNOWN, 0.0
        
        confidence = color_pixels[best_color] / max(total_colored_pixels, 1)
        confidence = min(confidence * 1.5, 1.0)  # Boost confidence slightly
        
        # Map to enum
        state_mapping = {
            'red': TrafficLightState.RED,
            'yellow': TrafficLightState.YELLOW,
            'green': TrafficLightState.GREEN
        }
        
        return state_mapping.get(best_color, TrafficLightState.UNKNOWN), confidence
    
    def _is_traffic_light_relevant(self, x: int, y: int, w: int, h: int, distance: float) -> bool:
        """Determine if traffic light is relevant for ego vehicle"""
        # Check if traffic light is in front and reasonably centered
        center_x = x + w / 2
        
        # Must be in central portion of image (not too far left/right)
        if center_x < self.image_width * 0.2 or center_x > self.image_width * 0.8:
            return False
        
        # Must be at reasonable distance
        if distance < 5.0 or distance > 100.0:
            return False
        
        # Must be in upper portion of image (traffic lights are mounted high)
        if y > self.image_height * 0.7:
            return False
        
        return True
    
    def _get_most_relevant_traffic_light(self, traffic_lights: List[TrafficLightInfo]) -> Optional[TrafficLightInfo]:
        """Select the most relevant traffic light for ego vehicle"""
        relevant_lights = [tl for tl in traffic_lights if tl.relevant_for_ego]
        
        if not relevant_lights:
            return None
        
        # Sort by distance and confidence
        relevant_lights.sort(key=lambda tl: (tl.distance, -tl.confidence))
        return relevant_lights[0]
    
    def _calculate_object_confidence(self, mask: np.ndarray, depth_image: Optional[np.ndarray], 
                                            object_type: ObjectType, bbox: Tuple[int, int, int, int]) -> float:
        """Confidence calculation with multiple factors"""
        base_confidence = 0.7
        
        # Size-based adjustment
        area = np.sum(mask)
        x1, y1, x2, y2 = bbox
        
        if object_type == ObjectType.VEHICLE:
            # Vehicles should be reasonably sized
            if 500 <= area <= 50000:
                base_confidence += 0.1
            elif area < 300:
                base_confidence -= 0.3
        elif object_type == ObjectType.PEDESTRIAN:
            # Pedestrians are typically smaller
            if 200 <= area <= 5000:
                base_confidence += 0.1
            elif area < 150:
                base_confidence -= 0.3
        
        # Aspect ratio check
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / max(height, 1)
        
        if object_type == ObjectType.VEHICLE:
            if 0.8 <= aspect_ratio <= 3.0:  # Reasonable vehicle aspect ratio
                base_confidence += 0.1
            else:
                base_confidence -= 0.2
        elif object_type == ObjectType.PEDESTRIAN:
            if 0.3 <= aspect_ratio <= 1.2:  # Reasonable pedestrian aspect ratio
                base_confidence += 0.1
            else:
                base_confidence -= 0.2
        
        # Depth consistency check
        if depth_image is not None:
            depths = depth_image[mask]
            valid_depths = depths[(depths > 1.0) & (depths < 200.0)]
            if len(valid_depths) > 10:
                depth_std = np.std(valid_depths)
                if depth_std < 3.0:  # Consistent depth
                    base_confidence += 0.15
                elif depth_std > 10.0:  # Inconsistent depth
                    base_confidence -= 0.15
        
        # Position-based adjustment (objects in center are more likely to be relevant)
        center_x = (x1 + x2) / 2
        distance_from_center = abs(center_x - self.image_width / 2) / (self.image_width / 2)
        if distance_from_center < 0.3:
            base_confidence += 0.05
        
        return max(0.0, min(1.0, base_confidence))
    
    def _estimate_object_motion(self, bbox: Tuple[int, int, int, int], 
                                       object_type: ObjectType) -> bool:
        """Motion estimation using simple tracking"""
        # This is a placeholder for more sophisticated tracking
        # In a real implementation, you would track objects across frames
        
        # For now, assume vehicles are more likely to be moving
        if object_type == ObjectType.VEHICLE:
            return True  # Could be refined with actual tracking
        elif object_type == ObjectType.PEDESTRIAN:
            return np.random.random() > 0.7  # Pedestrians less likely to be moving
        
        return False
    
    def _update_object_tracking(self, detected_objects: List[DetectedObject]) -> List[DetectedObject]:
        """Simple object tracking to assign track IDs"""
        # This is a simplified tracking implementation
        # A full implementation would use Kalman filters or similar
        
        for obj in detected_objects:
            # Find closest previous track
            min_distance = float('inf')
            best_track_id = None
            
            for track_id, prev_obj in self.object_history.items():
                # Calculate distance between current and previous detection
                curr_center = ((obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2)
                prev_center = ((prev_obj.bbox[0] + prev_obj.bbox[2]) / 2, (prev_obj.bbox[1] + prev_obj.bbox[3]) / 2)
                
                pixel_distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                       (curr_center[1] - prev_center[1])**2)
                
                if pixel_distance < min_distance and pixel_distance < 100:  # 100 pixel threshold
                    min_distance = pixel_distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                obj.track_id = best_track_id
                # Update relative speed estimate
                prev_obj = self.object_history[best_track_id]
                obj.relative_speed = prev_obj.relative_speed * 0.8 + \
                                   (prev_obj.distance - obj.distance) * 0.2  # Simple smoothing
            else:
                # New object
                obj.track_id = self.next_track_id
                self.next_track_id += 1
            
            # Update history
            self.object_history[obj.track_id] = obj
        
        # Clean old tracks (simple timeout)
        if len(self.object_history) > 50:
            # Keep only recent tracks
            current_tracks = {obj.track_id for obj in detected_objects}
            self.object_history = {tid: obj for tid, obj in self.object_history.items() 
                                 if tid in current_tracks}
        
        return detected_objects
    
    def _calculate_curvature(self, left_boundary: Optional[np.ndarray], 
                                    right_boundary: Optional[np.ndarray]) -> float:
        """Curvature calculation"""
        curvatures = []
        
        if left_boundary is not None and len(left_boundary) >= 3:
            # For polynomial ax^2 + bx + c, curvature = |2a| / (1 + (2ax + b)^2)^1.5
            # Approximate at y = image_height - 100
            y = self.image_height - 100
            a, b = left_boundary[0], left_boundary[1]
            denominator = (1 + (2*a*y + b)**2)**1.5
            if denominator > 0:
                curvatures.append(abs(2*a) / denominator)
        
        if right_boundary is not None and len(right_boundary) >= 3:
            y = self.image_height - 100
            a, b = right_boundary[0], right_boundary[1]
            denominator = (1 + (2*a*y + b)**2)**1.5
            if denominator > 0:
                curvatures.append(abs(2*a) / denominator)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _calculate_center_line(self, left_boundary: Optional[np.ndarray], 
                                      right_boundary: Optional[np.ndarray],
                                      lane_width: float) -> Optional[np.ndarray]:
        """Center line calculation with single boundary handling"""
        if left_boundary is not None and right_boundary is not None:
            # Both boundaries available
            return (left_boundary + right_boundary) / 2
        elif left_boundary is not None:
            # Only left boundary - estimate right boundary
            # Assume standard lane width in pixels
            lane_width_pixels = lane_width * 30  # Rough conversion
            # Shift left boundary to the right
            estimated_center = left_boundary.copy()
            estimated_center[-1] += lane_width_pixels / 2  # Adjust constant term
            return estimated_center
        elif right_boundary is not None:
            # Only right boundary - estimate left boundary
            lane_width_pixels = lane_width * 30
            estimated_center = right_boundary.copy()
            estimated_center[-1] -= lane_width_pixels / 2
            return estimated_center
        
        return None
    
    def _calculate_heading_angle(self, center_line: Optional[np.ndarray]) -> float:
        """Heading angle calculation"""
        if center_line is None or len(center_line) < 2:
            return 0.0
        
        # Calculate angle at multiple points and average
        y_positions = [self.image_height - 50, self.image_height - 100, self.image_height - 150]
        angles = []
        
        for y in y_positions:
            if y < 0:
                continue
            try:
                # For polynomial ax^2 + bx + c, derivative is 2ax + b
                slope = 2 * center_line[0] * y + center_line[1]
                angle = np.arctan(slope)
                angles.append(angle)
            except:
                continue
        
        return np.mean(angles) if angles else 0.0
    
    def _classify_lane_types(self, rgb_image: np.ndarray, 
                                    left_boundary: Optional[np.ndarray],
                                    right_boundary: Optional[np.ndarray],
                                    left_pixels: Optional[np.ndarray],
                                    right_pixels: Optional[np.ndarray]) -> Tuple[LaneType, LaneType]:
        """Lane type classification using RGB analysis"""
        left_type = LaneType.UNKNOWN
        right_type = LaneType.UNKNOWN
        
        # Analyze left boundary
        if left_pixels is not None and len(left_pixels) > 10:
            left_type = self._analyze_lane_marking_type(rgb_image, left_pixels)
        
        # Analyze right boundary  
        if right_pixels is not None and len(right_pixels) > 10:
            right_type = self._analyze_lane_marking_type(rgb_image, right_pixels)
        
        return left_type, right_type
    
    def _analyze_lane_marking_type(self, rgb_image: np.ndarray, pixels: np.ndarray) -> LaneType:
        """Analyze lane marking type from RGB pixels"""
        if len(pixels) < 10:
            return LaneType.UNKNOWN
        
        # Sample colors at lane marking pixels
        colors = []
        for pixel in pixels[:100]:  # Sample up to 100 pixels
            x, y = int(pixel[0]), int(pixel[1])
            if 0 <= x < rgb_image.shape[1] and 0 <= y < rgb_image.shape[0]:
                colors.append(rgb_image[y, x])
        
        if not colors:
            return LaneType.UNKNOWN
        
        colors = np.array(colors)
        avg_color = np.mean(colors, axis=0)
        
        # Simple color-based classification
        # Convert to HSV for better color analysis
        hsv_color = cv2.cvtColor(avg_color.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        
        # Check if it's yellow-ish
        if 15 <= hsv_color[0] <= 35 and hsv_color[1] > 100:  # Yellow hue range
            return LaneType.DASHED_YELLOW  # Assume dashed for simplicity
        else:
            # Default to white (most common)
            return LaneType.DASHED_WHITE
    
    def _calculate_lane_confidence(self, roadline_roi: np.ndarray, 
                                          left_boundary: Optional[np.ndarray],
                                          right_boundary: Optional[np.ndarray],
                                          left_pixels: Optional[np.ndarray],
                                          right_pixels: Optional[np.ndarray]) -> float:
        """Lane confidence calculation"""
        base_confidence = 0.3
        
        # Boundary detection quality
        if left_boundary is not None:
            base_confidence += 0.25
            if left_pixels is not None and len(left_pixels) > 50:
                base_confidence += 0.1
        
        if right_boundary is not None:
            base_confidence += 0.25
            if right_pixels is not None and len(right_pixels) > 50:
                base_confidence += 0.1
        
        # Total lane pixel count
        total_pixels = np.sum(roadline_roi)
        if total_pixels > 200:
            base_confidence += 0.1
        elif total_pixels > 400:
            base_confidence += 0.2
        
        # Polynomial fit quality (if we had fit errors, we could use those)
        # For now, assume good fits contribute to confidence
        
        return min(1.0, base_confidence)
    
    def _calculate_traffic_light_distance(self, x: int, y: int, w: int, h: int,
                                                 depth_image: Optional[np.ndarray]) -> float:
        """Traffic light distance calculation"""
        if depth_image is not None:
            # Sample multiple points in the traffic light region
            center_x, center_y = x + w // 2, y + h // 2
            sample_points = [
                (center_x, center_y),
                (x + w//4, y + h//4),
                (x + 3*w//4, y + h//4),
                (x + w//4, y + 3*h//4),
                (x + 3*w//4, y + 3*h//4)
            ]
            
            valid_distances = []
            for px, py in sample_points:
                if 0 <= px < depth_image.shape[1] and 0 <= py < depth_image.shape[0]:
                    distance = depth_image[py, px]
                    if 5.0 <= distance <= 150.0:  # Reasonable range for traffic lights
                        valid_distances.append(distance)
            
            if valid_distances:
                return float(np.median(valid_distances))
        
        # Fallback: estimate from size
        # Typical traffic light is ~0.3m diameter
        apparent_size = max(w, h)
        estimated_distance = (0.3 * self.focal_length) / max(apparent_size, 1)
        return max(10.0, min(estimated_distance, 100.0))
    
    def _is_road_clear(self, semantic_image: np.ndarray, depth_image: Optional[np.ndarray],
                              obstacles: List[DetectedObject], min_distance: float = 20.0) -> bool:
        """Road clearance check using detected objects"""
        # Check detected obstacles first
        for obstacle in obstacles:
            if (obstacle.lane_assignment == 0 and  # Same lane
                obstacle.distance < min_distance and
                obstacle.confidence > 0.5):
                return False
        
        # Fallback to semantic + depth analysis
        if depth_image is None:
            return True
        
        # Check center region ahead
        h, w = semantic_image.shape
        roi_y1, roi_y2 = h//2, h*3//4
        roi_x1, roi_x2 = w//3, w*2//3
        
        roi_semantic = semantic_image[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_depth = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Find non-road pixels at close distance
        road_mask = (roi_semantic == self.CLASS_TO_ID.get('road', 7))
        close_objects = (roi_depth < min_distance) & (roi_depth > 1.0) & ~road_mask
        
        # If more than 5% of ROI has close non-road objects, consider blocked
        blocked_ratio = np.sum(close_objects) / close_objects.size
        return blocked_ratio < 0.05
    
    def _detect_intersection(self, semantic_image: np.ndarray, 
                                    traffic_lights: List[TrafficLightInfo]) -> bool:
        """Intersection detection"""
        # Check for traffic lights
        if any(tl.relevant_for_ego and tl.distance < 50.0 for tl in traffic_lights):
            return True
        
        # Check for traffic signs
        has_traffic_sign = np.any(semantic_image == self.CLASS_TO_ID.get('traffic_sign', 12))
        if has_traffic_sign:
            return True
        
        # Check road topology (simplified)
        # Look for significant changes in road area in the forward direction
        road_mask = (semantic_image == self.CLASS_TO_ID.get('road', 7))
        
        # Compare road area in near vs far regions
        h, w = semantic_image.shape
        near_region = road_mask[h*2//3:, :]
        far_region = road_mask[h//3:h*2//3, :]
        
        near_road_ratio = np.sum(near_region) / near_region.size
        far_road_ratio = np.sum(far_region) / far_region.size
        
        # Significant increase in road area might indicate intersection
        return far_road_ratio > near_road_ratio * 1.5
    
    def _assess_sensor_quality(self, rgb_image: np.ndarray, semantic_image: np.ndarray, 
                             depth_image: Optional[np.ndarray]) -> Dict[str, float]:
        """Assess quality of sensor data for confidence estimation"""
        quality = {}
        
        # RGB image quality
        if rgb_image is not None:
            # Check for proper exposure and contrast
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)
            brightness = np.mean(gray)
            
            # Good contrast and reasonable brightness
            quality['rgb'] = min(1.0, contrast / 50.0) * (1.0 - abs(brightness - 128) / 128.0)
        else:
            quality['rgb'] = 0.0
        
        # Semantic image quality
        if semantic_image is not None:
            # Check for variety of classes (good segmentation)
            unique_classes = len(np.unique(semantic_image))
            quality['semantic'] = min(1.0, unique_classes / 10.0)  # Expect ~10 classes
        else:
            quality['semantic'] = 0.0
        
        # Depth image quality
        if depth_image is not None:
            valid_depth_ratio = np.sum((depth_image > 1.0) & (depth_image < 200.0)) / depth_image.size
            quality['depth'] = valid_depth_ratio
        else:
            quality['depth'] = 0.0
        
        return quality
    
    def _detect_speed_limit(self, rgb_image: np.ndarray, semantic_image: np.ndarray) -> Optional[float]:
        """Detect speed limit signs (placeholder for OCR implementation)"""
        # Check for traffic signs in semantic image
        has_signs = np.any(semantic_image == self.CLASS_TO_ID.get('traffic_sign', 12))
        
        if has_signs:
            # In a real implementation, this would use OCR or trained classifiers
            # For now, return a default speed limit
            return 50.0  # km/h
        
        return None
    
    def _empty_perception(self, frame_id: int = 0, timestamp: float = 0.0) -> PerceptionOutput:
        """Return empty perception output with proper initialization"""
        return PerceptionOutput(
            frame_id=frame_id,
            timestamp=timestamp,
            drivable_area=np.zeros((self.image_height, self.image_width), dtype=np.uint8),
            safety_metrics=SafetyMetrics(),
            sensor_quality={'rgb': 0.0, 'semantic': 0.0, 'depth': 0.0}
        )