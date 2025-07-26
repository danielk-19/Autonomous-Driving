"""
Perception system for autonomous driving.
Processes sensor data to understand the environment.
"""

import numpy as np
import cv2
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import sys

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Import utility functions
from utils.utils import (
    setup_logging, Timer, LoggingContext, clamp, normalize_angle,
    calculate_distance, is_collision_imminent, preprocess_image,
    combine_rgb_semantic, safe_float, safe_int, PerformanceMonitor
)

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
    Perception system with improved algorithms, safety focus, and utility integration
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
    
    def __init__(self, image_width: int = 800, image_height: int = 600, 
                 log_level: int = logging.INFO):
        """
        Initialize perception system with utility integration
        """
        # Setup logging using utils
        setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        self.image_width = image_width
        self.image_height = image_height
        
        # Camera calibration parameters (for distance estimation)
        self.camera_matrix = self._get_default_camera_matrix()
        self.focal_length = safe_float(400.0)  # Using utils safe conversion
        self.camera_height = safe_float(2.4)   # Camera height in meters
        
        # Traffic light color detection parameters (improved HSV ranges)
        self.color_ranges = {
            'red': [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                   (np.array([170, 120, 70]), np.array([180, 255, 255]))],
            'yellow': [(np.array([15, 120, 70]), np.array([35, 255, 255]))],
            'green': [(np.array([40, 120, 70]), np.array([80, 255, 255]))]
        }
        
        # Object detection parameters with safe conversions
        self.min_object_area = safe_int(100)
        self.min_vehicle_area = safe_int(500)
        self.min_pedestrian_area = safe_int(200)
        self.max_detection_distance = safe_float(100.0)
        
        # Lane detection parameters
        self.min_lane_pixels = safe_int(50)
        self.roi_height_ratio = safe_float(0.4)
        self.lane_width_pixels = safe_int(100)
        
        # Safety parameters
        self.safe_following_distance = safe_float(15.0)
        self.emergency_brake_distance = safe_float(8.0)
        self.lane_change_clearance = safe_float(20.0)
        
        # Temporal tracking
        self.object_history = {}
        self.next_track_id = 0
        
        # Frame counter for performance monitoring
        self.frame_count = 0
        
        self.logger.info("Perception system initialized")
    
    def _get_default_camera_matrix(self) -> np.ndarray:
        """Default camera matrix for CARLA setup using safe conversions"""
        fx = fy = safe_float(400.0)
        cx, cy = safe_float(self.image_width / 2), safe_float(self.image_height / 2)
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def process_sensors(self, sensor_data: Dict[str, Any], frame_id: int = 0, 
                       timestamp: float = 0.0) -> PerceptionOutput:
        """
        Main processing function with utility integration and performance monitoring
        """
        # Use Timer context manager from utils
        with Timer(f"Perception processing frame {frame_id}"):
            # Update performance monitoring
            self.performance_monitor.log_frame()
            self.frame_count += 1
            
            # Extract and validate sensor data with safe conversions
            rgb_image = sensor_data.get('rgb')
            semantic_image = sensor_data.get('semantic') 
            depth_image = sensor_data.get('depth')
            vehicle_speed = safe_float(sensor_data.get('speed', 0.0))
            gps_data = sensor_data.get('gps', {})
            imu_data = sensor_data.get('imu', {})
            
            if rgb_image is None or semantic_image is None:
                self.logger.warning(f"Missing essential sensor data for frame {frame_id}")
                return self._empty_perception(frame_id, timestamp)
            
            # Preprocess images using utils function
            try:
                # Create combined RGB+semantic visualization for debugging
                if rgb_image is not None and semantic_image is not None:
                    combined_viz = combine_rgb_semantic(rgb_image, semantic_image, alpha=0.7)
                    # This could be saved for debugging if needed
            except Exception as e:
                self.logger.warning(f"Failed to create combined visualization: {e}")
            
            # Assess sensor quality
            sensor_quality = self._assess_sensor_quality(rgb_image, semantic_image, depth_image)
            
            perception = PerceptionOutput(
                frame_id=frame_id,
                timestamp=timestamp,
                sensor_quality=sensor_quality
            )
            
            # Use debug logging context for detailed processing
            with LoggingContext(logging.DEBUG, self.logger.name):
                # Object detection
                all_objects = self._detect_objects(semantic_image, depth_image, rgb_image)
                
                # Temporal tracking
                all_objects = self._update_object_tracking(all_objects)
                
                # Categorize objects with filtering
                perception.detected_objects = all_objects
                perception.vehicles = [obj for obj in all_objects if obj.object_type == ObjectType.VEHICLE]
                perception.pedestrians = [obj for obj in all_objects if obj.object_type == ObjectType.PEDESTRIAN]
                perception.obstacles = [obj for obj in all_objects 
                                      if obj.distance < 50.0 and obj.confidence > 0.5]
                
                self.logger.debug(f"Detected: {len(perception.vehicles)} vehicles, "
                                f"{len(perception.pedestrians)} pedestrians, "
                                f"{len(perception.obstacles)} obstacles")
            
            # Lane detection with processing
            perception.lane_info = self._detect_lanes(semantic_image, rgb_image, depth_image)
            
            # Traffic light detection with multiple lights
            perception.traffic_lights = self._detect_traffic_lights(
                rgb_image, semantic_image, depth_image
            )
            perception.traffic_light = self._get_most_relevant_traffic_light(perception.traffic_lights)
            
            # Road analysis
            perception.drivable_area = self._get_drivable_area(semantic_image)
            perception.road_ahead_clear = self._is_road_clear(
                semantic_image, depth_image, perception.obstacles
            )
            perception.intersection_ahead = self._detect_intersection(
                semantic_image, perception.traffic_lights
            )
            
            # Calculate closest vehicle distance with safety clamping
            if perception.vehicles:
                closest_distance = min(v.distance for v in perception.vehicles)
                perception.closest_vehicle_distance = clamp(closest_distance, 0.0, 1000.0)
            
            # Safety metrics calculation with collision detection
            perception.safety_metrics = self._calculate_safety_metrics(
                perception.vehicles, perception.pedestrians, perception.lane_info, vehicle_speed
            )
            
            # Speed limit detection
            perception.speed_limit = self._detect_speed_limit(rgb_image, semantic_image)
            
            # Log performance metrics periodically
            if self.frame_count % 100 == 0:
                stats = self.performance_monitor.get_stats()
                self.logger.info(f"Perception performance: {stats['avg_fps']:.1f} FPS, "
                               f"avg frame time: {stats['avg_frame_time']*1000:.1f}ms")
            
            return perception
    
    def _detect_objects(self, semantic_image: np.ndarray, depth_image: Optional[np.ndarray],
                       rgb_image: np.ndarray) -> List[DetectedObject]:
        """Object detection with error handling and validation"""
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
                
                try:
                    # Create mask for this object type
                    object_mask = (semantic_image == class_id)
                    
                    if not np.any(object_mask):
                        continue
                    
                    # Extract objects
                    objects = self._extract_objects(
                        object_mask, obj_type, depth_image, rgb_image
                    )
                    detected_objects.extend(objects)
                    
                except Exception as e:
                    self.logger.error(f"Error detecting {obj_type.value} objects: {e}")
                    continue
        
        # Filter by distance and confidence with safe bounds
        valid_objects = []
        for obj in detected_objects:
            if (0.0 <= obj.distance <= self.max_detection_distance and 
                0.0 <= obj.confidence <= 1.0):
                valid_objects.append(obj)
        
        self.logger.debug(f"Detected {len(valid_objects)} valid objects from {len(detected_objects)} raw detections")
        return valid_objects
    
    def _extract_objects(self, mask: np.ndarray, object_type: ObjectType,
                        depth_image: Optional[np.ndarray], 
                        rgb_image: np.ndarray) -> List[DetectedObject]:
        """Object extraction with morphological operations and validation"""
        objects = []
        
        try:
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            
            # Use connected components for object separation
            num_labels, labels = cv2.connectedComponents(mask_cleaned)
            
            for label in range(1, num_labels):  # Skip background
                component_mask = (labels == label)
                area = np.sum(component_mask)
                
                # Filter by minimum area based on object type
                min_area = (self.min_vehicle_area if object_type == ObjectType.VEHICLE 
                           else self.min_pedestrian_area)
                if area < min_area:
                    continue
                
                # Get bounding box with safe indexing
                rows, cols = np.where(component_mask)
                if len(rows) == 0 or len(cols) == 0:
                    continue
                    
                x1, y1 = safe_int(cols.min()), safe_int(rows.min())
                x2, y2 = safe_int(cols.max()), safe_int(rows.max())
                
                # Clamp bounding box to image bounds
                x1 = clamp(x1, 0, self.image_width - 1)
                x2 = clamp(x2, 0, self.image_width - 1)
                y1 = clamp(y1, 0, self.image_height - 1)
                y2 = clamp(y2, 0, self.image_height - 1)
                
                bbox = (x1, y1, x2, y2)
                
                # Validate bounding box
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    continue
                
                # Filter by aspect ratio (basic sanity check)
                aspect_ratio = safe_float(width) / max(safe_float(height), 1.0)
                
                if object_type == ObjectType.VEHICLE:
                    if not (0.3 <= aspect_ratio <= 5.0):
                        continue
                elif object_type == ObjectType.PEDESTRIAN:
                    if not (0.2 <= aspect_ratio <= 2.0):
                        continue
                
                # Distance calculation with error handling
                distance = self._calculate_object_distance(
                    component_mask, depth_image, bbox, object_type
                )
                
                # Skip very distant or very close (likely noise) objects
                if distance < 2.0 or distance > self.max_detection_distance:
                    continue
                
                # Calculate relative position with safe conversions
                center_x = safe_float((x1 + x2) / 2)
                center_y = safe_float((y1 + y2) / 2)
                
                # Convert to real-world coordinates
                lateral_distance = self._pixel_to_lateral_distance(center_x, distance)
                
                # Assign lane based on lateral position
                lane_assignment = self._assign_lane(lateral_distance)
                
                # Confidence calculation
                confidence = self._calculate_object_confidence(
                    component_mask, depth_image, object_type, bbox
                )
                
                # Motion estimation
                is_moving = self._estimate_object_motion(bbox, object_type)
                
                detected_object = DetectedObject(
                    object_type=object_type,
                    bbox=bbox,
                    confidence=clamp(confidence, 0.0, 1.0),
                    distance=clamp(distance, 0.0, self.max_detection_distance),
                    relative_position=(lateral_distance, distance),
                    lane_assignment=lane_assignment,
                    size_2d=(safe_float(width), safe_float(height)),
                    is_moving=is_moving
                )
                
                objects.append(detected_object)
                
        except Exception as e:
            self.logger.error(f"Error in object extraction for {object_type.value}: {e}")
        
        return objects
    
    def _calculate_object_distance(self, mask: np.ndarray, depth_image: Optional[np.ndarray],
                                  bbox: Tuple[int, int, int, int], 
                                  object_type: ObjectType) -> float:
        """Distance calculation with fallbacks and validation"""
        if depth_image is not None:
            try:
                object_depths = depth_image[mask]
                # Filter for valid depth values with more robust bounds
                valid_depths = object_depths[
                    (object_depths > 1.0) & 
                    (object_depths < 200.0) & 
                    np.isfinite(object_depths)
                ]
                
                if len(valid_depths) > 10:
                    # Use percentile instead of median for outlier handling
                    distance = float(np.percentile(valid_depths, 50))  # Median
                    if 2.0 <= distance <= self.max_detection_distance:
                        return distance
            except Exception as e:
                self.logger.debug(f"Depth-based distance calculation failed: {e}")
        
        # Estimate from bounding box size with object-specific parameters
        x1, y1, x2, y2 = bbox
        height = safe_float(y2 - y1)
        
        # Object-specific size assumptions (more realistic)
        if object_type == ObjectType.VEHICLE:
            # Assume vehicle height ~1.6m (average car height)
            estimated_object_height = 1.6
        else:  # Pedestrian
            # Assume pedestrian height ~1.7m
            estimated_object_height = 1.7
        
        # Distance calculation considering camera angle
        if height > 1:
            distance = (estimated_object_height * self.focal_length) / height
            # Apply perspective correction (objects lower in image are closer)
            y_center = safe_float((y1 + y2) / 2)
            perspective_factor = 1.0 + (y_center - self.image_height / 2) / self.image_height * 0.2
            distance *= perspective_factor
        else:
            distance = 50.0  # Default distance for very small objects
        
        # Clamp to reasonable bounds with safety margins
        return clamp(distance, 3.0, 150.0)
    
    def _calculate_safety_metrics(self, vehicles: List[DetectedObject], 
                                 pedestrians: List[DetectedObject],
                                 lane_info: Optional[LaneInfo],
                                 ego_speed: float) -> SafetyMetrics:
        """Safety metrics calculation using utility functions"""
        safety = SafetyMetrics()
        
        # Convert ego vehicle position for collision detection
        ego_position_2d = (self.image_width / 2, self.image_height)  # Bottom center
        
        # Collision risk assessment
        for vehicle in vehicles:
            # Use utility function for collision imminence check
            vehicle_2d_pos = ((vehicle.bbox[0] + vehicle.bbox[2]) / 2, 
                             (vehicle.bbox[1] + vehicle.bbox[3]) / 2)
            
            # Convert to world coordinates (simplified)
            obstacle_locations = [type('Location', (), {'x': vehicle.relative_position[0], 
                                                       'y': vehicle.relative_position[1]})()]
            ego_location = type('Location', (), {'x': 0.0, 'y': 0.0})()
            
            # Check collision imminence using utils function
            if is_collision_imminent(ego_location, obstacle_locations, 
                                   threshold=self.emergency_brake_distance):
                safety.emergency_brake_needed = True
                safety.collision_risk_front = 1.0
            elif vehicle.distance < self.safe_following_distance:
                # Calculate risk with smooth falloff
                risk = 1.0 - (vehicle.distance / self.safe_following_distance)
                safety.collision_risk_front = max(safety.collision_risk_front, clamp(risk, 0.0, 1.0))
            
            # Update following distance for same-lane vehicles
            if vehicle.lane_assignment == 0:  # Same lane
                safety.following_distance = min(safety.following_distance, vehicle.distance)
        
        # Pedestrian risk assessment
        for pedestrian in pedestrians:
            if pedestrian.distance < 10.0:  # Close pedestrian threshold
                risk_level = 1.0 - (pedestrian.distance / 10.0)  # Linear falloff
                risk_level = clamp(risk_level * 0.8, 0.0, 1.0)  # Scale down slightly
                
                if pedestrian.relative_position[0] < 0:  # Left side
                    safety.collision_risk_left = max(safety.collision_risk_left, risk_level)
                else:  # Right side
                    safety.collision_risk_right = max(safety.collision_risk_right, risk_level)
        
        # Time to collision calculation
        front_vehicles = [v for v in vehicles 
                         if v.lane_assignment == 0 and v.distance < 50.0]
        
        if front_vehicles and ego_speed > 1.0:
            closest_vehicle = min(front_vehicles, key=lambda v: v.distance)
            # Assume relative speed (in real system, this would be tracked)
            relative_speed = max(ego_speed - closest_vehicle.relative_speed, 0.1)
            if relative_speed > 0:
                safety.time_to_collision = clamp(
                    closest_vehicle.distance / relative_speed, 
                    0.0, 30.0  # Cap at 30 seconds
                )
        
        # Lane change safety with distance calculations
        left_vehicles = [v for v in vehicles if v.lane_assignment == -1]
        right_vehicles = [v for v in vehicles if v.lane_assignment == 1]
        
        # More sophisticated lane change safety considering relative speeds
        safety.safe_to_change_left = all(
            v.distance > self.lane_change_clearance or 
            (v.distance > 10.0 and abs(v.relative_speed - ego_speed) < 10.0)
            for v in left_vehicles
        )
        
        safety.safe_to_change_right = all(
            v.distance > self.lane_change_clearance or 
            (v.distance > 10.0 and abs(v.relative_speed - ego_speed) < 10.0)
            for v in right_vehicles
        )
        
        # Clamp all risk values to valid range
        safety.collision_risk_front = clamp(safety.collision_risk_front, 0.0, 1.0)
        safety.collision_risk_left = clamp(safety.collision_risk_left, 0.0, 1.0)
        safety.collision_risk_right = clamp(safety.collision_risk_right, 0.0, 1.0)
        
        return safety
    
    def _pixel_to_lateral_distance(self, pixel_x: float, distance: float) -> float:
        """Pixel to lateral distance conversion with clamping"""
        try:
            # Simple pinhole camera model
            lateral_angle = (pixel_x - self.image_width / 2) / self.focal_length
            lateral_distance = distance * np.tan(lateral_angle)
            # Clamp to reasonable lateral bounds (±20m from vehicle center)
            return clamp(float(lateral_distance), -20.0, 20.0)
        except Exception as e:
            self.logger.debug(f"Lateral distance calculation failed: {e}")
            return 0.0
    
    def _assign_lane(self, lateral_distance: float) -> int:
        """Lane assignment with configurable lane width"""
        # Use more precise lane width (3.7m is US standard)
        lane_half_width = 1.85  # meters
        
        lateral_distance = safe_float(lateral_distance)
        
        if lateral_distance < -lane_half_width:
            return -1  # Left lane
        elif lateral_distance > lane_half_width:
            return 1   # Right lane
        else:
            return 0   # Same lane
    
    def _detect_lanes(self, semantic_image: np.ndarray, 
                     rgb_image: np.ndarray,
                     depth_image: Optional[np.ndarray]) -> Optional[LaneInfo]:
        """Lane detection with error handling"""
        try:
            # Get road and roadline masks with safe class ID lookup
            road_class_id = self.CLASS_TO_ID.get('road', 7)
            roadline_class_id = self.CLASS_TO_ID.get('roadline', 6)
            
            road_mask = (semantic_image == road_class_id)
            roadline_mask = (semantic_image == roadline_class_id)
            
            if not np.any(roadline_mask):
                self.logger.debug("No roadline pixels detected")
                return None
            
            # ROI selection
            roi_mask = self._create_lane_roi_mask()
            roadline_roi = roadline_mask & roi_mask
            
            if np.sum(roadline_roi) < self.min_lane_pixels:
                self.logger.debug(f"Insufficient lane pixels: {np.sum(roadline_roi)} < {self.min_lane_pixels}")
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
                
                # Heading angle calculation with angle normalization
                raw_heading = self._calculate_heading_angle(lane_info.center_line)
                lane_info.heading_angle = normalize_angle(raw_heading)
                
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
            
        except Exception as e:
            self.logger.error(f"Lane detection failed: {e}")
            return None
    
    def _detect_traffic_lights(self, rgb_image: np.ndarray, semantic_image: np.ndarray,
                              depth_image: Optional[np.ndarray]) -> List[TrafficLightInfo]:
        """Traffic light detection with error handling"""
        traffic_lights = []
        
        try:
            # Find traffic light regions with safe class lookup
            traffic_light_class_id = self.CLASS_TO_ID.get('traffic_light', 18)
            traffic_light_mask = (semantic_image == traffic_light_class_id)
            
            if not np.any(traffic_light_mask):
                return traffic_lights
            
            # Get connected components for multiple traffic lights
            mask_uint8 = traffic_light_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                try:
                    area = cv2.contourArea(contour)
                    if area < 100:  # Minimum size threshold
                        continue
                        
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Clamp bounding box to image bounds
                    x = clamp(x, 0, rgb_image.shape[1] - 1)
                    y = clamp(y, 0, rgb_image.shape[0] - 1)
                    w = clamp(w, 1, rgb_image.shape[1] - x)
                    h = clamp(h, 1, rgb_image.shape[0] - y)
                    
                    bbox = (x, y, x + w, y + h)
                    
                    # ROI extraction with safe padding
                    padding = 5
                    x_start = clamp(x - padding, 0, rgb_image.shape[1])
                    y_start = clamp(y - padding, 0, rgb_image.shape[0])
                    x_end = clamp(x + w + padding, 0, rgb_image.shape[1])
                    y_end = clamp(y + h + padding, 0, rgb_image.shape[0])
                    
                    if x_end <= x_start or y_end <= y_start:
                        continue
                        
                    tl_roi = rgb_image[y_start:y_end, x_start:x_end]
                    
                    if tl_roi.size == 0:
                        continue
                    
                    # Color classification
                    state, color_confidence = self._classify_traffic_light_color(tl_roi)
                    
                    if color_confidence < 0.3:  # Skip low confidence detections
                        continue
                    
                    # Distance calculation
                    distance = self._calculate_traffic_light_distance(x, y, w, h, depth_image)
                    distance = clamp(distance, 5.0, 200.0)  # Reasonable bounds
                    
                    # Determine relevance for ego vehicle
                    relevant_for_ego = self._is_traffic_light_relevant(x, y, w, h, distance)
                    
                    traffic_light_info = TrafficLightInfo(
                        state=state,
                        distance=distance,
                        confidence=clamp(color_confidence, 0.0, 1.0),
                        bbox=bbox,
                        relevant_for_ego=relevant_for_ego
                    )
                    
                    traffic_lights.append(traffic_light_info)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing traffic light contour: {e}")
                    continue
            
            # Sort by distance (closest first)
            traffic_lights.sort(key=lambda tl: tl.distance)
            
        except Exception as e:
            self.logger.error(f"Traffic light detection failed: {e}")
        
        return traffic_lights
    
    def _get_drivable_area(self, semantic_image: np.ndarray) -> Optional[np.ndarray]:
        """Drivable area detection with error handling"""
        try:
            # Get road class ID safely
            road_class_id = self.CLASS_TO_ID.get('road', 7)
            drivable_mask = (semantic_image == road_class_id)
            
            # Apply morphological operations for cleaner result
            kernel = np.ones((5, 5), np.uint8)
            drivable_area = cv2.morphologyEx(
                drivable_mask.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                kernel
            )
            
            return drivable_area
            
        except Exception as e:
            self.logger.error(f"Drivable area detection failed: {e}")
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
    
    def _create_lane_roi_mask(self) -> np.ndarray:
        """ROI mask creation with safer bounds"""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        
        try:
            # Trapezoidal ROI considering perspective with safe conversions
            bottom_width = self.image_width
            top_width = safe_int(self.image_width * 0.6)
            height_start = safe_int(self.image_height * 0.4)
            
            for y in range(height_start, self.image_height):
                progress = safe_float(y - height_start) / max(self.image_height - height_start, 1)
                width = safe_int(top_width + (bottom_width - top_width) * progress)
                x_start = clamp((self.image_width - width) // 2, 0, self.image_width)
                x_end = clamp(x_start + width, 0, self.image_width)
                
                if x_end > x_start:
                    mask[y, x_start:x_end] = True
                    
        except Exception as e:
            self.logger.error(f"ROI mask creation failed: {e}")
        
        return mask
    
    def _extract_lane_boundaries(self, roadline_roi: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], 
        Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Lane boundary extraction with error handling"""
        try:
            rows, cols = np.where(roadline_roi)
            
            if len(cols) < self.min_lane_pixels:
                return None, None, None, None
            
            # Use image center for initial separation with safe conversion
            img_center = safe_int(roadline_roi.shape[1] // 2)
            
            # Separate left and right with some overlap handling
            left_mask = cols < img_center + 50  # Allow some overlap
            right_mask = cols > img_center - 50
            
            left_boundary = None
            right_boundary = None
            left_pixels = None
            right_pixels = None
            
            # Fit polynomials to lane boundaries with validation
            if np.sum(left_mask) > 20:
                left_points = np.column_stack((cols[left_mask], rows[left_mask]))
                left_pixels = left_points
                left_boundary = self._fit_lane_polynomial(left_points)
            
            if np.sum(right_mask) > 20:
                right_points = np.column_stack((cols[right_mask], rows[right_mask]))
                right_pixels = right_points
                right_boundary = self._fit_lane_polynomial(right_points)
            
            return left_boundary, right_boundary, left_pixels, right_pixels
            
        except Exception as e:
            self.logger.error(f"Lane boundary extraction failed: {e}")
            return None, None, None, None
    
    def _fit_lane_polynomial(self, points: np.ndarray, degree: int = 2) -> Optional[np.ndarray]:
        """Polynomial fitting with validation"""
        if len(points) < degree + 1:
            return None
        
        try:
            best_poly = None
            best_score = 0
            
            # Multiple attempts for robust fitting with scoring
            attempts = min(10, len(points) // 10)  # Adaptive number of attempts
            
            for attempt in range(max(attempts, 3)):
                # Sample subset of points
                n_sample = clamp(
                    max(50, len(points) // 3), 
                    degree + 1, 
                    len(points)
                )
                
                if n_sample >= len(points):
                    sample_points = points
                else:
                    sample_indices = np.random.choice(len(points), n_sample, replace=False)
                    sample_points = points[sample_indices]
                
                # Fit polynomial with error handling
                try:
                    poly = np.polyfit(sample_points[:, 1], sample_points[:, 0], degree)
                    
                    # Quality evaluation
                    predicted_x = np.polyval(poly, points[:, 1])
                    errors = np.abs(predicted_x - points[:, 0])
                    
                    # Use multiple thresholds for assessment
                    inliers_strict = np.sum(errors < 5.0)   # 5 pixel threshold
                    inliers_loose = np.sum(errors < 15.0)   # 15 pixel threshold
                    
                    valid_errors = errors[errors < 15.0]
                    avg_error = np.mean(valid_errors) if len(valid_errors) > 0 else float('inf')
                    
                    # Scoring considering multiple factors
                    score = (inliers_strict * 2.0 + inliers_loose * 1.0 - 
                            avg_error * 0.2 - np.std(errors) * 0.1)
                    
                    if score > best_score and inliers_strict > len(points) * 0.3:
                        best_score = score
                        best_poly = poly
                        
                except np.linalg.LinAlgError:
                    continue
                except Exception as e:
                    self.logger.debug(f"Polynomial fitting attempt {attempt} failed: {e}")
                    continue
            
            return best_poly if best_score > 0 else None
            
        except Exception as e:
            self.logger.error(f"Polynomial fitting failed: {e}")
            return None
    
    def _calculate_lane_width(self, left_boundary: Optional[np.ndarray], 
                             right_boundary: Optional[np.ndarray],
                             depth_image: Optional[np.ndarray]) -> float:
        """Lane width calculation with error handling"""
        if left_boundary is None or right_boundary is None:
            return 3.7  # Standard lane width
        
        try:
            # Sample multiple y positions for robust measurement
            y_start = safe_float(self.image_height * 0.6)
            y_end = safe_float(self.image_height - 20)
            y_positions = np.linspace(y_start, y_end, 10)
            
            widths = []
            
            for y in y_positions:
                try:
                    left_x = np.polyval(left_boundary, y)
                    right_x = np.polyval(right_boundary, y)
                    width_pixels = abs(right_x - left_x)
                    
                    # Distance-to-meters conversion
                    if depth_image is not None and 0 <= int(y) < depth_image.shape[0]:
                        # Sample depth at lane boundaries with bounds checking
                        left_x_int = clamp(int(left_x), 0, self.image_width - 1)
                        right_x_int = clamp(int(right_x), 0, self.image_width - 1)
                        y_int = clamp(int(y), 0, self.image_height - 1)
                        
                        left_depth = safe_float(depth_image[y_int, left_x_int])
                        right_depth = safe_float(depth_image[y_int, right_x_int])
                        
                        if left_depth > 0 and right_depth > 0:
                            avg_depth = (left_depth + right_depth) / 2.0
                            
                            # Pixel-to-meter conversion
                            meters_per_pixel = avg_depth / self.focal_length
                            width_meters = width_pixels * meters_per_pixel
                            
                            # Sanity check for reasonable lane width
                            if 2.0 <= width_meters <= 6.0:
                                widths.append(width_meters)
                    else:
                        # Fallback approximation
                        estimated_depth = 50.0 - (y - y_start) * 0.5
                        meters_per_pixel = estimated_depth / self.focal_length
                        width_meters = width_pixels * meters_per_pixel * 0.01
                        
                        if 2.0 <= width_meters <= 6.0:
                            widths.append(width_meters)
                            
                except Exception as e:
                    self.logger.debug(f"Width calculation failed for y={y}: {e}")
                    continue
            
            if widths:
                # Use median for robustness and clamp to reasonable bounds
                median_width = safe_float(np.median(widths))
                return clamp(median_width, 2.5, 5.0)
            
            return 3.7  # Default standard lane width
            
        except Exception as e:
            self.logger.error(f"Lane width calculation failed: {e}")
            return 3.7
    
    def _detect_lane_departure(self, center_line: Optional[np.ndarray], 
                              lane_width: float) -> Tuple[bool, bool, float]:
        """Lane departure detection with error handling"""
        if center_line is None:
            return False, False, 0.0
        
        try:
            # Check vehicle position relative to lane center at bottom of image
            y_check = safe_float(self.image_height - 50)
            
            lane_center_x = np.polyval(center_line, y_check)
            vehicle_center_x = safe_float(self.image_width / 2)
            
            # Convert pixel offset to meters
            pixel_offset = vehicle_center_x - lane_center_x
            
            # Pixel-to-meter conversion (rough approximation)
            # This should be calibrated based on camera parameters
            estimated_distance = 10.0  # Assume 10m ahead for ground plane
            meters_per_pixel = estimated_distance / self.focal_length
            meter_offset = pixel_offset * meters_per_pixel * 0.02  # Calibration factor
            
            # Clamp offset to reasonable bounds
            meter_offset = clamp(meter_offset, -10.0, 10.0)
            
            # Departure thresholds with safety margin
            departure_threshold = clamp((lane_width / 2) - 0.5, 1.0, 3.0)  # 0.5m safety margin
            
            left_departure = meter_offset < -departure_threshold
            right_departure = meter_offset > departure_threshold
            
            return left_departure, right_departure, meter_offset
            
        except Exception as e:
            self.logger.error(f"Lane departure detection failed: {e}")
            return False, False, 0.0
    
    def _classify_traffic_light_color(self, tl_roi: np.ndarray) -> Tuple[TrafficLightState, float]:
        """Traffic light color classification with preprocessing"""
        if tl_roi.size == 0 or tl_roi.shape[0] < 10 or tl_roi.shape[1] < 10:
            return TrafficLightState.UNKNOWN, 0.0
        
        try:
            # Preprocessing for color detection
            # Apply bilateral filter to reduce noise while preserving edges
            if len(tl_roi.shape) == 3:
                filtered = cv2.bilateralFilter(tl_roi, 9, 75, 75)
            else:
                filtered = tl_roi
            
            # Convert to HSV with error handling
            try:
                if len(filtered.shape) == 3 and filtered.shape[2] == 3:
                    hsv = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)
                else:
                    return TrafficLightState.UNKNOWN, 0.0
            except cv2.error:
                return TrafficLightState.UNKNOWN, 0.0
            
            # Color detection with multiple metrics
            color_scores = {}
            color_pixels = {}
            color_intensities = {}
            
            for color_name, ranges in self.color_ranges.items():
                total_pixels = 0
                total_intensity = 0
                max_intensity = 0
                
                for lower, upper in ranges:
                    try:
                        mask = cv2.inRange(hsv, lower, upper)
                        pixel_count = np.sum(mask > 0)
                        total_pixels += pixel_count
                        
                        # Intensity calculation
                        if pixel_count > 0:
                            intensities = hsv[mask > 0, 2]  # V channel (brightness)
                            intensity_mean = np.mean(intensities)
                            intensity_max = np.max(intensities)
                            
                            total_intensity += intensity_mean * pixel_count
                            max_intensity = max(max_intensity, intensity_max)
                            
                    except Exception as e:
                        self.logger.debug(f"Color range processing failed for {color_name}: {e}")
                        continue
                
                color_pixels[color_name] = total_pixels
                color_intensities[color_name] = max_intensity
                
                if total_pixels > 0:
                    color_scores[color_name] = total_intensity / total_pixels
                else:
                    color_scores[color_name] = 0
            
            # Color selection with multiple criteria
            best_color = None
            best_combined_score = 0
            
            min_pixels_threshold = max(20, tl_roi.size * 0.05)  # Adaptive threshold
            
            for color_name in color_scores:
                if color_pixels[color_name] < min_pixels_threshold:
                    continue
                
                # Combined scoring
                pixel_ratio = color_pixels[color_name] / max(tl_roi.size, 1)
                intensity_score = color_scores[color_name] / 255.0
                brightness_boost = color_intensities[color_name] / 255.0
                
                combined_score = (pixel_ratio * 0.4 + 
                                intensity_score * 0.4 + 
                                brightness_boost * 0.2)
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_color = color_name
            
            if best_color is None:
                return TrafficLightState.UNKNOWN, 0.0
            
            # Confidence calculation
            total_colored_pixels = sum(color_pixels.values())
            if total_colored_pixels == 0:
                return TrafficLightState.UNKNOWN, 0.0
            
            pixel_confidence = color_pixels[best_color] / max(total_colored_pixels, 1)
            intensity_confidence = color_scores[best_color] / 255.0
            
            # Combined confidence with clamping
            confidence = clamp((pixel_confidence + intensity_confidence) / 2.0 * 1.3, 0.0, 1.0)
            
            # State mapping
            state_mapping = {
                'red': TrafficLightState.RED,
                'yellow': TrafficLightState.YELLOW,
                'green': TrafficLightState.GREEN
            }
            
            final_state = state_mapping.get(best_color, TrafficLightState.UNKNOWN)
            
            return final_state, confidence
            
        except Exception as e:
            self.logger.error(f"Traffic light color classification failed: {e}")
            return TrafficLightState.UNKNOWN, 0.0
    
    def _is_traffic_light_relevant(self, x: int, y: int, w: int, h: int, distance: float) -> bool:
        """Traffic light relevance determination"""
        try:
            # Center calculation with safe conversions
            center_x = safe_float(x + w / 2)
            center_y = safe_float(y + h / 2)
            
            # Must be in central portion of image (not too far left/right)
            lateral_threshold = 0.25  # 25% from edges
            if (center_x < self.image_width * lateral_threshold or 
                center_x > self.image_width * (1 - lateral_threshold)):
                return False
            
            # Distance validation
            if not (5.0 <= distance <= 100.0):
                return False
            
            # Must be in upper portion of image (traffic lights are mounted high)
            if center_y > self.image_height * 0.7:
                return False
            
            # Additional size-based validation
            min_size = 10  # Minimum pixel size
            max_size = min(self.image_width, self.image_height) * 0.3  # Max 30% of image
            
            if w < min_size or h < min_size or w > max_size or h > max_size:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Traffic light relevance check failed: {e}")
            return False
    
    def _get_most_relevant_traffic_light(self, traffic_lights: List[TrafficLightInfo]) -> Optional[TrafficLightInfo]:
        """Traffic light selection with prioritization"""
        try:
            relevant_lights = [tl for tl in traffic_lights if tl.relevant_for_ego]
            
            if not relevant_lights:
                return None
            
            # Sorting with multiple criteria
            def relevance_score(tl: TrafficLightInfo) -> Tuple[float, float, float]:
                # Primary: distance (closer is better)
                distance_score = 1.0 / max(tl.distance, 0.1)
                # Secondary: confidence (higher is better)  
                confidence_score = tl.confidence
                # Tertiary: state priority (red > yellow > green > unknown)
                state_priority = {
                    TrafficLightState.RED: 3.0,
                    TrafficLightState.YELLOW: 2.0,
                    TrafficLightState.GREEN: 1.0,
                    TrafficLightState.UNKNOWN: 0.0
                }.get(tl.state, 0.0)
                
                return (state_priority, confidence_score, distance_score)
            
            # Sort by relevance score (descending)
            relevant_lights.sort(key=relevance_score, reverse=True)
            return relevant_lights[0]
            
        except Exception as e:
            self.logger.error(f"Traffic light selection failed: {e}")
            return None
    
    def _calculate_object_confidence(self, mask: np.ndarray, depth_image: Optional[np.ndarray], 
                                    object_type: ObjectType, bbox: Tuple[int, int, int, int]) -> float:
        """Confidence calculation with multiple validation factors"""
        try:
            base_confidence = 0.7
            
            # Size-based adjustment
            area = np.sum(mask)
            x1, y1, x2, y2 = bbox
            
            # Object-specific size validation
            if object_type == ObjectType.VEHICLE:
                if 500 <= area <= 50000:
                    base_confidence += 0.15
                elif area < 300:
                    base_confidence -= 0.3
                elif area > 80000:  # Too large, likely noise
                    base_confidence -= 0.4
            elif object_type == ObjectType.PEDESTRIAN:
                if 200 <= area <= 5000:
                    base_confidence += 0.15
                elif area < 150:
                    base_confidence -= 0.3
                elif area > 10000:  # Too large for pedestrian
                    base_confidence -= 0.4
            
            # Aspect ratio validation
            width, height = safe_float(x2 - x1), safe_float(y2 - y1)
            if width > 0 and height > 0:
                aspect_ratio = width / height
                
                if object_type == ObjectType.VEHICLE:
                    # Vehicles: wider range for different orientations
                    if 0.5 <= aspect_ratio <= 4.0:
                        base_confidence += 0.1
                    else:
                        base_confidence -= 0.2
                elif object_type == ObjectType.PEDESTRIAN:
                    # Pedestrians: typically taller than wide
                    if 0.3 <= aspect_ratio <= 1.5:
                        base_confidence += 0.1
                    else:
                        base_confidence -= 0.2
            
            # Depth consistency check
            if depth_image is not None:
                try:
                    depths = depth_image[mask]
                    valid_depths = depths[(depths > 1.0) & (depths < 200.0) & np.isfinite(depths)]
                    
                    if len(valid_depths) > 10:
                        depth_std = np.std(valid_depths)
                        depth_mean = np.mean(valid_depths)
                        
                        # Consistent depth boosts confidence
                        if depth_std < 2.0:
                            base_confidence += 0.2
                        elif depth_std < 5.0:
                            base_confidence += 0.1
                        else:
                            base_confidence -= 0.1
                        
                        # Reasonable depth range
                        if 3.0 <= depth_mean <= 100.0:
                            base_confidence += 0.05
                        
                except Exception as e:
                    self.logger.debug(f"Depth consistency check failed: {e}")
            
            # Position-based adjustment
            center_x = (x1 + x2) / 2
            distance_from_center = abs(center_x - self.image_width / 2) / (self.image_width / 2)
            
            # Objects in center are more likely to be relevant
            if distance_from_center < 0.2:
                base_confidence += 0.1
            elif distance_from_center < 0.4:
                base_confidence += 0.05
            
            # Vertical position consideration (objects on ground more likely valid)
            center_y = (y1 + y2) / 2
            if center_y > self.image_height * 0.3:  # Lower in image
                base_confidence += 0.05
            
            # Clamp final confidence to valid range
            return clamp(base_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Object confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence
    
    def _estimate_object_motion(self, bbox: Tuple[int, int, int, int], 
                               object_type: ObjectType) -> bool:
        """Motion estimation with simple tracking history"""
        try:
            # This is a placeholder for motion estimation
            # In a real implementation, this would use proper object tracking
            
            current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Check if we have history for similar objects
            # This is a simplified approach - real implementation would use proper tracking
            
            if object_type == ObjectType.VEHICLE:
                # Vehicles are more likely to be moving, especially if detected consistently
                return True
            elif object_type == ObjectType.PEDESTRIAN:
                # Pedestrians movement depends on location and context
                # Near sidewalks less likely to be moving into road
                center_x = current_center[0]
                
                # If pedestrian is in center lanes, more likely moving (crossing)
                if 0.3 * self.image_width <= center_x <= 0.7 * self.image_width:
                    return True
                else:
                    return False  # Likely on sidewalk
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Motion estimation failed: {e}")
            return False
    
    def _update_object_tracking(self, detected_objects: List[DetectedObject]) -> List[DetectedObject]:
        """Object tracking with association and cleanup"""
        try:
            # Tracking with distance calculation
            for obj in detected_objects:
                min_distance = float('inf')
                best_track_id = None
                
                # Current object center
                curr_center = ((obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2)
                
                # Find best matching previous track
                for track_id, prev_obj in self.object_history.items():
                    # Only match same object types
                    if prev_obj.object_type != obj.object_type:
                        continue
                    
                    # Previous object center
                    prev_center = ((prev_obj.bbox[0] + prev_obj.bbox[2]) / 2, 
                                  (prev_obj.bbox[1] + prev_obj.bbox[3]) / 2)
                    
                    # Distance calculation
                    pixel_distance = calculate_distance(curr_center, prev_center)
                    
                    # Size consistency check
                    curr_size = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
                    prev_size = (prev_obj.bbox[2] - prev_obj.bbox[0]) * (prev_obj.bbox[3] - prev_obj.bbox[1])
                    size_ratio = min(curr_size, prev_size) / max(curr_size, prev_size, 1)
                    
                    # Combined tracking score
                    if pixel_distance < 150 and size_ratio > 0.5:  # Reasonable thresholds
                        tracking_score = pixel_distance / (size_ratio + 0.1)  # Lower is better
                        
                        if tracking_score < min_distance:
                            min_distance = tracking_score
                            best_track_id = track_id
                
                # Assign track ID
                if best_track_id is not None:
                    obj.track_id = best_track_id
                    
                    # Relative speed estimation
                    prev_obj = self.object_history[best_track_id]
                    if hasattr(prev_obj, 'distance') and hasattr(prev_obj, 'relative_speed'):
                        # Simple speed estimation based on distance change
                        distance_change = prev_obj.distance - obj.distance
                        # Smooth with previous estimate (simple low-pass filter)
                        obj.relative_speed = prev_obj.relative_speed * 0.7 + distance_change * 0.3
                        obj.relative_speed = clamp(obj.relative_speed, -50.0, 50.0)  # Reasonable speed bounds
                else:
                    # New object
                    obj.track_id = self.next_track_id
                    self.next_track_id += 1
                    obj.relative_speed = 0.0  # Unknown initial speed
                
                # Update history with current object
                self.object_history[obj.track_id] = obj
            
            # History cleanup
            if len(self.object_history) > 100:  # Prevent memory buildup
                # Keep only recent tracks and current detections
                current_tracks = {obj.track_id for obj in detected_objects}
                # Also keep some recent tracks for continuity
                all_track_ids = list(self.object_history.keys())
                recent_tracks = set(all_track_ids[-50:])  # Keep last 50 tracks
                
                tracks_to_keep = current_tracks.union(recent_tracks)
                self.object_history = {
                    tid: obj for tid, obj in self.object_history.items() 
                    if tid in tracks_to_keep
                }
            
            return detected_objects
            
        except Exception as e:
            self.logger.error(f"Object tracking update failed: {e}")
            return detected_objects
    
    def _calculate_curvature(self, left_boundary: Optional[np.ndarray], 
                            right_boundary: Optional[np.ndarray]) -> float:
        """Curvature calculation with error handling"""
        try:
            curvatures = []
            
            # Curvature calculation for polynomial curves
            evaluation_points = [
                self.image_height - 50,   # Near field
                self.image_height - 100,  # Medium field
                self.image_height - 150   # Far field (if available)
            ]
            
            for boundary, boundary_name in [(left_boundary, "left"), (right_boundary, "right")]:
                if boundary is not None and len(boundary) >= 3:
                    for y in evaluation_points:
                        if y < 0:
                            continue
                            
                        try:
                            # For polynomial ax^2 + bx + c, curvature = |2a| / (1 + (2ax + b)^2)^1.5
                            a, b = safe_float(boundary[0]), safe_float(boundary[1])
                            
                            # Calculate derivative at point
                            derivative = 2 * a * y + b
                            denominator = (1 + derivative**2)**1.5
                            
                            if denominator > 1e-6:  # Avoid division by very small numbers
                                curvature = abs(2 * a) / denominator
                                # Clamp curvature to reasonable bounds
                                curvature = clamp(curvature, 0.0, 0.1)  # Max curvature limit
                                curvatures.append(curvature)
                                
                        except Exception as e:
                            self.logger.debug(f"Curvature calculation failed for {boundary_name} at y={y}: {e}")
                            continue
            
            if curvatures:
                # Use median for robustness
                return safe_float(np.median(curvatures))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Curvature calculation failed: {e}")
            return 0.0
    
    def _calculate_center_line(self, left_boundary: Optional[np.ndarray], 
                              right_boundary: Optional[np.ndarray],
                              lane_width: float) -> Optional[np.ndarray]:
        """Center line calculation with single boundary handling"""
        try:
            if left_boundary is not None and right_boundary is not None:
                # Both boundaries available - simple average
                return (left_boundary + right_boundary) / 2
                
            elif left_boundary is not None:
                # Only left boundary - estimate center using lane width
                lane_width_pixels = safe_float(lane_width * 30)  # Rough pixel conversion
                estimated_center = left_boundary.copy()
                # Shift right by half lane width (adjust constant term)
                estimated_center[-1] += lane_width_pixels / 2
                return estimated_center
                
            elif right_boundary is not None:
                # Only right boundary - estimate center using lane width  
                lane_width_pixels = safe_float(lane_width * 30)
                estimated_center = right_boundary.copy()
                # Shift left by half lane width
                estimated_center[-1] -= lane_width_pixels / 2
                return estimated_center
            
            return None
            
        except Exception as e:
            self.logger.error(f"Center line calculation failed: {e}")
            return None
    
    def _calculate_heading_angle(self, center_line: Optional[np.ndarray]) -> float:
        """Heading angle calculation with multiple point averaging"""
        if center_line is None or len(center_line) < 2:
            return 0.0
        
        try:
            # Calculate angle at multiple points and average for stability
            y_positions = [
                self.image_height - 30,   # Very near
                self.image_height - 60,   # Near  
                self.image_height - 100,  # Medium
                self.image_height - 140   # Far (if available)
            ]
            
            angles = []
            
            for y in y_positions:
                if y < 0:
                    continue
                    
                try:
                    # For polynomial ax^2 + bx + c, derivative is 2ax + b
                    a, b = safe_float(center_line[0]), safe_float(center_line[1])
                    slope = 2 * a * y + b
                    
                    # Convert slope to angle (radians)
                    angle = np.arctan(slope)
                    
                    # Clamp angle to reasonable bounds  
                    angle = clamp(angle, -np.pi/4, np.pi/4)  # ±45 degrees max
                    angles.append(angle)
                    
                except Exception as e:
                    self.logger.debug(f"Heading angle calculation failed at y={y}: {e}")
                    continue
            
            if angles:
                # Weighted average (closer points have more weight)
                weights = np.array([1.0, 0.8, 0.6, 0.4][:len(angles)])
                weighted_angle = np.average(angles, weights=weights)
                return safe_float(weighted_angle)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Heading angle calculation failed: {e}")
            return 0.0
    
    def _classify_lane_types(self, rgb_image: np.ndarray, 
                            left_boundary: Optional[np.ndarray],
                            right_boundary: Optional[np.ndarray],
                            left_pixels: Optional[np.ndarray],
                            right_pixels: Optional[np.ndarray]) -> Tuple[LaneType, LaneType]:
        """Lane type classification with color analysis"""
        try:
            left_type = LaneType.UNKNOWN
            right_type = LaneType.UNKNOWN
            
            # Analyze left boundary
            if left_pixels is not None and len(left_pixels) > 10:
                left_type = self._analyze_lane_marking_type(rgb_image, left_pixels)
            
            # Analyze right boundary
            if right_pixels is not None and len(right_pixels) > 10:
                right_type = self._analyze_lane_marking_type(rgb_image, right_pixels)
            
            return left_type, right_type
            
        except Exception as e:
            self.logger.error(f"Lane type classification failed: {e}")
            return LaneType.UNKNOWN, LaneType.UNKNOWN
    
    def _analyze_lane_marking_type(self, rgb_image: np.ndarray, pixels: np.ndarray) -> LaneType:
        """Lane marking analysis with color classification"""
        if len(pixels) < 10:
            return LaneType.UNKNOWN
        
        try:
            # Sample colors at lane marking pixels with bounds checking
            colors = []
            sample_size = min(100, len(pixels))  # Sample up to 100 pixels
            
            # Use stratified sampling
            indices = np.linspace(0, len(pixels)-1, sample_size, dtype=int)
            
            for idx in indices:
                pixel = pixels[idx]
                x, y = int(pixel[0]), int(pixel[1])
                
                # Bounds checking
                if (0 <= x < rgb_image.shape[1] and 
                    0 <= y < rgb_image.shape[0]):
                    colors.append(rgb_image[y, x])
            
            if not colors:
                return LaneType.UNKNOWN
            
            colors = np.array(colors)
            
            # Color analysis
            # Convert to HSV for color classification
            try:
                # Reshape for cv2 conversion
                colors_reshaped = colors.reshape(-1, 1, 3).astype(np.uint8)
                hsv_colors = cv2.cvtColor(colors_reshaped, cv2.COLOR_RGB2HSV)
                hsv_colors = hsv_colors.reshape(-1, 3)
                
                # Analyze color distribution
                avg_hsv = np.mean(hsv_colors, axis=0)
                hue_std = np.std(hsv_colors[:, 0])
                
                # Color classification logic
                avg_hue = safe_float(avg_hsv[0])
                avg_sat = safe_float(avg_hsv[1])
                avg_val = safe_float(avg_hsv[2])
                
                # Yellow detection
                if (15 <= avg_hue <= 35 and avg_sat > 80 and avg_val > 100):
                    # Check for dashed pattern (simplified - would need temporal analysis)
                    # For now, assume most yellow lines are solid center lines
                    return LaneType.SOLID_YELLOW
                
                # White detection
                elif (avg_sat < 50 and avg_val > 150):  # Low saturation, high brightness
                    # Simple dashed vs solid classification based on pixel density
                    # This is a rough approximation - real implementation would analyze gaps
                    pixel_density = len(colors) / max(len(pixels), 1)
                    
                    if pixel_density > 0.7:  # Dense pixels = likely solid
                        return LaneType.SOLID_WHITE
                    else:  # Sparse pixels = likely dashed
                        return LaneType.DASHED_WHITE
                
                # Default to dashed white for unclassified but visible markings
                else:
                    return LaneType.DASHED_WHITE
                    
            except Exception as e:
                self.logger.debug(f"HSV conversion failed in lane marking analysis: {e}")
                
                # Fallback to RGB analysis
                avg_color = np.mean(colors, axis=0)
                
                # Simple RGB-based classification
                r, g, b = avg_color
                
                # Yellow-ish (high red and green, low blue)
                if r > 150 and g > 150 and b < 100:
                    return LaneType.SOLID_YELLOW
                # White-ish (high all channels)
                elif r > 180 and g > 180 and b > 180:
                    return LaneType.SOLID_WHITE
                # Light colors default to dashed white
                elif (r + g + b) / 3 > 120:
                    return LaneType.DASHED_WHITE
                else:
                    return LaneType.UNKNOWN
                    
        except Exception as e:
            self.logger.error(f"Lane marking type analysis failed: {e}")
            return LaneType.UNKNOWN
    
    def _calculate_lane_confidence(self, roadline_roi: np.ndarray, 
                                  left_boundary: Optional[np.ndarray],
                                  right_boundary: Optional[np.ndarray],
                                  left_pixels: Optional[np.ndarray],
                                  right_pixels: Optional[np.ndarray]) -> float:
        """Lane confidence calculation with multiple factors"""
        try:
            base_confidence = 0.2
            
            # Boundary detection quality
            boundaries_detected = 0
            if left_boundary is not None:
                base_confidence += 0.3
                boundaries_detected += 1
                if left_pixels is not None and len(left_pixels) > 50:
                    base_confidence += 0.1
                if left_pixels is not None and len(left_pixels) > 100:
                    base_confidence += 0.05
            
            if right_boundary is not None:
                base_confidence += 0.3
                boundaries_detected += 1
                if right_pixels is not None and len(right_pixels) > 50:
                    base_confidence += 0.1
                if right_pixels is not None and len(right_pixels) > 100:
                    base_confidence += 0.05
            
            # Both boundaries detected bonus
            if boundaries_detected == 2:
                base_confidence += 0.15
            
            # Total lane pixel count assessment
            total_pixels = np.sum(roadline_roi)
            if total_pixels > 200:
                base_confidence += 0.1
            if total_pixels > 500:
                base_confidence += 0.1
            if total_pixels > 1000:
                base_confidence += 0.05
            
            # Pixel distribution quality (more spread out is better)
            if total_pixels > 0:
                rows, cols = np.where(roadline_roi) 
                if len(rows) > 0:
                    row_spread = np.max(rows) - np.min(rows)
                    col_spread = np.max(cols) - np.min(cols)
                    
                    # Good vertical spread indicates lane extends into distance
                    if row_spread > self.image_height * 0.3:
                        base_confidence += 0.05
                    if row_spread > self.image_height * 0.5:
                        base_confidence += 0.05
                    
                    # Reasonable horizontal spread
                    if self.image_width * 0.2 < col_spread < self.image_width * 0.8:
                        base_confidence += 0.05
            
            # Clamp to valid confidence range
            return clamp(base_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Lane confidence calculation failed: {e}")
            return 0.3  # Default moderate confidence
    
    def _calculate_traffic_light_distance(self, x: int, y: int, w: int, h: int,
                                         depth_image: Optional[np.ndarray]) -> float:
        """Traffic light distance calculation with multiple fallbacks"""
        try:
            if depth_image is not None:
                # Sampling strategy for accuracy
                center_x, center_y = x + w // 2, y + h // 2
                
                # Sample multiple points with different strategies
                sample_points = []
                
                # Center point
                sample_points.append((center_x, center_y))
                
                # Edge points (traffic lights often have bright edges)
                margin = max(2, min(w, h) // 4)
                sample_points.extend([
                    (x + margin, y + margin),
                    (x + w - margin, y + margin), 
                    (x + margin, y + h - margin),
                    (x + w - margin, y + h - margin)
                ])
                
                # Additional center-region points
                for dx in [-w//4, 0, w//4]:
                    for dy in [-h//4, 0, h//4]:
                        px, py = center_x + dx, center_y + dy
                        if x <= px < x + w and y <= py < y + h:
                            sample_points.append((px, py))
                
                valid_distances = []
                
                for px, py in sample_points:
                    # Bounds checking
                    if (0 <= px < depth_image.shape[1] and 
                        0 <= py < depth_image.shape[0]):
                        distance = safe_float(depth_image[py, px])
                        
                        # Validation for traffic light distances
                        if 5.0 <= distance <= 200.0:  # Reasonable range for traffic lights
                            valid_distances.append(distance)
                
                if valid_distances:
                    # Use median for robustness against outliers
                    median_distance = np.median(valid_distances)
                    
                    # Additional validation - traffic lights shouldn't be too close/far
                    if 8.0 <= median_distance <= 150.0:
                        return safe_float(median_distance)
            
            # Fallback estimation using size and position
            # Traffic lights have known approximate sizes
            apparent_size = max(safe_float(w), safe_float(h))
            
            # Size-based estimation
            # Typical traffic light diameter: 20-30cm, assume 25cm
            typical_light_diameter = 0.25  # meters
            
            if apparent_size > 1:
                estimated_distance = (typical_light_diameter * self.focal_length) / apparent_size
                
                # Position-based adjustment (higher = farther typically)
                y_center = safe_float(y + h / 2)
                height_factor = 1.0 + (self.image_height - y_center) / self.image_height * 0.3
                estimated_distance *= height_factor
                
                # Clamp to reasonable bounds for traffic lights
                return clamp(estimated_distance, 10.0, 120.0)
            
            # Final fallback
            return 40.0  # Default reasonable distance
            
        except Exception as e:
            self.logger.error(f"Traffic light distance calculation failed: {e}")
            return 40.0
    
    def _is_road_clear(self, semantic_image: np.ndarray, depth_image: Optional[np.ndarray],
                      obstacles: List[DetectedObject], min_distance: float = 20.0) -> bool:
        """Road clearance check with obstacle analysis"""
        try:
            # Primary check: detected obstacles in same lane
            same_lane_obstacles = [
                obs for obs in obstacles 
                if (obs.lane_assignment == 0 and  # Same lane
                    obs.distance < min_distance and
                    obs.confidence > 0.6)  # Higher confidence threshold
            ]
            
            if same_lane_obstacles:
                # Check if obstacles are actually blocking (not just detected noise)
                blocking_obstacles = [
                    obs for obs in same_lane_obstacles
                    if (obs.object_type in [ObjectType.VEHICLE, ObjectType.PEDESTRIAN] and
                        obs.distance < min_distance * 0.8)  # Closer threshold for blocking
                ]
                
                if blocking_obstacles:
                    return False
            
            # Secondary check: semantic + depth analysis for missed objects
            if depth_image is None:
                return len(same_lane_obstacles) == 0
            
            try:
                # ROI for road ahead analysis
                h, w = semantic_image.shape
                
                # More focused ROI - central driving area
                roi_y1 = safe_int(h * 0.4)   # Start from 40% down
                roi_y2 = safe_int(h * 0.8)   # End at 80% down  
                roi_x1 = safe_int(w * 0.35)  # 35% from left
                roi_x2 = safe_int(w * 0.65)  # 65% from left
                
                # Ensure valid ROI bounds
                roi_y1 = clamp(roi_y1, 0, h)
                roi_y2 = clamp(roi_y2, roi_y1, h)
                roi_x1 = clamp(roi_x1, 0, w)
                roi_x2 = clamp(roi_x2, roi_x1, w)
                
                if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
                    return True  # Invalid ROI, assume clear
                
                roi_semantic = semantic_image[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_depth = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Obstacle detection in ROI
                road_class_id = self.CLASS_TO_ID.get('road', 7)
                vehicle_class_id = self.CLASS_TO_ID.get('vehicles', 10)
                pedestrian_class_id = self.CLASS_TO_ID.get('pedestrian', 4)
                
                # Create masks for different object types
                road_mask = (roi_semantic == road_class_id)
                vehicle_mask = (roi_semantic == vehicle_class_id)
                pedestrian_mask = (roi_semantic == pedestrian_class_id)
                
                # Check for close vehicles/pedestrians
                close_depth_mask = (roi_depth < min_distance) & (roi_depth > 2.0)
                
                # Vehicles in path
                close_vehicles = vehicle_mask & close_depth_mask
                if np.sum(close_vehicles) > 100:  # Significant vehicle presence
                    return False
                
                # Pedestrians in path (lower threshold as they're smaller)
                close_pedestrians = pedestrian_mask & close_depth_mask
                if np.sum(close_pedestrians) > 50:  # Significant pedestrian presence
                    return False
                
                # General obstacle check (non-road objects at close distance)
                non_road_close = ~road_mask & close_depth_mask
                obstacle_ratio = np.sum(non_road_close) / max(non_road_close.size, 1)
                
                # If more than 8% of ROI has close non-road objects, consider blocked
                return obstacle_ratio < 0.08
                
            except Exception as e:
                self.logger.debug(f"Semantic road clearance check failed: {e}")
                return len(same_lane_obstacles) == 0
            
        except Exception as e:
            self.logger.error(f"Road clearance check failed: {e}")
            return True  # Default to clear if check fails
    
    def _detect_intersection(self, semantic_image: np.ndarray, 
                            traffic_lights: List[TrafficLightInfo]) -> bool:
        """Intersection detection with multiple indicators"""
        try:
            # Primary indicator: relevant traffic lights nearby
            nearby_relevant_lights = [
                tl for tl in traffic_lights 
                if tl.relevant_for_ego and tl.distance < 60.0
            ]
            
            if nearby_relevant_lights:
                return True
            
            # Secondary indicator: traffic signs
            traffic_sign_class_id = self.CLASS_TO_ID.get('traffic_sign', 12)
            traffic_sign_mask = (semantic_image == traffic_sign_class_id)
            
            if np.sum(traffic_sign_mask) > 200:  # Significant traffic sign presence
                return True
            
            # Tertiary indicator: road topology analysis
            road_class_id = self.CLASS_TO_ID.get('road', 7)
            road_mask = (semantic_image == road_class_id)
            
            # Road topology analysis
            h, w = semantic_image.shape
            
            # Analyze road area in horizontal strips
            strip_height = h // 10
            road_areas = []
            
            for i in range(3, 8):  # Middle portion of image
                y_start = i * strip_height
                y_end = (i + 1) * strip_height
                
                if y_end <= h:
                    strip = road_mask[y_start:y_end, :]
                    road_area = np.sum(strip)
                    road_areas.append(road_area)
            
            if len(road_areas) >= 3:
                # Look for significant changes in road area (intersection expansion)
                area_changes = []
                for i in range(1, len(road_areas)):
                    change_ratio = road_areas[i] / max(road_areas[i-1], 1)
                    area_changes.append(change_ratio)
                
                # Significant expansion might indicate intersection
                max_expansion = max(area_changes) if area_changes else 1.0
                if max_expansion > 1.4:  # 40% increase in road area
                    return True
            
            # Additional check: lane marking complexity
            roadline_class_id = self.CLASS_TO_ID.get('roadline', 6)
            roadline_mask = (semantic_image == roadline_class_id)
            
            # Count connected components in lane markings
            if np.sum(roadline_mask) > 0:
                roadline_uint8 = roadline_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    roadline_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Many separate lane marking components might indicate intersection
                if len(contours) > 8:  # Complex lane marking pattern
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Intersection detection failed: {e}")
            return False
    
    def _assess_sensor_quality(self, rgb_image: np.ndarray, semantic_image: np.ndarray, 
                              depth_image: Optional[np.ndarray]) -> Dict[str, float]:
        """Sensor quality assessment with more comprehensive metrics"""
        quality = {}
        
        try:
            # RGB image quality assessment
            if rgb_image is not None:
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                
                # Contrast assessment (using standard deviation)
                contrast = safe_float(np.std(gray))
                contrast_score = clamp(contrast / 50.0, 0.0, 1.0)
                
                # Brightness assessment (prefer values around 128)
                brightness = safe_float(np.mean(gray))
                brightness_score = 1.0 - abs(brightness - 128.0) / 128.0
                brightness_score = clamp(brightness_score, 0.0, 1.0)
                
                # Sharpness assessment (using Laplacian variance)
                laplacian_var = safe_float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_score = clamp(laplacian_var / 1000.0, 0.0, 1.0)
                
                # Combined RGB quality score
                quality['rgb'] = (contrast_score * 0.4 + 
                                brightness_score * 0.3 + 
                                sharpness_score * 0.3)
            else:
                quality['rgb'] = 0.0
            
            # Semantic image quality assessment
            if semantic_image is not None:
                unique_classes = len(np.unique(semantic_image))
                class_diversity_score = clamp(unique_classes / 15.0, 0.0, 1.0)  # Expect ~15 classes
                
                # Check for reasonable class distribution
                class_counts = np.bincount(semantic_image.flatten())
                if len(class_counts) > 1:
                    # Entropy-based diversity measure
                    probabilities = class_counts / np.sum(class_counts)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    entropy_score = clamp(entropy / 4.0, 0.0, 1.0)  # Normalize by max entropy
                else:
                    entropy_score = 0.0
                
                quality['semantic'] = (class_diversity_score * 0.6 + entropy_score * 0.4)
            else:
                quality['semantic'] = 0.0
            
            # Depth image quality assessment
            if depth_image is not None:
                # Valid depth ratio (finite, positive, reasonable range)
                valid_mask = (
                    np.isfinite(depth_image) & 
                    (depth_image > 0.5) & 
                    (depth_image < 300.0)
                )
                valid_ratio = np.sum(valid_mask) / depth_image.size
                
                # Depth range assessment (good depth should have reasonable spread)
                if np.sum(valid_mask) > 100:
                    valid_depths = depth_image[valid_mask]
                    depth_range = np.max(valid_depths) - np.min(valid_depths)
                    range_score = clamp(depth_range / 100.0, 0.0, 1.0)  # Normalize by 100m range
                    
                    # Depth smoothness (not too noisy)
                    depth_gradient = np.gradient(depth_image)
                    gradient_magnitude = np.sqrt(depth_gradient[0]**2 + depth_gradient[1]**2)
                    smoothness_score = 1.0 - clamp(np.mean(gradient_magnitude) / 10.0, 0.0, 1.0)
                else:
                    range_score = 0.0
                    smoothness_score = 0.0
                
                quality['depth'] = (valid_ratio * 0.5 + range_score * 0.3 + smoothness_score * 0.2)
            else:
                quality['depth'] = 0.0
            
            # Overall sensor quality
            weights = {'rgb': 0.4, 'semantic': 0.4, 'depth': 0.2}
            quality['overall'] = sum(quality[sensor] * weights[sensor] for sensor in weights)
            
        except Exception as e:
            self.logger.error(f"Sensor quality assessment failed: {e}")
            # Return default low quality scores
            quality = {'rgb': 0.3, 'semantic': 0.3, 'depth': 0.3, 'overall': 0.3}
        
        return quality
    
    def _detect_speed_limit(self, rgb_image: np.ndarray, semantic_image: np.ndarray) -> Optional[float]:
        """Speed limit detection with sign analysis"""
        try:
            # Check for traffic signs in semantic image
            traffic_sign_class_id = self.CLASS_TO_ID.get('traffic_sign', 12)
            sign_mask = (semantic_image == traffic_sign_class_id)
            
            if not np.any(sign_mask):
                return None
            
            # Find sign regions
            sign_uint8 = sign_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(sign_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Minimum size for readable sign
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic sign shape validation (roughly square/rectangular)
                aspect_ratio = w / max(h, 1)
                if not (0.7 <= aspect_ratio <= 1.5):
                    continue
                
                # Extract sign ROI
                x = clamp(x, 0, rgb_image.shape[1] - 1)
                y = clamp(y, 0, rgb_image.shape[0] - 1)
                w = clamp(w, 1, rgb_image.shape[1] - x)
                h = clamp(h, 1, rgb_image.shape[0] - y)
                
                sign_roi = rgb_image[y:y+h, x:x+w]
                
                if sign_roi.size == 0:
                    continue
                
                # Placeholder for OCR/pattern recognition
                # In a real implementation, this would use:
                # 1. OCR to read numbers on signs
                # 2. Template matching for standard speed limit signs
                # 3. Machine learning classifier for sign recognition
                
                # For now, return common speed limits based on context
                # This is a simplified approach
                return 50.0  # km/h - common urban speed limit
            
            return None
            
        except Exception as e:
            self.logger.error(f"Speed limit detection failed: {e}")
            return None
    
    def _empty_perception(self, frame_id: int = 0, timestamp: float = 0.0) -> PerceptionOutput:
        """Empty perception output with proper initialization"""
        try:
            return PerceptionOutput(
                frame_id=frame_id,
                timestamp=timestamp,
                drivable_area=np.zeros((self.image_height, self.image_width), dtype=np.uint8),
                safety_metrics=SafetyMetrics(),
                sensor_quality={'rgb': 0.0, 'semantic': 0.0, 'depth': 0.0, 'overall': 0.0}
            )
        except Exception as e:
            self.logger.error(f"Failed to create empty perception output: {e}")
            # Return minimal valid output
            return PerceptionOutput()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the monitor"""
        try:
            stats = self.performance_monitor.get_stats()
            stats['total_frames_processed'] = self.frame_count
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {'error': str(e)}
    
    def reset_performance_monitor(self):
        """Reset performance monitoring"""
        try:
            self.performance_monitor.reset()
            self.frame_count = 0
            self.logger.info("Performance monitor reset")
        except Exception as e:
            self.logger.error(f"Failed to reset performance monitor: {e}")
    
    def set_logging_level(self, level: int):
        """Dynamically change logging level"""
        try:
            self.logger.setLevel(level)
            logging.getLogger().setLevel(level)
            self.logger.info(f"Logging level changed to {logging.getLevelName(level)}")
        except Exception as e:
            self.logger.error(f"Failed to set logging level: {e}")
    
    def cleanup(self):
        """Cleanup resources and log final statistics"""
        try:
            # Log final performance statistics
            if self.frame_count > 0:
                final_stats = self.get_performance_stats()
                self.logger.info(f"Perception system final stats: {final_stats}")
            
            # Clear tracking history
            self.object_history.clear()
            
            self.logger.info("Perception system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Export the main class and data structures for easy importing
__all__ = [
    'PerceptionSystem',
    'PerceptionOutput', 
    'DetectedObject',
    'LaneInfo',
    'TrafficLightInfo',
    'SafetyMetrics',
    'TrafficLightState',
    'ObjectType',
    'LaneType'
]