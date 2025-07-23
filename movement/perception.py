"""
Perception system for autonomous driving.
Processes sensor data to understand the environment.
"""

import numpy as np
import cv2
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

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

@dataclass
class TrafficLightInfo:
    """Traffic light information"""
    state: TrafficLightState
    distance: float
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    time_to_change: Optional[float] = None  # Estimated seconds until state change

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
    
    # Road analysis
    drivable_area: Optional[np.ndarray] = None
    road_ahead_clear: bool = True
    intersection_ahead: bool = False
    closest_vehicle_distance: float = float('inf')
    
    # Speed and limits
    speed_limit: Optional[float] = None
    
    # Metadata
    frame_id: int = 0
    timestamp: float = 0.0
    processing_time: float = 0.0

class PerceptionSystem:
    """
    Perception system
    """
    
    # CARLA semantic segmentation classes (comprehensive mapping)
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
        
        # Traffic light color detection parameters (improved ranges)
        self.color_ranges = {
            'red': [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                   (np.array([170, 50, 50]), np.array([180, 255, 255]))],
            'yellow': [(np.array([15, 50, 50]), np.array([35, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
        }
        
        # Object detection parameters
        self.min_object_area = 100
        self.min_vehicle_area = 200
        self.min_pedestrian_area = 50
        
        # Lane detection parameters
        self.min_lane_pixels = 20
        self.roi_height_ratio = 0.5  # Bottom half of image
        
    def process_sensors(self, sensor_data: Dict[str, Any], frame_id: int = 0, 
                       timestamp: float = 0.0) -> PerceptionOutput:
        """
        Main processing function combining both approaches
        """
        import time
        start_time = time.time()
        
        rgb_image = sensor_data.get('rgb')
        semantic_image = sensor_data.get('semantic') 
        depth_image = sensor_data.get('depth')
        vehicle_speed = sensor_data.get('speed', 0.0)
        
        if rgb_image is None or semantic_image is None:
            return self._empty_perception(frame_id, timestamp)
            
        perception = PerceptionOutput(
            frame_id=frame_id,
            timestamp=timestamp
        )
        
        # Detect all objects using method
        all_objects = self._detect_objects(semantic_image, depth_image, rgb_image)
        
        # Categorize objects
        perception.detected_objects = all_objects
        perception.vehicles = [obj for obj in all_objects if obj.object_type == ObjectType.VEHICLE]
        perception.pedestrians = [obj for obj in all_objects if obj.object_type == ObjectType.PEDESTRIAN]
        perception.obstacles = [obj for obj in all_objects 
                              if obj.object_type in [ObjectType.VEHICLE, ObjectType.PEDESTRIAN] 
                              and obj.distance < 30.0]
        
        # Lane detection
        perception.lane_info = self._detect_lanes(semantic_image, rgb_image)
        
        # Traffic light detection
        perception.traffic_light = self._detect_traffic_lights(
            rgb_image, semantic_image, depth_image
        )
        
        # Road analysis
        perception.drivable_area = self._get_drivable_area(semantic_image)
        perception.road_ahead_clear = self._is_road_clear(semantic_image, depth_image)
        perception.intersection_ahead = self._detect_intersection(semantic_image)
        
        # Calculate closest vehicle
        if perception.vehicles:
            perception.closest_vehicle_distance = min(v.distance for v in perception.vehicles)
        
        # Speed limit detection (placeholder for future implementation)
        perception.speed_limit = self._detect_speed_limit(rgb_image, semantic_image)
        
        # Record processing time
        perception.processing_time = time.time() - start_time
        
        return perception
    
    def _detect_objects(self, semantic_image: np.ndarray, depth_image: Optional[np.ndarray],
                                rgb_image: np.ndarray) -> List[DetectedObject]:
        """Object detection combining both approaches"""
        detected_objects = []
        
        # Define object class mappings
        object_mappings = {
            ObjectType.VEHICLE: [self.CLASS_TO_ID.get('vehicles', 10)],
            ObjectType.PEDESTRIAN: [self.CLASS_TO_ID.get('pedestrian', 4)],
            ObjectType.CYCLIST: [],  # Add if available in semantic classes
        }
        
        for obj_type, class_ids in object_mappings.items():
            for class_id in class_ids:
                if class_id is None:
                    continue
                    
                # Create mask for this object type
                object_mask = (semantic_image == class_id)
                
                if not np.any(object_mask):
                    continue
                
                # Extract objects using connected components
                objects = self._extract_objects_from_mask(
                    object_mask, obj_type, depth_image, rgb_image
                )
                detected_objects.extend(objects)
        
        return detected_objects
    
    def _extract_objects_from_mask(self, mask: np.ndarray, object_type: ObjectType,
                                          depth_image: Optional[np.ndarray], 
                                          rgb_image: np.ndarray) -> List[DetectedObject]:
        """Object extraction with connected components"""
        objects = []
        
        # Use connected components for better object separation
        mask_uint8 = mask.astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        
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
            
            # Calculate distance using depth image
            distance = self._calculate_object_distance(component_mask, depth_image)
            
            # Calculate relative position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            lateral_pos = (center_x - self.image_width / 2) / (self.image_width / 2)
            longitudinal_pos = distance
            
            # Assign lane based on lateral position
            lane_assignment = self._assign_lane(center_x)
            
            # Calculate confidence based on size and depth consistency
            confidence = self._calculate_object_confidence(component_mask, depth_image, object_type)
            
            # Determine if object is moving (placeholder - would need tracking)
            is_moving = self._estimate_object_motion(component_mask, object_type)
            
            detected_object = DetectedObject(
                object_type=object_type,
                bbox=bbox,
                confidence=confidence,
                distance=distance,
                relative_position=(lateral_pos, longitudinal_pos),
                lane_assignment=lane_assignment,
                size_2d=(x2 - x1, y2 - y1),
                is_moving=is_moving
            )
            
            objects.append(detected_object)
        
        return objects
    
    def _detect_lanes(self, semantic_image: np.ndarray, 
                              rgb_image: np.ndarray) -> Optional[LaneInfo]:
        """Lane detection combining both approaches"""
        # Get road and roadline masks
        road_mask = (semantic_image == self.CLASS_TO_ID.get('road', 7))
        roadline_mask = (semantic_image == self.CLASS_TO_ID.get('roadline', 6))
        
        if not np.any(roadline_mask):
            return None
        
        # Focus on region of interest (bottom half)
        roi_start = int(self.image_height * self.roi_height_ratio)
        roadline_roi = roadline_mask[roi_start:, :]
        
        if np.sum(roadline_roi) < self.min_lane_pixels:
            return None
        
        lane_info = LaneInfo()
        
        # Extract lane pixels
        lane_pixels = np.where(roadline_roi)
        lane_y = lane_pixels[0] + roi_start
        lane_x = lane_pixels[1]
        
        # Store raw pixels
        lane_info.left_lane_pixels = np.column_stack((lane_x, lane_y))
        lane_info.right_lane_pixels = np.column_stack((lane_x, lane_y))
        
        # Boundary detection with polynomial fitting
        left_boundary, right_boundary = self._extract_lane_boundaries(roadline_roi)
        
        lane_info.left_boundary = left_boundary
        lane_info.right_boundary = right_boundary
        
        # Calculate metrics
        if left_boundary is not None and right_boundary is not None:
            lane_info.lane_width = self._calculate_lane_width(left_boundary, right_boundary)
            lane_info.curvature = self._calculate_curvature(left_boundary, right_boundary)
            lane_info.center_line = self._calculate_center_line(left_boundary, right_boundary)
            lane_info.heading_angle = self._calculate_heading_angle(lane_info.center_line)
            
            # Classify lane types using RGB analysis
            lane_info.lane_type_left, lane_info.lane_type_right = self._classify_lane_types(
                rgb_image, left_boundary, right_boundary
            )
            
            # Calculate confidence
            lane_info.confidence = self._calculate_lane_confidence(roadline_roi, left_boundary, right_boundary)
        
        return lane_info
    
    def _detect_traffic_lights(self, rgb_image: np.ndarray, semantic_image: np.ndarray,
                                       depth_image: Optional[np.ndarray]) -> Optional[TrafficLightInfo]:
        """Traffic light detection"""
        # Find traffic light regions
        traffic_light_mask = (semantic_image == self.CLASS_TO_ID.get('traffic_light', 18))
        
        if not np.any(traffic_light_mask):
            return None
        
        # Get connected components for multiple traffic lights
        mask_uint8 = traffic_light_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find the most relevant traffic light (largest/closest)
        best_traffic_light = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Too small
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            
            # Extract ROI from RGB image
            tl_roi = rgb_image[y:y+h, x:x+w]
            
            # Classify color with method
            state, color_confidence = self._classify_traffic_light_color(tl_roi)
            
            # Calculate distance
            distance = self._calculate_traffic_light_distance(x, y, w, h, depth_image)
            
            # Score based on size, confidence, and proximity
            score = area * color_confidence / max(distance, 1.0)
            
            if score > best_score:
                best_score = score
                best_traffic_light = TrafficLightInfo(
                    state=state,
                    distance=distance,
                    confidence=color_confidence,
                    bbox=bbox
                )
        
        return best_traffic_light
    
    def _extract_lane_boundaries(self, roadline_roi: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Lane boundary extraction with polynomial fitting"""
        rows, cols = np.where(roadline_roi)
        
        if len(cols) < self.min_lane_pixels:
            return None, None
        
        # Split into left and right based on image center
        img_center = roadline_roi.shape[1] // 2
        
        left_mask = cols < img_center
        right_mask = cols >= img_center
        
        left_boundary = None
        right_boundary = None
        
        # Fit polynomials to lane boundaries
        if np.sum(left_mask) > 10:
            left_points = np.column_stack((cols[left_mask], rows[left_mask]))
            left_boundary = self._fit_lane_polynomial(left_points)
        
        if np.sum(right_mask) > 10:
            right_points = np.column_stack((cols[right_mask], rows[right_mask]))
            right_boundary = self._fit_lane_polynomial(right_points)
        
        return left_boundary, right_boundary
    
    def _fit_lane_polynomial(self, points: np.ndarray, degree: int = 2) -> Optional[np.ndarray]:
        """Fit polynomial to lane points with outlier rejection"""
        if len(points) < degree + 1:
            return None
        
        try:
            # Use RANSAC-like approach for robust fitting
            best_poly = None
            best_inliers = 0
            
            for _ in range(5):  # Multiple attempts
                # Sample points for initial fit
                sample_indices = np.random.choice(len(points), min(len(points), 20), replace=False)
                sample_points = points[sample_indices]
                
                # Fit polynomial
                poly = np.polyfit(sample_points[:, 1], sample_points[:, 0], degree)
                
                # Count inliers
                predicted_x = np.polyval(poly, points[:, 1])
                errors = np.abs(predicted_x - points[:, 0])
                inliers = np.sum(errors < 5.0)  # 5 pixel threshold
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_poly = poly
            
            return best_poly
            
        except Exception:
            return None
    
    def _calculate_lane_width(self, left_boundary: np.ndarray, 
                                     right_boundary: np.ndarray) -> float:
        """Calculate lane width using polynomial boundaries"""
        if left_boundary is None or right_boundary is None:
            return 3.5  # Default lane width
        
        # Sample multiple y positions
        y_positions = np.linspace(0, self.image_height // 3, 10)
        widths = []
        
        for y in y_positions:
            try:
                left_x = np.polyval(left_boundary, y)
                right_x = np.polyval(right_boundary, y)
                width_pixels = abs(right_x - left_x)
                # Convert to meters (rough approximation)
                width_meters = width_pixels * 0.01  # Adjust based on camera calibration
                widths.append(width_meters)
            except:
                continue
        
        return np.median(widths) if widths else 3.5
    
    def _calculate_curvature(self, left_boundary: np.ndarray, 
                                    right_boundary: np.ndarray) -> float:
        """Curvature calculation"""
        if left_boundary is None or right_boundary is None:
            return 0.0
        
        # Use average curvature of both boundaries
        left_curvature = abs(2 * left_boundary[0]) if len(left_boundary) > 2 else 0.0
        right_curvature = abs(2 * right_boundary[0]) if len(right_boundary) > 2 else 0.0
        
        return (left_curvature + right_curvature) / 2
    
    def _calculate_center_line(self, left_boundary: np.ndarray, 
                                      right_boundary: np.ndarray) -> np.ndarray:
        """Calculate center line"""
        if left_boundary is None or right_boundary is None:
            return None
        
        # Average the polynomials
        center_poly = (left_boundary + right_boundary) / 2
        return center_poly
    
    def _classify_traffic_light_color(self, tl_roi: np.ndarray) -> Tuple[TrafficLightState, float]:
        """Traffic light color classification"""
        if tl_roi.size == 0:
            return TrafficLightState.UNKNOWN, 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(tl_roi, cv2.COLOR_RGB2HSV)
        
        # Test each color
        color_scores = {}
        
        for color_name, ranges in self.color_ranges.items():
            total_pixels = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_pixels += np.sum(mask > 0)
            color_scores[color_name] = total_pixels
        
        # Find dominant color
        max_color = max(color_scores.keys(), key=lambda x: color_scores[x])
        max_pixels = color_scores[max_color]
        
        if max_pixels < 10:  # Minimum threshold
            return TrafficLightState.UNKNOWN, 0.0
        
        # Calculate confidence
        total_colored_pixels = sum(color_scores.values())
        confidence = max_pixels / max(total_colored_pixels, 1)
        
        # Map to enum
        state_mapping = {
            'red': TrafficLightState.RED,
            'yellow': TrafficLightState.YELLOW,
            'green': TrafficLightState.GREEN
        }
        
        return state_mapping.get(max_color, TrafficLightState.UNKNOWN), confidence
    
    # Helper methods
    def _calculate_object_distance(self, mask: np.ndarray, depth_image: Optional[np.ndarray]) -> float:
        """Calculate object distance using depth image"""
        if depth_image is None:
            return 50.0  # Default distance
        
        object_depths = depth_image[mask]
        valid_depths = object_depths[object_depths > 0]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        return 50.0
    
    def _assign_lane(self, center_x: float) -> int:
        """Assign lane based on lateral position"""
        img_center = self.image_width / 2
        threshold = 50  # pixels
        
        if center_x < img_center - threshold:
            return -1  # Left lane
        elif center_x > img_center + threshold:
            return 1   # Right lane
        else:
            return 0   # Same lane
    
    def _calculate_object_confidence(self, mask: np.ndarray, depth_image: Optional[np.ndarray], 
                                   object_type: ObjectType) -> float:
        """Calculate object detection confidence"""
        base_confidence = 0.8
        
        # Adjust based on size
        area = np.sum(mask)
        if area > 1000:
            base_confidence += 0.1
        elif area < 200:
            base_confidence -= 0.2
        
        # Adjust based on depth consistency
        if depth_image is not None:
            depths = depth_image[mask]
            valid_depths = depths[depths > 0]
            if len(valid_depths) > 10:
                depth_std = np.std(valid_depths)
                if depth_std < 2.0:  # Consistent depth
                    base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _estimate_object_motion(self, mask: np.ndarray, object_type: ObjectType) -> bool:
        """Estimate if object is moving (placeholder for tracking)"""
        # This would require temporal tracking - placeholder implementation
        return object_type == ObjectType.VEHICLE
    
    def _calculate_heading_angle(self, center_line: Optional[np.ndarray]) -> float:
        """Calculate heading angle relative to lane"""
        if center_line is None or len(center_line) < 2:
            return 0.0
        
        # Calculate derivative at bottom of image to get heading
        y = self.image_height - 50
        try:
            # For polynomial ax^2 + bx + c, derivative is 2ax + b
            slope = 2 * center_line[0] * y + center_line[1]
            angle = np.arctan(slope)
            return angle
        except:
            return 0.0
    
    def _classify_lane_types(self, rgb_image: np.ndarray, left_boundary: Optional[np.ndarray],
                           right_boundary: Optional[np.ndarray]) -> Tuple[LaneType, LaneType]:
        """Classify lane line types using RGB analysis"""
        # Placeholder implementation - would need more sophisticated analysis
        return LaneType.DASHED_WHITE, LaneType.DASHED_WHITE
    
    def _calculate_lane_confidence(self, roadline_roi: np.ndarray, left_boundary: Optional[np.ndarray],
                                 right_boundary: Optional[np.ndarray]) -> float:
        """Calculate lane detection confidence"""
        base_confidence = 0.5
        
        if left_boundary is not None:
            base_confidence += 0.25
        if right_boundary is not None:
            base_confidence += 0.25
        
        # Adjust based on number of lane pixels
        lane_pixel_count = np.sum(roadline_roi)
        if lane_pixel_count > 100:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_traffic_light_distance(self, x: int, y: int, w: int, h: int,
                                        depth_image: Optional[np.ndarray]) -> float:
        """Calculate traffic light distance"""
        if depth_image is None:
            return 20.0  # Default distance
        
        center_x, center_y = x + w // 2, y + h // 2
        distance = depth_image[center_y, center_x]
        
        return float(distance) if distance > 0 else 20.0
    
    def _get_drivable_area(self, semantic_image: np.ndarray) -> np.ndarray:
        """Extract drivable area"""
        return (semantic_image == self.CLASS_TO_ID.get('road', 7)).astype(np.uint8)
    
    def _is_road_clear(self, semantic_image: np.ndarray, depth_image: Optional[np.ndarray],
                      min_distance: float = 20.0) -> bool:
        """Check if road ahead is clear"""
        if depth_image is None:
            return True
        
        # Check center region ahead
        h, w = semantic_image.shape
        roi = semantic_image[h//2:h*3//4, w//3:w*2//3]
        depth_roi = depth_image[h//2:h*3//4, w//3:w*2//3]
        
        # Find non-road pixels at close distance
        road_mask = (roi == self.CLASS_TO_ID.get('road', 7))
        obstacle_mask = ~road_mask & (depth_roi < min_distance) & (depth_roi > 0)
        
        return not np.any(obstacle_mask)
    
    def _detect_intersection(self, semantic_image: np.ndarray) -> bool:
        """Detect intersection ahead"""
        has_traffic_light = np.any(semantic_image == self.CLASS_TO_ID.get('traffic_light', 18))
        has_traffic_sign = np.any(semantic_image == self.CLASS_TO_ID.get('traffic_sign', 12))
        
        return has_traffic_light or has_traffic_sign
    
    def _detect_speed_limit(self, rgb_image: np.ndarray, semantic_image: np.ndarray) -> Optional[float]:
        """Detect speed limit signs (placeholder)"""
        # This would require OCR or sign classification
        return None
    
    def _empty_perception(self, frame_id: int = 0, timestamp: float = 0.0) -> PerceptionOutput:
        """Return empty perception output"""
        return PerceptionOutput(
            frame_id=frame_id,
            timestamp=timestamp,
            drivable_area=np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        )