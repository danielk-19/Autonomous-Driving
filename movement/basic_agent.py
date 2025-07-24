"""
Hybrid autonomous driving agent combining rule-based safety with ML-based control.
This agent prioritizes safety through hard-coded traffic rules while using ML for smooth driving.
"""
import carla
import numpy as np
import torch
import cv2
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import math
import time
from pathlib import Path
import sys

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from .perception import PerceptionSystem, PerceptionOutput, DetectedObject, TrafficLightState

class DrivingState(Enum):
    """Current driving state of the agent"""
    NORMAL_DRIVING = "normal_driving"
    FOLLOWING_VEHICLE = "following_vehicle"
    STOPPING_FOR_TRAFFIC_LIGHT = "stopping_for_traffic_light"
    EMERGENCY_BRAKE = "emergency_brake"
    LANE_CHANGE = "lane_change"
    INTERSECTION_APPROACH = "intersection_approach"

@dataclass
class VehicleAction:
    """Vehicle control action"""
    throttle: float  # 0.0 to 1.0
    steer: float     # -1.0 to 1.0
    brake: float     # 0.0 to 1.0
    hand_brake: bool = False
    reverse: bool = False

class SafetyMonitor:
    """
    Rule-based safety system that enforces traffic laws and collision avoidance
    """
    
    def __init__(self):
        # Safety parameters
        self.min_following_distance = 3.0  # meters
        self.emergency_brake_distance = 5.0  # meters
        self.max_speed = 50.0  # km/h
        self.traffic_light_stop_distance = 50.0  # meters
        self.reaction_time = 0.3  # seconds
        
        # State tracking
        self.previous_actions = []
        self.brake_count = 0
        self.safety_violations = 0
        
    def validate_action(self, proposed_action: VehicleAction, 
                       perception: PerceptionOutput, 
                       current_speed: float,
                       vehicle_location: carla.Location) -> VehicleAction:
        """
        Validate and potentially override proposed action for safety
        Returns safe action with safety intervention tracking
        """
        safe_action = VehicleAction(
            throttle=proposed_action.throttle,
            steer=proposed_action.steer,
            brake=proposed_action.brake
        )
        
        safety_intervention = False
        
        # Rule 1: Emergency collision avoidance (HIGHEST PRIORITY)
        if self._check_emergency_collision(perception):
            safe_action = VehicleAction(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
            safety_intervention = True
            self.safety_violations += 1
        
        # Rule 2: Traffic light compliance (MANDATORY)
        elif self._check_traffic_light_violation(perception, current_speed):
            safe_action = VehicleAction(
                throttle=0.0, 
                steer=self._calculate_safe_steering(proposed_action.steer),
                brake=0.8
            )
            safety_intervention = True
        
        # Rule 3: Speed limit enforcement (MANDATORY)
        elif current_speed > self.max_speed:
            safe_action.throttle = 0.0
            safe_action.brake = max(safe_action.brake, 0.4)
            safety_intervention = True
        
        # Rule 4: Following distance enforcement
        else:
            following_action = self._enforce_following_distance(safe_action, perception, current_speed)
            if following_action:
                safe_action = following_action
                safety_intervention = True
        
        # Rule 5: Lateral safety (prevent dangerous steering)
        original_steer = safe_action.steer
        safe_action.steer = np.clip(safe_action.steer, -0.8, 0.8)
        if abs(original_steer - safe_action.steer) > 0.1:
            safety_intervention = True
        
        # Rule 6: Lane departure prevention
        if perception.lane_info and (perception.lane_info.lane_departure_left or 
                                   perception.lane_info.lane_departure_right):
            # Apply corrective steering
            correction = self._calculate_lane_departure_correction(perception.lane_info)
            safe_action.steer = np.clip(safe_action.steer + correction, -0.8, 0.8)
            if abs(correction) > 0.05:
                safety_intervention = True
        
        # Track interventions
        if safety_intervention:
            self.brake_count += 1
        
        # Store action history for analysis
        self.previous_actions.append({
            'proposed': proposed_action,
            'safe': safe_action,
            'intervention': safety_intervention,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.previous_actions) > 100:
            self.previous_actions = self.previous_actions[-50:]
        
        return safe_action
    
    def _check_emergency_collision(self, perception: PerceptionOutput) -> bool:
        """Check for imminent collision requiring emergency brake"""
        # Use safety metrics from perception system
        if perception.safety_metrics.emergency_brake_needed:
            return True
        
        # Additional check using obstacles
        for obstacle in perception.obstacles:
            if (obstacle.distance < self.emergency_brake_distance and 
                obstacle.lane_assignment == 0 and  # Same lane
                abs(obstacle.relative_position[0]) < 1.0):  # In front of us
                return True
        
        # Check time to collision
        if perception.safety_metrics.time_to_collision < 2.0:  # Less than 2 seconds
            return True
        
        return False
    
    def _check_traffic_light_violation(self, perception: PerceptionOutput, current_speed: float) -> bool:
        """Check if we should stop for a red traffic light"""
        if not perception.traffic_light or not perception.traffic_light.relevant_for_ego:
            return False
        
        traffic_light = perception.traffic_light
        
        # Red light - must stop if we can safely do so
        if traffic_light.state == TrafficLightState.RED:
            # Calculate stopping distance
            stopping_distance = (current_speed / 3.6) ** 2 / (2 * 4.0) + current_speed / 3.6 * self.reaction_time
            
            # If we're too close and moving fast, might be safer to continue
            if traffic_light.distance < stopping_distance and current_speed > 20.0:
                return False  # Too late to stop safely
            elif traffic_light.distance < self.traffic_light_stop_distance:
                return True
        
        # Yellow light - stop if we can do so safely
        elif traffic_light.state == TrafficLightState.YELLOW:
            stopping_distance = (current_speed / 3.6) ** 2 / (2 * 4.0) + current_speed / 3.6 * self.reaction_time
            if traffic_light.distance > stopping_distance and traffic_light.distance < 30.0:
                return True
        
        return False
    
    def _enforce_following_distance(self, action: VehicleAction, 
                                   perception: PerceptionOutput, 
                                   current_speed: float) -> Optional[VehicleAction]:
        """Enforce safe following distance behind vehicles"""
        # Use perception system's safety metrics
        if perception.safety_metrics.following_distance == float('inf'):
            return None
        
        following_distance = perception.safety_metrics.following_distance
        
        # Calculate required following distance based on speed (time-based)
        required_distance = max(self.min_following_distance, 
                              current_speed / 3.6 * 2.0)  # 2-second rule
        
        if following_distance < required_distance:
            # Too close - reduce throttle and apply brakes
            distance_ratio = following_distance / required_distance
            brake_intensity = 1.0 - distance_ratio
            brake_intensity = np.clip(brake_intensity, 0.0, 0.8)
            
            return VehicleAction(
                throttle=0.0,
                steer=action.steer,
                brake=brake_intensity
            )
        
        return None
    
    def _calculate_safe_steering(self, proposed_steer: float) -> float:
        """Calculate safe steering that prevents sharp turns"""
        # Limit steering rate of change
        if self.previous_actions:
            last_steer = self.previous_actions[-1]['safe'].steer
            max_steer_change = 0.3  # Maximum steering change per frame
            
            steer_change = proposed_steer - last_steer
            if abs(steer_change) > max_steer_change:
                return last_steer + np.sign(steer_change) * max_steer_change
        
        return proposed_steer
    
    def _calculate_lane_departure_correction(self, lane_info) -> float:
        """Calculate steering correction for lane departure"""
        if lane_info.lane_departure_left:
            return 0.2  # Steer right
        elif lane_info.lane_departure_right:
            return -0.2  # Steer left
        
        # Gentle center correction based on offset
        if abs(lane_info.lane_center_offset) > 0.3:  # More than 30cm off center
            return -np.sign(lane_info.lane_center_offset) * 0.1
        
        return 0.0

class MLController:
    """
    Machine Learning controller for smooth driving behavior
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained behavioral cloning model"""
        try:
            # Import model architecture
            from models.bc_model import BehavioralCloningModel
            
            self.model = BehavioralCloningModel()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"Loaded ML model from {model_path}")
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            self.model_loaded = False
    
    def predict_action(self, rgb_image: np.ndarray, 
                      semantic_image: np.ndarray,
                      current_speed: float) -> VehicleAction:
        """Predict vehicle action using ML model"""
        if not self.model_loaded or self.model is None:
            return self._fallback_action(rgb_image, current_speed)
        
        try:
            # Preprocess image (match training preprocessing)
            processed_image = self._preprocess_image(rgb_image, semantic_image)
            
            # Model inference
            with torch.no_grad():
                processed_image = processed_image.unsqueeze(0).to(self.device)
                predictions = self.model(processed_image)
                
                steering = float(predictions[0, 0].cpu().numpy())
                throttle = float(predictions[0, 1].cpu().numpy())
                brake = float(predictions[0, 2].cpu().numpy())
                
                # Ensure valid ranges
                steering = np.clip(steering, -1.0, 1.0)
                throttle = np.clip(throttle, 0.0, 1.0)
                brake = np.clip(brake, 0.0, 1.0)
                
                return VehicleAction(
                    throttle=throttle,
                    steer=steering,
                    brake=brake
                )
        
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self._fallback_action(rgb_image, current_speed)
    
    def _preprocess_image(self, rgb_image: np.ndarray, semantic_image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input (match training preprocessing)"""
        # Resize to model input size (224x224)
        rgb_resized = cv2.resize(rgb_image, (224, 224))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(rgb_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Apply ImageNet normalization if model was trained with it
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor
    
    def _fallback_action(self, rgb_image: np.ndarray, current_speed: float) -> VehicleAction:
        """Fallback action when ML model is not available"""
        # Simple rule-based fallback
        target_speed = 25.0  # km/h
        
        if current_speed < target_speed:
            throttle = 0.4
            brake = 0.0
        elif current_speed > target_speed + 5.0:
            throttle = 0.0
            brake = 0.3
        else:
            throttle = 0.2
            brake = 0.0
        
        return VehicleAction(
            throttle=throttle,
            steer=0.0,  # Straight driving
            brake=brake
        )

class MotionPlanner:
    """
    High-level motion planner that decides driving strategy
    """
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.waypoint_buffer = []
        self.target_speed = 25.0  # km/h
        
        # Get the map for waypoint navigation
        self.map = world.get_map()
        
        # Planning parameters
        self.lookahead_distance = 10.0  # meters
        self.comfortable_deceleration = 3.0  # m/s²
        
    def plan_motion(self, perception: PerceptionOutput, 
                   ml_action: VehicleAction,
                   destination: Optional[carla.Location] = None) -> Tuple[VehicleAction, DrivingState]:
        """
        Plan vehicle motion based on perception, ML suggestion, and destination
        """
        current_location = self.vehicle.get_location()
        current_speed = self._get_current_speed()
        
        # Determine driving state
        driving_state = self._determine_driving_state(perception, current_speed)
        
        # Plan based on current state
        if driving_state == DrivingState.EMERGENCY_BRAKE:
            action = VehicleAction(throttle=0.0, steer=0.0, brake=1.0)
        
        elif driving_state == DrivingState.STOPPING_FOR_TRAFFIC_LIGHT:
            action = self._plan_traffic_light_stop(perception, current_speed)
        
        elif driving_state == DrivingState.FOLLOWING_VEHICLE:
            action = self._plan_vehicle_following(perception, current_speed, ml_action)
        
        elif driving_state == DrivingState.INTERSECTION_APPROACH:
            action = self._plan_intersection_approach(perception, current_speed, ml_action)
        
        else:  # NORMAL_DRIVING
            action = self._plan_normal_driving(perception, current_speed, ml_action, destination)
        
        return action, driving_state
    
    def _determine_driving_state(self, perception: PerceptionOutput, current_speed: float) -> DrivingState:
        """Determine current driving state based on environment"""
        
        # Check for emergency situations first
        if perception.safety_metrics.emergency_brake_needed:
            return DrivingState.EMERGENCY_BRAKE
        
        # Check for immediate collision risk
        for obstacle in perception.obstacles:
            if (obstacle.distance < 5.0 and 
                obstacle.lane_assignment == 0 and
                obstacle.confidence > 0.7):
                return DrivingState.EMERGENCY_BRAKE
        
        # Check for traffic lights
        if (perception.traffic_light and 
            perception.traffic_light.relevant_for_ego and
            perception.traffic_light.state in [TrafficLightState.RED, TrafficLightState.YELLOW]):
            if perception.traffic_light.distance < 50.0:
                return DrivingState.STOPPING_FOR_TRAFFIC_LIGHT
        
        # Check for vehicle following
        if perception.safety_metrics.following_distance < 20.0:
            return DrivingState.FOLLOWING_VEHICLE
        
        # Check for intersection approach
        if perception.intersection_ahead:
            return DrivingState.INTERSECTION_APPROACH
        
        return DrivingState.NORMAL_DRIVING
    
    def _plan_traffic_light_stop(self, perception: PerceptionOutput, current_speed: float) -> VehicleAction:
        """Plan smooth stop for traffic light"""
        if not perception.traffic_light:
            return VehicleAction(throttle=0.0, steer=0.0, brake=0.4)
        
        distance = perception.traffic_light.distance
        
        # Calculate required deceleration for smooth stop
        if distance > 0 and current_speed > 1.0:
            current_speed_ms = current_speed / 3.6
            required_deceleration = (current_speed_ms ** 2) / (2 * distance)
            
            # Limit to comfortable deceleration
            if required_deceleration > self.comfortable_deceleration:
                brake_intensity = 0.8  # Strong braking needed
            elif required_deceleration > 1.0:
                brake_intensity = 0.5  # Moderate braking
            else:
                brake_intensity = 0.3  # Gentle braking
        else:
            brake_intensity = 0.6
        
        # Lane keeping while braking
        steer = self._calculate_lane_keeping_steer(perception)
        
        return VehicleAction(
            throttle=0.0,
            steer=steer,
            brake=brake_intensity
        )
    
    def _plan_vehicle_following(self, perception: PerceptionOutput, 
                               current_speed: float, ml_action: VehicleAction) -> VehicleAction:
        """Plan action for following another vehicle using adaptive cruise control"""
        following_distance = perception.safety_metrics.following_distance
        
        if following_distance == float('inf'):
            # No vehicle to follow, use normal driving
            return self._plan_normal_driving(perception, current_speed, ml_action, None)
        
        # Adaptive cruise control logic
        desired_distance = max(3.0, current_speed / 3.6 * 1.5)  # 1.5-second rule
        distance_error = following_distance - desired_distance
        
        # PID-like controller for following
        if distance_error > 3.0:  # Too far - speed up gradually
            throttle = min(0.5, ml_action.throttle + 0.1)
            brake = 0.0
        elif distance_error < -3.0:  # Too close - slow down
            throttle = 0.0
            brake = min(0.6, abs(distance_error) * 0.1)
        else:  # Good distance - maintain
            # Use ML action but limit acceleration
            throttle = min(ml_action.throttle, 0.3)
            brake = ml_action.brake
        
        # Lane keeping with ML steering bias
        lane_steer = self._calculate_lane_keeping_steer(perception)
        combined_steer = 0.7 * lane_steer + 0.3 * ml_action.steer
        
        return VehicleAction(
            throttle=throttle,
            steer=combined_steer,
            brake=brake
        )
    
    def _plan_intersection_approach(self, perception: PerceptionOutput, 
                                   current_speed: float, ml_action: VehicleAction) -> VehicleAction:
        """Plan approach to intersection with caution"""
        # Reduce speed when approaching intersection
        target_speed = 15.0  # Slower speed for intersection
        
        if current_speed > target_speed:
            throttle = 0.0
            brake = 0.4
        else:
            # Use ML action but be more conservative
            throttle = min(ml_action.throttle, 0.3)
            brake = max(ml_action.brake, 0.1)  # Slight brake bias
        
        # Careful steering - prefer lane keeping over ML
        steer = self._calculate_lane_keeping_steer(perception)
        if abs(steer) < 0.1:  # Only use ML steering if lane keeping is stable
            steer = 0.5 * steer + 0.5 * ml_action.steer
        
        return VehicleAction(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
    
    def _plan_normal_driving(self, perception: PerceptionOutput, 
                           current_speed: float, 
                           ml_action: VehicleAction,
                           destination: Optional[carla.Location]) -> VehicleAction:
        """Plan normal driving behavior combining ML and rules"""
        
        # Speed control with ML bias
        if current_speed < self.target_speed:
            throttle = ml_action.throttle
            brake = 0.0
        elif current_speed > self.target_speed + 10.0:
            throttle = 0.0
            brake = max(ml_action.brake, 0.3)
        else:
            throttle = ml_action.throttle
            brake = ml_action.brake
        
        # Steering combination
        steer = self._calculate_steering(perception, ml_action, destination)
        
        return VehicleAction(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
    
    def _calculate_steering(self, perception: PerceptionOutput, 
                          ml_action: VehicleAction,
                          destination: Optional[carla.Location]) -> float:
        """Calculate steering combining ML prediction with lane keeping and navigation"""
        
        # Lane keeping component
        lane_steer = self._calculate_lane_keeping_steer(perception)
        
        # Navigation component
        waypoint_steer = 0.0
        if destination:
            waypoint_steer = self._calculate_waypoint_steer(destination)
        
        # ML component
        ml_steer = ml_action.steer
        
        # Weighted combination based on lane detection confidence
        if perception.lane_info and perception.lane_info.confidence > 0.7:
            # High confidence in lane detection - prioritize lane keeping
            combined_steer = 0.6 * lane_steer + 0.3 * ml_steer + 0.1 * waypoint_steer
        else:
            # Low confidence - rely more on ML
            combined_steer = 0.7 * ml_steer + 0.2 * lane_steer + 0.1 * waypoint_steer
        
        return np.clip(combined_steer, -1.0, 1.0)
    
    def _calculate_lane_keeping_steer(self, perception: PerceptionOutput) -> float:
        """Calculate steering to stay in lane center"""
        if not perception.lane_info or not perception.lane_info.center_line:
            return 0.0
        
        lane_info = perception.lane_info
        
        # Proportional controller for lane keeping
        lateral_error = lane_info.lane_center_offset
        heading_error = lane_info.heading_angle
        
        # PD controller
        steer_command = -0.5 * lateral_error - 0.3 * heading_error
        
        return np.clip(steer_command, -0.6, 0.6)
    
    def _calculate_waypoint_steer(self, destination: carla.Location) -> float:
        """Calculate steering towards waypoint/destination"""
        current_location = self.vehicle.get_location()
        current_rotation = self.vehicle.get_transform().rotation
        
        # Get waypoint towards destination
        current_waypoint = self.map.get_waypoint(current_location)
        
        if not current_waypoint:
            return 0.0
        
        # Simple waypoint following - get next waypoint
        next_waypoints = current_waypoint.next(self.lookahead_distance)
        if not next_waypoints:
            return 0.0
        
        next_waypoint = next_waypoints[0]
        target_location = next_waypoint.transform.location
        
        # Calculate angle to target
        dx = target_location.x - current_location.x
        dy = target_location.y - current_location.y
        target_angle = math.atan2(dy, dx)
        
        # Convert vehicle rotation to radians
        current_angle = math.radians(current_rotation.yaw)
        
        # Calculate steering angle
        angle_diff = target_angle - current_angle
        
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Convert to steering command
        steer_command = angle_diff / math.pi  # Normalize to [-1, 1]
        
        return np.clip(steer_command, -0.8, 0.8)
    
    def _get_current_speed(self) -> float:
        """Get current vehicle speed in km/h"""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6
        return speed_kmh

class HybridAgent:
    """
    Main autonomous driving agent that combines perception, ML, planning, and safety
    """
    
    def __init__(self, world, vehicle, model_path: Optional[str] = None):
        self.world = world
        self.vehicle = vehicle
        
        # Initialize subsystems
        self.perception = PerceptionSystem()
        self.ml_controller = MLController(model_path)
        self.motion_planner = MotionPlanner(world, vehicle)
        self.safety_monitor = SafetyMonitor()
        
        # State tracking
        self.current_state = DrivingState.NORMAL_DRIVING
        self.destination = None
        
        # Performance metrics
        self.total_frames = 0
        self.safety_interventions = 0
        self.ml_predictions = 0
        self.successful_ml_predictions = 0
        
        # Sensor data storage
        self.latest_sensor_data = {}
        
    def set_destination(self, destination: carla.Location):
        """Set navigation destination"""
        self.destination = destination
        print(f"Destination set to: ({destination.x:.2f}, {destination.y:.2f})")
    
    def step(self, sensor_data: Dict[str, Any]) -> carla.VehicleControl:
        """
        Main control step - processes sensors and returns vehicle control
        """
        self.total_frames += 1
        self.latest_sensor_data = sensor_data
        
        # 1. Perception - understand environment
        perception_output = self.perception.process_sensors(
            sensor_data, 
            frame_id=self.total_frames,
            timestamp=time.time()
        )
        
        # 2. ML Prediction - get ML-based control suggestion
        rgb_image = sensor_data.get('rgb')
        semantic_image = sensor_data.get('semantic')
        current_speed = self.motion_planner._get_current_speed()
        
        if rgb_image is not None and semantic_image is not None:
            ml_action = self.ml_controller.predict_action(rgb_image, semantic_image, current_speed)
            self.ml_predictions += 1
            if ml_action.throttle > 0 or abs(ml_action.steer) > 0.05:
                self.successful_ml_predictions += 1
        else:
            ml_action = VehicleAction(throttle=0.3, steer=0.0, brake=0.0)
        
        # 3. Motion Planning - combine ML with rule-based planning
        planned_action, driving_state = self.motion_planner.plan_motion(
            perception_output, ml_action, self.destination
        )
        self.current_state = driving_state
        
        # 4. Safety Validation - ensure action is safe (FINAL OVERRIDE)
        safe_action = self.safety_monitor.validate_action(
            planned_action, 
            perception_output, 
            current_speed,
            self.vehicle.get_location()
        )
        
        # Track safety interventions
        if (abs(safe_action.throttle - planned_action.throttle) > 0.1 or
            abs(safe_action.brake - planned_action.brake) > 0.1 or
            abs(safe_action.steer - planned_action.steer) > 0.1):
            self.safety_interventions += 1
        
        # 5. Convert to CARLA control
        control = carla.VehicleControl(
            throttle=safe_action.throttle,
            steer=safe_action.steer,
            brake=safe_action.brake,
            hand_brake=safe_action.hand_brake,
            reverse=safe_action.reverse
        )
        
        return control
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status for debugging/monitoring"""
        ml_success_rate = (self.successful_ml_predictions / max(self.ml_predictions, 1))
        intervention_rate = self.safety_interventions / max(self.total_frames, 1)
        
        status = {
            'driving_state': self.current_state.value,
            'total_frames': self.total_frames,
            'safety_interventions': self.safety_interventions,
            'intervention_rate': intervention_rate,
            'current_speed': self.motion_planner._get_current_speed(),
            'has_destination': self.destination is not None,
            'ml_model_loaded': self.ml_controller.model_loaded,
            'ml_success_rate': ml_success_rate,
            'safety_violations': self.safety_monitor.safety_violations
        }
        
        # Add sensor quality if available
        if self.latest_sensor_data:
            status['sensor_data_available'] = {
                'rgb': 'rgb' in self.latest_sensor_data,
                'semantic': 'semantic' in self.latest_sensor_data,
                'depth': 'depth' in self.latest_sensor_data,
                'gps': 'gps' in self.latest_sensor_data,
                'imu': 'imu' in self.latest_sensor_data
            }
        
        return status
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information"""
        return {
            'perception_stats': {
                'vehicles_detected': len(getattr(self.perception, 'last_perception', PerceptionOutput()).vehicles),
                'pedestrians_detected': len(getattr(self.perception, 'last_perception', PerceptionOutput()).pedestrians),
                'traffic_lights_detected': len(getattr(self.perception, 'last_perception', PerceptionOutput()).traffic_lights),
                'lane_detected': getattr(self.perception, 'last_perception', PerceptionOutput()).lane_info is not None
            },
            'safety_stats': {
                'total_interventions': self.safety_interventions,
                'recent_violations': self.safety_monitor.safety_violations,
                'brake_count': self.safety_monitor.brake_count
            },
            'ml_stats': {
                'model_loaded': self.ml_controller.model_loaded,
                'total_predictions': self.ml_predictions,
                'successful_predictions': self.successful_ml_predictions
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_frames = 0
        self.safety_interventions = 0
        self.ml_predictions = 0
        self.successful_ml_predictions = 0
        self.safety_monitor.safety_violations = 0
        self.safety_monitor.brake_count = 0
        print("Agent metrics reset")

# Utility functions for integration with main system
def create_hybrid_agent(world, vehicle, model_path: Optional[str] = None) -> HybridAgent:
    """Factory function to create a properly configured hybrid agent"""
    return HybridAgent(world, vehicle, model_path)

def get_agent_capabilities() -> Dict[str, bool]:
    """Get agent capabilities for system validation"""
    return {
        'ml_prediction': True,
        'rule_based_safety': True,
        'traffic_light_detection': True,
        'lane_keeping': True,
        'collision_avoidance': True,
        'speed_limit_enforcement': True,
        'following_distance_control': True,
        'intersection_handling': True,
        'waypoint_navigation': True
    }