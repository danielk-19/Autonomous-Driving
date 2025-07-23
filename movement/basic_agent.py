"""
Hybrid autonomous driving agent combining rule-based safety with ML-based control.
This agent prioritizes safety through hard-coded traffic rules while using ML for smooth driving.
"""
import carla
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple
import math

from .perception import PerceptionSystem, PerceptionOutput, DetectedObject

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
        self.min_following_distance = 5.0  # meters
        self.emergency_brake_distance = 3.0  # meters
        self.max_speed = 30.0  # km/h (city speed limit)
        self.traffic_light_stop_distance = 50.0  # meters
        
        # State tracking
        self.previous_actions = []
        self.brake_count = 0
        
    def validate_action(self, proposed_action: VehicleAction, 
                       perception: PerceptionOutput, 
                       current_speed: float) -> VehicleAction:
        """
        Validate and potentially override proposed action for safety
        """
        safe_action = VehicleAction(
            throttle=proposed_action.throttle,
            steer=proposed_action.steer,
            brake=proposed_action.brake
        )
        
        # Rule 1: Emergency collision avoidance
        if self._check_emergency_collision(perception):
            return VehicleAction(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
        
        # Rule 2: Traffic light compliance
        if self._check_traffic_light_violation(perception, current_speed):
            return VehicleAction(throttle=0.0, steer=proposed_action.steer, brake=0.8)
        
        # Rule 3: Speed limit enforcement
        if current_speed > self.max_speed:
            safe_action.throttle = 0.0
            safe_action.brake = max(safe_action.brake, 0.3)
        
        # Rule 4: Following distance
        following_action = self._enforce_following_distance(safe_action, perception, current_speed)
        if following_action:
            safe_action = following_action
        
        # Rule 5: Lateral safety (prevent dangerous steering)
        safe_action.steer = np.clip(safe_action.steer, -0.8, 0.8)
        
        return safe_action
    
    def _check_emergency_collision(self, perception: PerceptionOutput) -> bool:
        """Check for imminent collision requiring emergency brake"""
        for obstacle in perception.obstacles:
            if (obstacle.distance < self.emergency_brake_distance and 
                abs(obstacle.relative_position[0]) < 0.5):  # In front of us
                return True
        return False
    
    def _check_traffic_light_violation(self, perception: PerceptionOutput, current_speed: float) -> bool:
        """Check if we should stop for a red traffic light"""
        if perception.traffic_light is None:
            return False
        
        traffic_light = perception.traffic_light
        
        # Red light - must stop if we can safely do so
        if traffic_light.state == 'red':
            # If we're close and moving fast, might be safer to continue
            if traffic_light.distance < 5.0 and current_speed > 15.0:
                return False  # Too late to stop safely
            elif traffic_light.distance < self.traffic_light_stop_distance:
                return True
        
        # Yellow light - stop if we can do so safely
        elif traffic_light.state == 'yellow':
            if traffic_light.distance < 20.0 and current_speed < 10.0:
                return True
        
        return False
    
    def _enforce_following_distance(self, action: VehicleAction, 
                                   perception: PerceptionOutput, 
                                   current_speed: float) -> Optional[VehicleAction]:
        """Enforce safe following distance behind vehicles"""
        # Find closest vehicle in front
        closest_vehicle = None
        min_distance = float('inf')
        
        for obj in perception.obstacles:
            if (obj.type == 'vehicle' and 
                obj.relative_position[1] > 0 and  # In front
                abs(obj.relative_position[0]) < 0.3 and  # In our lane
                obj.distance < min_distance):
                min_distance = obj.distance
                closest_vehicle = obj
        
        if closest_vehicle is None:
            return None
        
        # Calculate required following distance based on speed
        required_distance = max(self.min_following_distance, current_speed * 0.5)
        
        if closest_vehicle.distance < required_distance:
            # Too close - reduce throttle and apply brakes
            brake_intensity = 1.0 - (closest_vehicle.distance / required_distance)
            brake_intensity = np.clip(brake_intensity, 0.0, 0.8)
            
            return VehicleAction(
                throttle=0.0,
                steer=action.steer,
                brake=brake_intensity
            )
        
        return None

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
        
    def plan_motion(self, perception: PerceptionOutput, 
                   destination: Optional[carla.Location] = None) -> Tuple[VehicleAction, DrivingState]:
        """
        Plan vehicle motion based on perception and destination
        """
        current_location = self.vehicle.get_location()
        current_speed = self._get_current_speed()
        
        # Determine driving state
        driving_state = self._determine_driving_state(perception, current_speed)
        
        # Plan based on current state
        if driving_state == DrivingState.EMERGENCY_BRAKE:
            action = VehicleAction(throttle=0.0, steer=0.0, brake=1.0)
        
        elif driving_state == DrivingState.STOPPING_FOR_TRAFFIC_LIGHT:
            action = self._plan_traffic_light_stop(perception)
        
        elif driving_state == DrivingState.FOLLOWING_VEHICLE:
            action = self._plan_vehicle_following(perception, current_speed)
        
        elif driving_state == DrivingState.INTERSECTION_APPROACH:
            action = self._plan_intersection_approach(perception, current_speed)
        
        else:  # NORMAL_DRIVING
            action = self._plan_normal_driving(perception, current_speed, destination)
        
        return action, driving_state
    
    def _determine_driving_state(self, perception: PerceptionOutput, current_speed: float) -> DrivingState:
        """Determine current driving state based on environment"""
        
        # Check for emergency situations
        for obstacle in perception.obstacles:
            if obstacle.distance < 3.0 and abs(obstacle.relative_position[0]) < 0.5:
                return DrivingState.EMERGENCY_BRAKE
        
        # Check for traffic lights
        if perception.traffic_light and perception.traffic_light.state in ['red', 'yellow']:
            if perception.traffic_light.distance < 50.0:
                return DrivingState.STOPPING_FOR_TRAFFIC_LIGHT
        
        # Check for vehicle following
        for obj in perception.obstacles:
            if (obj.type == 'vehicle' and 
                obj.distance < 20.0 and 
                abs(obj.relative_position[0]) < 0.4):
                return DrivingState.FOLLOWING_VEHICLE
        
        # Check for intersection approach (simplified)
        if perception.traffic_light and perception.traffic_light.distance < 30.0:
            return DrivingState.INTERSECTION_APPROACH
        
        return DrivingState.NORMAL_DRIVING
    
    def _plan_traffic_light_stop(self, perception: PerceptionOutput) -> VehicleAction:
        """Plan smooth stop for traffic light"""
        if not perception.traffic_light:
            return VehicleAction(throttle=0.0, steer=0.0, brake=0.3)
        
        distance = perception.traffic_light.distance
        current_speed = self._get_current_speed()
        
        # Calculate brake intensity based on distance and speed
        if distance > 20.0:
            brake_intensity = 0.2
        elif distance > 10.0:
            brake_intensity = 0.4
        else:
            brake_intensity = 0.6
        
        # Smooth deceleration
        if current_speed > 5.0:
            brake_intensity = min(brake_intensity, 0.5)
        
        return VehicleAction(
            throttle=0.0,
            steer=0.0,
            brake=brake_intensity
        )
    
    def _plan_vehicle_following(self, perception: PerceptionOutput, current_speed: float) -> VehicleAction:
        """Plan action for following another vehicle"""
        # Find lead vehicle
        lead_vehicle = None
        for obj in perception.obstacles:
            if (obj.type == 'vehicle' and 
                obj.distance < 20.0 and 
                abs(obj.relative_position[0]) < 0.4):
                if lead_vehicle is None or obj.distance < lead_vehicle.distance:
                    lead_vehicle = obj
        
        if not lead_vehicle:
            return self._plan_normal_driving(perception, current_speed, None)
        
        # Adaptive cruise control logic
        desired_distance = max(5.0, current_speed * 0.6)  # Time-based following distance
        distance_error = lead_vehicle.distance - desired_distance
        
        if distance_error > 2.0:  # Too far - speed up
            throttle = min(0.4, distance_error * 0.1)
            brake = 0.0
        elif distance_error < -2.0:  # Too close - slow down
            throttle = 0.0
            brake = min(0.5, abs(distance_error) * 0.1)
        else:  # Maintain current speed
            throttle = 0.2
            brake = 0.0
        
        # Lateral control - stay centered in lane
        steer = self._calculate_lane_keeping_steer(perception)
        
        return VehicleAction(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
    
    def _plan_intersection_approach(self, perception: PerceptionOutput, current_speed: float) -> VehicleAction:
        """Plan approach to intersection"""
        # Reduce speed when approaching intersection
        target_speed = 15.0  # Slower speed for intersection
        
        if current_speed > target_speed:
            return VehicleAction(
                throttle=0.0,
                steer=self._calculate_lane_keeping_steer(perception),
                brake=0.3
            )
        else:
            return VehicleAction(
                throttle=0.2,
                steer=self._calculate_lane_keeping_steer(perception),
                brake=0.0
            )
    
    def _plan_normal_driving(self, perception: PerceptionOutput, 
                           current_speed: float, 
                           destination: Optional[carla.Location]) -> VehicleAction:
        """Plan normal driving behavior"""
        
        # Speed control
        if current_speed < self.target_speed:
            throttle = 0.5
            brake = 0.0
        elif current_speed > self.target_speed + 5.0:
            throttle = 0.0
            brake = 0.2
        else:
            throttle = 0.3
            brake = 0.0
        
        # Lateral control
        steer = self._calculate_steering(perception, destination)
        
        return VehicleAction(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
    
    def _calculate_steering(self, perception: PerceptionOutput, 
                          destination: Optional[carla.Location]) -> float:
        """Calculate steering angle based on lane keeping and navigation"""
        
        # Primary: Lane keeping
        lane_steer = self._calculate_lane_keeping_steer(perception)
        
        # Secondary: Waypoint following if destination provided
        waypoint_steer = 0.0
        if destination:
            waypoint_steer = self._calculate_waypoint_steer(destination)
        
        # Combine steering inputs (lane keeping has priority)
        combined_steer = 0.7 * lane_steer + 0.3 * waypoint_steer
        
        return np.clip(combined_steer, -1.0, 1.0)
    
    def _calculate_lane_keeping_steer(self, perception: PerceptionOutput) -> float:
        """Calculate steering to stay in lane center"""
        if not perception.lane_info:
            return 0.0
        
        lane_info = perception.lane_info
        
        # Simple lane keeping - steer towards lane center
        image_center = 400  # Image width / 2
        lane_center_x = lane_info.lane_center[0] if len(lane_info.lane_center) > 0 else image_center
        
        # Calculate lateral offset from lane center
        lateral_offset = (lane_center_x - image_center) / image_center
        
        # Proportional controller for lane keeping
        steer_gain = 0.5
        steer_command = -lateral_offset * steer_gain
        
        # Add heading angle compensation if available
        if lane_info.heading_angle != 0:
            steer_command += lane_info.heading_angle * 0.3
        
        return np.clip(steer_command, -0.5, 0.5)
    
    def _calculate_waypoint_steer(self, destination: carla.Location) -> float:
        """Calculate steering towards waypoint/destination"""
        current_location = self.vehicle.get_location()
        current_rotation = self.vehicle.get_transform().rotation
        
        # Get waypoint towards destination
        current_waypoint = self.map.get_waypoint(current_location)
        
        if not current_waypoint:
            return 0.0
        
        # Simple waypoint following - get next waypoint
        next_waypoints = current_waypoint.next(5.0)  # 5 meters ahead
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
    Main autonomous driving agent that combines perception, planning, and safety
    """
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        
        # Initialize subsystems
        self.perception = PerceptionSystem()
        self.motion_planner = MotionPlanner(world, vehicle)
        self.safety_monitor = SafetyMonitor()
        
        # State tracking
        self.current_state = DrivingState.NORMAL_DRIVING
        self.destination = None
        
        # Performance metrics
        self.total_frames = 0
        self.safety_interventions = 0
        
    def set_destination(self, destination: carla.Location):
        """Set navigation destination"""
        self.destination = destination
    
    def step(self, sensor_data) -> carla.VehicleControl:
        """
        Main control step - processes sensors and returns vehicle control
        """
        self.total_frames += 1
        
        # 1. Perception - understand environment
        perception_output = self.perception.process_sensors(sensor_data)
        
        # 2. Motion Planning - decide what to do
        planned_action, driving_state = self.motion_planner.plan_motion(
            perception_output, self.destination
        )
        self.current_state = driving_state
        
        # 3. Safety Validation - ensure action is safe
        current_speed = self.motion_planner._get_current_speed()
        safe_action = self.safety_monitor.validate_action(
            planned_action, perception_output, current_speed
        )
        
        # Track safety interventions
        if (abs(safe_action.throttle - planned_action.throttle) > 0.1 or
            abs(safe_action.brake - planned_action.brake) > 0.1):
            self.safety_interventions += 1
        
        # 4. Convert to CARLA control
        control = carla.VehicleControl(
            throttle=safe_action.throttle,
            steer=safe_action.steer,
            brake=safe_action.brake,
            hand_brake=safe_action.hand_brake,
            reverse=safe_action.reverse
        )
        
        return control
    
    def get_status(self) -> dict:
        """Get current agent status for debugging/monitoring"""
        return {
            'driving_state': self.current_state.value,
            'total_frames': self.total_frames,
            'safety_interventions': self.safety_interventions,
            'intervention_rate': self.safety_interventions / max(self.total_frames, 1),
            'current_speed': self.motion_planner._get_current_speed(),
            'has_destination': self.destination is not None
        }