"""
CARLA Agent Evaluation System
Evaluates autonomous driving agent performance across multiple metrics
"""

import carla
import json
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from movement.basic_agent import HybridAgent
from movement.perception import PerceptionSystem
from sensors.capture_sensors import SensorManager

class AgentEvaluator:
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.map = world.get_map()
        
        # Evaluation metrics
        self.reset_metrics()
        
        # Test scenarios
        self.scenarios = [
            'urban_driving',
            'highway_driving', 
            'traffic_lights',
            'intersections',
            'lane_changes'
        ]
    
    def reset_metrics(self):
        """Reset all evaluation metrics"""
        self.metrics = {
            'total_distance': 0.0,
            'total_time': 0.0,
            'average_speed': 0.0,
            'traffic_violations': {
                'red_light_violations': 0,
                'speed_violations': 0,
                'collision_count': 0,
                'lane_violations': 0
            },
            'navigation_accuracy': {
                'waypoints_reached': 0,
                'waypoints_missed': 0,
                'route_completion': 0.0
            },
            'driving_quality': {
                'smooth_steering': 0.0,
                'smooth_acceleration': 0.0,
                'following_distance': 0.0
            },
            'safety_metrics': {
                'emergency_stops': 0,
                'close_calls': 0,
                'rule_overrides': 0
            }
        }
        
        self.start_time = None
        self.last_location = None
        self.steering_history = []
        self.speed_history = []
        
    def setup_test_route(self, scenario_type='urban_driving'):
        """Setup test route based on scenario type"""
        spawn_points = self.map.get_spawn_points()
        
        if scenario_type == 'urban_driving':
            # Select spawn points in urban areas
            start_point = spawn_points[0]
            end_point = spawn_points[10]
        elif scenario_type == 'highway_driving':
            # Select highway spawn points
            start_point = spawn_points[5]
            end_point = spawn_points[15]
        elif scenario_type == 'traffic_lights':
            # Route through intersections with traffic lights
            start_point = spawn_points[2]
            end_point = spawn_points[8]
        else:
            # Default route
            start_point = spawn_points[0]
            end_point = spawn_points[20]
            
        return start_point, end_point
    
    def spawn_test_vehicle(self, spawn_point):
        """Spawn vehicle for testing"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.world.tick()
            vehicle.set_simulate_physics(True)
            return vehicle
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return None
    
    def setup_traffic_scenario(self, density='medium'):
        """Add background traffic for more realistic testing"""
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        
        # Spawn background vehicles
        spawn_points = self.map.get_spawn_points()
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        
        num_vehicles = {'low': 20, 'medium': 50, 'high': 100}[density]
        
        background_vehicles = []
        for i in range(min(num_vehicles, len(spawn_points))):
            try:
                bp = np.random.choice(vehicle_bps)
                vehicle = self.world.spawn_actor(bp, spawn_points[i])
                vehicle.set_autopilot(True)
                background_vehicles.append(vehicle)
            except:
                continue
        
        return background_vehicles
    
    def update_metrics(self, vehicle, control, perception_data):
        """Update evaluation metrics during testing"""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            self.last_location = vehicle.get_location()
            return
        
        # Calculate distance traveled
        current_location = vehicle.get_location()
        if self.last_location:
            distance = current_location.distance(self.last_location)
            self.metrics['total_distance'] += distance
        
        # Update time
        self.metrics['total_time'] = current_time - self.start_time
        
        # Speed tracking
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        self.speed_history.append(speed)
        
        # Steering smoothness
        self.steering_history.append(control.steer)
        
        # Check for traffic violations
        self.check_traffic_violations(vehicle, perception_data, speed)
        
        # Update driving quality metrics
        self.update_driving_quality()
        
        self.last_location = current_location
    
    def check_traffic_violations(self, vehicle, perception_data, speed):
        """Check for various traffic violations"""
        # Speed limit violations
        speed_limit = 50  # km/h default urban speed limit
        if speed > speed_limit * 1.1:  # 10% tolerance
            self.metrics['traffic_violations']['speed_violations'] += 1
        
        # Red light violations
        if hasattr(perception_data, 'traffic_lights'):
            for light_state in perception_data.traffic_lights:
                if light_state == 'red' and speed > 5:  # Moving through red light
                    self.metrics['traffic_violations']['red_light_violations'] += 1
        
        # Collision detection
        if len(vehicle.get_world().get_actors().filter('sensor.other.collision*')) > 0:
            self.metrics['traffic_violations']['collision_count'] += 1
    
    def update_driving_quality(self):
        """Calculate driving quality metrics"""
        if len(self.steering_history) > 10:
            # Steering smoothness (lower is better)
            steering_changes = np.diff(self.steering_history[-10:])
            self.metrics['driving_quality']['smooth_steering'] = np.std(steering_changes)
        
        if len(self.speed_history) > 10:
            # Speed smoothness
            speed_changes = np.diff(self.speed_history[-10:])
            self.metrics['driving_quality']['smooth_acceleration'] = np.std(speed_changes)
    
    def check_route_completion(self, vehicle, target_location, tolerance=5.0):
        """Check if vehicle reached target destination"""
        current_location = vehicle.get_location()
        distance_to_target = current_location.distance(target_location)
        
        if distance_to_target < tolerance:
            return True
        return False
    
    def calculate_final_scores(self):
        """Calculate final evaluation scores"""
        scores = {}
        
        # Safety Score (0-100, higher is better)
        total_violations = sum(self.metrics['traffic_violations'].values())
        safety_score = max(0, 100 - (total_violations * 10))
        scores['safety'] = safety_score
        
        # Efficiency Score (based on time and distance)
        if self.metrics['total_time'] > 0:
            avg_speed = (self.metrics['total_distance'] / 1000) / (self.metrics['total_time'] / 3600)
            efficiency_score = min(100, (avg_speed / 30) * 100)  # 30 km/h as baseline
        else:
            efficiency_score = 0
        scores['efficiency'] = efficiency_score
        
        # Comfort Score (based on smoothness)
        steering_smoothness = self.metrics['driving_quality']['smooth_steering']
        comfort_score = max(0, 100 - (steering_smoothness * 1000))
        scores['comfort'] = comfort_score
        
        # Overall Score
        scores['overall'] = (safety_score * 0.5 + efficiency_score * 0.3 + comfort_score * 0.2)
        
        return scores
    
    def run_evaluation(self, scenario='urban_driving', duration=300, traffic_density='medium'):
        """Run complete evaluation of the agent"""
        print(f"Starting evaluation - Scenario: {scenario}, Duration: {duration}s")
        
        # Setup
        self.reset_metrics()
        start_point, end_point = self.setup_test_route(scenario)
        
        # Spawn vehicle
        vehicle = self.spawn_test_vehicle(start_point)
        if not vehicle:
            return None
        
        # Setup background traffic
        background_vehicles = self.setup_traffic_scenario(traffic_density)
        
        # Initialize agent and sensors
        sensor_manager = SensorManager(vehicle, self.world)
        sensor_manager.setup_sensors()
        
        perception = PerceptionSystem()
        agent = HybridAgent(vehicle, self.map)
        
        # Set destination
        agent.set_destination(end_point.location)
        
        try:
            # Main evaluation loop
            start_time = time.time()
            route_completed = False
            
            while time.time() - start_time < duration and not route_completed:
                self.world.tick()
                
                # Get sensor data
                sensor_data = sensor_manager.get_latest_data()
                if not sensor_data:
                    continue
                
                # Run perception
                perception_data = perception.process_frame(
                    sensor_data['rgb'],
                    sensor_data['semantic'],
                    sensor_data.get('lidar')
                )
                
                # Get agent decision
                control = agent.run_step(perception_data)
                vehicle.apply_control(control)
                
                # Update metrics
                self.update_metrics(vehicle, control, perception_data)
                
                # Check route completion
                route_completed = self.check_route_completion(vehicle, end_point.location)
                
                # Print progress every 30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    print(f"Progress: {elapsed:.0f}s, Distance: {self.metrics['total_distance']:.1f}m")
            
            # Calculate final scores
            scores = self.calculate_final_scores()
            
            # Update route completion metric
            if route_completed:
                self.metrics['navigation_accuracy']['route_completion'] = 1.0
            else:
                # Partial completion based on distance to target
                distance_to_target = vehicle.get_location().distance(end_point.location)
                initial_distance = start_point.location.distance(end_point.location)
                completion_ratio = max(0, 1 - (distance_to_target / initial_distance))
                self.metrics['navigation_accuracy']['route_completion'] = completion_ratio
            
            return {
                'scenario': scenario,
                'metrics': self.metrics,
                'scores': scores,
                'route_completed': route_completed,
                'duration': time.time() - start_time
            }
            
        finally:
            # Cleanup
            sensor_manager.cleanup()
            vehicle.destroy()
            for bg_vehicle in background_vehicles:
                try:
                    bg_vehicle.destroy()
                except:
                    pass
    
    def generate_report(self, results):
        """Generate detailed evaluation report"""
        report = {
            'evaluation_summary': {
                'scenario': results['scenario'],
                'duration': f"{results['duration']:.1f}s",
                'route_completed': results['route_completed'],
                'total_distance': f"{results['metrics']['total_distance']:.1f}m"
            },
            'scores': results['scores'],
            'traffic_violations': results['metrics']['traffic_violations'],
            'driving_quality': results['metrics']['driving_quality'],
            'navigation_accuracy': results['metrics']['navigation_accuracy']
        }
        
        return report

def main():
    """Run agent evaluation"""
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Initialize evaluator
        evaluator = AgentEvaluator(client, world)
        
        # Run evaluations for different scenarios
        scenarios = ['urban_driving', 'traffic_lights', 'intersections']
        all_results = []
        
        for scenario in scenarios:
            print(f"\n{'='*50}")
            print(f"Evaluating scenario: {scenario}")
            print(f"{'='*50}")
            
            results = evaluator.run_evaluation(
                scenario=scenario,
                duration=180,  # 3 minutes per scenario
                traffic_density='medium'
            )
            
            if results:
                all_results.append(results)
                report = evaluator.generate_report(results)
                
                print(f"\nResults for {scenario}:")
                print(f"Overall Score: {report['scores']['overall']:.1f}/100")
                print(f"Safety Score: {report['scores']['safety']:.1f}/100")
                print(f"Efficiency Score: {report['scores']['efficiency']:.1f}/100")
                print(f"Comfort Score: {report['scores']['comfort']:.1f}/100")
                print(f"Route Completed: {report['evaluation_summary']['route_completed']}")
                
                # Save detailed results
                output_file = f"evaluation_results_{scenario}_{int(time.time())}.json"
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Detailed results saved to: {output_file}")
        
        # Generate overall summary
        if all_results:
            overall_score = np.mean([r['scores']['overall'] for r in all_results])
            print(f"\n{'='*50}")
            print(f"OVERALL EVALUATION COMPLETE")
            print(f"Average Score Across All Scenarios: {overall_score:.1f}/100")
            print(f"{'='*50}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
    finally:
        try:
            # Reset to asynchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except:
            pass

if __name__ == '__main__':
    main()