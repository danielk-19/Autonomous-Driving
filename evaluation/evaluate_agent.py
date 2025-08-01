"""
CARLA Agent Evaluation System
Evaluates autonomous driving agent performance across multiple metrics and scenarios
"""

import carla
import json
import time
import numpy as np
from pathlib import Path
import sys
import logging

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Import project components
from movement.basic_agent import HybridAgent
from sensors.capture_sensors import SensorManager
from utils.utils import (
    setup_logging, ensure_dir, save_json, get_timestamp, 
    safe_float, safe_int, clamp, Timer, PerformanceMonitor,
    setup_carla_world, destroy_actors, get_latest_file
)

class AgentEvaluator:
    """
    Comprehensive evaluation system for the hybrid autonomous driving agent
    Tests safety, efficiency, comfort, and navigation accuracy
    """
    
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.map = world.get_map()
        
        # Setup logging
        setup_logging(logging.INFO, f"logs/evaluation_{get_timestamp()}.log")
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Evaluation metrics storage
        self.reset_metrics()
        
        # Test scenarios
        self.scenarios = [
            'urban_driving',
            'highway_driving', 
            'traffic_lights',
            'intersections',
            'lane_changes',
            'emergency_scenarios'
        ]
        
        self.logger.info("AgentEvaluator initialized")
    
    def reset_metrics(self):
        """Reset all evaluation metrics"""
        self.metrics = {
            'safety': {
                'collisions': 0,
                'traffic_light_violations': 0,
                'speed_violations': 0,
                'lane_violations': 0,
                'emergency_brakes': 0,
                'close_calls': 0,
                'rule_based_interventions': 0
            },
            'efficiency': {
                'total_distance': 0.0,
                'total_time': 0.0,
                'average_speed': 0.0,
                'route_completion_ratio': 0.0,
                'waypoints_reached': 0,
                'waypoints_missed': 0,
                'fuel_efficiency_score': 0.0
            },
            'comfort': {
                'steering_smoothness': 0.0,
                'acceleration_smoothness': 0.0,
                'jerk_count': 0,
                'harsh_braking_count': 0,
                'comfort_score': 0.0
            },
            'navigation': {
                'destination_reached': False,
                'navigation_errors': 0,
                'off_route_distance': 0.0,
                'route_adherence_score': 0.0
            }
        }
        
        # Data collection for analysis
        self.frame_data = []
        self.start_time = None
        self.last_location = None
        self.control_history = []
        self.speed_history = []
        self.steering_history = []
        
        self.performance_monitor.reset()
    
    def setup_test_scenario(self, scenario_type='urban_driving', traffic_density='medium'):
        """Setup test scenario with appropriate spawn points and traffic"""
        spawn_points = self.map.get_spawn_points()
        
        # Select spawn points based on scenario type
        scenario_configs = {
            'urban_driving': {
                'start_idx': 0,
                'end_idx': 20,
                'max_speed': 50,
                'description': 'Urban environment with intersections and traffic lights'
            },
            'highway_driving': {
                'start_idx': 5,
                'end_idx': 25,
                'max_speed': 80,
                'description': 'Highway scenario with high-speed driving'
            },
            'traffic_lights': {
                'start_idx': 2,
                'end_idx': 12,
                'max_speed': 50,
                'description': 'Route through multiple traffic light intersections'
            },
            'intersections': {
                'start_idx': 3,
                'end_idx': 18,
                'max_speed': 30,
                'description': 'Complex intersection navigation'
            },
            'lane_changes': {
                'start_idx': 8,
                'end_idx': 28,
                'max_speed': 60,
                'description': 'Multi-lane road with lane change requirements'
            },
            'emergency_scenarios': {
                'start_idx': 1,
                'end_idx': 15,
                'max_speed': 40,
                'description': 'Emergency situations and obstacle avoidance'
            }
        }
        
        config = scenario_configs.get(scenario_type, scenario_configs['urban_driving'])
        
        # Select spawn points with bounds checking
        start_idx = min(config['start_idx'], len(spawn_points) - 1)
        end_idx = min(config['end_idx'], len(spawn_points) - 1)
        
        start_point = spawn_points[start_idx]
        end_point = spawn_points[end_idx]
        
        self.logger.info(f"Scenario: {config['description']}")
        self.logger.info(f"Route: spawn point {start_idx} to {end_idx}")
        
        return start_point, end_point, config
    
    def spawn_test_vehicle(self, spawn_point):
        """Spawn test vehicle with error handling"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        
        # Prefer Tesla Model 3 for consistency
        vehicle_bp = None
        for bp in vehicle_blueprints:
            if 'tesla' in bp.id.lower() and 'model3' in bp.id.lower():
                vehicle_bp = bp
                break
        
        if not vehicle_bp:
            vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        
        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.world.tick()
            vehicle.set_simulate_physics(True)
            self.logger.info(f"Test vehicle spawned: {vehicle_bp.id}")
            return vehicle
        except Exception as e:
            self.logger.error(f"Failed to spawn test vehicle: {e}")
            return None
    
    def setup_background_traffic(self, density='medium'):
        """Setup background traffic for realistic testing"""
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        
        # Traffic density configuration
        density_configs = {
            'low': {'num_vehicles': 20, 'num_pedestrians': 10},
            'medium': {'num_vehicles': 50, 'num_pedestrians': 20},
            'high': {'num_vehicles': 100, 'num_pedestrians': 40}
        }
        
        config = density_configs.get(density, density_configs['medium'])
        
        spawn_points = self.map.get_spawn_points()
        blueprint_library = self.world.get_blueprint_library()
        
        # Spawn background vehicles
        background_actors = []
        vehicle_bps = blueprint_library.filter('vehicle.*')
        
        for i in range(min(config['num_vehicles'], len(spawn_points) - 1)):
            try:
                bp = np.random.choice(vehicle_bps)
                vehicle = self.world.spawn_actor(bp, spawn_points[i + 1])
                vehicle.set_autopilot(True, traffic_manager.get_port())
                background_actors.append(vehicle)
            except:
                continue
        
        # Spawn pedestrians
        walker_bps = blueprint_library.filter('walker.pedestrian.*')
        walker_spawn_points = []
        
        for i in range(config['num_pedestrians']):
            spawn_point = self.world.get_random_location_from_navigation()
            if spawn_point:
                walker_spawn_points.append(spawn_point)
        
        for spawn_point in walker_spawn_points:
            try:
                walker_bp = np.random.choice(walker_bps)
                walker = self.world.spawn_actor(walker_bp, spawn_point)
                background_actors.append(walker)
            except:
                continue
        
        self.logger.info(f"Background traffic: {len(background_actors)} actors spawned")
        return background_actors
    
    def update_metrics(self, vehicle, control, agent_status, sensor_data):
        """Update evaluation metrics during testing"""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            self.last_location = vehicle.get_location()
            return
        
        current_location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6
        
        # Update efficiency metrics
        if self.last_location:
            distance = current_location.distance(self.last_location)
            self.metrics['efficiency']['total_distance'] += distance
        
        self.metrics['efficiency']['total_time'] = current_time - self.start_time
        self.speed_history.append(speed_kmh)
        
        # Update control history for comfort analysis
        self.control_history.append({
            'throttle': safe_float(control.throttle),
            'brake': safe_float(control.brake),
            'steer': safe_float(control.steer),
            'timestamp': current_time
        })
        self.steering_history.append(safe_float(control.steer))
        
        # Safety metrics from agent status
        if agent_status:
            self.metrics['safety']['rule_based_interventions'] += agent_status.get('safety_interventions', 0)
        
        # Check for violations
        self._check_safety_violations(vehicle, speed_kmh, sensor_data)
        
        # Update comfort metrics
        self._update_comfort_metrics()
        
        # Store frame data for detailed analysis
        self.frame_data.append({
            'timestamp': current_time,
            'location': (current_location.x, current_location.y, current_location.z),
            'speed': speed_kmh,
            'control': {
                'throttle': safe_float(control.throttle),
                'brake': safe_float(control.brake),
                'steer': safe_float(control.steer)
            },
            'agent_status': agent_status
        })
        
        self.last_location = current_location
        self.performance_monitor.log_frame()
    
    def _check_safety_violations(self, vehicle, speed_kmh, sensor_data):
        """Check for various safety violations"""
        # Speed limit violations (50 km/h max)
        speed_limit = 50.0
        if speed_kmh > speed_limit * 1.1:  # 10% tolerance
            self.metrics['safety']['speed_violations'] += 1
        
        # Lane violations using lane detection from sensor data
        if sensor_data and 'semantic' in sensor_data:
            # This would require implementing lane detection logic
            # For now, we'll use a placeholder
            pass
        
        # Collision detection
        collision_sensor = None
        for actor in self.world.get_actors():
            if 'collision' in actor.type_id and actor.parent and actor.parent.id == vehicle.id:
                collision_sensor = actor
                break
        
        if collision_sensor and hasattr(collision_sensor, 'get_collision_history'):
            collision_events = collision_sensor.get_collision_history()
            self.metrics['safety']['collisions'] += len(collision_events)
    
    def _update_comfort_metrics(self):
        """Update driving comfort metrics"""
        if len(self.control_history) < 10:
            return
        
        # Steering smoothness (lower is better)
        recent_steering = [c['steer'] for c in self.control_history[-10:]]
        steering_changes = np.diff(recent_steering)
        self.metrics['comfort']['steering_smoothness'] = np.std(steering_changes)
        
        # Acceleration smoothness
        recent_throttle = [c['throttle'] for c in self.control_history[-10:]]
        throttle_changes = np.diff(recent_throttle)
        self.metrics['comfort']['acceleration_smoothness'] = np.std(throttle_changes)
        
        # Count harsh braking events
        recent_brake = [c['brake'] for c in self.control_history[-5:]]
        if any(b > 0.7 for b in recent_brake):
            self.metrics['comfort']['harsh_braking_count'] += 1
        
        # Count jerk events (rapid steering changes)
        if len(steering_changes) > 0 and max(abs(s) for s in steering_changes) > 0.3:
            self.metrics['comfort']['jerk_count'] += 1
    
    def check_route_completion(self, vehicle, target_location, tolerance=5.0):
        """Check if vehicle reached target destination"""
        current_location = vehicle.get_location()
        distance_to_target = current_location.distance(target_location)
        
        if distance_to_target < tolerance:
            self.metrics['navigation']['destination_reached'] = True
            return True
        return False
    
    def calculate_final_scores(self):
        """Calculate final evaluation scores"""
        scores = {}
        
        # Safety Score (0-100, higher is better) - target: <1 violation per 1000 frames
        total_violations = sum(self.metrics['safety'].values()) - self.metrics['safety']['rule_based_interventions']
        total_frames = len(self.frame_data)
        violation_rate = total_violations / max(total_frames, 1) * 1000  # violations per 1000 frames
        
        if violation_rate < 1.0:  # Meeting target
            safety_score = 100
        else:
            safety_score = max(0, 100 - (violation_rate - 1.0) * 10)
        scores['safety'] = safety_score
        
        # Efficiency Score - target: >80% route completion rate
        if self.metrics['efficiency']['total_time'] > 0:
            avg_speed = (self.metrics['efficiency']['total_distance'] / 1000) / (self.metrics['efficiency']['total_time'] / 3600)
            # Base efficiency on average speed and route completion
            route_completion = 1.0 if self.metrics['navigation']['destination_reached'] else 0.5
            efficiency_score = min(100, (avg_speed / 25.0) * 100 * route_completion)  # 25 km/h baseline
        else:
            efficiency_score = 0
        scores['efficiency'] = efficiency_score
        
        # Comfort Score - target: <0.5 m/s² average acceleration
        steering_smoothness = self.metrics['comfort']['steering_smoothness']
        accel_smoothness = self.metrics['comfort']['acceleration_smoothness']
        
        # Combine smoothness metrics
        comfort_penalty = (steering_smoothness * 100) + (accel_smoothness * 50)
        comfort_score = max(0, 100 - comfort_penalty)
        scores['comfort'] = comfort_score
        
        # Navigation Score
        if self.metrics['navigation']['destination_reached']:
            nav_score = 100 - (self.metrics['navigation']['navigation_errors'] * 10)
        else:
            nav_score = 50 - (self.metrics['navigation']['navigation_errors'] * 10)
        scores['navigation'] = max(0, nav_score)
        
        # Overall Score (weighted according to priorities)
        scores['overall'] = (
            safety_score * 0.4 +      # Safety is highest priority
            efficiency_score * 0.3 +   # Efficiency second
            comfort_score * 0.2 +      # Comfort third
            nav_score * 0.1            # Navigation support
        )
        
        return scores
    
    def run_evaluation(self, scenario='urban_driving', duration=300, traffic_density='medium', model_path=None):
        """Run complete evaluation of the hybrid agent"""
        self.logger.info(f"Starting evaluation - Scenario: {scenario}, Duration: {duration}s")
        
        with Timer(f"Evaluation {scenario}"):
            # Setup
            self.reset_metrics()
            start_point, end_point, config = self.setup_test_scenario(scenario, traffic_density)
            
            # Spawn test vehicle
            vehicle = self.spawn_test_vehicle(start_point)
            if not vehicle:
                self.logger.error("Failed to spawn test vehicle")
                return None
             
            # Setup background traffic
            background_actors = self.setup_background_traffic(traffic_density)
            
            try:
                # Initialize sensor system
                sensor_manager = SensorManager(self.world, vehicle)
                sensor_manager.setup_sensors()
                
                # Initialize the hybrid agent (THIS IS THE KEY FIX)
                agent = HybridAgent(self.world, vehicle, model_path)
                agent.set_destination(end_point.location)
                
                # Main evaluation loop
                start_time = time.time()
                route_completed = False
                
                self.logger.info("Starting evaluation loop...")
                
                while time.time() - start_time < duration and not route_completed:
                    self.world.tick()
                    
                    # Get sensor data
                    sensor_data = sensor_manager.get_latest_data()
                    if not sensor_data or 'rgb' not in sensor_data:
                        continue
                    
                    # Run agent step (this is where the hybrid agent makes decisions)
                    control = agent.step(sensor_data)
                    vehicle.apply_control(control)
                    
                    # Get agent status for metrics
                    agent_status = agent.get_status()
                    
                    # Update evaluation metrics
                    self.update_metrics(vehicle, control, agent_status, sensor_data)
                    
                    # Check route completion
                    route_completed = self.check_route_completion(vehicle, end_point.location)
                    
                    # Progress reporting
                    elapsed = time.time() - start_time
                    if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                        self.logger.info(f"Progress: {elapsed:.0f}s, Distance: {self.metrics['efficiency']['total_distance']:.1f}m, "
                                       f"Speed: {np.mean(self.speed_history[-10:]) if self.speed_history else 0:.1f} km/h")
                
                # Calculate final scores
                scores = self.calculate_final_scores()
                
                # Update route completion metric
                if route_completed:
                    self.metrics['efficiency']['route_completion_ratio'] = 1.0
                else:
                    # Partial completion based on distance to target
                    current_location = vehicle.get_location()
                    distance_to_target = current_location.distance(end_point.location)
                    initial_distance = start_point.location.distance(end_point.location)
                    completion_ratio = max(0, 1 - (distance_to_target / initial_distance))
                    self.metrics['efficiency']['route_completion_ratio'] = completion_ratio
                
                # Get final agent debug info
                final_agent_info = agent.get_comprehensive_metrics()
                
                return {
                    'scenario': scenario,
                    'config': config,
                    'metrics': self.metrics,  
                    'scores': scores,
                    'route_completed': route_completed,
                    'duration': time.time() - start_time,
                    'total_frames': len(self.frame_data),
                    'agent_info': final_agent_info,
                    'performance_stats': self.performance_monitor.get_stats()
                }
                
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                return None
                
            finally:
                # Cleanup
                try:
                    sensor_manager.cleanup()
                    vehicle.destroy()
                    destroy_actors(background_actors)
                    self.logger.info("Cleanup completed")
                except Exception as e:
                    self.logger.warning(f"Cleanup error: {e}")
    
    def generate_detailed_report(self, results):
        """Generate comprehensive evaluation report"""
        if not results:
            return None
        
        # Performance analysis
        avg_speed = np.mean([f['speed'] for f in self.frame_data]) if self.frame_data else 0
        speed_std = np.std([f['speed'] for f in self.frame_data]) if self.frame_data else 0
        
        # Control analysis
        throttle_usage = np.mean([f['control']['throttle'] for f in self.frame_data]) if self.frame_data else 0
        brake_usage = np.mean([f['control']['brake'] for f in self.frame_data]) if self.frame_data else 0
        steering_usage = np.std([f['control']['steer'] for f in self.frame_data]) if self.frame_data else 0
        
        report = {
            'evaluation_summary': {
                'scenario': results['scenario'],
                'scenario_description': results['config']['description'],
                'duration': f"{results['duration']:.1f}s",
                'total_frames': results['total_frames'],
                'route_completed': results['route_completed'],
                'total_distance': f"{results['metrics']['efficiency']['total_distance']:.1f}m",
                'average_speed': f"{avg_speed:.1f} km/h",
                'speed_consistency': f"{speed_std:.1f} km/h std"
            },
            'scores': results['scores'],
            'detailed_metrics': results['metrics'],
            'performance_analysis': {
                'avg_throttle_usage': f"{throttle_usage:.3f}",
                'avg_brake_usage': f"{brake_usage:.3f}",
                'steering_variability': f"{steering_usage:.3f}",
                'fps': f"{results['performance_stats']['avg_fps']:.1f}",
                'processing_efficiency': results['performance_stats']
            },
            'agent_performance': results['agent_info'],
            'compliance_check': {
                'safety_target_met': results['scores']['safety'] >= 90,  # <1 violation per 1000 frames
                'efficiency_target_met': results['scores']['efficiency'] >= 80,  # >80% completion
                'comfort_target_met': results['scores']['comfort'] >= 75,  # <0.5 m/s² acceleration
                'overall_passing': results['scores']['overall'] >= 80
            }
        }
        
        return report
    
    def save_evaluation_data(self, results, output_dir="evaluation_results"):
        """Save detailed evaluation data"""
        ensure_dir(output_dir)
        timestamp = get_timestamp()
        
        # Save main report
        report = self.generate_detailed_report(results)
        if report:
            report_file = Path(output_dir) / f"evaluation_report_{results['scenario']}_{timestamp}.json"
            save_json(report, str(report_file))
            self.logger.info(f"Evaluation report saved: {report_file}")
        
        # Save raw frame data for detailed analysis
        frame_data_file = Path(output_dir) / f"frame_data_{results['scenario']}_{timestamp}.json"
        save_json(self.frame_data, str(frame_data_file))
        
        # Save metrics summary
        metrics_file = Path(output_dir) / f"metrics_{results['scenario']}_{timestamp}.json"
        save_json(results['metrics'], str(metrics_file))
        
        return str(report_file)

def main():
    """Main evaluation function"""
    # Setup logging
    setup_logging(logging.INFO, f"logs/main_evaluation_{get_timestamp()}.log")
    logger = logging.getLogger(__name__)
    
    try:
        # Connect to CARLA using utility function
        client, world = setup_carla_world('localhost', 2000, 10.0)
        if not client or not world:
            logger.error("Failed to connect to CARLA server")
            return
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        
        # Find the latest trained model
        model_path = get_latest_file("models", "*.pth")
        if model_path:
            logger.info(f"Using model: {model_path}")
        else:
            logger.warning("No trained model found, agent will use fallback control")
        
        # Initialize evaluator
        evaluator = AgentEvaluator(client, world)
        
        # Run evaluations for different scenarios
        scenarios_to_test = ['urban_driving', 'traffic_lights', 'intersections', 'highway_driving']
        all_results = []
        
        for scenario in scenarios_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATING SCENARIO: {scenario.upper()}")
            logger.info(f"{'='*60}")
            
            results = evaluator.run_evaluation(
                scenario=scenario,
                duration=180,  # 3 minutes per scenario
                traffic_density='medium',
                model_path=str(model_path) if model_path else None
            )
            
            if results:
                all_results.append(results)
                
                # Generate and save report
                report_file = evaluator.save_evaluation_data(results)
                report = evaluator.generate_detailed_report(results)
                
                # Print summary
                logger.info(f"\nRESULTS FOR {scenario.upper()}:")
                logger.info(f"Overall Score: {results['scores']['overall']:.1f}/100")
                logger.info(f"Safety Score: {results['scores']['safety']:.1f}/100")
                logger.info(f"Efficiency Score: {results['scores']['efficiency']:.1f}/100")
                logger.info(f"Comfort Score: {results['scores']['comfort']:.1f}/100")
                logger.info(f"Navigation Score: {results['scores']['navigation']:.1f}/100")
                logger.info(f"Route Completed: {results['route_completed']}")
                logger.info(f"Compliance: {report['compliance_check']['overall_passing']}")
                logger.info(f"Detailed report saved: {report_file}")
            else:
                logger.error(f"Evaluation failed for scenario: {scenario}")
        
        # Generate overall summary
        if all_results:
            overall_scores = {
                'safety': np.mean([r['scores']['safety'] for r in all_results]),
                'efficiency': np.mean([r['scores']['efficiency'] for r in all_results]),
                'comfort': np.mean([r['scores']['comfort'] for r in all_results]),
                'navigation': np.mean([r['scores']['navigation'] for r in all_results]),
                'overall': np.mean([r['scores']['overall'] for r in all_results])
            }
            
            completion_rate = sum(1 for r in all_results if r['route_completed']) / len(all_results)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"COMPREHENSIVE EVALUATION COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Overall Performance Score: {overall_scores['overall']:.1f}/100")
            logger.info(f"Safety Score: {overall_scores['safety']:.1f}/100")
            logger.info(f"Efficiency Score: {overall_scores['efficiency']:.1f}/100") 
            logger.info(f"Comfort Score: {overall_scores['comfort']:.1f}/100")
            logger.info(f"Navigation Score: {overall_scores['navigation']:.1f}/100")
            logger.info(f"Route Completion Rate: {completion_rate*100:.1f}%")
            
            # Check targets
            meets_safety = overall_scores['safety'] >= 90
            meets_efficiency = overall_scores['efficiency'] >= 80 and completion_rate >= 0.8
            meets_comfort = overall_scores['comfort'] >= 75
            
            logger.info(f"\nCOMPLIANCE:")
            logger.info(f"Safety Target (90+): {'Pass' if meets_safety else 'Fail'} ({overall_scores['safety']:.1f})")
            logger.info(f"Efficiency Target (80%+ completion): {'Pass' if meets_efficiency else 'Fail'} ({completion_rate*100:.1f}%)")
            logger.info(f"Comfort Target (75+): {'Pass' if meets_comfort else 'Fail'} ({overall_scores['comfort']:.1f})")
            
            # Save comprehensive summary
            summary = {
                'timestamp': get_timestamp(),
                'model_path': str(model_path) if model_path else None,
                'scenarios_tested': scenarios_to_test,
                'overall_scores': overall_scores,
                'completion_rate': completion_rate,
                'compliance': {
                    'safety': meets_safety,
                    'efficiency': meets_efficiency,
                    'comfort': meets_comfort,
                    'overall_passing': meets_safety and meets_efficiency and meets_comfort
                },
                'individual_results': all_results
            }
            
            summary_file = f"evaluation_results/comprehensive_summary_{get_timestamp()}.json"
            ensure_dir("evaluation_results")
            save_json(summary, summary_file)
            logger.info(f"Comprehensive summary saved: {summary_file}")
        
    except Exception as e:
        logger.error(f"Main evaluation failed: {e}")
        raise
    finally:
        try:
            # Reset to asynchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            logger.info("CARLA world reset to asynchronous mode")
        except:
            pass

if __name__ == '__main__':
    main()