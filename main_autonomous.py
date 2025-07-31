"""
Main execution script for CARLA Autonomous Driving System
Combines hybrid ML + rule-based agent with real-time monitoring and data collection
"""

import argparse
import carla
import pygame
import numpy as np
import json
import time
import random
import logging
from pathlib import Path
import sys
import cv2

# Add project root to path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

# Import project components
from sensors.capture_sensors import SensorManager
from movement.basic_agent import HybridAgent, create_hybrid_agent, validate_agent_configuration
from utils.utils import (
    setup_logging, ensure_dir, save_json, get_timestamp, 
    Timer, PerformanceMonitor, setup_carla_world, destroy_actors,
    get_latest_file, check_gpu_availability, get_system_info,
    safe_float, safe_int, clamp
)

class AutonomousDrivingSystem:
    """
    Complete autonomous driving system with hybrid ML+rule-based control,
    real-time visualization, performance monitoring, and data collection
    """
    
    def __init__(self, host='localhost', port=2000, model_path=None, save_data=True, display_mode=True):
        # Setup logging
        log_file = f"logs/main_autonomous_{get_timestamp()}.log"
        setup_logging(logging.INFO, log_file)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Autonomous Driving System")
        
        # System info
        system_info = get_system_info()
        gpu_available = check_gpu_availability()
        self.logger.info(f"System: {system_info['platform']}, GPU: {gpu_available}")
        
        # CARLA connection using utility function
        self.client, self.world = setup_carla_world(host, port, 10.0)
        if not self.client or not self.world:
            raise ConnectionError("Failed to connect to CARLA server")
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        settings.no_rendering_mode = not display_mode
        self.world.apply_settings(settings)
        
        self.logger.info("CARLA world configured: synchronous mode, 20 FPS")
        
        # Initialize components
        self.vehicle = None
        self.agent = None
        self.sensor_manager = None
        self.background_actors = []
        
        # Model path handling
        if model_path is None:
            model_path = get_latest_file("models", "*.pth")
            if model_path:
                self.logger.info(f"Auto-detected model: {model_path}")
            else:
                self.logger.warning("No trained model found, using fallback control")
        self.model_path = str(model_path) if model_path else None
        
        # Display and visualization
        self.display_mode = display_mode
        if display_mode:
            pygame.init()
            self.display_width = 800
            self.display_height = 600
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption("CARLA Hybrid Autonomous Driving System")
            self.font = pygame.font.Font(None, 24)
            self.logger.info("Display initialized")
        
        # Data collection setup
        self.save_data = save_data
        self.session_id = f"autonomous_session_{get_timestamp()}"
        
        if save_data:
            self.data_path = Path("data/raw_sessions") / self.session_id
            ensure_dir(self.data_path)
            self.logger.info(f"Data collection enabled: {self.data_path}")
        else:
            self.data_path = None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.frame_count = 0
        self.start_time = None
        self.last_status_time = 0
        
        # Runtime statistics
        self.stats = {
            'total_distance': 0.0,
            'safety_interventions': 0,
            'emergency_stops': 0,
            'destination_reached': False,
            'max_speed': 0.0,
            'avg_speed': 0.0
        }
        
        self.logger.info("Autonomous Driving System initialized successfully")
    
    def setup_vehicle_and_sensors(self, spawn_point_index=None):
        """Setup vehicle and sensor system"""
        with Timer("Vehicle and sensor setup"):
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in the map")
            
            # Select spawn point
            if spawn_point_index is not None and 0 <= spawn_point_index < len(spawn_points):
                spawn_point = spawn_points[spawn_point_index]
            else:
                spawn_point = random.choice(spawn_points)
            
            # Get vehicle blueprint
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
            
            # Set vehicle color
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
            
            # Spawn vehicle
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.world.tick()  # Ensure proper initialization
                self.vehicle.set_simulate_physics(True)
                
                self.logger.info(f"Vehicle spawned: {vehicle_bp.id} at {spawn_point.location}")
            except Exception as e:
                raise RuntimeError(f"Failed to spawn vehicle: {e}")
            
            # Setup sensor manager
            try:
                # Pass output directory if data saving is enabled
                output_dir = str(self.data_path) if self.save_data and self.data_path else None
                
                # Initializes sensor manager
                self.sensor_manager = SensorManager(self.world, self.vehicle, output_dir)
                
                if not self.sensor_manager.setup_sensors():
                    raise RuntimeError("Sensor setup failed")
                
                # Wait for sensors to initialize
                self.logger.info("Waiting for sensor initialization...")
                for _ in range(10):
                    self.world.tick()
                    time.sleep(0.05)
                
                # Wait for initial sensor data
                if not self.sensor_manager.wait_for_data(['rgb'], timeout=10.0):
                    self.logger.warning("RGB sensor data not ready within timeout")
                else:
                    self.logger.info("Sensors initialized and data ready")
                    
            except Exception as e:
                if self.vehicle:
                    self.vehicle.destroy()
                raise RuntimeError(f"Failed to setup sensors: {e}")
            
            # Initialize hybrid agent
            try:
                self.agent = create_hybrid_agent(self.world, self.vehicle, self.model_path)
                
                # Validate agent configuration
                validation_results = validate_agent_configuration(self.agent)
                if not validation_results['configuration_valid']:
                    self.logger.warning(f"Agent validation issues: {validation_results['issues']}")
                
                self.logger.info("Hybrid agent initialized and validated")
            except Exception as e:
                if self.sensor_manager:
                    self.sensor_manager.cleanup()
                if self.vehicle:
                    self.vehicle.destroy()
                raise RuntimeError(f"Failed to initialize agent: {e}")
    
    def setup_background_traffic(self, num_vehicles=30, num_pedestrians=10):
        """Setup background traffic for realistic environment"""
        try:
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            
            spawn_points = self.world.get_map().get_spawn_points()
            blueprint_library = self.world.get_blueprint_library()
            
            # Spawn background vehicles
            vehicle_bps = blueprint_library.filter('vehicle.*')
            vehicles_spawned = 0
            
            for i in range(min(num_vehicles, len(spawn_points) - 1)):
                try:
                    bp = random.choice(vehicle_bps)
                    # Avoid spawning at the same location as our vehicle
                    spawn_point = spawn_points[i + 1]
                    vehicle = self.world.spawn_actor(bp, spawn_point)
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    self.background_actors.append(vehicle)
                    vehicles_spawned += 1
                except:
                    continue
            
            # Spawn pedestrians
            walker_bps = blueprint_library.filter('walker.pedestrian.*')
            pedestrians_spawned = 0
            
            for _ in range(num_pedestrians):
                try:
                    spawn_point = self.world.get_random_location_from_navigation()
                    if spawn_point:
                        walker_bp = random.choice(walker_bps)
                        walker = self.world.spawn_actor(walker_bp, spawn_point)
                        self.background_actors.append(walker)
                        pedestrians_spawned += 1
                except:
                    continue
            
            self.logger.info(f"Background traffic: {vehicles_spawned} vehicles, {pedestrians_spawned} pedestrians")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup background traffic: {e}")
    
    def set_destination(self, destination_index=None, min_distance=100.0):
        """Set navigation destination"""
        spawn_points = self.world.get_map().get_spawn_points()
        current_location = self.vehicle.get_location()
        
        if destination_index is not None and 0 <= destination_index < len(spawn_points):
            destination = spawn_points[destination_index].location
        else:
            # Find destinations that are far enough away
            valid_destinations = []
            for point in spawn_points:
                distance = current_location.distance(point.location)
                if distance >= min_distance:
                    valid_destinations.append(point.location)
            
            if valid_destinations:
                destination = random.choice(valid_destinations)
            else:
                # Fallback to any destination
                destination = random.choice(spawn_points).location
        
        self.agent.set_destination(destination)
        distance_to_dest = current_location.distance(destination)
        
        self.logger.info(f"Destination set: ({destination.x:.1f}, {destination.y:.1f}), "
                        f"Distance: {distance_to_dest:.1f}m")
        
        return destination
    
    def update_statistics(self):
        """Update runtime statistics"""
        if not self.vehicle or not self.agent:
            return
        
        # Get current status
        agent_status = self.agent.get_status()
        current_speed = safe_float(agent_status.get('current_speed', 0))
        
        # Update stats
        self.stats['max_speed'] = max(self.stats['max_speed'], current_speed)
        self.stats['safety_interventions'] = safe_int(agent_status.get('safety_interventions', 0))
        
        # Calculate average speed
        if self.frame_count > 0:
            total_time = time.time() - self.start_time
            if total_time > 0:
                # Estimate distance from speed over time
                self.stats['avg_speed'] = self.stats['max_speed'] * 0.7  # Rough estimate
        
        # Check if destination reached
        if agent_status.get('has_destination', False):
            # This would need to be implemented based on distance to destination
            pass
    
    def draw_hud(self, sensor_data):
        """Draw heads-up display with system information"""
        if not self.display_mode:
            return
        
        # Get current status
        agent_status = self.agent.get_status() if self.agent else {}
        agent_debug = self.agent.get_debug_info() if self.agent else {}
        
        # Performance stats
        perf_stats = self.performance_monitor.get_stats()
        
        # Create HUD information
        hud_info = [
            f"Frame: {self.frame_count}",
            f"FPS: {perf_stats['avg_fps']:.1f}",
            f"Time: {perf_stats['total_time']:.1f}s",
            "",
            f"Driving State: {agent_status.get('driving_state', 'N/A')}",
            f"Speed: {agent_status.get('current_speed', 0):.1f} km/h",
            f"Max Speed: {self.stats['max_speed']:.1f} km/h",
            "",
            f"Safety Interventions: {self.stats['safety_interventions']}",
            f"Intervention Rate: {agent_status.get('intervention_rate', 0)*100:.1f}%",
            f"ML Model: {'Loaded' if agent_status.get('ml_model_loaded', False) else 'Fallback'}",
            "",
            f"Data Collection: {'ON' if self.save_data else 'OFF'}",
            f"Model: {Path(self.model_path).name if self.model_path else 'None'}",
        ]
        
        # Add sensor status if available
        if hasattr(self.sensor_manager, 'get_sensor_status'):
            sensor_status = self.sensor_manager.get_sensor_status()
            hud_info.extend([
                "",
                "Sensors:",
                f"  RGB: {'✓' if sensor_status.get('rgb', False) else '✗'}",
                f"  Semantic: {'✓' if sensor_status.get('semantic', False) else '✗'}",
                f"  Depth: {'✓' if sensor_status.get('depth', False) else '✗'}",
                f"  GPS: {'✓' if sensor_status.get('gps', False) else '✗'}",
                f"  IMU: {'✓' if sensor_status.get('imu', False) else '✗'}",
            ])
        
        # Vehicle control information
        if self.vehicle:
            control = self.vehicle.get_control()
            hud_info.extend([
                "",
                "Controls:",
                f"  Throttle: {control.throttle:.3f}",
                f"  Brake: {control.brake:.3f}",
                f"  Steer: {control.steer:.3f}",
            ])
        
        # Draw HUD
        y_offset = 10
        for line in hud_info:
            if line:  # Skip empty lines for spacing
                # Create text surface
                text_surface = self.font.render(line, True, (255, 255, 255))
                
                # Add semi-transparent background
                bg_surface = pygame.Surface((text_surface.get_width() + 10, text_surface.get_height() + 4))
                bg_surface.fill((0, 0, 0))
                bg_surface.set_alpha(128)
                
                # Blit to screen
                self.screen.blit(bg_surface, (5, y_offset - 2))
                self.screen.blit(text_surface, (10, y_offset))
            
            y_offset += 20
        
        # Draw controls help at bottom
        help_text = [
            "ESC: Exit | R: New Destination | S: Toggle Data | T: Traffic | H: Agent Health"
        ]
        
        for i, text in enumerate(help_text):
            text_surface = self.font.render(text, True, (200, 200, 200))
            bg_surface = pygame.Surface((text_surface.get_width() + 10, text_surface.get_height() + 4))
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(150)
            
            y_pos = self.display_height - (len(help_text) - i) * 25
            self.screen.blit(bg_surface, (5, y_pos - 2))
            self.screen.blit(text_surface, (10, y_pos))
    
    def handle_user_input(self):
        """Handle pygame events and user input"""
        if not self.display_mode:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # Set new random destination
                    self.set_destination()
                    self.logger.info("New destination set by user")
                elif event.key == pygame.K_s:
                    # Toggle data saving
                    self.save_data = not self.save_data
                    self.logger.info(f"Data saving {'enabled' if self.save_data else 'disabled'}")
                elif event.key == pygame.K_t:
                    # Add more traffic
                    self.setup_background_traffic(10, 5)
                    self.logger.info("Additional traffic spawned")
                elif event.key == pygame.K_h:
                    # Print agent health check
                    if self.agent:
                        health = self.agent.health_check()
                        self.logger.info(f"Agent health: {health}")
                        print(f"Agent Health Check: {health}")
        
        return True
    
    def save_session_data(self, sensor_data, control_command, agent_status):
        """Save frame data for this session"""
        if not self.save_data or not self.data_path:
            return
        
        try:
            # Prepare frame data
            frame_data = {
                'frame_id': self.frame_count,
                'timestamp': get_timestamp(),
                'control': {
                    'steering': safe_float(control_command.steer),
                    'throttle': safe_float(control_command.throttle),
                    'brake': safe_float(control_command.brake)
                },
                'agent_status': agent_status,
                'vehicle_state': {
                    'speed': safe_float(agent_status.get('current_speed', 0)),
                    'location': self.vehicle.get_location() if self.vehicle else None
                }
            }
            
            # Save sensor data if available
            if sensor_data:
                # RGB image
                if 'rgb' in sensor_data:
                    rgb_path = self.data_path / 'rgb' / f"{self.frame_count:06d}.png"
                    ensure_dir(rgb_path.parent)
                    cv2.imwrite(str(rgb_path), sensor_data['rgb'])
                
                # GPS data
                if 'gps' in sensor_data:
                    gps_path = self.data_path / 'gps' / f"{self.frame_count:06d}.json"
                    ensure_dir(gps_path.parent)
                    save_json(sensor_data['gps'], str(gps_path))
                
                # Control data
                control_path = self.data_path / 'control' / f"{self.frame_count:06d}.json"
                ensure_dir(control_path.parent)
                save_json(frame_data['control'], str(control_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to save session data: {e}")
    
    def run_autonomous_driving(self, max_frames=10000, destination_index=None, spawn_point=None, traffic=True):
        """Main autonomous driving loop"""
        self.logger.info("Starting autonomous driving system")
        
        try:
            # Setup phase
            with Timer("System setup"):
                self.setup_vehicle_and_sensors(spawn_point)
                
                if traffic:
                    self.setup_background_traffic()
                
                destination = self.set_destination(destination_index)
            
            # Initialize runtime
            self.start_time = time.time()
            self.performance_monitor.reset()
            running = True
            
            if self.display_mode:
                clock = pygame.time.Clock()
                self.logger.info("Display mode enabled")
            
            print("\n" + "="*60)
            print("CARLA HYBRID AUTONOMOUS DRIVING SYSTEM")
            print("="*60)
            print("Controls:")
            print("  ESC - Exit system")
            print("  R - Set new random destination") 
            print("  S - Toggle data collection")
            print("  T - Spawn additional traffic")
            print("  H - Print agent health status")
            print("="*60)
            
            # Main driving loop
            while running and self.frame_count < max_frames:
                # Handle user input
                running = self.handle_user_input()
                if not running:
                    break
                
                # Get sensor data
                sensor_data = self.sensor_manager.get_latest_data()
                
                if sensor_data and 'rgb' in sensor_data:
                    # Run autonomous agent step
                    control_command = self.agent.step(sensor_data)
                    self.vehicle.apply_control(control_command)
                    
                    # Update statistics
                    self.update_statistics()
                    
                    # Get agent status for monitoring
                    agent_status = self.agent.get_status()
                    
                    # Save session data if enabled
                    self.save_session_data(sensor_data, control_command, agent_status)
                    
                    # Display update
                    if self.display_mode and sensor_data.get('rgb') is not None:
                        # Convert and display RGB image
                        rgb_array = sensor_data['rgb']
                        if rgb_array.shape[0] == 3:  # CHW format
                            rgb_array = rgb_array.transpose(1, 2, 0)  # Convert to HWC
                        
                        # Convert BGR to RGB for pygame
                        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
                        
                        # Create pygame surface
                        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
                        surface = pygame.transform.scale(surface, (self.display_width, self.display_height))
                        
                        # Draw to screen
                        self.screen.blit(surface, (0, 0))
                        self.draw_hud(sensor_data)
                        pygame.display.flip()
                        
                        # Limit display FPS
                        clock.tick(30)
                    
                    # Performance monitoring
                    self.performance_monitor.log_frame()
                    self.frame_count += 1
                    
                    # Periodic status logging
                    current_time = time.time()
                    if current_time - self.last_status_time > 30.0:  # Every 30 seconds
                        perf_stats = self.performance_monitor.get_stats()
                        self.logger.info(f"Status - Frame: {self.frame_count}, "
                                       f"FPS: {perf_stats['avg_fps']:.1f}, "
                                       f"Speed: {agent_status.get('current_speed', 0):.1f} km/h, "
                                       f"Interventions: {self.stats['safety_interventions']}")
                        self.last_status_time = current_time
                
                # Advance CARLA simulation
                self.world.tick()
            
            # Final statistics
            self.print_final_statistics()
            
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
            print("\nSystem interrupted by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
            print(f"System error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def print_final_statistics(self):
        """Print final performance statistics"""
        if self.frame_count == 0:
            return
        
        elapsed_time = time.time() - self.start_time
        perf_stats = self.performance_monitor.get_stats()
        
        print(f"\n{'='*60}")
        print("AUTONOMOUS DRIVING SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Time: {elapsed_time:.1f}s")
        print(f"Average FPS: {perf_stats['avg_fps']:.2f}")
        print(f"Max Speed: {self.stats['max_speed']:.1f} km/h")
        print(f"Safety Interventions: {self.stats['safety_interventions']}")
        
        if self.agent:
            agent_status = self.agent.get_status()
            print(f"Intervention Rate: {agent_status.get('intervention_rate', 0)*100:.1f}%")
            print(f"ML Model Used: {'Yes' if agent_status.get('ml_model_loaded', False) else 'No (Fallback)'}")
        
        if self.save_data:
            print(f"Data Saved: {self.data_path}")
        
        print(f"{'='*60}")
        
        # Log final statistics
        self.logger.info(f"Session complete - Frames: {self.frame_count}, "
                        f"Time: {elapsed_time:.1f}s, FPS: {perf_stats['avg_fps']:.2f}")
    
    def cleanup(self):
        """Clean up all resources"""
        self.logger.info("Starting system cleanup...")
        
        try:
            # Save agent metrics if available
            if self.agent and self.save_data:
                self.agent.save_metrics_to_file(str(self.data_path.parent) if self.data_path else "logs")
            
            # Cleanup sensor manager
            if self.sensor_manager:
                self.sensor_manager.cleanup()
                self.logger.info("Sensor manager cleaned up")
            
            # Destroy vehicle
            if self.vehicle:
                self.vehicle.destroy()
                self.logger.info("Vehicle destroyed")
            
            # Destroy background actors
            if self.background_actors:
                destroy_actors(self.background_actors)
                self.logger.info(f"Destroyed {len(self.background_actors)} background actors")
            
            # Reset CARLA world settings
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                self.logger.info("CARLA world reset to asynchronous mode")
            
            # Cleanup pygame
            if self.display_mode:
                pygame.quit()
                self.logger.info("Display system cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        self.logger.info("System cleanup complete")

def main():
    """Main function with comprehensive command line interface"""
    parser = argparse.ArgumentParser(
        description='CARLA Hybrid Autonomous Driving System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_autonomous.py                           # Run with default settings
  python main_autonomous.py --model models/best.pth  # Use model
  python main_autonomous.py --no-display --frames 5000  # Headless mode
  python main_autonomous.py --spawn-point 5 --dest 20   # Route
        """
    )
    
    # Connection settings
    parser.add_argument('--host', default='localhost', 
                       help='IP address of CARLA server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                       help='TCP port of CARLA server (default: 2000)')
    
    # Model and data settings
    parser.add_argument('--model', '--model-path', dest='model_path',
                       help='Path to trained ML model (auto-detects if not specified)')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable data collection and saving')
    
    # Simulation settings  
    parser.add_argument('--frames', default=10000, type=int,
                       help='Maximum number of frames to run (default: 10000)')
    parser.add_argument('--spawn-point', type=int,
                       help='Vehicle spawn point index')
    parser.add_argument('--dest', '--destination', type=int, dest='destination',
                       help='Destination spawn point index')
    parser.add_argument('--no-traffic', action='store_true',
                       help='Disable background traffic')
    
    # Display settings
    parser.add_argument('--no-display', action='store_true',
                       help='Run in headless mode (no visualization)')
    
    args = parser.parse_args()
    
    try:
        # Initialize autonomous driving system
        ads = AutonomousDrivingSystem(
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            save_data=not args.no_save,
            display_mode=not args.no_display
        )
        
        # Run the system
        ads.run_autonomous_driving(
            max_frames=args.frames,
            destination_index=args.destination,
            spawn_point=args.spawn_point,
            traffic=not args.no_traffic
        )
        
    except Exception as e:
        print(f"Failed to start autonomous driving system: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()