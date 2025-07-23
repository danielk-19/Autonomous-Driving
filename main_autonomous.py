"""
Complete autonomous driving system for CARLA.
Combines hybrid agent with data collection and real-time visualization.
"""
import argparse
import carla
import pygame
import numpy as np
import json
import time
import random
from pathlib import Path

# Import our custom modules
from sensors.capture_sensors import SensorManager
from movement.basic_agent import HybridAgent
from movement.perception import PerceptionSystem

class AutonomousDrivingSystem:
    """
    Complete autonomous driving system with visualization and data collection
    """
    
    def __init__(self, host='127.0.0.1', port=2000, save_data=True):
        # CARLA connection
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Initialize components
        self.vehicle = None
        self.agent = None
        self.sensor_manager = None
        
        # Pygame for visualization
        pygame.init()
        self.display_width = 800
        self.display_height = 600
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Hybrid Autonomous Driving - CARLA")
        
        # Data collection
        self.save_data = save_data
        self.data_path = Path("./autonomous_data")
        if save_data:
            self.data_path.mkdir(exist_ok=True, parents=True)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_vehicle_and_sensors(self):
        """Setup vehicle and sensors in CARLA world"""
        # Get vehicle blueprint
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.world.tick()  # Important for proper initialization
        
        # Enable physics
        self.vehicle.set_simulate_physics(True)
        
        # Setup sensor manager
        data_save_path = self.data_path if self.save_data else None
        self.sensor_manager = SensorManager(self.world, self.vehicle, data_save_path)
        
        # Initialize hybrid agent
        self.agent = HybridAgent(self.world, self.vehicle)
        
        print(f"Vehicle spawned at: {spawn_point.location}")
        print(f"Agent initialized with hybrid control system")
        
    def set_destination(self, destination_index=None):
        """Set a random destination or specific waypoint"""
        spawn_points = self.world.get_map().get_spawn_points()
        
        if destination_index is not None and destination_index < len(spawn_points):
            destination = spawn_points[destination_index].location
        else:
            # Random destination far from current location
            current_location = self.vehicle.get_location()
            valid_destinations = []
            
            for point in spawn_points:
                distance = current_location.distance(point.location)
                if distance > 100:  # At least 100m away
                    valid_destinations.append(point.location)
            
            if valid_destinations:
                destination = random.choice(valid_destinations)
            else:
                destination = random.choice(spawn_points).location
        
        self.agent.set_destination(destination)
        print(f"Destination set to: {destination}")
        return destination
    
    def draw_overlay(self, sensor_data):
        """Draw debugging overlay on the screen"""
        # Get agent status
        status = self.agent.get_status()
        
        # Create text overlay
        font = pygame.font.Font(None, 24)
        y_offset = 10
        
        # Agent information
        texts = [
            f"Frame: {self.frame_count}",
            f"State: {status['driving_state']}",
            f"Speed: {status['current_speed']:.1f} km/h",
            f"Safety Interventions: {status['safety_interventions']} ({status['intervention_rate']*100:.1f}%)",
            f"FPS: {self.frame_count / (time.time() - self.start_time):.1f}",
        ]
        
        # Vehicle control info
        control = self.vehicle.get_control()
        texts.extend([
            f"Throttle: {control.throttle:.2f}",
            f"Brake: {control.brake:.2f}",
            f"Steer: {control.steer:.2f}",
        ])
        
        # GPS info if available
        if sensor_data.get('gps'):
            gps = sensor_data['gps']
            texts.append(f"GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}")
        
        # Draw text
        for text in texts:
            text_surface = font.render(text, True, (255, 255, 255))
            # Add background for readability
            bg_surface = pygame.Surface((text_surface.get_width() + 10, text_surface.get_height() + 4))
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(128)
            
            self.screen.blit(bg_surface, (5, y_offset - 2))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def run_autonomous_driving(self, max_frames=10000, destination_index=None):
        """Run the autonomous driving system"""
        try:
            # Setup
            self.setup_vehicle_and_sensors()
            self.set_destination(destination_index)
            
            print("Starting autonomous driving...")
            print("Controls:")
            print("  ESC - Exit")
            print("  R - Set new random destination")
            print("  S - Toggle data saving")
            
            running = True
            clock = pygame.time.Clock()
            
            while running and self.frame_count < max_frames:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.set_destination()
                        elif event.key == pygame.K_s:
                            self.save_data = not self.save_data
                            print(f"Data saving: {'ON' if self.save_data else 'OFF'}")
                
                # Get sensor data
                sensor_data = self.sensor_manager.get_sensor_data()
                
                if self.sensor_manager.data_ready():
                    # Run autonomous agent
                    control = self.agent.step(sensor_data['sensors'])
                    self.vehicle.apply_control(control)
                    
                    # Save data if enabled
                    if self.save_data:
                        additional_data = {
                            'agent_status': self.agent.get_status(),
                            'control_command': {
                                'throttle': control.throttle,
                                'steer': control.steer,
                                'brake': control.brake
                            }
                        }
                        self.sensor_manager.save_frame_data(additional_data)
                    
                    # Visualization
                    if sensor_data['sensors']['rgb'] is not None:
                        # Display RGB image
                        rgb_array = sensor_data['sensors']['rgb']
                        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
                        surface = pygame.transform.scale(surface, (self.display_width, self.display_height))
                        self.screen.blit(surface, (0, 0))
                        
                        # Draw overlay
                        self.draw_overlay(sensor_data)
                        
                        pygame.display.flip()
                    
                    self.frame_count += 1
                
                # Advance simulation
                self.world.tick()
                clock.tick(60)  # Limit to 60 FPS for display
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print(f"\nCleaning up... Processed {self.frame_count} frames")
        
        if self.sensor_manager:
            self.sensor_manager.cleanup()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        # Reset world settings
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        pygame.quit()
        
        # Print performance stats
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"Average FPS: {fps:.2f}")
            
            if hasattr(self, 'agent') and self.agent:
                status = self.agent.get_status()
                print(f"Safety interventions: {status['safety_interventions']} ({status['intervention_rate']*100:.1f}%)")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='CARLA Hybrid Autonomous Driving')
    parser.add_argument('--host', default='127.0.0.1', help='IP of the CARLA server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port of the CARLA server')
    parser.add_argument('--frames', default=10000, type=int, help='Maximum number of frames to run')
    parser.add_argument('--destination', type=int, help='Specific destination spawn point index')
    parser.add_argument('--no-save', action='store_true', help='Disable data saving')
    
    args = parser.parse_args()
    
    # Initialize system
    ads = AutonomousDrivingSystem(
        host=args.host,
        port=args.port,
        save_data=not args.no_save
    )
    
    # Run autonomous driving
    ads.run_autonomous_driving(
        max_frames=args.frames,
        destination_index=args.destination
    )

if __name__ == '__main__':
    main()