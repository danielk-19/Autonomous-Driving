"""
Enhanced sensor data collection for autonomous driving.
Collects RGB, semantic segmentation, depth, GPS, IMU, and traffic light data.
"""

import os
import sys
import json
import time
import numpy as np
import pygame
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

try:
    import carla
except ImportError:
    raise RuntimeError('Cannot import CARLA')

class SensorManager:
    def __init__(self, world, vehicle, output_dir):
        self.world = world
        self.vehicle = vehicle
        self.output_dir = output_dir
        self.sensors = {}
        self.sensor_data = {}
        self.frame_count = 0
        
        # Create output directories
        for sensor_type in ['rgb', 'semantic', 'depth', 'gps', 'imu', 'control']:
            os.makedirs(os.path.join(output_dir, sensor_type), exist_ok=True)
    
    def setup_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        
        # RGB Camera
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '800')
        rgb_bp.set_attribute('image_size_y', '600')
        rgb_bp.set_attribute('fov', '90')
        rgb_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensors['rgb'] = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors['rgb'].listen(lambda data: self._rgb_callback(data))
        
        # Semantic Segmentation Camera
        sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '800')
        sem_bp.set_attribute('image_size_y', '600')
        sem_bp.set_attribute('fov', '90')
        self.sensors['semantic'] = self.world.spawn_actor(sem_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors['semantic'].listen(lambda data: self._semantic_callback(data))
        
        # Depth Camera
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '800')
        depth_bp.set_attribute('image_size_y', '600')
        depth_bp.set_attribute('fov', '90')
        self.sensors['depth'] = self.world.spawn_actor(depth_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors['depth'].listen(lambda data: self._depth_callback(data))
        
        # GPS
        gps_bp = bp_lib.find('sensor.other.gnss')
        self.sensors['gps'] = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['gps'].listen(lambda data: self._gps_callback(data))
        
        # IMU
        imu_bp = bp_lib.find('sensor.other.imu')
        self.sensors['imu'] = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['imu'].listen(lambda data: self._imu_callback(data))
        
        print("All sensors initialized successfully!")
    
    def _rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Remove alpha
        self.sensor_data['rgb'] = array.copy()
    
    def _semantic_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, 2]  # Class ID channel
        self.sensor_data['semantic'] = array.copy()
    
    def _depth_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        # Convert to actual depth values
        array = array.astype(np.float32)
        normalized = (array[:, :, 0] + array[:, :, 1] * 256 + array[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = normalized * 1000  # CARLA depth range
        self.sensor_data['depth'] = depth_in_meters.copy()
    
    def _gps_callback(self, data):
        self.sensor_data['gps'] = {
            'lat': data.latitude,
            'lon': data.longitude,
            'alt': data.altitude
        }
    
    def _imu_callback(self, data):
        self.sensor_data['imu'] = {
            'accel_x': data.accelerometer.x,
            'accel_y': data.accelerometer.y,
            'accel_z': data.accelerometer.z,
            'gyro_x': data.gyroscope.x,
            'gyro_y': data.gyroscope.y,
            'gyro_z': data.gyroscope.z,
            'compass': data.compass
        }
    
    def save_frame_data(self):
        """Save all sensor data for current frame"""
        frame_id = f"{self.frame_count:06d}"
        
        # Save RGB image
        if 'rgb' in self.sensor_data:
            rgb_path = os.path.join(self.output_dir, 'rgb', f"{frame_id}.png")
            pygame.image.save(
                pygame.surfarray.make_surface(self.sensor_data['rgb'].swapaxes(0,1)), 
                rgb_path
            )
        
        # Save semantic segmentation
        if 'semantic' in self.sensor_data:
            sem_path = os.path.join(self.output_dir, 'semantic', f"{frame_id}.png")
            pygame.image.save(
                pygame.surfarray.make_surface(self.sensor_data['semantic'].swapaxes(0,1)), 
                sem_path
            )
        
        # Save depth map
        if 'depth' in self.sensor_data:
            depth_path = os.path.join(self.output_dir, 'depth', f"{frame_id}.npy")
            np.save(depth_path, self.sensor_data['depth'])
        
        # Save GPS data
        if 'gps' in self.sensor_data:
            gps_path = os.path.join(self.output_dir, 'gps', f"{frame_id}.json")
            with open(gps_path, 'w') as f:
                json.dump(self.sensor_data['gps'], f)
        
        # Save IMU data
        if 'imu' in self.sensor_data:
            imu_path = os.path.join(self.output_dir, 'imu', f"{frame_id}.json")
            with open(imu_path, 'w') as f:
                json.dump(self.sensor_data['imu'], f)
        
        # Save control data
        control = self.vehicle.get_control()
        control_data = {
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            'hand_brake': control.hand_brake,
            'reverse': control.reverse,
            'manual_gear_shift': control.manual_gear_shift,
            'gear': control.gear
        }
        control_path = os.path.join(self.output_dir, 'control', f"{frame_id}.json")
        with open(control_path, 'w') as f:
            json.dump(control_data, f)
        
        self.frame_count += 1
        return frame_id
    
    def cleanup(self):
        """Destroy all sensors"""
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        print("All sensors cleaned up!")

def main():
    pygame.init()
    
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
        
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Setup sensors
        output_dir = 'data/sensor_test'
        sensor_manager = SensorManager(world, vehicle, output_dir)
        sensor_manager.setup_sensors()
        
        # Let sensors initialize
        world.tick()
        time.sleep(1)
        
        # Enable autopilot
        vehicle.set_autopilot(True)
        world.tick()
        
        print("Starting data collection... Press Ctrl+C to stop")
        
        # Data collection loop
        while True:
            world.tick()
            time.sleep(0.05)  # Match fixed_delta_seconds
            
            frame_id = sensor_manager.save_frame_data()
            if int(frame_id) % 100 == 0:
                print(f"Collected {frame_id} frames")
    
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    
    finally:
        # Cleanup
        if 'sensor_manager' in locals():
            sensor_manager.cleanup()
        if 'vehicle' in locals():
            vehicle.destroy()
        
        # Reset world settings
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        pygame.quit()
        print("Cleanup complete!")

if __name__ == "__main__":
    main()