"""
Sensor data collection for autonomous driving.
Collects RGB, semantic segmentation, depth, GPS, IMU, and control data.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2

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
        
        print(f"Output directory created: {output_dir}")
    
    def setup_sensors(self):
        """Setup all sensors"""
        bp_lib = self.world.get_blueprint_library()
        
        # RGB Camera - 800x600
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '800')
        rgb_bp.set_attribute('image_size_y', '600')
        rgb_bp.set_attribute('fov', '90')
        rgb_bp.set_attribute('sensor_tick', '0.05')  # 20 FPS
        rgb_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensors['rgb'] = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors['rgb'].listen(lambda data: self._rgb_callback(data))
        
        # Semantic Segmentation Camera
        sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '800')
        sem_bp.set_attribute('image_size_y', '600')
        sem_bp.set_attribute('fov', '90')
        sem_bp.set_attribute('sensor_tick', '0.05')
        self.sensors['semantic'] = self.world.spawn_actor(sem_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors['semantic'].listen(lambda data: self._semantic_callback(data))
        
        # Depth Camera
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '800')
        depth_bp.set_attribute('image_size_y', '600')
        depth_bp.set_attribute('fov', '90')
        depth_bp.set_attribute('sensor_tick', '0.05')
        self.sensors['depth'] = self.world.spawn_actor(depth_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors['depth'].listen(lambda data: self._depth_callback(data))
        
        # GPS
        gps_bp = bp_lib.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.05')
        self.sensors['gps'] = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['gps'].listen(lambda data: self._gps_callback(data))
        
        # IMU
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        self.sensors['imu'] = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['imu'].listen(lambda data: self._imu_callback(data))
        
        print("All sensors initialized successfully!")
    
    def _rgb_callback(self, image):
        """Process RGB camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Remove alpha
        array = array[:, :, ::-1]  # Convert BGR to RGB
        self.sensor_data['rgb'] = array.copy()
    
    def _semantic_callback(self, image):
        """Process semantic segmentation data"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, 2]  # Class ID channel
        self.sensor_data['semantic'] = array.copy()
    
    def _depth_callback(self, image):
        """Process depth camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        # Convert to actual depth values
        array = array.astype(np.float32)
        normalized = (array[:, :, 0] + array[:, :, 1] * 256 + array[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = normalized * 1000  # CARLA depth range
        self.sensor_data['depth'] = depth_in_meters.copy()
    
    def _gps_callback(self, data):
        """Process GPS data"""
        self.sensor_data['gps'] = {
            'lat': float(data.latitude),
            'lon': float(data.longitude),
            'alt': float(data.altitude)
        }
    
    def _imu_callback(self, data):
        """Process IMU data"""
        self.sensor_data['imu'] = {
            'accel_x': float(data.accelerometer.x),
            'accel_y': float(data.accelerometer.y),
            'accel_z': float(data.accelerometer.z),
            'gyro_x': float(data.gyroscope.x),
            'gyro_y': float(data.gyroscope.y),
            'gyro_z': float(data.gyroscope.z),
            'compass': float(data.compass)
        }
    
    def save_frame_data(self):
        """Save all sensor data for current frame - using proper file naming convention"""
        frame_id = f"{self.frame_count:06d}"  # 6-digit zero-padded
        
        # Save RGB image as PNG
        if 'rgb' in self.sensor_data:
            rgb_path = os.path.join(self.output_dir, 'rgb', f"{frame_id}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(self.sensor_data['rgb'], cv2.COLOR_RGB2BGR))
        
        # Save semantic segmentation as PNG
        if 'semantic' in self.sensor_data:
            sem_path = os.path.join(self.output_dir, 'semantic', f"{frame_id}.png")
            cv2.imwrite(sem_path, self.sensor_data['semantic'])
        
        # Save depth map as NPY
        if 'depth' in self.sensor_data:
            depth_path = os.path.join(self.output_dir, 'depth', f"{frame_id}.npy")
            np.save(depth_path, self.sensor_data['depth'])
        
        # Save GPS data as JSON
        if 'gps' in self.sensor_data:
            gps_path = os.path.join(self.output_dir, 'gps', f"{frame_id}.json")
            with open(gps_path, 'w') as f:
                json.dump(self.sensor_data['gps'], f, indent=2)
        
        # Save IMU data as JSON
        if 'imu' in self.sensor_data:
            imu_path = os.path.join(self.output_dir, 'imu', f"{frame_id}.json")
            with open(imu_path, 'w') as f:
                json.dump(self.sensor_data['imu'], f, indent=2)
        
        # Save control data as JSON
        control = self.vehicle.get_control()
        control_data = {
            'steer': float(control.steer),
            'throttle': float(control.throttle),
            'brake': float(control.brake),
            'hand_brake': bool(control.hand_brake),
            'reverse': bool(control.reverse),
            'manual_gear_shift': bool(control.manual_gear_shift),
            'gear': int(control.gear)
        }
        control_path = os.path.join(self.output_dir, 'control', f"{frame_id}.json")
        with open(control_path, 'w') as f:
            json.dump(control_data, f, indent=2)
        
        self.frame_count += 1
        return frame_id
    
    def get_stats(self):
        """Get collection statistics"""
        return {
            'total_frames': self.frame_count,
            'output_dir': self.output_dir
        }
    
    def cleanup(self):
        """Destroy all sensors"""
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        print("All sensors cleaned up!")

def create_session_directory():
    """Create timestamped session directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"session_{timestamp}"
    session_dir = os.path.join("data", "raw_sessions", session_name)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def main():
    parser = argparse.ArgumentParser(description="CARLA Sensor Data Collection")
    parser.add_argument('--output', help='Output directory (default: auto-generated timestamp)')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--timeout', type=float, default=10.0, help='Connection timeout')
    parser.add_argument('--autopilot', action='store_true', help='Enable autopilot')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to collect')
    args = parser.parse_args()

    # Create output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = create_session_directory()
    
    print(f"Data will be saved to: {output_dir}")

    try:
        # Connect to CARLA
        print(f"Connecting to CARLA server at {args.host}:{args.port}")
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
        
        # Set synchronous mode with 20 FPS (0.05s tick)
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        print("Set synchronous mode: 20 FPS")
        
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")
        
        spawn_point = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        
        # Setup sensors
        sensor_manager = SensorManager(world, vehicle, output_dir)
        sensor_manager.setup_sensors()
        
        # Let sensors initialize
        for _ in range(5):
            world.tick()
            time.sleep(0.1)
        
        # Enable autopilot if requested
        if args.autopilot:
            vehicle.set_autopilot(True)
            print("Autopilot enabled")
        
        print("Starting data collection... Press Ctrl+C to stop")
        
        # Data collection loop
        while True:
            world.tick()
            
            # Save frame data
            frame_id = sensor_manager.save_frame_data()
            
            # Progress reporting
            frame_num = int(frame_id)
            if frame_num % 100 == 0:
                stats = sensor_manager.get_stats()
                print(f"Collected {frame_num} frames - Total: {stats['total_frames']}")
            
            # Check max frames limit
            if args.max_frames and frame_num >= args.max_frames:
                print(f"Reached maximum frames limit: {args.max_frames}")
                break
    
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    
    except Exception as e:
        print(f"Error during data collection: {e}")
        raise
    
    finally:
        # Cleanup
        if 'sensor_manager' in locals():
            stats = sensor_manager.get_stats()
            print(f"Final stats: {stats['total_frames']} frames collected")
            sensor_manager.cleanup()
        
        if 'vehicle' in locals():
            vehicle.destroy()
            print("Vehicle destroyed")
        
        # Restore original world settings
        if 'world' in locals() and 'original_settings' in locals():
            world.apply_settings(original_settings)
            print("World settings restored")
        
        print("Cleanup complete!")

if __name__ == "__main__":
    main()