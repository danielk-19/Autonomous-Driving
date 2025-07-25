"""
Sensor data collection for autonomous driving.
Collects RGB, semantic segmentation, depth, GPS, IMU, and control data.
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2

# Add project root to path and import utilities
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from utils.utils import (
    setup_logging, save_json, ensure_dir, get_timestamp,
    safe_float, safe_int, save_image, setup_carla_world,
    get_carla_vehicle_blueprint, spawn_vehicle, destroy_actors,
    Timer, PerformanceMonitor
)

try:
    import carla
except ImportError:
    raise RuntimeError('Cannot import CARLA')

import logging
logger = logging.getLogger(__name__)

class SensorManager:
    """Manages all sensors for data collection with optimized performance"""
    
    def __init__(self, world, vehicle, output_dir):
        self.world = world
        self.vehicle = vehicle
        self.output_dir = Path(output_dir)
        self.sensors = {}
        self.sensor_data = {}
        self.frame_count = 0
        
        # Performance monitoring
        self.performance = PerformanceMonitor()
        
        # Create output directories using utils
        for sensor_type in ['rgb', 'semantic', 'depth', 'gps', 'imu', 'control']:
            ensure_dir(self.output_dir / sensor_type)
        
        logger.info(f"Output directory created: {output_dir}")
    
    def setup_sensors(self):
        """Setup all sensors with optimized configuration"""
        bp_lib = self.world.get_blueprint_library()
        
        # Common sensor configuration
        sensor_config = {
            'image_size_x': '800',
            'image_size_y': '600',
            'fov': '90',
            'sensor_tick': '0.05'  # 20 FPS
        }
        
        # Camera mount position
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        # RGB Camera
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        for key, value in sensor_config.items():
            rgb_bp.set_attribute(key, value)
        self.sensors['rgb'] = self.world.spawn_actor(rgb_bp, camera_transform, attach_to=self.vehicle)
        self.sensors['rgb'].listen(lambda data: self._rgb_callback(data))
        
        # Semantic Segmentation Camera
        sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        for key, value in sensor_config.items():
            sem_bp.set_attribute(key, value)
        self.sensors['semantic'] = self.world.spawn_actor(sem_bp, camera_transform, attach_to=self.vehicle)
        self.sensors['semantic'].listen(lambda data: self._semantic_callback(data))
        
        # Depth Camera
        depth_bp = bp_lib.find('sensor.camera.depth')
        for key, value in sensor_config.items():
            depth_bp.set_attribute(key, value)
        self.sensors['depth'] = self.world.spawn_actor(depth_bp, camera_transform, attach_to=self.vehicle)
        self.sensors['depth'].listen(lambda data: self._depth_callback(data))
        
        # GPS Sensor
        gps_bp = bp_lib.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.05')
        self.sensors['gps'] = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['gps'].listen(lambda data: self._gps_callback(data))
        
        # IMU Sensor
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        self.sensors['imu'] = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['imu'].listen(lambda data: self._imu_callback(data))
        
        logger.info(f"All {len(self.sensors)} sensors initialized successfully")
        return True
    
    def _rgb_callback(self, image):
        """Process RGB camera data"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            # Remove alpha channel and convert to RGB
            rgb_array = array[:, :, :3]
            self.sensor_data['rgb'] = rgb_array.copy()
        except Exception as e:
            logger.error(f"RGB callback error: {e}")
    
    def _semantic_callback(self, image):
        """Process semantic segmentation data"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            # Extract semantic class ID from red channel
            semantic_array = array[:, :, 2]
            self.sensor_data['semantic'] = semantic_array.copy()
        except Exception as e:
            logger.error(f"Semantic callback error: {e}")
    
    def _depth_callback(self, image):
        """Process depth camera data"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            
            # Convert to depth in meters (CARLA format)
            array = array.astype(np.float32)
            normalized = (array[:, :, 0] + array[:, :, 1] * 256 + 
                         array[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
            depth_meters = normalized * 1000  # CARLA depth range
            
            self.sensor_data['depth'] = depth_meters.copy()
        except Exception as e:
            logger.error(f"Depth callback error: {e}")
    
    def _gps_callback(self, data):
        """Process GPS data"""
        try:
            self.sensor_data['gps'] = {
                'lat': float(data.latitude),
                'lon': float(data.longitude), 
                'alt': float(data.altitude)
            }
        except Exception as e:
            logger.error(f"GPS callback error: {e}")
    
    def _imu_callback(self, data):
        """Process IMU data"""
        try:
            self.sensor_data['imu'] = {
                'accel_x': float(data.accelerometer.x),
                'accel_y': float(data.accelerometer.y),
                'accel_z': float(data.accelerometer.z),
                'gyro_x': float(data.gyroscope.x),
                'gyro_y': float(data.gyroscope.y),
                'gyro_z': float(data.gyroscope.z),
                'compass': float(data.compass)
            }
        except Exception as e:
            logger.error(f"IMU callback error: {e}")
    
    def save_frame_data(self):
        """Save all sensor data for current frame"""
        try:
            self.performance.log_frame()
            
            # Use 6-digit zero-padded frame ID
            frame_id = f"{self.frame_count:06d}"
            
            # Save RGB image as PNG (convert RGB to BGR for OpenCV)
            if 'rgb' in self.sensor_data:
                rgb_path = self.output_dir / 'rgb' / f"{frame_id}.png"
                rgb_bgr = cv2.cvtColor(self.sensor_data['rgb'], cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(rgb_path), rgb_bgr)
                if not success:
                    logger.warning(f"Failed to save RGB frame {frame_id}")
            
            # Save semantic segmentation as PNG
            if 'semantic' in self.sensor_data:
                sem_path = self.output_dir / 'semantic' / f"{frame_id}.png"
                success = cv2.imwrite(str(sem_path), self.sensor_data['semantic'])
                if not success:
                    logger.warning(f"Failed to save semantic frame {frame_id}")
            
            # Save depth map as NPY
            if 'depth' in self.sensor_data:
                depth_path = self.output_dir / 'depth' / f"{frame_id}.npy"
                try:
                    np.save(depth_path, self.sensor_data['depth'])
                except Exception as e:
                    logger.warning(f"Failed to save depth frame {frame_id}: {e}")
            
            # Save GPS data as JSON using utils function
            if 'gps' in self.sensor_data:
                gps_path = self.output_dir / 'gps' / f"{frame_id}.json"
                if not save_json(self.sensor_data['gps'], gps_path):
                    logger.warning(f"Failed to save GPS frame {frame_id}")
            
            # Save IMU data as JSON using utils function
            if 'imu' in self.sensor_data:
                imu_path = self.output_dir / 'imu' / f"{frame_id}.json"
                if not save_json(self.sensor_data['imu'], imu_path):
                    logger.warning(f"Failed to save IMU frame {frame_id}")
            
            # Save control data as JSON
            control = self.vehicle.get_control()
            vehicle_transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h
            
            control_data = {
                'steer': float(control.steer),
                'throttle': float(control.throttle),
                'brake': float(control.brake),
                'speed': float(speed),
                'timestamp': time.time(),  # Add timestamp
                'hand_brake': bool(control.hand_brake),
                'reverse': bool(control.reverse),
                'gear': int(control.gear),
                'location': {
                    'x': float(vehicle_transform.location.x),
                    'y': float(vehicle_transform.location.y),
                    'z': float(vehicle_transform.location.z)
                },
                'rotation': {
                    'pitch': float(vehicle_transform.rotation.pitch),
                    'yaw': float(vehicle_transform.rotation.yaw),
                    'roll': float(vehicle_transform.rotation.roll)
                }
            }
            
            control_path = self.output_dir / 'control' / f"{frame_id}.json"
            if not save_json(control_data, control_path):
                logger.warning(f"Failed to save control frame {frame_id}")
            
            self.frame_count += 1
            return frame_id
            
        except Exception as e:
            logger.error(f"Failed to save frame data: {e}")
            return None
    
    def get_stats(self):
        """Get collection statistics"""
        perf_stats = self.performance.get_stats()
        return {
            'total_frames': self.frame_count,
            'output_dir': str(self.output_dir),
            'sensors_active': len(self.sensors),
            'data_types': list(self.sensor_data.keys()),
            'avg_fps': perf_stats['avg_fps'],
            'total_time': perf_stats['total_time']
        }
    
    def cleanup(self):
        """Safely destroy all sensors using utils function"""
        if self.sensors:
            destroy_actors(list(self.sensors.values()))
            self.sensors.clear()
            logger.info("All sensors cleaned up")

def create_session_directory():
    """Create timestamped session directory using utils functions"""
    timestamp = get_timestamp()
    session_name = f"session_{timestamp}"
    session_dir = Path("data") / "raw_sessions" / session_name
    ensure_dir(session_dir)
    return session_dir

def main():
    """Main data collection function with comprehensive error handling"""
    parser = argparse.ArgumentParser(description="CARLA Sensor Data Collection")
    parser.add_argument('--output', help='Output directory (default: auto-generated timestamp)')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--timeout', type=float, default=10.0, help='Connection timeout')
    parser.add_argument('--autopilot', action='store_true', help='Enable autopilot')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to collect')
    parser.add_argument('--vehicle-filter', default='vehicle.tesla.model3', 
                       help='Vehicle blueprint filter')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', help='Log file path (optional)')
    args = parser.parse_args()

    # Setup logging using utils function
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level=log_level, log_file=args.log_file)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
        ensure_dir(output_dir)
    else:
        output_dir = create_session_directory()
    
    logger.info(f"CARLA Sensor Data Collection")
    logger.info(f"Data will be saved to: {output_dir}")
    logger.info(f"Vehicle filter: {args.vehicle_filter}")
    logger.info("=" * 50)

    # Initialize variables for cleanup
    client = None
    world = None
    vehicle = None
    sensor_manager = None
    original_settings = None
    performance_monitor = PerformanceMonitor()

    try:
        # Connect to CARLA using utility function
        logger.info(f"Connecting to CARLA server at {args.host}:{args.port}")
        client, world = setup_carla_world(args.host, args.port, args.timeout)
        
        if client is None or world is None:
            raise RuntimeError("Failed to connect to CARLA server")
        
        # Configure synchronous mode for consistent data collection
        with Timer("World setup"):
            original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            settings.no_rendering_mode = False  # Need rendering for sensors
            world.apply_settings(settings)
            logger.info("Set synchronous mode: 20 FPS")
        
        # Spawn vehicle using utility functions
        with Timer("Vehicle spawning"):
            blueprint = get_carla_vehicle_blueprint(world, args.vehicle_filter)
            if blueprint is None:
                raise RuntimeError(f"No vehicle found matching filter: {args.vehicle_filter}")
            
            vehicle = spawn_vehicle(world, blueprint)
            if vehicle is None:
                raise RuntimeError("Failed to spawn vehicle")
        
        # Setup sensors
        with Timer("Sensor setup"):
            sensor_manager = SensorManager(world, vehicle, output_dir)
            if not sensor_manager.setup_sensors():
                raise RuntimeError("Failed to setup sensors")
        
        # Let sensors initialize
        logger.info("Initializing sensors...")
        for i in range(10):
            world.tick()
            time.sleep(0.05)
        
        # Enable autopilot if requested
        if args.autopilot:
            vehicle.set_autopilot(True)
            logger.info("Autopilot enabled - vehicle will drive automatically")
        else:
            logger.info("Manual control mode - vehicle will remain stationary")
        
        logger.info("Starting data collection... Press Ctrl+C to stop")
        performance_monitor.reset()
        
        # Data collection loop
        frame_errors = 0
        max_errors = 50  # Allow some frame errors before stopping
        last_report_time = time.time()
        report_interval = 10.0  # Report every 10 seconds
        
        while True:
            try:
                world.tick()
                performance_monitor.log_frame()
                
                # Save frame data
                frame_id = sensor_manager.save_frame_data()
                
                if frame_id is None:
                    frame_errors += 1
                    if frame_errors > max_errors:
                        logger.error(f"Too many frame errors ({frame_errors}), stopping collection")
                        break
                    continue
                else:
                    frame_errors = 0  # Reset error count on successful frame
                
                # Progress reporting (both per-frame and time-based)
                frame_num = int(frame_id)
                current_time = time.time()
                
                # Per-frame reporting
                if frame_num > 0 and frame_num % 100 == 0:
                    stats = sensor_manager.get_stats()
                    logger.info(f"Frame {frame_num:06d} - FPS: {stats['avg_fps']:.1f} - "
                              f"Sensors: {stats['sensors_active']} - "
                              f"Data types: {len(stats['data_types'])}")
                
                # Time-based reporting
                if current_time - last_report_time >= report_interval:
                    stats = sensor_manager.get_stats()
                    logger.info(f"Collection status - Frame: {frame_num:06d}, "
                              f"FPS: {stats['avg_fps']:.1f}, "
                              f"Total time: {stats['total_time']:.1f}s")
                    last_report_time = current_time
                
                # Check max frames limit
                if args.max_frames and frame_num >= args.max_frames:
                    logger.info(f"Reached maximum frames limit: {args.max_frames}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                frame_errors += 1
                if frame_errors > max_errors:
                    logger.error("Too many consecutive errors, stopping")
                    break
    
    except KeyboardInterrupt:
        logger.info("\nData collection stopped by user")
    
    except Exception as e:
        logger.error(f"Critical error during data collection: {e}")
        raise
    
    finally:
        # Comprehensive cleanup with timing
        logger.info("Starting cleanup...")
        
        if sensor_manager:
            with Timer("Sensor cleanup"):
                stats = sensor_manager.get_stats()
                logger.info(f"Final collection stats:")
                logger.info(f"  Total frames: {stats['total_frames']}")
                logger.info(f"  Collection time: {stats['total_time']:.1f}s")
                logger.info(f"  Average FPS: {stats['avg_fps']:.1f}")
                logger.info(f"  Output directory: {stats['output_dir']}")
                sensor_manager.cleanup()
        
        if vehicle:
            with Timer("Vehicle cleanup"):
                destroy_actors([vehicle])
                logger.info("Vehicle destroyed")
        
        if world and original_settings:
            with Timer("World settings restore"):
                world.apply_settings(original_settings)
                logger.info("World settings restored")
        
        # Final performance report
        if 'performance_monitor' in locals():
            final_stats = performance_monitor.get_stats()
            logger.info("=" * 50)
            logger.info("FINAL PERFORMANCE REPORT:")
            logger.info(f"  Total collection time: {final_stats['total_time']:.1f} seconds")
            logger.info(f"  Total frames processed: {final_stats['total_frames']}")
            logger.info(f"  Average FPS: {final_stats['avg_fps']:.1f}")
            logger.info(f"  Frame time - Avg: {final_stats['avg_frame_time']:.3f}s, "
                       f"Min: {final_stats['min_frame_time']:.3f}s, "
                       f"Max: {final_stats['max_frame_time']:.3f}s")
        
        logger.info("Data collection session complete!")

if __name__ == "__main__":
    main()