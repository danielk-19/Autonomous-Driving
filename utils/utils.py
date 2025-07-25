"""
Utility functions for CARLA Autonomous Driving System
Common utilities used across the project for data handling, visualization, and system operations.
"""

import json
import numpy as np
import cv2
from pathlib import Path
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import carla
import random
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)

# Data utilities
def load_json(file_path):
    """Safely load JSON file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return None

def save_json(data, file_path, indent=2):
    """Safely save data to JSON file"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_timestamp():
    """Get current timestamp in standard format"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_float(value, default=0.0):
    """Safely convert value to float with default"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int with default"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# Image processing utilities
def load_image(image_path, target_size=None):
    """Load image with optional resizing"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None
        
        if target_size:
            image = cv2.resize(image, target_size)
        
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, output_path):
    """Save image with error handling"""
    try:
        ensure_dir(Path(output_path).parent)
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        logging.error(f"Failed to save image to {output_path}: {e}")
        return False

def preprocess_image(image, target_size=(224, 224), normalize=True):
    """Preprocess image for neural network input"""
    # Resize
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize
    image = image.astype(np.float32)
    if normalize:
        image = image / 255.0
    
    # Convert to CHW format for PyTorch
    image = np.transpose(image, (2, 0, 1))
    
    return image

def combine_rgb_semantic(rgb_image, semantic_image, alpha=0.7):
    """Combine RGB and semantic images with transparency"""
    try:
        # Ensure same size
        if rgb_image.shape[:2] != semantic_image.shape[:2]:
            semantic_image = cv2.resize(semantic_image, 
                                      (rgb_image.shape[1], rgb_image.shape[0]))
        
        # Convert semantic to colormap if grayscale
        if len(semantic_image.shape) == 2:
            semantic_image = cv2.applyColorMap(semantic_image, cv2.COLORMAP_JET)
        elif semantic_image.shape[2] == 1:
            semantic_image = cv2.applyColorMap(semantic_image[:,:,0], cv2.COLORMAP_JET)
        
        # Blend images
        combined = cv2.addWeighted(rgb_image, alpha, semantic_image, 1-alpha, 0)
        return combined
    except Exception as e:
        logging.error(f"Error combining RGB and semantic images: {e}")
        return rgb_image

# CARLA utilities
def setup_carla_world(host='localhost', port=2000, timeout=10.0):
    """Setup CARLA world connection with error handling"""
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        
        # Test connection
        world = client.get_world()
        logging.info(f"Connected to CARLA server at {host}:{port}")
        return client, world
    except Exception as e:
        logging.error(f"Failed to connect to CARLA server: {e}")
        return None, None

def get_carla_vehicle_blueprint(world, filter_pattern='vehicle.*'):
    """Get a random vehicle blueprint"""
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter(filter_pattern)
    return random.choice(vehicle_blueprints)

def spawn_vehicle(world, blueprint, spawn_point=None):
    """Spawn vehicle at spawn point with error handling"""
    try:
        if spawn_point is None:
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
        
        vehicle = world.spawn_actor(blueprint, spawn_point)
        logging.info(f"Vehicle spawned at location: {spawn_point.location}")
        return vehicle
    except Exception as e:
        logging.error(f"Failed to spawn vehicle: {e}")
        return None

def destroy_actors(actors):
    """Safely destroy CARLA actors"""
    destroyed_count = 0
    for actor in actors:
        if actor is not None:
            try:
                actor.destroy()
                destroyed_count += 1
            except Exception as e:
                logging.warning(f"Failed to destroy actor: {e}")
    
    logging.info(f"Destroyed {destroyed_count} actors")

# Model utilities
def load_pytorch_model(model_path, model_class, device='cpu'):
    """Load PyTorch model with error handling"""
    try:
        model = model_class()
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return None

def save_pytorch_model(model, optimizer, epoch, loss, model_path):
    """Save PyTorch model checkpoint"""
    try:
        ensure_dir(Path(model_path).parent)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, model_path)
        logging.info(f"Model checkpoint saved to {model_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save model to {model_path}: {e}")
        return False

# Data analysis utilities
def calculate_steering_distribution(measurements):
    """Calculate steering angle distribution statistics"""
    steering_angles = [m['steering'] for m in measurements if 'steering' in m]
    
    if not steering_angles:
        return None
    
    return {
        'mean': np.mean(steering_angles),
        'std': np.std(steering_angles),
        'min': np.min(steering_angles),
        'max': np.max(steering_angles),
        'median': np.median(steering_angles),
        'straight_ratio': sum(1 for s in steering_angles if abs(s) < 0.05) / len(steering_angles)
    }

def balance_steering_samples(samples, threshold=0.05, balance_ratio=0.3):
    """Balance steering samples to reduce straight-driving bias"""
    straight_samples = [s for s in samples if abs(s.get('steering', 0)) < threshold]
    turn_samples = [s for s in samples if abs(s.get('steering', 0)) >= threshold]
    
    # Calculate target straight samples
    target_straight = int(len(turn_samples) * (1/balance_ratio - 1))
    
    if len(straight_samples) > target_straight:
        straight_samples = random.sample(straight_samples, target_straight)
    
    balanced_samples = straight_samples + turn_samples
    random.shuffle(balanced_samples)
    
    logging.info(f"Balanced dataset: {len(straight_samples)} straight + {len(turn_samples)} turning = {len(balanced_samples)} total")
    
    return balanced_samples

# Visualization utilities
def plot_training_history(history, save_path=None):
    """Plot training and validation loss history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.get('train_loss', []), label='Train Loss', color='blue')
    ax1.plot(history.get('val_loss', []), label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot (if available)
    if 'train_acc' in history and 'val_acc' in history:
        ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")
    
    return fig

def visualize_predictions(images, true_steering, pred_steering, save_path=None, max_samples=9):
    """Visualize model predictions vs ground truth"""
    n_samples = min(len(images), max_samples)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Convert image format if needed
        img = images[i]
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_steering[i]:.3f}\nPred: {pred_steering[i]:.3f}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Predictions visualization saved to {save_path}")
    
    return fig

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor system performance during training/inference"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.frame_times = []
        self.memory_usage = []
    
    def log_frame(self):
        """Log processing time for current frame"""
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
    
    def get_fps(self):
        """Calculate average FPS"""
        if not self.frame_times:
            return 0
        return 1.0 / np.mean(self.frame_times)
    
    def get_stats(self):
        """Get performance statistics"""
        total_time = time.time() - self.start_time
        avg_fps = self.get_fps()
        
        return {
            'total_time': total_time,
            'total_frames': len(self.frame_times),
            'avg_fps': avg_fps,
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'max_frame_time': np.max(self.frame_times) if self.frame_times else 0,
            'min_frame_time': np.min(self.frame_times) if self.frame_times else 0
        }

# Configuration utilities
def load_config(config_path, default_config=None):
    """Load configuration from JSON file with defaults"""
    if Path(config_path).exists():
        config = load_json(config_path)
        if config is None:
            config = default_config or {}
    else:
        config = default_config or {}
        # Save default config
        if default_config:
            save_json(default_config, config_path)
            logging.info(f"Created default config file: {config_path}")
    
    return config

def merge_configs(base_config, override_config):
    """Merge two configuration dictionaries"""
    merged = base_config.copy()
    if override_config:
        merged.update(override_config)
    return merged

# Safety utilities
def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_collision_imminent(vehicle_location, obstacle_locations, threshold=3.0):
    """Check if collision is imminent with any obstacle"""
    for obstacle_loc in obstacle_locations:
        distance = calculate_distance(
            (vehicle_location.x, vehicle_location.y),
            (obstacle_loc.x, obstacle_loc.y)
        )
        if distance < threshold:
            return True
    return False

# File system utilities
def get_latest_file(directory, pattern="*.pth"):
    """Get the most recently modified file matching pattern"""
    files = list(Path(directory).glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

def cleanup_old_files(directory, pattern="*.log", keep_count=5):
    """Keep only the most recent N files matching pattern"""
    files = list(Path(directory).glob(pattern))
    if len(files) <= keep_count:
        return
    
    # Sort by modification time, newest first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Remove old files
    for old_file in files[keep_count:]:
        try:
            old_file.unlink()
            logging.info(f"Removed old file: {old_file}")
        except Exception as e:
            logging.warning(f"Failed to remove {old_file}: {e}")

# System utilities
def check_gpu_availability():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        logging.info(f"GPU available: {gpu_name} ({gpu_count} devices)")
        return True
    else:
        logging.info("No GPU available, using CPU")
        return False

def get_system_info():
    """Get system information"""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
        'gpu_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__
    }

# Context managers
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logging.info(f"{self.operation_name} completed in {self.duration:.3f} seconds")

class LoggingContext:
    """Context manager for temporary logging level changes"""
    
    def __init__(self, level=logging.DEBUG, logger_name=None):
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)

# Data validation utilities
def validate_measurements_format(measurements):
    """Validate measurements follow the correct format"""
    required_fields = ['frame_id', 'timestamp', 'steering', 'throttle', 'brake', 'speed', 'gps', 'imu']
    required_gps_fields = ['lat', 'lon', 'alt']
    required_imu_fields = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'compass']
    
    errors = []
    
    for i, measurement in enumerate(measurements):
        # Check top-level fields
        for field in required_fields:
            if field not in measurement:
                errors.append(f"Frame {i}: Missing field '{field}'")
        
        # Check GPS fields
        if 'gps' in measurement:
            for gps_field in required_gps_fields:
                if gps_field not in measurement['gps']:
                    errors.append(f"Frame {i}: Missing GPS field '{gps_field}'")
        
        # Check IMU fields
        if 'imu' in measurement:
            for imu_field in required_imu_fields:
                if imu_field not in measurement['imu']:
                    errors.append(f"Frame {i}: Missing IMU field '{imu_field}'")
        
        # Validate data types
        try:
            float(measurement.get('steering', 0))
            float(measurement.get('throttle', 0))
            float(measurement.get('brake', 0))
            float(measurement.get('speed', 0))
            int(measurement.get('frame_id', 0))
        except (ValueError, TypeError) as e:
            errors.append(f"Frame {i}: Invalid numeric value - {e}")
    
    return errors

# Export utilities for easy importing
__all__ = [
    'setup_logging', 'load_json', 'save_json', 'ensure_dir', 'get_timestamp',
    'safe_float', 'safe_int', 'load_image', 'save_image', 'preprocess_image',
    'combine_rgb_semantic', 'setup_carla_world', 'get_carla_vehicle_blueprint',
    'spawn_vehicle', 'destroy_actors', 'load_pytorch_model', 'save_pytorch_model',
    'calculate_steering_distribution', 'balance_steering_samples', 'plot_training_history',
    'visualize_predictions', 'PerformanceMonitor', 'load_config', 'merge_configs',
    'clamp', 'normalize_angle', 'calculate_distance', 'is_collision_imminent',
    'get_latest_file', 'cleanup_old_files', 'check_gpu_availability', 'get_system_info',
    'Timer', 'LoggingContext', 'validate_measurements_format'
]