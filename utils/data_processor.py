"""
Data Processor for CARLA Autonomous Driving System
Converts raw session data to ML-ready processed format.

Handles:
- Raw session data → Processed episodes conversion
- RGB + semantic image combination
- Metadata consolidation into measurements.json
- Train/val/test split creation
- Data quality validation
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import random
import shutil
from datetime import datetime
import logging

# Import utilities from our utils module
import sys
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from utils import (
    setup_logging, load_json, save_json, ensure_dir, get_timestamp,
    safe_float, safe_int, load_image, save_image, combine_rgb_semantic,
    validate_measurements_format, Timer, PerformanceMonitor
)

# Setup logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes raw CARLA session data into ML-ready format
    
    Input format (Raw sessions):
    data/raw_sessions/session_YYYYMMDD_HHMMSS/
    ├── rgb/           # RGB images: 000000.png, 000001.png, ...
    ├── semantic/      # Semantic segmentation: 000000.png, 000001.png, ...
    ├── depth/         # Depth maps: 000000.npy, 000001.npy, ...
    ├── gps/           # GPS data: 000000.json, 000001.json, ...
    ├── imu/           # IMU data: 000000.json, 000001.json, ...
    └── control/       # Control data: 000000.json, 000001.json, ...
    
    Output format (Processed episodes):
    data/processed/episodes/episode_001/
    ├── images/           # Combined RGB+semantic: 000000.png, 000001.png, ...
    └── measurements.json # All frame metadata in single file
    """
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.raw_sessions_dir = self.data_root / "raw_sessions"
        self.processed_dir = self.data_root / "processed"
        self.episodes_dir = self.processed_dir / "episodes"
        
        # Create processed directories
        ensure_dir(self.processed_dir)
        ensure_dir(self.episodes_dir)
        
        # Initialize performance monitor
        self.performance = PerformanceMonitor()
        
        logger.info(f"DataProcessor initialized with data root: {self.data_root}")
    
    def process_all_sessions(self):
        """Process all raw sessions into episodes"""
        logger.info("Starting processing of all raw sessions...")
        
        session_dirs = [d for d in self.raw_sessions_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('session_')]
        
        if not session_dirs:
            logger.warning(f"No session directories found in {self.raw_sessions_dir}")
            return 0, 0
        
        logger.info(f"Found {len(session_dirs)} sessions to process")
        
        episode_counter = 1
        total_frames = 0
        
        with Timer("All sessions processing"):
            for session_dir in sorted(session_dirs):
                logger.info(f"Processing session: {session_dir.name}")
                
                try:
                    frames_processed = self.process_session(session_dir, episode_counter)
                    if frames_processed > 0:
                        total_frames += frames_processed
                        episode_counter += 1
                        logger.info(f"Session processed: {frames_processed} frames")
                    else:
                        logger.warning(f"No valid frames found in session: {session_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to process session {session_dir.name}: {e}")
                    continue
        
        logger.info(f"Processing complete! Created {episode_counter-1} episodes with {total_frames} total frames")
        
        # Create data splits after processing all sessions
        self.create_data_splits()
        
        return episode_counter - 1, total_frames
    
    def process_session(self, session_dir, episode_id):
        """Process a single raw session into a processed episode"""
        
        # Create episode directory
        episode_dir = self.episodes_dir / f"episode_{episode_id:03d}"
        ensure_dir(episode_dir)
        images_dir = episode_dir / "images"
        ensure_dir(images_dir)
        
        # Get all frame IDs from RGB directory
        rgb_dir = session_dir / "rgb"
        if not rgb_dir.exists():
            logger.warning(f"No RGB directory found in {session_dir}")
            return 0
        
        rgb_files = sorted(rgb_dir.glob("*.png"))
        frame_ids = [f.stem for f in rgb_files]
        
        if not frame_ids:
            logger.warning(f"No RGB images found in {rgb_dir}")
            return 0
        
        logger.info(f"Processing {len(frame_ids)} frames for episode {episode_id}")
        
        measurements = []
        valid_frames = 0
        
        for frame_id in frame_ids:
            try:
                self.performance.log_frame()
                
                # Validate that all required data exists for this frame
                if not self._validate_frame_data(session_dir, frame_id):
                    logger.debug(f"Skipping frame {frame_id} - missing data")
                    continue
                
                # Combine RGB and semantic images using utils function
                combined_image = self._combine_images_for_frame(session_dir, frame_id)
                if combined_image is None:
                    logger.debug(f"Failed to combine images for frame {frame_id}")
                    continue
                
                # Save combined image using utils function
                output_image_path = images_dir / f"{frame_id}.png"
                if not save_image(combined_image, output_image_path):
                    logger.debug(f"Failed to save image for frame {frame_id}")
                    continue
                
                # Load and consolidate metadata
                frame_measurements = self._load_frame_measurements(session_dir, frame_id)
                if frame_measurements is None:
                    logger.debug(f"Failed to load measurements for frame {frame_id}")
                    continue
                
                measurements.append(frame_measurements)
                valid_frames += 1
                
                if valid_frames % 100 == 0:
                    fps = self.performance.get_fps()
                    logger.info(f"Processed {valid_frames} frames... (FPS: {fps:.1f})")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_id}: {e}")
                continue
        
        # Save measurements.json using utils function
        measurements_file = episode_dir / "measurements.json"
        if not save_json(measurements, measurements_file):
            logger.error(f"Failed to save measurements for episode {episode_id}")
            return 0
        
        logger.info(f"Episode {episode_id} complete: {valid_frames} valid frames")
        return valid_frames
    
    def _validate_frame_data(self, session_dir, frame_id):
        """Validate that all required data files exist for a frame"""
        required_files = [
            session_dir / "rgb" / f"{frame_id}.png",
            session_dir / "semantic" / f"{frame_id}.png",
            session_dir / "gps" / f"{frame_id}.json",
            session_dir / "imu" / f"{frame_id}.json",
            session_dir / "control" / f"{frame_id}.json"
        ]
        
        return all(f.exists() for f in required_files)
    
    def _combine_images_for_frame(self, session_dir, frame_id):
        """Combine RGB and semantic images for training using utils function"""
        rgb_path = session_dir / "rgb" / f"{frame_id}.png"
        semantic_path = session_dir / "semantic" / f"{frame_id}.png"
        
        try:
            # Load images using utils function
            rgb_img = load_image(rgb_path)
            semantic_img = load_image(semantic_path)
            
            if rgb_img is None or semantic_img is None:
                return None
            
            # Resize to standard training size (224x224)
            target_size = (224, 224)
            rgb_resized = cv2.resize(rgb_img, target_size)
            semantic_resized = cv2.resize(semantic_img, target_size)
            
            # Use utils function to combine images
            combined = combine_rgb_semantic(rgb_resized, semantic_resized, alpha=0.7)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining images for frame {frame_id}: {e}")
            return None
    
    def _load_frame_measurements(self, session_dir, frame_id):
        """Load and consolidate all measurements for a single frame using utils functions"""
        try:
            # Load data using utils functions
            control_data = load_json(session_dir / "control" / f"{frame_id}.json")
            gps_data = load_json(session_dir / "gps" / f"{frame_id}.json")
            imu_data = load_json(session_dir / "imu" / f"{frame_id}.json")
            
            if not all([control_data, gps_data, imu_data]):
                return None
            
            # Create consolidated measurement using safe conversion functions
            measurement = {
                "frame_id": safe_int(frame_id),
                "timestamp": control_data.get("timestamp", datetime.now().isoformat()),
                "steering": safe_float(control_data.get("steer", 0.0)),
                "throttle": safe_float(control_data.get("throttle", 0.0)),
                "brake": safe_float(control_data.get("brake", 0.0)),
                "speed": safe_float(control_data.get("speed", 0.0)),
                "gps": {
                    "lat": safe_float(gps_data.get("lat", 0.0)),
                    "lon": safe_float(gps_data.get("lon", 0.0)),
                    "alt": safe_float(gps_data.get("alt", 0.0))
                },
                "imu": {
                    "accel_x": safe_float(imu_data.get("accel_x", 0.0)),
                    "accel_y": safe_float(imu_data.get("accel_y", 0.0)),
                    "accel_z": safe_float(imu_data.get("accel_z", 9.8)),
                    "gyro_x": safe_float(imu_data.get("gyro_x", 0.0)),
                    "gyro_y": safe_float(imu_data.get("gyro_y", 0.0)),
                    "gyro_z": safe_float(imu_data.get("gyro_z", 0.0)),
                    "compass": safe_float(imu_data.get("compass", 0.0))
                }
            }
            
            return measurement
            
        except Exception as e:
            logger.error(f"Error loading measurements for frame {frame_id}: {e}")
            return None
    
    def create_data_splits(self, test_size=0.2, val_size=0.1):
        """Create train/validation/test splits by episode"""
        logger.info("Creating data splits...")
        
        # Get all episodes
        episode_dirs = [d for d in self.episodes_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('episode_')]
        
        if not episode_dirs:
            logger.warning("No episodes found for splitting")
            return None
        
        episode_ids = sorted([d.name for d in episode_dirs])
        logger.info(f"Found {len(episode_ids)} episodes for splitting")
        
        # Split episodes (not individual frames)
        train_val_episodes, test_episodes = train_test_split(
            episode_ids, test_size=test_size, random_state=42
        )
        
        train_episodes, val_episodes = train_test_split(
            train_val_episodes, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Count total samples in each split
        def count_episode_samples(episode_list):
            total = 0
            for episode_id in episode_list:
                measurements_file = self.episodes_dir / episode_id / "measurements.json"
                measurements = load_json(measurements_file)
                if measurements:
                    total += len(measurements)
            return total
        
        train_samples = count_episode_samples(train_episodes)
        val_samples = count_episode_samples(val_episodes)
        test_samples = count_episode_samples(test_episodes)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(train_episodes)} episodes, {train_samples} samples")
        logger.info(f"  Validation: {len(val_episodes)} episodes, {val_samples} samples")
        logger.info(f"  Test: {len(test_episodes)} episodes, {test_samples} samples")
        
        # Create sample lists for each split
        train_sample_list = self._create_sample_list(train_episodes)
        val_sample_list = self._create_sample_list(val_episodes)
        test_sample_list = self._create_sample_list(test_episodes)
        
        # Save splits using utils functions
        splits_data = {
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "test_episodes": test_episodes,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "created_at": datetime.now().isoformat()
        }
        
        # Save all split files
        split_files = [
            (self.processed_dir / "data_splits.json", splits_data),
            (self.processed_dir / "train_samples.json", train_sample_list),
            (self.processed_dir / "val_samples.json", val_sample_list),
            (self.processed_dir / "test_samples.json", test_sample_list)
        ]
        
        for file_path, data in split_files:
            if not save_json(data, file_path):
                logger.error(f"Failed to save {file_path}")
                return None
        
        logger.info("Data splits saved successfully")
        return splits_data
    
    def _create_sample_list(self, episode_list):
        """Create list of individual samples for a split"""
        samples = []
        
        for episode_id in episode_list:
            measurements_file = self.episodes_dir / episode_id / "measurements.json"
            measurements = load_json(measurements_file)
            
            if measurements:
                for measurement in measurements:
                    sample = {
                        "episode": episode_id,
                        "frame_id": measurement["frame_id"],
                        "image_path": f"episodes/{episode_id}/images/{measurement['frame_id']:06d}.png",
                        "steering": measurement["steering"],
                        "throttle": measurement["throttle"],
                        "brake": measurement["brake"],
                        "speed": measurement["speed"]
                    }
                    samples.append(sample)
        
        return samples
    
    def analyze_processed_data(self):
        """Analyze the processed dataset and generate statistics"""
        logger.info("Analyzing processed dataset...")
        
        # Count episodes and total frames
        episode_dirs = [d for d in self.episodes_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('episode_')]
        
        total_frames = 0
        steering_data = []
        speed_data = []
        
        for episode_dir in episode_dirs:
            measurements_file = episode_dir / "measurements.json"
            measurements = load_json(measurements_file)
            
            if measurements:
                total_frames += len(measurements)
                
                for m in measurements:
                    steering_data.append(m["steering"])
                    speed_data.append(m["speed"])
        
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Episodes: {len(episode_dirs)}")
        logger.info(f"  Total frames: {total_frames}")
        
        stats = {
            'episodes': len(episode_dirs),
            'total_frames': total_frames,
            'steering_stats': {},
            'speed_stats': {}
        }
        
        if steering_data:
            stats['steering_stats'] = {
                'mean': np.mean(steering_data),
                'std': np.std(steering_data),
                'min': np.min(steering_data),
                'max': np.max(steering_data),
                'median': np.median(steering_data)
            }
            
            stats['speed_stats'] = {
                'mean': np.mean(speed_data),
                'std': np.std(speed_data),
                'min': np.min(speed_data),
                'max': np.max(speed_data),
                'median': np.median(speed_data)
            }
            
            logger.info(f"  Steering - Mean: {stats['steering_stats']['mean']:.3f}, "
                       f"Std: {stats['steering_stats']['std']:.3f}, "
                       f"Range: [{stats['steering_stats']['min']:.3f}, {stats['steering_stats']['max']:.3f}]")
            
            logger.info(f"  Speed - Mean: {stats['speed_stats']['mean']:.3f}, "
                       f"Std: {stats['speed_stats']['std']:.3f}, "
                       f"Range: [{stats['speed_stats']['min']:.3f}, {stats['speed_stats']['max']:.3f}]")
            
            # Plot distributions
            self._plot_data_distributions(steering_data, speed_data)
        
        return stats
    
    def _plot_data_distributions(self, steering_data, speed_data):
        """Plot steering and speed distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Steering distribution
        axes[0, 0].hist(steering_data, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Steering Angle')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Steering Angle Distribution')
        
        # Steering over time (sample)
        sample_size = min(1000, len(steering_data))
        axes[0, 1].plot(steering_data[:sample_size])
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Steering Angle')
        axes[0, 1].set_title(f'Steering Sequence (First {sample_size} frames)')
        
        # Speed distribution
        axes[1, 0].hist(speed_data, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Speed (km/h)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Speed Distribution')
        
        # Speed over time (sample)
        axes[1, 1].plot(speed_data[:sample_size])
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Speed (km/h)')
        axes[1, 1].set_title(f'Speed Sequence (First {sample_size} frames)')
        
        plt.tight_layout()
        plot_path = self.processed_dir / 'data_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Data analysis plots saved to: {plot_path}")
    
    def validate_processed_data(self):
        """Validate processed data integrity using utils functions"""
        logger.info("Validating processed data...")
        
        issues = []
        
        # Check episodes
        episode_dirs = [d for d in self.episodes_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('episode_')]
        
        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            
            # Check required files
            measurements_file = episode_dir / "measurements.json"
            images_dir = episode_dir / "images"
            
            if not measurements_file.exists():
                issues.append(f"Missing measurements.json in {episode_id}")
                continue
            
            if not images_dir.exists():
                issues.append(f"Missing images directory in {episode_id}")
                continue
            
            # Load measurements using utils function
            measurements = load_json(measurements_file)
            if not measurements:
                issues.append(f"Corrupted measurements.json in {episode_id}")
                continue
            
            # Validate measurements format using utils function
            format_errors = validate_measurements_format(measurements)
            if format_errors:
                issues.extend([f"{episode_id}: {error}" for error in format_errors[:5]])  # Limit errors
            
            # Check image-measurement alignment
            image_files = list(images_dir.glob("*.png"))
            
            if len(image_files) != len(measurements):
                issues.append(f"Image count mismatch in {episode_id}: "
                            f"{len(image_files)} images vs {len(measurements)} measurements")
            
            # Check for missing images (sample check)
            for i, measurement in enumerate(measurements[:10]):  # Check first 10
                frame_id = measurement["frame_id"]
                image_path = images_dir / f"{frame_id:06d}.png"
                if not image_path.exists():
                    issues.append(f"Missing image {frame_id:06d}.png in {episode_id}")
        
        # Check split files
        required_split_files = [
            "data_splits.json",
            "train_samples.json",
            "val_samples.json",
            "test_samples.json"
        ]
        
        for split_file in required_split_files:
            if not (self.processed_dir / split_file).exists():
                issues.append(f"Missing split file: {split_file}")
        
        if issues:
            logger.warning(f"Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more issues")
        else:
            logger.info("✓ Data validation passed - no issues found")
        
        return issues
    
    def clean_processed_data(self):
        """Clean up processed data by removing incomplete episodes"""
        logger.info("Cleaning processed data...")
        
        removed_episodes = []
        
        episode_dirs = [d for d in self.episodes_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('episode_')]
        
        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            measurements_file = episode_dir / "measurements.json"
            images_dir = episode_dir / "images"
            
            should_remove = False
            
            # Check if essential files exist
            if not measurements_file.exists() or not images_dir.exists():
                should_remove = True
                logger.info(f"Removing {episode_id}: missing essential files")
            else:
                # Check if measurements can be loaded using utils function
                measurements = load_json(measurements_file)
                
                if not measurements:
                    should_remove = True
                    logger.info(f"Removing {episode_id}: corrupted measurements")
                elif len(measurements) < 10:
                    should_remove = True
                    logger.info(f"Removing {episode_id}: insufficient data ({len(measurements)} frames)")
            
            if should_remove:
                shutil.rmtree(episode_dir)
                removed_episodes.append(episode_id)
        
        logger.info(f"Cleaned {len(removed_episodes)} incomplete episodes")
        
        # Recreate splits after cleaning
        if removed_episodes:
            logger.info("Recreating data splits after cleaning...")
            self.create_data_splits()
        
        return removed_episodes

def main():
    """Main data processor interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CARLA Data Processor')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--action', type=str, choices=[
        'process', 'analyze', 'validate', 'clean', 'all'
    ], default='all', help='Action to perform')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging using utils function
    setup_logging(getattr(logging, args.log_level))
    
    # Initialize processor
    processor = DataProcessor(args.data_root)
    
    logger.info(f"CARLA Data Processor")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Action: {args.action}")
    logger.info("=" * 50)
    
    try:
        if args.action in ['process', 'all']:
            episodes, frames = processor.process_all_sessions()
            logger.info(f"Processing complete: {episodes} episodes, {frames} frames")
        
        if args.action in ['analyze', 'all']:
            stats = processor.analyze_processed_data()
        
        if args.action in ['validate', 'all']:
            issues = processor.validate_processed_data()
        
        if args.action in ['clean', 'all']:
            removed = processor.clean_processed_data()
        
        logger.info("Data processing pipeline complete!")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise

if __name__ == '__main__':
    main()