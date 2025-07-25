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

import json
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        self.processed_dir.mkdir(exist_ok=True)
        self.episodes_dir.mkdir(exist_ok=True)
        
        logger.info(f"DataProcessor initialized with data root: {self.data_root}")
    
    def process_all_sessions(self):
        """Process all raw sessions into episodes"""
        logger.info("Starting processing of all raw sessions...")
        
        session_dirs = [d for d in self.raw_sessions_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('session_')]
        
        if not session_dirs:
            logger.warning(f"No session directories found in {self.raw_sessions_dir}")
            return
        
        logger.info(f"Found {len(session_dirs)} sessions to process")
        
        episode_counter = 1
        total_frames = 0
        
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
        episode_dir.mkdir(exist_ok=True)
        images_dir = episode_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
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
                # Validate that all required data exists for this frame
                if not self.validate_frame_data(session_dir, frame_id):
                    logger.debug(f"Skipping frame {frame_id} - missing data")
                    continue
                
                # Combine RGB and semantic images
                combined_image = self.combine_rgb_semantic(session_dir, frame_id)
                if combined_image is None:
                    logger.debug(f"Failed to combine images for frame {frame_id}")
                    continue
                
                # Save combined image
                output_image_path = images_dir / f"{frame_id}.png"
                cv2.imwrite(str(output_image_path), combined_image)
                
                # Load and consolidate metadata
                frame_measurements = self.load_frame_measurements(session_dir, frame_id)
                if frame_measurements is None:
                    logger.debug(f"Failed to load measurements for frame {frame_id}")
                    continue
                
                measurements.append(frame_measurements)
                valid_frames += 1
                
                if valid_frames % 100 == 0:
                    logger.info(f"Processed {valid_frames} frames...")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_id}: {e}")
                continue
        
        # Save measurements.json
        measurements_file = episode_dir / "measurements.json"
        with open(measurements_file, 'w') as f:
            json.dump(measurements, f, indent=2)
        
        logger.info(f"Episode {episode_id} complete: {valid_frames} valid frames")
        return valid_frames
    
    def validate_frame_data(self, session_dir, frame_id):
        """Validate that all required data files exist for a frame"""
        required_files = [
            session_dir / "rgb" / f"{frame_id}.png",
            session_dir / "semantic" / f"{frame_id}.png",
            session_dir / "gps" / f"{frame_id}.json",
            session_dir / "imu" / f"{frame_id}.json",
            session_dir / "control" / f"{frame_id}.json"
        ]
        
        return all(f.exists() for f in required_files)
    
    def combine_rgb_semantic(self, session_dir, frame_id):
        """Combine RGB and semantic images for training"""
        rgb_path = session_dir / "rgb" / f"{frame_id}.png"
        semantic_path = session_dir / "semantic" / f"{frame_id}.png"
        
        try:
            # Load images
            rgb_img = cv2.imread(str(rgb_path))
            semantic_img = cv2.imread(str(semantic_path))
            
            if rgb_img is None or semantic_img is None:
                return None
            
            # Resize to standard training size (224x224)
            target_size = (224, 224)
            rgb_resized = cv2.resize(rgb_img, target_size)
            semantic_resized = cv2.resize(semantic_img, target_size)
            
            # Combine: RGB as main + semantic as overlay with transparency
            # Convert semantic to single channel for overlay
            semantic_gray = cv2.cvtColor(semantic_resized, cv2.COLOR_BGR2GRAY)
            semantic_colored = cv2.applyColorMap(semantic_gray, cv2.COLORMAP_JET)
            
            # Blend images (70% RGB, 30% semantic overlay)
            combined = cv2.addWeighted(rgb_resized, 0.7, semantic_colored, 0.3, 0)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining images for frame {frame_id}: {e}")
            return None
    
    def load_frame_measurements(self, session_dir, frame_id):
        """Load and consolidate all measurements for a single frame"""
        try:
            # Load control data
            control_path = session_dir / "control" / f"{frame_id}.json"
            with open(control_path, 'r') as f:
                control_data = json.load(f)
            
            # Load GPS data
            gps_path = session_dir / "gps" / f"{frame_id}.json"
            with open(gps_path, 'r') as f:
                gps_data = json.load(f)
            
            # Load IMU data
            imu_path = session_dir / "imu" / f"{frame_id}.json"
            with open(imu_path, 'r') as f:
                imu_data = json.load(f)
            
            # Create consolidated measurement
            measurement = {
                "frame_id": int(frame_id),
                "timestamp": control_data.get("timestamp", datetime.now().isoformat()),
                "steering": float(control_data.get("steer", 0.0)),
                "throttle": float(control_data.get("throttle", 0.0)),
                "brake": float(control_data.get("brake", 0.0)),
                "speed": float(control_data.get("speed", 0.0)),
                "gps": {
                    "lat": float(gps_data.get("lat", 0.0)),
                    "lon": float(gps_data.get("lon", 0.0)),
                    "alt": float(gps_data.get("alt", 0.0))
                },
                "imu": {
                    "accel_x": float(imu_data.get("accel_x", 0.0)),
                    "accel_y": float(imu_data.get("accel_y", 0.0)),
                    "accel_z": float(imu_data.get("accel_z", 9.8)),
                    "gyro_x": float(imu_data.get("gyro_x", 0.0)),
                    "gyro_y": float(imu_data.get("gyro_y", 0.0)),
                    "gyro_z": float(imu_data.get("gyro_z", 0.0)),
                    "compass": float(imu_data.get("compass", 0.0))
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
            return
        
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
                if measurements_file.exists():
                    with open(measurements_file, 'r') as f:
                        measurements = json.load(f)
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
        train_sample_list = self.create_sample_list(train_episodes)
        val_sample_list = self.create_sample_list(val_episodes)
        test_sample_list = self.create_sample_list(test_episodes)
        
        # Save splits
        splits_data = {
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "test_episodes": test_episodes,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples
        }
        
        # Save main splits file
        with open(self.processed_dir / "data_splits.json", 'w') as f:
            json.dump(splits_data, f, indent=2)
        
        # Save individual sample lists
        with open(self.processed_dir / "train_samples.json", 'w') as f:
            json.dump(train_sample_list, f, indent=2)
        
        with open(self.processed_dir / "val_samples.json", 'w') as f:
            json.dump(val_sample_list, f, indent=2)
        
        with open(self.processed_dir / "test_samples.json", 'w') as f:
            json.dump(test_sample_list, f, indent=2)
        
        logger.info("Data splits saved successfully")
        return splits_data
    
    def create_sample_list(self, episode_list):
        """Create list of individual samples for a split"""
        samples = []
        
        for episode_id in episode_list:
            measurements_file = self.episodes_dir / episode_id / "measurements.json"
            if measurements_file.exists():
                with open(measurements_file, 'r') as f:
                    measurements = json.load(f)
                    
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
            if measurements_file.exists():
                with open(measurements_file, 'r') as f:
                    measurements = json.load(f)
                    total_frames += len(measurements)
                    
                    for m in measurements:
                        steering_data.append(m["steering"])
                        speed_data.append(m["speed"])
        
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Episodes: {len(episode_dirs)}")
        logger.info(f"  Total frames: {total_frames}")
        
        if steering_data:
            logger.info(f"  Steering - Mean: {np.mean(steering_data):.3f}, "
                       f"Std: {np.std(steering_data):.3f}, "
                       f"Range: [{np.min(steering_data):.3f}, {np.max(steering_data):.3f}]")
            
            logger.info(f"  Speed - Mean: {np.mean(speed_data):.3f}, "
                       f"Std: {np.std(speed_data):.3f}, "
                       f"Range: [{np.min(speed_data):.3f}, {np.max(speed_data):.3f}]")
            
            # Plot distributions
            self.plot_data_distributions(steering_data, speed_data)
        
        return {
            'episodes': len(episode_dirs),
            'total_frames': total_frames,
            'steering_stats': {
                'mean': np.mean(steering_data) if steering_data else 0,
                'std': np.std(steering_data) if steering_data else 0,
                'min': np.min(steering_data) if steering_data else 0,
                'max': np.max(steering_data) if steering_data else 0
            },
            'speed_stats': {
                'mean': np.mean(speed_data) if speed_data else 0,
                'std': np.std(speed_data) if speed_data else 0,
                'min': np.min(speed_data) if speed_data else 0,
                'max': np.max(speed_data) if speed_data else 0
            }
        }
    
    def plot_data_distributions(self, steering_data, speed_data):
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
        """Validate processed data integrity"""
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
            
            # Load measurements
            try:
                with open(measurements_file, 'r') as f:
                    measurements = json.load(f)
            except Exception as e:
                issues.append(f"Corrupted measurements.json in {episode_id}: {e}")
                continue
            
            # Check image-measurement alignment
            image_files = list(images_dir.glob("*.png"))
            
            if len(image_files) != len(measurements):
                issues.append(f"Image count mismatch in {episode_id}: "
                            f"{len(image_files)} images vs {len(measurements)} measurements")
            
            # Check for missing images
            for measurement in measurements:
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
                # Check if measurements can be loaded
                try:
                    with open(measurements_file, 'r') as f:
                        measurements = json.load(f)
                    
                    # Check if episode has sufficient data (at least 10 frames)
                    if len(measurements) < 10:
                        should_remove = True
                        logger.info(f"Removing {episode_id}: insufficient data ({len(measurements)} frames)")
                    
                except Exception as e:
                    should_remove = True
                    logger.info(f"Removing {episode_id}: corrupted measurements ({e})")
            
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
    
    args = parser.parse_args()
    
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