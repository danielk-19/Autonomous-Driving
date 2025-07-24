"""
Data Management utilities for CARLA autonomous driving project
Handles dataset creation, augmentation, and validation
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

class DataManager:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.semantic_dir = self.data_root / "semantic"
        self.rgb_dir = self.data_root / "rgb" 
        self.metadata_dir = self.data_root / "metadata"
        
    def analyze_dataset(self):
        """Analyze the collected dataset and generate statistics"""
        print("Analyzing dataset...")
        
        # Count files
        semantic_files = list(self.semantic_dir.glob("*.png"))
        rgb_files = list(self.rgb_dir.glob("*.png")) if self.rgb_dir.exists() else []
        gps_files = list(self.metadata_dir.glob("*_gps.json"))
        steer_files = list(self.metadata_dir.glob("*_steer.json"))
        
        print(f"Dataset Statistics:")
        print(f"  Semantic images: {len(semantic_files)}")
        print(f"  RGB images: {len(rgb_files)}")
        print(f"  GPS files: {len(gps_files)}")
        print(f"  Steering files: {len(steer_files)}")
        
        # Analyze steering distribution
        steering_data = []
        for steer_file in steer_files:
            with open(steer_file, 'r') as f:
                data = json.load(f)
                steering_data.append(data.get('steer', 0))
        
        if steering_data:
            print(f"\nSteering Analysis:")
            print(f"  Mean steering: {np.mean(steering_data):.3f}")
            print(f"  Std steering: {np.std(steering_data):.3f}")
            print(f"  Min steering: {np.min(steering_data):.3f}")
            print(f"  Max steering: {np.max(steering_data):.3f}")
            
            # Plot steering distribution
            self.plot_steering_distribution(steering_data)
        
        # Check for missing data
        self.check_data_integrity()
        
        return {
            'total_frames': len(semantic_files),
            'steering_stats': {
                'mean': np.mean(steering_data) if steering_data else 0,
                'std': np.std(steering_data) if steering_data else 0,
                'distribution': steering_data
            }
        }
    
    def plot_steering_distribution(self, steering_data):
        """Plot steering angle distribution"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(steering_data, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Steering Angle')
        plt.ylabel('Frequency')
        plt.title('Steering Angle Distribution')
        
        plt.subplot(1, 2, 2)
        plt.plot(steering_data[:1000])  # Plot first 1000 samples
        plt.xlabel('Frame')
        plt.ylabel('Steering Angle')
        plt.title('Steering Sequence (First 1000 frames)')
        
        plt.tight_layout()
        plt.savefig(self.data_root / 'steering_analysis.png')
        plt.close()
        print(f"  Saved steering analysis plot to: {self.data_root / 'steering_analysis.png'}")
    
    def check_data_integrity(self):
        """Check for missing or corrupted data files"""
        print("\nChecking data integrity...")
        
        semantic_files = set([f.stem for f in self.semantic_dir.glob("*.png")])
        gps_files = set([f.stem.replace('_gps', '') for f in self.metadata_dir.glob("*_gps.json")])
        steer_files = set([f.stem.replace('_steer', '') for f in self.metadata_dir.glob("*_steer.json")])
        
        # Find missing files
        all_frames = semantic_files | gps_files | steer_files
        missing_semantic = all_frames - semantic_files
        missing_gps = all_frames - gps_files
        missing_steer = all_frames - steer_files
        
        if missing_semantic:
            print(f"  Warning: {len(missing_semantic)} missing semantic images")
        if missing_gps:
            print(f"  Warning: {len(missing_gps)} missing GPS files")
        if missing_steer:
            print(f"  Warning: {len(missing_steer)} missing steering files")
        
        if not (missing_semantic or missing_gps or missing_steer):
            print("  ✓ All data files present")
        
        return {
            'missing_semantic': list(missing_semantic),
            'missing_gps': list(missing_gps),
            'missing_steer': list(missing_steer)
        }
    
    def create_training_split(self, test_size=0.2, val_size=0.1):
        """Create train/validation/test splits"""
        print(f"\nCreating data splits (train/val/test: {1-test_size-val_size:.1f}/{val_size:.1f}/{test_size:.1f})")
        
        # Get all valid frame IDs (frames with all required data)
        semantic_files = set([f.stem for f in self.semantic_dir.glob("*.png")])
        gps_files = set([f.stem.replace('_gps', '') for f in self.metadata_dir.glob("*_gps.json")])
        steer_files = set([f.stem.replace('_steer', '') for f in self.metadata_dir.glob("*_steer.json")])
        
        # Only use frames that have all data types
        valid_frames = list(semantic_files & gps_files & steer_files)
        
        print(f"  Found {len(valid_frames)} valid frames")
        
        # First split: separate test set
        train_val_frames, test_frames = train_test_split(
            valid_frames, 
            test_size=test_size, 
            random_state=42
        )
        
        # Second split: separate validation from training
        train_frames, val_frames = train_test_split(
            train_val_frames, 
            test_size=val_size/(1-test_size), 
            random_state=42
        )
        
        print(f"  Train: {len(train_frames)} frames")
        print(f"  Validation: {len(val_frames)} frames")  
        print(f"  Test: {len(test_frames)} frames")
        
        # Save splits to JSON
        splits = {
            'train': sorted(train_frames),
            'validation': sorted(val_frames),
            'test': sorted(test_frames)
        }
        
        splits_file = self.data_root / 'data_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"  Saved data splits to: {splits_file}")
        return splits
    
    def balance_steering_data(self, steering_threshold=0.05, balance_ratio=0.3):
        """Balance dataset to reduce straight-driving bias"""
        print(f"\nBalancing steering data (threshold: {steering_threshold}, ratio: {balance_ratio})")
        
        # Load steering data
        steer_files = list(self.metadata_dir.glob("*_steer.json"))
        straight_frames = []
        turn_frames = []
        
        for steer_file in steer_files:
            with open(steer_file, 'r') as f:
                data = json.load(f)
                steer_angle = abs(data.get('steer', 0))
                frame_id = steer_file.stem.replace('_steer', '')
                
                if steer_angle < steering_threshold:
                    straight_frames.append(frame_id)
                else:
                    turn_frames.append(frame_id)
        
        print(f"  Straight driving frames: {len(straight_frames)}")
        print(f"  Turning frames: {len(turn_frames)}")
        
        # Balance by randomly sampling straight frames
        if len(straight_frames) > len(turn_frames) * (1/balance_ratio - 1):
            target_straight = int(len(turn_frames) * (1/balance_ratio - 1))
            straight_frames = random.sample(straight_frames, target_straight)
            print(f"  Reduced straight frames to: {len(straight_frames)}")
        
        balanced_frames = straight_frames + turn_frames
        print(f"  Final balanced dataset: {len(balanced_frames)} frames")
        
        # Save balanced frame list
        balanced_file = self.data_root / 'balanced_frames.json'
        with open(balanced_file, 'w') as f:
            json.dump(sorted(balanced_frames), f, indent=2)
        
        return balanced_frames
    
    def augment_data(self, frame_ids, augmentation_factor=2):
        """Create augmented versions of data for training"""
        print(f"\nCreating augmented dataset (factor: {augmentation_factor})")
        
        augmented_dir = self.data_root / 'augmented'
        augmented_dir.mkdir(exist_ok=True)
        
        augmentations = []
        
        for i, frame_id in enumerate(frame_ids[:100]):  # Limit for demo
            if i % 50 == 0:
                print(f"  Processing frame {i+1}/{len(frame_ids[:100])}")
            
            # Load original image
            sem_path = self.semantic_dir / f"{frame_id}.png"
            if not sem_path.exists():
                continue
                
            image = cv2.imread(str(sem_path))
            
            # Load original steering
            steer_path = self.metadata_dir / f"{frame_id}_steer.json"
            with open(steer_path, 'r') as f:
                steer_data = json.load(f)
            
            original_steer = steer_data['steer']
            
            # Apply augmentations
            for aug_idx in range(augmentation_factor):
                aug_image, aug_steer = self.apply_augmentation(image, original_steer)
                
                # Save augmented data
                aug_frame_id = f"{frame_id}_aug_{aug_idx}"
                
                aug_img_path = augmented_dir / f"{aug_frame_id}.png"
                cv2.imwrite(str(aug_img_path), aug_image)
                
                aug_steer_path = augmented_dir / f"{aug_frame_id}_steer.json"
                with open(aug_steer_path, 'w') as f:
                    json.dump({'steer': aug_steer}, f)
                
                augmentations.append(aug_frame_id)
        
        print(f"  Created {len(augmentations)} augmented frames")
        return augmentations
    
    def apply_augmentation(self, image, steering_angle):
        """Apply single augmentation to image and adjust steering accordingly"""
        aug_type = random.choice(['flip', 'brightness', 'blur', 'noise'])
        
        if aug_type == 'flip':
            # Horizontal flip
            aug_image = cv2.flip(image, 1)
            aug_steer = -steering_angle  # Flip steering direction
        
        elif aug_type == 'brightness':
            # Adjust brightness
            factor = random.uniform(0.7, 1.3)
            aug_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            aug_steer = steering_angle  # Steering unchanged
        
        elif aug_type == 'blur':
            # Add slight blur
            kernel_size = random.choice([3, 5])
            aug_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            aug_steer = steering_angle
        
        else:  # noise
            # Add noise
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            aug_image = cv2.add(image, noise)
            aug_steer = steering_angle
        
        return aug_image, aug_steer
    
    def export_for_training(self, splits_file=None, output_format='carla_dataset'):
        """Export data in format suitable for training"""
        print(f"\nExporting data for training (format: {output_format})")
        
        if splits_file is None:
            splits_file = self.data_root / 'data_splits.json'
        
        if not splits_file.exists():
            print("  No splits file found, creating default splits...")
            splits = self.create_training_split()
        else:
            with open(splits_file, 'r') as f:
                splits = json.load(f)
        
        # Create training-ready directory structure
        export_dir = self.data_root / 'training_ready'
        export_dir.mkdir(exist_ok=True)
        
        for split_name, frame_ids in splits.items():
            split_dir = export_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create CSV file with image paths and labels
            data_list = []
            
            for frame_id in frame_ids:
                # Check if all required files exist
                sem_path = self.semantic_dir / f"{frame_id}.png"
                steer_path = self.metadata_dir / f"{frame_id}_steer.json"
                gps_path = self.metadata_dir / f"{frame_id}_gps.json"
                
                if not all([sem_path.exists(), steer_path.exists(), gps_path.exists()]):
                    continue
                
                # Load steering data
                with open(steer_path, 'r') as f:
                    steer_data = json.load(f)
                
                # Load GPS data
                with open(gps_path, 'r') as f:
                    gps_data = json.load(f)
                
                data_list.append({
                    'frame_id': frame_id,
                    'semantic_path': str(sem_path),
                    'steer': steer_data['steer'],
                    'throttle': steer_data.get('throttle', 0),
                    'brake': steer_data.get('brake', 0),
                    'lat': gps_data.get('lat', 0),
                    'lon': gps_data.get('lon', 0)
                })
            
            # Save to CSV
            df = pd.DataFrame(data_list)
            csv_path = split_dir / f'{split_name}_data.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"  {split_name}: {len(data_list)} samples -> {csv_path}")
        
        print(f"  Training data exported to: {export_dir}")
        
        return export_dir
    
    def visualize_samples(self, num_samples=9):
        """Visualize random samples from the dataset"""
        print(f"\nVisualizing {num_samples} random samples...")
        
        semantic_files = list(self.semantic_dir.glob("*.png"))
        sample_files = random.sample(semantic_files, min(num_samples, len(semantic_files)))
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, sem_file in enumerate(sample_files):
            frame_id = sem_file.stem
            
            # Load semantic image
            sem_image = cv2.imread(str(sem_file))
            sem_image = cv2.cvtColor(sem_image, cv2.COLOR_BGR2RGB)
            
            # Load steering data
            steer_file = self.metadata_dir / f"{frame_id}_steer.json"
            steer_angle = 0
            if steer_file.exists():
                with open(steer_file, 'r') as f:
                    steer_data = json.load(f)
                    steer_angle = steer_data.get('steer', 0)
            
            # Display
            axes[i].imshow(sem_image)
            axes[i].set_title(f"Frame {frame_id}\nSteering: {steer_angle:.3f}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(sample_files), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        viz_path = self.data_root / 'sample_visualization.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Sample visualization saved to: {viz_path}")
        return viz_path
    
    def clean_dataset(self, min_file_size=1000):
        """Clean dataset by removing corrupted or invalid files"""
        print(f"\nCleaning dataset (min file size: {min_file_size} bytes)...")
        
        removed_files = []
        
        # Check semantic images
        for sem_file in self.semantic_dir.glob("*.png"):
            if sem_file.stat().st_size < min_file_size:
                print(f"  Removing corrupted semantic image: {sem_file.name}")
                sem_file.unlink()
                removed_files.append(str(sem_file))
                
                # Remove associated metadata files
                frame_id = sem_file.stem
                gps_file = self.metadata_dir / f"{frame_id}_gps.json"
                steer_file = self.metadata_dir / f"{frame_id}_steer.json"
                
                if gps_file.exists():
                    gps_file.unlink()
                    removed_files.append(str(gps_file))
                    
                if steer_file.exists():
                    steer_file.unlink()
                    removed_files.append(str(steer_file))
        
        # Check for orphaned metadata files
        semantic_frames = set([f.stem for f in self.semantic_dir.glob("*.png")])
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            if metadata_file.name.endswith('_gps.json'):
                frame_id = metadata_file.stem.replace('_gps', '')
            elif metadata_file.name.endswith('_steer.json'):
                frame_id = metadata_file.stem.replace('_steer', '')
            else:
                continue
            
            if frame_id not in semantic_frames:
                print(f"  Removing orphaned metadata: {metadata_file.name}")
                metadata_file.unlink()
                removed_files.append(str(metadata_file))
        
        print(f"  Removed {len(removed_files)} corrupted/orphaned files")
        return removed_files

def main():
    """Main data management interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CARLA Dataset Management')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--action', type=str, choices=[
        'analyze', 'clean', 'split', 'balance', 'augment', 'export', 'visualize', 'all'
    ], default='all', help='Action to perform')
    
    args = parser.parse_args()
    
    # Initialize data manager
    dm = DataManager(args.data_root)
    
    print(f"CARLA Dataset Manager")
    print(f"Data root: {args.data_root}")
    print(f"Action: {args.action}")
    print("="*50)
    
    if args.action in ['analyze', 'all']:
        stats = dm.analyze_dataset()
    
    if args.action in ['clean', 'all']:
        dm.clean_dataset()
    
    if args.action in ['split', 'all']:
        splits = dm.create_training_split()
    
    if args.action in ['balance', 'all']:
        balanced = dm.balance_steering_data()
    
    if args.action in ['augment', 'all']:
        # Only augment a subset for demo
        sample_frames = list(dm.semantic_dir.glob("*.png"))[:50]
        frame_ids = [f.stem for f in sample_frames]
        dm.augment_data(frame_ids, augmentation_factor=2)
    
    if args.action in ['visualize', 'all']:
        dm.visualize_samples()
    
    if args.action in ['export', 'all']:
        dm.export_for_training()
    
    print("\nDataset management complete!")

if __name__ == '__main__':
    main()