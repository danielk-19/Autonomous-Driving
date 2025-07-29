"""
CARLA Dataset for Behavioral Cloning
Handles loading images and corresponding control commands from processed episodes.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path
import sys

# Add project root to path and import utils
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from utils.utils import (
    setup_logging, load_json, save_json, ensure_dir, get_timestamp,
    safe_float, safe_int, load_image, preprocess_image, 
    calculate_steering_distribution, balance_steering_samples,
    validate_measurements_format, Timer, PerformanceMonitor
)

# Setup logging
logger = logging.getLogger(__name__)


class CARLADataset(Dataset):
    """
    CARLA Dataset for behavioral cloning
    Handles loading images and corresponding control commands from processed episodes
    
    Expected data structure:
    data/processed/episodes/episode_XXX/images/XXXXXX.png
    data/processed/episodes/episode_XXX/measurements.json
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 balance_steering: bool = True,
                 steering_threshold: float = 0.1,
                 max_samples: Optional[int] = None):
        """
        Initialize CARLA dataset
        
        Args:
            data_dir: Directory containing processed data (data/processed/)
            split: 'train', 'val', or 'test'
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            balance_steering: Whether to balance steering distribution
            steering_threshold: Threshold for considering steering as straight
            max_samples: Maximum number of samples to load (for testing)
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = (safe_int(image_size[0], 224), safe_int(image_size[1], 224))
        self.augment = augment and split == 'train'
        self.balance_steering = balance_steering
        self.steering_threshold = safe_float(steering_threshold, 0.1)
        
        logger.info(f"Initializing CARLADataset for {split} split from {data_dir}")
        
        # Load dataset samples using utils
        with Timer(f"Loading {split} samples"):
            self.samples = self._load_samples()
        
        if not self.samples:
            logger.error(f"No samples found for {split} split!")
            raise ValueError(f"No samples found for {split} split in {data_dir}")
        
        # Balance steering distribution for training
        if balance_steering and split == 'train':
            logger.info("Balancing steering distribution...")
            self.samples = balance_steering_samples(
                self.samples, 
                threshold=self.steering_threshold,
                balance_ratio=0.3
            )
        
        # Limit samples if requested
        if max_samples and max_samples < len(self.samples):
            logger.info(f"Limiting dataset to {max_samples} samples")
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Setup transforms
        self.transforms = self._get_transforms()
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from split files or create from episodes"""
        samples = []
        
        # First, try to load from existing split files (preferred method)
        split_file = os.path.join(self.data_dir, f'{self.split}_samples.json')
        
        if os.path.exists(split_file):
            logger.info(f"Loading samples from existing split file: {split_file}")
            samples = load_json(split_file)
            if samples is None:
                logger.warning(f"Failed to load split file {split_file}, scanning episodes...")
                samples = []
            else:
                # Validate sample paths exist
                valid_samples = []
                for sample in samples:
                    if os.path.exists(sample.get('image_path', '')):
                        valid_samples.append(sample)
                    else:
                        logger.warning(f"Image not found: {sample.get('image_path', 'unknown')}")
                samples = valid_samples
        
        # If no split file or empty, scan episodes directory structure
        if not samples:
            logger.info("Scanning episodes directory structure...")
            samples = self._scan_episodes()
        
        return samples
    
    def _scan_episodes(self) -> List[Dict]:
        """
        Scan processed episodes directory structure:
        data/processed/episodes/episode_XXX/images/XXXXXX.png
        data/processed/episodes/episode_XXX/measurements.json
        """
        samples = []
        episodes_dir = os.path.join(self.data_dir, 'episodes')
        
        if not os.path.exists(episodes_dir):
            logger.error(f"Episodes directory not found: {episodes_dir}")
            logger.info("Expected structure: data/processed/episodes/episode_XXX/")
            return []
        
        episode_dirs = sorted([d for d in os.listdir(episodes_dir) 
                              if d.startswith('episode_') and 
                              os.path.isdir(os.path.join(episodes_dir, d))])
        
        if not episode_dirs:
            logger.warning(f"No episode directories found in {episodes_dir}")
            return []
        
        logger.info(f"Found {len(episode_dirs)} episodes")
        
        for episode_dir in episode_dirs:
            episode_path = os.path.join(episodes_dir, episode_dir)
            images_dir = os.path.join(episode_path, 'images')
            measurements_file = os.path.join(episode_path, 'measurements.json')
            
            # Check required structure
            if not os.path.exists(images_dir):
                logger.warning(f"Images directory not found: {images_dir}")
                continue
                
            if not os.path.exists(measurements_file):
                logger.warning(f"Measurements file not found: {measurements_file}")
                continue
            
            # Load measurements using utils
            measurements = load_json(measurements_file)
            if measurements is None:
                logger.warning(f"Failed to load measurements from {measurements_file}")
                continue
            
            # Validate measurements format
            validation_errors = validate_measurements_format(measurements)
            if validation_errors:
                logger.warning(f"Measurements validation errors in {episode_dir}: {validation_errors[:3]}...")  # Show first 3 errors
            
            # Process each frame
            for measurement in measurements:
                frame_id = safe_int(measurement.get('frame_id', 0))
                image_path = os.path.join(images_dir, f'{frame_id:06d}.png')
                
                if os.path.exists(image_path):
                    sample = {
                        'image_path': image_path,
                        'steering': safe_float(measurement.get('steering', 0.0)),
                        'throttle': safe_float(measurement.get('throttle', 0.0)),
                        'brake': safe_float(measurement.get('brake', 0.0)),
                        'speed': safe_float(measurement.get('speed', 0.0)),
                        'episode': episode_dir,
                        'frame_id': frame_id,
                        'timestamp': measurement.get('timestamp', ''),
                        'gps': measurement.get('gps', {}),
                        'imu': measurement.get('imu', {})
                    }
                    samples.append(sample)
                else:
                    logger.debug(f"Image not found: {image_path}")
        
        logger.info(f"Scanned {len(samples)} samples from episodes")
        return samples
    
    def _get_transforms(self):
        """Get data augmentation transforms using albumentations"""
        if self.augment:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, 
                        scale_limit=0.1, 
                        rotate_limit=5, 
                        p=0.3
                    ),
                ], p=0.5),
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.2, 
                        contrast=0.2, 
                        saturation=0.2, 
                        hue=0.1, 
                        p=0.8
                    ),
                    A.GaussNoise(var_limit=10.0, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ], p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample
        
        Returns:
            Tuple of (image_tensor, control_dict)
        """
        self.perf_monitor.log_frame()
        
        sample = self.samples[idx]
        
        # Load image using utils
        image = load_image(sample['image_path'])
        if image is None:
            # Fallback to PIL
            try:
                image = np.array(Image.open(sample['image_path']).convert('RGB'))
            except Exception as e:
                logger.error(f"Failed to load image {sample['image_path']}: {e}")
                # Return black image as fallback
                image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB (load_image returns BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original steering for flip correction
        original_steering = sample['steering']
        
        # Apply transforms
        try:
            transformed = self.transforms(image=image)
            image_tensor = transformed['image']
        except Exception as e:
            logger.error(f"Transform failed for image {sample['image_path']}: {e}")
            # Create fallback tensor
            image_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
        
        # Handle horizontal flip for steering (albumentations doesn't provide easy access to applied transforms)
        # For now, we'll use original steering. In practice, you might want to implement
        # a custom transform that tracks flips.
        steering = original_steering
        
        # Prepare control targets with proper clamping using utils
        controls = {
            'steering': torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32).unsqueeze(0),
            'throttle': torch.tensor(np.clip(sample['throttle'], 0.0, 1.0), dtype=torch.float32).unsqueeze(0),
            'brake': torch.tensor(np.clip(sample['brake'], 0.0, 1.0), dtype=torch.float32).unsqueeze(0)
        }
        
        return image_tensor, controls
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics using utils"""
        if not self.samples:
            return {}
        
        # Use utils function for steering distribution
        steering_stats = calculate_steering_distribution(self.samples)
        
        # Calculate other statistics
        throttle_values = [s['throttle'] for s in self.samples]
        brake_values = [s['brake'] for s in self.samples]
        speed_values = [s['speed'] for s in self.samples]
        
        stats = {
            'total_samples': len(self.samples),
            'episodes': len(set(s['episode'] for s in self.samples)),
            'steering': steering_stats,
            'throttle': {
                'mean': np.mean(throttle_values),
                'std': np.std(throttle_values),
                'min': np.min(throttle_values),
                'max': np.max(throttle_values)
            },
            'brake': {
                'mean': np.mean(brake_values),
                'std': np.std(brake_values),
                'brake_ratio': np.mean(np.array(brake_values) > 0.1)
            },
            'speed': {
                'mean': np.mean(speed_values),
                'std': np.std(speed_values),
                'min': np.min(speed_values),
                'max': np.max(speed_values)
            },
            'performance': self.perf_monitor.get_stats()
        }
        
        return stats


class CARLADataModule:
    """
    Data module for handling train/val/test splits and data loaders
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: Tuple[int, int] = (224, 224),
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 balance_steering: bool = True):
        """
        Initialize data module
        
        Args:
            data_dir: Directory containing processed data (data/processed/)
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            image_size: Target image size
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            balance_steering: Whether to balance steering distribution
        """
        self.data_dir = data_dir
        self.batch_size = safe_int(batch_size, 32)
        self.num_workers = safe_int(num_workers, 4)
        self.image_size = (safe_int(image_size[0], 224), safe_int(image_size[1], 224))
        self.val_split = safe_float(val_split, 0.15)
        self.test_split = safe_float(test_split, 0.15)
        self.balance_steering = balance_steering
        
        logger.info(f"Initializing CARLADataModule with data_dir: {data_dir}")
        
        # Ensure data directory structure exists
        ensure_dir(os.path.join(data_dir, 'episodes'))
        
        # Create splits if they don't exist
        with Timer("Creating/loading data splits"):
            self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits"""
        splits_file = os.path.join(self.data_dir, 'data_splits.json')
        
        # Check if splits already exist and are valid
        if os.path.exists(splits_file):
            logger.info("Checking existing data splits...")
            splits_data = load_json(splits_file)
            
            # Validate existing splits
            if (splits_data and 
                all(key in splits_data for key in ['train_samples', 'val_samples', 'test_samples']) and
                all(os.path.exists(os.path.join(self.data_dir, f'{split}_samples.json')) 
                    for split in ['train', 'val', 'test'])):
                logger.info("Using existing valid data splits")
                return
            else:
                logger.warning("Existing splits are invalid, creating new ones...")
        
        logger.info("Creating new data splits...")
        
        # Scan all episodes to create splits
        episodes_dir = os.path.join(self.data_dir, 'episodes')
        if not os.path.exists(episodes_dir):
            raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
        
        # Get all episode directories
        episode_dirs = sorted([d for d in os.listdir(episodes_dir) 
                              if d.startswith('episode_') and 
                              os.path.isdir(os.path.join(episodes_dir, d))])
        
        if not episode_dirs:
            raise ValueError(f"No episode directories found in {episodes_dir}")
        
        logger.info(f"Found {len(episode_dirs)} episodes for splitting")
        
        # Split episodes (not individual samples) to avoid data leakage
        train_episodes, temp_episodes = train_test_split(
            episode_dirs, 
            test_size=(self.val_split + self.test_split), 
            random_state=42
        )
        
        val_episodes, test_episodes = train_test_split(
            temp_episodes, 
            test_size=(self.test_split / (self.val_split + self.test_split)), 
            random_state=42
        )
        
        logger.info(f"Episode splits: {len(train_episodes)} train, {len(val_episodes)} val, {len(test_episodes)} test")
        
        # Create sample lists for each split
        train_samples = self._collect_episode_samples(train_episodes)
        val_samples = self._collect_episode_samples(val_episodes)
        test_samples = self._collect_episode_samples(test_episodes)
        
        logger.info(f"Sample counts: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        # Save individual split files using utils
        save_json(train_samples, os.path.join(self.data_dir, 'train_samples.json'))
        save_json(val_samples, os.path.join(self.data_dir, 'val_samples.json'))
        save_json(test_samples, os.path.join(self.data_dir, 'test_samples.json'))
        
        # Save combined splits info
        splits_info = {
            'train_episodes': train_episodes,
            'val_episodes': val_episodes,
            'test_episodes': test_episodes,
            'train_samples_count': len(train_samples),
            'val_samples_count': len(val_samples),
            'test_samples_count': len(test_samples),
            'created_at': get_timestamp(),
            'splits_config': {
                'val_split': self.val_split,
                'test_split': self.test_split,
                'balance_steering': self.balance_steering
            }
        }
        
        save_json(splits_info, splits_file)
        logger.info("Data splits created and saved successfully")
    
    def _collect_episode_samples(self, episode_list: List[str]) -> List[Dict]:
        """Collect samples from a list of episodes"""
        samples = []
        episodes_dir = os.path.join(self.data_dir, 'episodes')
        
        for episode_dir in episode_list:
            episode_path = os.path.join(episodes_dir, episode_dir)
            images_dir = os.path.join(episode_path, 'images')
            measurements_file = os.path.join(episode_path, 'measurements.json')
            
            if not os.path.exists(measurements_file):
                logger.warning(f"Measurements file not found: {measurements_file}")
                continue
            
            measurements = load_json(measurements_file)
            if measurements is None:
                logger.warning(f"Failed to load measurements from {measurements_file}")
                continue
            
            for measurement in measurements:
                frame_id = safe_int(measurement.get('frame_id', 0))
                image_path = os.path.join(images_dir, f'{frame_id:06d}.png')
                
                if os.path.exists(image_path):
                    sample = {
                        'image_path': image_path,
                        'steering': safe_float(measurement.get('steering', 0.0)),
                        'throttle': safe_float(measurement.get('throttle', 0.0)),
                        'brake': safe_float(measurement.get('brake', 0.0)),
                        'speed': safe_float(measurement.get('speed', 0.0)),
                        'episode': episode_dir,
                        'frame_id': frame_id,
                        'timestamp': measurement.get('timestamp', ''),
                        'gps': measurement.get('gps', {}),
                        'imu': measurement.get('imu', {})
                    }
                    samples.append(sample)
        
        return samples
    
    def get_datasets(self) -> Tuple[CARLADataset, CARLADataset, CARLADataset]:
        """Get train, validation, and test datasets"""
        train_dataset = CARLADataset(
            self.data_dir, 
            split='train',
            image_size=self.image_size,
            augment=True,
            balance_steering=self.balance_steering
        )
        
        val_dataset = CARLADataset(
            self.data_dir,
            split='val',
            image_size=self.image_size,
            augment=False,
            balance_steering=False
        )
        
        test_dataset = CARLADataset(
            self.data_dir,
            split='test',
            image_size=self.image_size,
            augment=False,
            balance_steering=False
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders"""
        train_dataset, val_dataset, test_dataset = self.get_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self, split: str = 'train') -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample batch for testing"""
        dataset = CARLADataset(
            self.data_dir, 
            split=split, 
            image_size=self.image_size,
            augment=(split == 'train'),
            balance_steering=False  # Don't balance for sample batch
        )
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        return next(iter(loader))
    
    def get_data_info(self) -> Dict:
        """Get comprehensive data information"""
        try:
            train_dataset, val_dataset, test_dataset = self.get_datasets()
            
            return {
                'data_dir': self.data_dir,
                'image_size': self.image_size,
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'splits': {
                    'train': len(train_dataset),
                    'val': len(val_dataset),
                    'test': len(test_dataset),
                    'total': len(train_dataset) + len(val_dataset) + len(test_dataset)
                },
                'train_stats': train_dataset.get_statistics(),
                'val_stats': val_dataset.get_statistics(),
                'test_stats': test_dataset.get_statistics(),
                'config': {
                    'val_split': self.val_split,
                    'test_split': self.test_split,
                    'balance_steering': self.balance_steering
                }
            }
        except Exception as e:
            logger.error(f"Failed to get data info: {e}")
            return {'error': str(e)}


def validate_data_structure(data_dir: str) -> Dict[str, Any]:
    """
    Validate that data directory follows the correct structure
    
    Args:
        data_dir: Path to data/processed/ directory
        
    Returns:
        Validation results
    """
    logger.info(f"Validating data structure in {data_dir}")
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'structure_found': {},
        'episodes_found': 0,
        'total_samples': 0
    }
    
    # Check main structure
    episodes_dir = os.path.join(data_dir, 'episodes')
    if not os.path.exists(episodes_dir):
        results['errors'].append(f"Episodes directory not found: {episodes_dir}")
        results['valid'] = False
        return results
    
    results['structure_found']['episodes_dir'] = True
    
    # Check episode directories
    episode_dirs = [d for d in os.listdir(episodes_dir) 
                   if d.startswith('episode_') and 
                   os.path.isdir(os.path.join(episodes_dir, d))]
    
    if not episode_dirs:
        results['errors'].append("No episode directories found")
        results['valid'] = False
        return results
    
    results['episodes_found'] = len(episode_dirs)
    
    # Validate episode structure
    sample_count = 0
    for episode_dir in episode_dirs[:5]:  # Check first 5 episodes
        episode_path = os.path.join(episodes_dir, episode_dir)
        images_dir = os.path.join(episode_path, 'images')
        measurements_file = os.path.join(episode_path, 'measurements.json')
        
        if not os.path.exists(images_dir):
            results['errors'].append(f"Images directory missing in {episode_dir}")
            results['valid'] = False
        
        if not os.path.exists(measurements_file):
            results['errors'].append(f"Measurements file missing in {episode_dir}")
            results['valid'] = False
        else:
            # Validate measurements format
            measurements = load_json(measurements_file)
            if measurements:
                validation_errors = validate_measurements_format(measurements)
                if validation_errors:
                    results['warnings'].extend(validation_errors[:3])  # First 3 errors
                sample_count += len(measurements)
    
    results['total_samples'] = sample_count * (len(episode_dirs) / min(5, len(episode_dirs)))  # Estimate
    
    # Check split files
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(data_dir, f'{split}_samples.json')
        results['structure_found'][f'{split}_samples'] = os.path.exists(split_file)
    
    logger.info(f"Data structure validation complete. Valid: {results['valid']}")
    return results


# Utility functions for easy testing
def test_dataset_loading(data_dir: str, split: str = 'train', max_samples: int = 10):
    """Test dataset loading with small sample"""
    logger.info(f"Testing dataset loading from {data_dir}")
    
    try:
        dataset = CARLADataset(
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            balance_steering=False
        )
        
        logger.info(f"Successfully loaded {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            image, controls = dataset[0]
            logger.info(f"Sample image shape: {image.shape}")
            logger.info(f"Sample controls: {controls}")
            
            # Test statistics
            stats = dataset.get_statistics()
            logger.info(f"Dataset statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset loading test failed: {e}")
        return False


def create_test_datamodule(data_dir: str, batch_size: int = 4):
    """Create a test data module with small batch size"""
    logger.info("Creating test data module")
    
    try:
        data_module = CARLADataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=0,  # No multiprocessing for testing
            val_split=0.2,
            test_split=0.2
        )
        
        # Test getting dataloaders
        train_loader, val_loader, test_loader = data_module.get_dataloaders()
        
        logger.info(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Test getting a batch
        sample_batch = data_module.get_sample_batch()
        images, controls = sample_batch
        logger.info(f"Sample batch - Images: {images.shape}, Controls keys: {list(controls.keys())}")
        
        return data_module
        
    except Exception as e:
        logger.error(f"Test data module creation failed: {e}")
        return None


if __name__ == "__main__":
    # Basic testing
    setup_logging()
    
    # Test with sample data directory
    test_data_dir = "data/processed"
    
    if os.path.exists(test_data_dir):
        logger.info("Running basic dataset tests...")
        
        # Validate structure
        validation_results = validate_data_structure(test_data_dir)
        logger.info(f"Structure validation: {validation_results}")
        
        if validation_results['valid']:
            # Test dataset loading
            success = test_dataset_loading(test_data_dir, max_samples=5)
            if success:
                logger.info("Dataset loading test passed")
                
                # Test data module
                data_module = create_test_datamodule(test_data_dir, batch_size=2)
                if data_module:
                    logger.info("Data module test passed")
                    
                    # Print data info
                    info = data_module.get_data_info()
                    logger.info(f"Data module info: {info}")
                else:
                    logger.error("Data module test failed")
            else:
                logger.error("Dataset loading test failed")
        else:
            logger.error("Data structure validation failed")
            logger.error(f"Errors: {validation_results['errors']}")
    else:
        logger.warning(f"Test data directory not found: {test_data_dir}")
        logger.info("To test the dataset:")
        logger.info("1. Ensure you have processed data in data/processed/episodes/")
        logger.info("2. Each episode should have images/ directory and measurements.json")
        logger.info("3. Run this script again")