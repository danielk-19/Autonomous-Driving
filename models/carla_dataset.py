import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CARLADataset(Dataset):
    """
    CARLA Dataset for behavioral cloning
    Handles loading images and corresponding control commands
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
            data_dir: Directory containing the dataset
            split: 'train', 'val', or 'test'
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            balance_steering: Whether to balance steering distribution
            steering_threshold: Threshold for considering steering as straight
            max_samples: Maximum number of samples to load (for testing)
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == 'train'
        self.balance_steering = balance_steering
        self.steering_threshold = steering_threshold
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        if balance_steering and split == 'train':
            self.samples = self._balance_steering_distribution()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Setup transforms
        self.transforms = self._get_transforms()
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from JSON files"""
        samples = []
        
        # Look for processed data files
        split_file = os.path.join(self.data_dir, f'{self.split}_samples.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                samples = json.load(f)
        else:
            # Fallback: scan directory structure
            samples = self._scan_directory()
        
        return samples
    
    def _scan_directory(self) -> List[Dict]:
        """Scan directory structure to find samples"""
        samples = []
        
        # Expected structure: data_dir/episode_*/frame_*.png and measurements.json
        for episode_dir in sorted(os.listdir(self.data_dir)):
            episode_path = os.path.join(self.data_dir, episode_dir)
            if not os.path.isdir(episode_path):
                continue
            
            measurements_file = os.path.join(episode_path, 'measurements.json')
            if not os.path.exists(measurements_file):
                continue
            
            # Load measurements
            with open(measurements_file, 'r') as f:
                measurements = json.load(f)
            
            # Match images with measurements
            for frame_data in measurements:
                frame_id = frame_data.get('frame_id', 0)
                image_path = os.path.join(episode_path, f'frame_{frame_id:06d}.png')
                
                if os.path.exists(image_path):
                    sample = {
                        'image_path': image_path,
                        'steering': float(frame_data.get('steering', 0.0)),
                        'throttle': float(frame_data.get('throttle', 0.0)),
                        'brake': float(frame_data.get('brake', 0.0)),
                        'speed': float(frame_data.get('speed', 0.0)),
                        'episode': episode_dir
                    }
                    samples.append(sample)
        
        return samples
    
    def _balance_steering_distribution(self) -> List[Dict]:
        """Balance steering distribution to reduce straight-driving bias"""
        samples = self.samples.copy()
        
        # Separate samples by steering magnitude
        straight_samples = []
        turn_samples = []
        
        for sample in samples:
            if abs(sample['steering']) < self.steering_threshold:
                straight_samples.append(sample)
            else:
                turn_samples.append(sample)
        
        print(f"Before balancing: {len(straight_samples)} straight, {len(turn_samples)} turning")
        
        # Keep all turning samples, subsample straight samples
        if len(straight_samples) > len(turn_samples) * 2:
            np.random.seed(42)  # Reproducible sampling
            straight_samples = np.random.choice(
                straight_samples, 
                size=len(turn_samples) * 2, 
                replace=False
            ).tolist()
        
        balanced_samples = straight_samples + turn_samples
        np.random.shuffle(balanced_samples)
        
        print(f"After balancing: {len(balanced_samples)} total samples")
        
        return balanced_samples
    
    def _get_transforms(self):
        """Get data augmentation transforms"""
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
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            # Fallback to PIL if opencv fails
            image = np.array(Image.open(sample['image_path']).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transforms(image=image)
        image_tensor = transformed['image']
        
        # Handle horizontal flip augmentation for steering
        steering = sample['steering']
        if self.augment and hasattr(transformed, 'replay') and 'HorizontalFlip' in str(transformed.replay):
            # Check if horizontal flip was applied
            for transform_info in transformed.replay.replay:
                if 'HorizontalFlip' in str(transform_info):
                    steering = -steering  # Flip steering for horizontal flip
                    break
        
        # Prepare control targets
        controls = {
            'steering': torch.tensor(steering, dtype=torch.float32).unsqueeze(0),
            'throttle': torch.tensor(sample['throttle'], dtype=torch.float32).unsqueeze(0),
            'brake': torch.tensor(sample['brake'], dtype=torch.float32).unsqueeze(0)
        }
        
        return image_tensor, controls
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            return {}
        
        steering_values = [s['steering'] for s in self.samples]
        throttle_values = [s['throttle'] for s in self.samples]
        brake_values = [s['brake'] for s in self.samples]
        speed_values = [s['speed'] for s in self.samples]
        
        stats = {
            'total_samples': len(self.samples),
            'steering': {
                'mean': np.mean(steering_values),
                'std': np.std(steering_values),
                'min': np.min(steering_values),
                'max': np.max(steering_values),
                'straight_driving_ratio': np.mean(np.abs(steering_values) < self.steering_threshold)
            },
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
            }
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
            data_dir: Directory containing the dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            image_size: Target image size
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            balance_steering: Whether to balance steering distribution
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split
        self.balance_steering = balance_steering
        
        # Create splits if they don't exist
        self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits"""
        splits_file = os.path.join(self.data_dir, 'data_splits.json')
        
        if os.path.exists(splits_file):
            print("Using existing data splits")
            return
        
        print("Creating new data splits...")
        
        # Load all samples
        temp_dataset = CARLADataset(self.data_dir, split='train', balance_steering=False)
        all_samples = temp_dataset.samples
        
        # Group by episode to avoid data leakage
        episodes = {}
        for sample in all_samples:
            episode = sample['episode']
            if episode not in episodes:
                episodes[episode] = []
            episodes[episode].append(sample)
        
        episode_names = list(episodes.keys())
        
        # Split episodes
        train_episodes, temp_episodes = train_test_split(
            episode_names, test_size=(self.val_split + self.test_split), random_state=42
        )
        
        val_episodes, test_episodes = train_test_split(
            temp_episodes, 
            test_size=(self.test_split / (self.val_split + self.test_split)), 
            random_state=42
        )
        
        # Create sample lists for each split
        train_samples = []
        val_samples = []
        test_samples = []
        
        for episode in train_episodes:
            train_samples.extend(episodes[episode])
        
        for episode in val_episodes:
            val_samples.extend(episodes[episode])
        
        for episode in test_episodes:
            test_samples.extend(episodes[episode])
        
        # Save splits
        splits = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'train_episodes': train_episodes,
            'val_episodes': val_episodes,
            'test_episodes': test_episodes
        }
        
        # Save individual split files
        with open(os.path.join(self.data_dir, 'train_samples.json'), 'w') as f:
            json.dump(train_samples, f)
        
        with open(os.path.join(self.data_dir, 'val_samples.json'), 'w') as f:
            json.dump(val_samples, f)
        
        with open(os.path.join(self.data_dir, 'test_samples.json'), 'w') as f:
            json.dump(test_samples, f)
        
        with open(splits_file, 'w') as f:
            json.dump(splits, f)
        
        print(f"Created splits: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    
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
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self, split: str = 'train') -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample batch for testing"""
        if split == 'train':
            dataset = CARLADataset(self.data_dir, split='train', image_size=self.image_size)
        elif split == 'val':
            dataset = CARLADataset(self.data_dir, split='val', image_size=self.image_size, augment=False)
        else:
            dataset = CARLADataset(self.data_dir, split='test', image_size=self.image_size, augment=False)
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        return next(iter(loader))