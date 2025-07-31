"""
Model Training Script for CARLA Behavioral Cloning
Trains neural network models for autonomous driving using processed CARLA data.
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add project root to path and import project modules
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from models.bc_model import (
    BehavioralCloningModel, LightweightBCModel, ControlTaskLoss, 
    AdaptiveLoss, create_model, create_loss_function, validate_model_output
)
from models.carla_dataset import CARLADataModule, CARLADataset, validate_data_structure

# Import utilities
from utils.utils import (
    setup_logging, ensure_dir, get_timestamp, safe_float, safe_int,
    save_pytorch_model, load_pytorch_model, plot_training_history,
    visualize_predictions, Timer, PerformanceMonitor, check_gpu_availability,
    get_system_info, cleanup_old_files, get_latest_file
)

# Setup logging
logger = logging.getLogger(__name__)


class BCTrainer:
    """
    Behavioral Cloning trainer for CARLA autonomous driving
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler_patience: int = 5,
                 early_stopping_patience: int = 10,
                 save_dir: str = 'checkpoints',
                 loss_type: str = 'control_task',
                 target_val_mse: float = 0.1):  # Performance target
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.target_val_mse = safe_float(target_val_mse, 0.1)
        
        # Create save directory using utils
        ensure_dir(save_dir)
        
        logger.info(f"Initializing BCTrainer with device: {device}")
        logger.info(f"Target validation MSE: {self.target_val_mse}")
        
        # Loss function using factory from bc_model
        self.loss_type = loss_type
        if loss_type == 'control_task':
            self.criterion = create_loss_function(
                loss_type='control_task',
                steering_weight=2.0,
                throttle_weight=1.0,
                brake_weight=1.0,
                smoothness_weight=0.1
            )
        elif loss_type == 'adaptive':
            self.criterion = create_loss_function(loss_type='adaptive', num_tasks=3)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=safe_float(learning_rate, 1e-3), 
            weight_decay=safe_float(weight_decay, 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=safe_int(scheduler_patience, 5),
            verbose=True,
            min_lr=1e-7
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_patience = safe_int(early_stopping_patience, 10)
        self.target_reached = False
        
        # Metrics tracking
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'steering_loss': [],
            'throttle_loss': [],
            'brake_loss': [],
            'steering_mae': [],
            'learning_rates': []
        }
        
        # Performance monitoring using utils
        self.perf_monitor = PerformanceMonitor()
        
        # TensorBoard logging
        log_dir = os.path.join('logs', f'training_{get_timestamp()}')
        ensure_dir(log_dir)
        self.writer = SummaryWriter(log_dir)
        
        # System info logging
        system_info = get_system_info()
        logger.info(f"System info: {system_info}")
        
        param_count = self.count_parameters()
        logger.info(f"Model has {param_count:,} trainable parameters")
        
        # Validate model output format
        self._validate_model()
        
        # Save training configuration
        self._save_config()
    
    def _validate_model(self):
        """Validate model output using utils"""
        try:
            # Get a sample batch
            sample_images, _ = next(iter(self.train_loader))
            sample_images = sample_images[:1].to(self.device)  # Single sample
            
            validation_results = validate_model_output(self.model, sample_images)
            
            if not validation_results['overall_valid']:
                raise ValueError(f"Model validation failed: {validation_results}")
            
            logger.info("Model output validation passed")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise
    
    def _save_config(self):
        """Save training configuration"""
        config = {
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'training_config': {
                'loss_type': self.loss_type,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[1]['weight_decay'] if len(self.optimizer.param_groups) > 1 else 0,
                'scheduler_patience': self.scheduler.patience,
                'early_stopping_patience': self.early_stopping_patience,
                'target_val_mse': self.target_val_mse,
                'device': str(self.device)
            },
            'data_info': {
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'batch_size': self.train_loader.batch_size
            },
            'system_info': get_system_info(),
            'created_at': get_timestamp()
        }
        
        config_path = os.path.join(self.save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved to {config_path}")
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.perf_monitor.reset()
        
        total_loss = 0.0
        component_losses = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0, 'smoothness': 0.0}
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            self.perf_monitor.log_frame()
            
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            target_tensor = torch.cat([
                targets['steering'].to(self.device, non_blocking=True),
                targets['throttle'].to(self.device, non_blocking=True),
                targets['brake'].to(self.device, non_blocking=True)
            ], dim=1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss using criterion
            loss_dict = self.criterion(predictions, target_tensor)
            total_batch_loss = loss_dict['total_loss']
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            if 'steering_loss' in loss_dict:
                component_losses['steering'] += loss_dict['steering_loss'].item()
                component_losses['throttle'] += loss_dict['throttle_loss'].item()
                component_losses['brake'] += loss_dict['brake_loss'].item()
                if 'smoothness_loss' in loss_dict:
                    component_losses['smoothness'] += loss_dict['smoothness_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'FPS': f'{self.perf_monitor.get_fps():.1f}'
            })
            
            # Log to tensorboard periodically
            if batch_idx % 100 == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/BatchLoss', total_batch_loss.item(), global_step)
                self.writer.add_scalar('Train/FPS', self.perf_monitor.get_fps(), global_step)
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'steering_loss': component_losses['steering'] / num_batches,
            'throttle_loss': component_losses['throttle'] / num_batches,
            'brake_loss': component_losses['brake'] / num_batches,
            'smoothness_loss': component_losses['smoothness'] / num_batches
        }
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        component_losses = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}
        mae_metrics = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}
        mse_metrics = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}
        
        num_batches = len(self.val_loader)
        
        # For visualization
        sample_images, sample_predictions, sample_targets = [], [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                target_tensor = torch.cat([
                    targets['steering'].to(self.device, non_blocking=True),
                    targets['throttle'].to(self.device, non_blocking=True),
                    targets['brake'].to(self.device, non_blocking=True)
                ], dim=1)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(predictions, target_tensor)
                
                # Accumulate losses
                total_loss += loss_dict['total_loss'].item()
                if 'steering_loss' in loss_dict:
                    component_losses['steering'] += loss_dict['steering_loss'].item()
                    component_losses['throttle'] += loss_dict['throttle_loss'].item()
                    component_losses['brake'] += loss_dict['brake_loss'].item()
                
                # Calculate metrics
                pred_steering = predictions[:, 0]
                pred_throttle = predictions[:, 1]
                pred_brake = predictions[:, 2]
                
                true_steering = target_tensor[:, 0]
                true_throttle = target_tensor[:, 1]
                true_brake = target_tensor[:, 2]
                
                # MAE
                mae_metrics['steering'] += torch.mean(torch.abs(pred_steering - true_steering)).item()
                mae_metrics['throttle'] += torch.mean(torch.abs(pred_throttle - true_throttle)).item()
                mae_metrics['brake'] += torch.mean(torch.abs(pred_brake - true_brake)).item()
                
                # MSE
                mse_metrics['steering'] += torch.mean((pred_steering - true_steering) ** 2).item()
                mse_metrics['throttle'] += torch.mean((pred_throttle - true_throttle) ** 2).item()
                mse_metrics['brake'] += torch.mean((pred_brake - true_brake) ** 2).item()
                
                # Collect samples for visualization (first batch only)
                if batch_idx == 0 and len(sample_images) == 0:
                    sample_images = images[:4].cpu().numpy()  # First 4 images
                    sample_predictions = predictions[:4, 0].cpu().numpy()  # Steering predictions
                    sample_targets = target_tensor[:4, 0].cpu().numpy()  # Steering targets
                
                pbar.set_postfix({'Val Loss': f'{loss_dict["total_loss"].item():.4f}'})
        
        # Calculate average metrics
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'steering_loss': component_losses['steering'] / num_batches,
            'throttle_loss': component_losses['throttle'] / num_batches,
            'brake_loss': component_losses['brake'] / num_batches,
            'steering_mae': mae_metrics['steering'] / num_batches,
            'throttle_mae': mae_metrics['throttle'] / num_batches,
            'brake_mae': mae_metrics['brake'] / num_batches,
            'steering_mse': mse_metrics['steering'] / num_batches,
            'throttle_mse': mse_metrics['throttle'] / num_batches,
            'brake_mse': mse_metrics['brake'] / num_batches
        }
        
        # Create visualization if we have samples
        if len(sample_images) > 0:
            try:
                viz_path = os.path.join(self.save_dir, f'predictions_epoch_{self.current_epoch + 1}.png')
                visualize_predictions(
                    sample_images, sample_targets, sample_predictions, 
                    save_path=viz_path, max_samples=4
                )
            except Exception as e:
                logger.warning(f"Failed to create prediction visualization: {e}")
        
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """Save model checkpoint using utils"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch + 1}.pth'
        
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'target_reached': self.target_reached,
            'model_config': {
                'model_type': type(self.model).__name__,
                'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
            },
            'training_config': {
                'loss_type': self.loss_type,
                'target_val_mse': self.target_val_mse
            }
        }
        
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            torch.save(checkpoint_data, filepath)
            logger.info(f"Checkpoint saved to {filepath}")
            
            if is_best:
                best_filepath = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(checkpoint_data, best_filepath)
                logger.info(f"New best model saved to {best_filepath}")
            
            # Cleanup old checkpoints (keep only 5 most recent)
            cleanup_old_files(self.save_dir, pattern="checkpoint_epoch_*.pth", keep_count=5)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_history = checkpoint.get('train_history', {
                'train_loss': [], 'val_loss': [], 'steering_loss': [],
                'throttle_loss': [], 'brake_loss': [], 'steering_mae': [],
                'learning_rates': []
            })
            self.target_reached = checkpoint.get('target_reached', False)
            
            logger.info(f"Checkpoint loaded from {filepath}")
            logger.info(f"Resuming from epoch {self.current_epoch + 1}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            raise
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Target validation MSE: {self.target_val_mse}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            
            with Timer(f"Epoch {epoch + 1}"):
                # Train epoch
                train_losses = self.train_epoch()
                
                # Validate epoch
                val_losses = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_losses['total_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store losses in history
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['steering_loss'].append(val_losses['steering_loss'])
            self.train_history['throttle_loss'].append(val_losses['throttle_loss'])
            self.train_history['brake_loss'].append(val_losses['brake_loss'])
            self.train_history['steering_mae'].append(val_losses['steering_mae'])
            self.train_history['learning_rates'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/TotalLoss', train_losses['total_loss'], epoch)
            self.writer.add_scalar('Train/SteeringLoss', train_losses['steering_loss'], epoch)
            self.writer.add_scalar('Train/ThrottleLoss', train_losses['throttle_loss'], epoch)
            self.writer.add_scalar('Train/BrakeLoss', train_losses['brake_loss'], epoch)
            
            self.writer.add_scalar('Val/TotalLoss', val_losses['total_loss'], epoch)
            self.writer.add_scalar('Val/SteeringLoss', val_losses['steering_loss'], epoch)
            self.writer.add_scalar('Val/ThrottleLoss', val_losses['throttle_loss'], epoch)
            self.writer.add_scalar('Val/BrakeLoss', val_losses['brake_loss'], epoch)
            self.writer.add_scalar('Val/SteeringMAE', val_losses['steering_mae'], epoch)
            self.writer.add_scalar('Val/SteeringMSE', val_losses['steering_mse'], epoch)
            
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Check if target performance reached
            if val_losses['total_loss'] < self.target_val_mse and not self.target_reached:
                self.target_reached = True
                logger.info(f"TARGET REACHED! Validation MSE: {val_losses['total_loss']:.4f} < {self.target_val_mse}")
            
            # Print epoch summary
            logger.info(f'\nEpoch {epoch + 1}/{self.current_epoch + num_epochs}:')
            logger.info(f'  Train Loss: {train_losses["total_loss"]:.4f}')
            logger.info(f'  Val Loss: {val_losses["total_loss"]:.4f}')
            logger.info(f'  Steering MAE: {val_losses["steering_mae"]:.4f}')
            logger.info(f'  Steering MSE: {val_losses["steering_mse"]:.4f}')
            logger.info(f'  Learning Rate: {current_lr:.2e}')
            logger.info(f'  Target Reached: {self.target_reached}')
            
            # Save checkpoint logic
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best or self.target_reached:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                break
            
            # If target reached and several epochs passed, can optionally stop
            if self.target_reached and self.early_stopping_counter >= 3:
                logger.info("Target reached and performance stable, stopping training")
                break
        
        # Save final checkpoint
        self.save_checkpoint(filename='final_model.pth')
        
        # Training summary
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Target MSE ({self.target_val_mse}) reached: {self.target_reached}")
        
        # Close tensorboard writer
        self.writer.close()
        
        # Plot training curves using utils
        self.plot_training_curves()
        
        # Return training summary
        return {
            'best_val_loss': self.best_val_loss,
            'target_reached': self.target_reached,
            'total_epochs': self.current_epoch + 1,
            'total_time_hours': total_time / 3600,
            'final_lr': current_lr
        }
    
    def plot_training_curves(self):
        """Plot training curves using utils"""
        try:
            plot_path = os.path.join(self.save_dir, 'training_curves.png')
            fig = plot_training_history(self.train_history, save_path=plot_path)
            logger.info(f"Training curves saved to {plot_path}")
            plt.close(fig)  # Close to free memory
        except Exception as e:
            logger.error(f"Failed to plot training curves: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Behavioral Cloning model for CARLA')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed CARLA dataset (data/processed/)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Image size [height, width]')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='full', 
                       choices=['full', 'lightweight'],
                       help='Type of model to use (bc_full or bc_lightweight)')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use attention mechanism in full model')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                       help='Patience for learning rate scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--target_val_mse', type=float, default=0.1,
                       help='Target validation MSE (performance target)')
    
    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default='control_task',
                       choices=['control_task', 'adaptive'],
                       help='Type of loss function to use')
    
    # Data processing arguments
    parser.add_argument('--balance_steering', action='store_true', default=True,
                       help='Balance steering distribution')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--auto_resume', action='store_true',
                       help='Automatically resume from latest checkpoint in save_dir')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cuda, cpu)')
    
    # Debugging arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--validate_data', action='store_true',
                       help='Validate data structure before training')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = os.path.join(args.save_dir, f'training_{get_timestamp()}.log')
    ensure_dir(args.save_dir)
    setup_logging(log_level=log_level, log_file=log_file)
    
    logger.info("="*60)
    logger.info("CARLA Behavioral Cloning Training")
    logger.info("="*60)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if check_gpu_availability() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Validate data structure if requested
    if args.validate_data:
        logger.info("Validating data structure...")
        validation_results = validate_data_structure(args.data_dir)
        
        if not validation_results['valid']:
            logger.error("Data structure validation failed!")
            logger.error(f"Errors: {validation_results['errors']}")
            return
        else:
            logger.info("Data structure validation passed ✓")
            logger.info(f"Found {validation_results['episodes_found']} episodes with ~{validation_results['total_samples']:.0f} samples")
    
    # Initialize data module
    logger.info("Initializing data module...")
    try:
        with Timer("Data module initialization"):
            data_module = CARLADataModule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=tuple(args.image_size),
                val_split=args.val_split,
                test_split=args.test_split,
                balance_steering=args.balance_steering
            )
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_module.get_dataloaders()
        
        # Log data information
        data_info = data_module.get_data_info()
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training samples: {data_info['splits']['train']:,}")
        logger.info(f"  Validation samples: {data_info['splits']['val']:,}")
        logger.info(f"  Test samples: {data_info['splits']['test']:,}")
        logger.info(f"  Total samples: {data_info['splits']['total']:,}")
        
        # Log training data statistics
        if 'train_stats' in data_info and 'steering' in data_info['train_stats']:
            steering_stats = data_info['train_stats']['steering']
            logger.info(f"  Steering distribution: mean={steering_stats.get('mean', 0):.3f}, "
                       f"std={steering_stats.get('std', 0):.3f}, "
                       f"straight_ratio={steering_stats.get('straight_ratio', 0):.3f}")
        
    except Exception as e:
        logger.error(f"Failed to initialize data module: {e}")
        return
    
    # Initialize model
    logger.info(f"Initializing {args.model_type} model...")
    try:
        with Timer("Model initialization"):
            model = create_model(
                model_type=args.model_type,
                device=device,
                input_channels=3,
                image_height=args.image_size[0],
                image_width=args.image_size[1],
                use_attention=args.use_attention,
                dropout_rate=args.dropout_rate
            )
        
        # Log model information
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    try:
        trainer = BCTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            early_stopping_patience=args.early_stopping_patience,
            save_dir=args.save_dir,
            loss_type=args.loss_type,
            target_val_mse=args.target_val_mse
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return
    
    # Handle checkpoint resuming
    resume_path = args.resume
    
    # Auto-resume logic
    if args.auto_resume and not args.resume:
        latest_checkpoint = get_latest_file(args.save_dir, pattern="checkpoint_epoch_*.pth")
        if latest_checkpoint:
            resume_path = str(latest_checkpoint)
            logger.info(f"Auto-resuming from latest checkpoint: {resume_path}")
    
    # Load checkpoint
    if resume_path:
        if os.path.exists(resume_path):
            logger.info(f"Resuming training from {resume_path}")
            try:
                trainer.load_checkpoint(resume_path)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch...")
        else:
            logger.warning(f"Checkpoint file not found: {resume_path}")
            logger.info("Starting training from scratch...")
    
    # Start training
    logger.info("Starting training...")
    try:
        training_summary = trainer.train(
            num_epochs=args.num_epochs, 
            save_every=args.save_every
        )
        
        # Log final results
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        logger.info(f"Best validation loss: {training_summary['best_val_loss']:.4f}")
        logger.info(f"Target MSE ({args.target_val_mse}) reached: {training_summary['target_reached']}")
        logger.info(f"Total epochs: {training_summary['total_epochs']}")
        logger.info(f"Total training time: {training_summary['total_time_hours']:.2f} hours")
        logger.info(f"Final learning rate: {training_summary['final_lr']:.2e}")
        
        # Performance assessment based on targets
        if training_summary['target_reached']:
            logger.info("SUCCESS: Model reached performance target!")
        else:
            logger.warning("Model did not reach performance target")
        
        # Save final summary
        summary_path = os.path.join(args.save_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'training_summary': training_summary,
                'args': vars(args),
                'data_info': data_info,
                'completed_at': get_timestamp()
            }, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(filename='interrupted_model.pth')
        logger.info("Model saved before interruption")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        return
    
    logger.info("Training script completed successfully!")


def validate_args(args):
    """Validate command line arguments"""
    errors = []
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        errors.append(f"Data directory does not exist: {args.data_dir}")
    
    # Check episodes directory
    episodes_dir = os.path.join(args.data_dir, 'episodes')
    if not os.path.exists(episodes_dir):
        errors.append(f"Episodes directory does not exist: {episodes_dir}")
    
    # Validate numeric arguments
    if args.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if args.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if not (0 <= args.dropout_rate <= 1):
        errors.append("Dropout rate must be between 0 and 1")
    
    if not (0 < args.val_split < 1):
        errors.append("Validation split must be between 0 and 1")
    
    if not (0 < args.test_split < 1):
        errors.append("Test split must be between 0 and 1")
    
    if args.val_split + args.test_split >= 1:
        errors.append("Val split + test split must be less than 1")
    
    # Check image size
    if len(args.image_size) != 2 or any(s <= 0 for s in args.image_size):
        errors.append("Image size must be two positive integers")
    
    # Check resume checkpoint
    if args.resume and not os.path.exists(args.resume):
        errors.append(f"Resume checkpoint does not exist: {args.resume}")
    
    return errors


def test_training_setup(args):
    """Test training setup with minimal configuration"""
    logger.info("Testing training setup...")
    
    try:
        # Test data loading
        test_data_module = CARLADataModule(
            data_dir=args.data_dir,
            batch_size=2,
            num_workers=0,
            image_size=tuple(args.image_size),
            val_split=0.2,
            test_split=0.2,
            balance_steering=False
        )
        
        train_loader, val_loader, _ = test_data_module.get_dataloaders()
        
        # Test model creation
        test_model = create_model(
            model_type=args.model_type,
            device='cpu',  # Use CPU for testing
            input_channels=3,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
            use_attention=False,  # Disable for faster testing
            dropout_rate=0.1
        )
        
        # Test forward pass
        sample_batch = test_data_module.get_sample_batch('train')
        images, targets = sample_batch
        
        with torch.no_grad():
            predictions = test_model(images)
        
        logger.info(f"Test batch - Images: {images.shape}, Predictions: {predictions.shape}")
        logger.info("Training setup test passed")
        return True
        
    except Exception as e:
        logger.error(f"Training setup test failed: {e}")
        return False


if __name__ == '__main__':
    args = parse_args()
    
    # Validate arguments
    validation_errors = validate_args(args)
    if validation_errors:
        print("Argument validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Test setup if in debug mode
    if args.debug:
        setup_logging(log_level=logging.DEBUG)
        logger.info("Debug mode enabled, testing setup...")
        
        if not test_training_setup(args):
            logger.error("Setup test failed, exiting")
            sys.exit(1)
        
        logger.info("Setup test passed, proceeding with training...")
    
    # Run main training
    main()