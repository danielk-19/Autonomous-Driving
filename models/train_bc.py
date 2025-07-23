import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bc_model import BehavioralCloningModel, LightweightBCModel, MultiTaskLoss, create_model
from carla_dataset import CARLADataModule, CARLADataset


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
                 save_dir: str = 'checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = MultiTaskLoss(use_adaptive_weights=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_patience = early_stopping_patience
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # TensorBoard logging
        log_dir = os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)
        
        print(f"Trainer initialized. Logs will be saved to {log_dir}")
        print(f"Model has {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        steering_loss_sum = 0.0
        throttle_loss_sum = 0.0
        brake_loss_sum = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss
            loss_dict = self.criterion(predictions, targets)
            total_batch_loss = loss_dict['total_loss']
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            steering_loss_sum += loss_dict['steering_loss'].item()
            throttle_loss_sum += loss_dict['throttle_loss'].item()
            brake_loss_sum += loss_dict['brake_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/BatchLoss', total_batch_loss.item(), global_step)
            
            if batch_idx % 100 == 0:  # Log weights every 100 batches
                weights = loss_dict['weights']
                for key, value in weights.items():
                    self.writer.add_scalar(f'Train/Weight_{key}', value, global_step)
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'steering_loss': steering_loss_sum / num_batches,
            'throttle_loss': throttle_loss_sum / num_batches,
            'brake_loss': brake_loss_sum / num_batches
        }
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        steering_loss_sum = 0.0
        throttle_loss_sum = 0.0
        brake_loss_sum = 0.0
        
        steering_mae = 0.0
        throttle_mae = 0.0
        brake_mae = 0.0
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            
            for images, targets in pbar:
                # Move data to device
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(predictions, targets)
                
                # Accumulate losses
                total_loss += loss_dict['total_loss'].item()
                steering_loss_sum += loss_dict['steering_loss'].item()
                throttle_loss_sum += loss_dict['throttle_loss'].item()
                brake_loss_sum += loss_dict['brake_loss'].item()
                
                # Calculate MAE
                steering_mae += torch.mean(torch.abs(predictions['steering'] - targets['steering'])).item()
                throttle_mae += torch.mean(torch.abs(predictions['throttle'] - targets['throttle'])).item()
                brake_mae += torch.mean(torch.abs(predictions['brake'] - targets['brake'])).item()
                
                pbar.set_postfix({'Val Loss': f'{loss_dict["total_loss"].item():.4f}'})
        
        # Calculate average metrics
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'steering_loss': steering_loss_sum / num_batches,
            'throttle_loss': throttle_loss_sum / num_batches,
            'brake_loss': brake_loss_sum / num_batches,
            'steering_mae': steering_mae / num_batches,
            'throttle_mae': throttle_mae / num_batches,
            'brake_mae': brake_mae / num_batches
        }
        
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'model_type': type(self.model).__name__,
                'input_channels': getattr(self.model, 'input_channels', 3),
                'image_height': getattr(self.model, 'image_height', 224),
                'image_width': getattr(self.model, 'image_width', 224)
            }
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
            print(f"New best model saved to {best_filepath}")
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate epoch
            val_losses = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_losses['total_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Store losses
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
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
            self.writer.add_scalar('Val/ThrottleMAE', val_losses['throttle_mae'], epoch)
            self.writer.add_scalar('Val/BrakeMAE', val_losses['brake_mae'], epoch)
            
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch summary
            print(f'\nEpoch {epoch + 1}/{self.current_epoch + num_epochs}:')
            print(f'  Train Loss: {train_losses["total_loss"]:.4f}')
            print(f'  Val Loss: {val_losses["total_loss"]:.4f}')
            print(f'  Steering MAE: {val_losses["steering_mae"]:.4f}')
            print(f'  Learning Rate: {current_lr:.2e}')
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break
        
        # Save final checkpoint
        self.save_checkpoint(filename='final_model.pth')
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(epochs, [x['total_loss'] for x in self.train_losses], label='Train')
        axes[0, 0].plot(epochs, [x['total_loss'] for x in self.val_losses], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Steering loss
        axes[0, 1].plot(epochs, [x['steering_loss'] for x in self.train_losses], label='Train')
        axes[0, 1].plot(epochs, [x['steering_loss'] for x in self.val_losses], label='Validation')
        axes[0, 1].set_title('Steering Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # MAE metrics
        axes[1, 1].plot(epochs, [x['steering_mae'] for x in self.val_losses], label='Steering MAE')
        axes[1, 1].plot(epochs, [x['throttle_mae'] for x in self.val_losses], label='Throttle MAE')
        axes[1, 1].plot(epochs, [x['brake_mae'] for x in self.val_losses], label='Brake MAE')
        axes[1, 1].set_title('Validation MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {plot_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Behavioral Cloning model for CARLA')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the CARLA dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Image size [height, width]')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='full', 
                       choices=['full', 'lightweight'],
                       help='Type of model to use')
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
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize data module
    print("Initializing data module...")
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
    
    print(f"Data loaded:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print(f"Initializing {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_channels=3,
        image_height=args.image_size[0],
        image_width=args.image_size[1],
        use_attention=args.use_attention,
        dropout_rate=args.dropout_rate
    )
    
    # Initialize trainer
    trainer = BCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir
    )
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(num_epochs=args.num_epochs, save_every=args.save_every)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()