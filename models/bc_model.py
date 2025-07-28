"""
Behavioral Cloning Model for CARLA Autonomous Driving System
Neural network architecture for predicting vehicle control commands from camera images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Import utilities
from utils.utils import (
    setup_logging, safe_float, safe_int, clamp, 
    save_pytorch_model, Timer, get_timestamp,
    check_gpu_availability, get_system_info
)

# Setup logging
logger = logging.getLogger(__name__)

class BehavioralCloningModel(nn.Module):
    """
    Behavioral Cloning model for autonomous driving in CARLA.
    Takes RGB camera input and outputs steering, throttle, and brake commands.
    
    Architecture:
    - Input: RGB images (3, 224, 224)
    - Output: [steering, throttle, brake]
    - Backbone: ResNet18-inspired CNN
    - Loss: Weighted MSE (steering weight = 2.0)
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 image_height: int = 224,
                 image_width: int = 224,
                 use_attention: bool = True,
                 dropout_rate: float = 0.3,
                 steering_weight: float = 2.0):
        super(BehavioralCloningModel, self).__init__()
        
        # Store configuration
        self.input_channels = safe_int(input_channels, 3)
        self.image_height = safe_int(image_height, 224)
        self.image_width = safe_int(image_width, 224)
        self.use_attention = use_attention
        self.dropout_rate = clamp(safe_float(dropout_rate), 0.0, 0.8)
        self.steering_weight = safe_float(steering_weight, 2.0)
        
        # Model metadata for compliance
        self.model_info = {
            'architecture': 'ResNet18-inspired CNN',
            'input_shape': (self.input_channels, self.image_height, self.image_width),
            'output_shape': (3,),  # [steering, throttle, brake]
            'parameters': 0,  # Will be calculated after initialization
            'created_at': get_timestamp(),
            'device_info': get_system_info()
        }
        
        logger.info(f"Initializing BehavioralCloningModel with input shape: {self.model_info['input_shape']}")
        
        # CNN Feature Extractor (ResNet18-inspired)
        self.conv_layers = self._build_cnn_backbone()
        
        # Calculate flattened size dynamically
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Attention mechanism (optional for improved performance)
        if self.use_attention:
            self.attention = self._build_attention_module()
        
        # Fully connected layers for control prediction
        self.fc_layers = self._build_fc_layers()
        
        # Output heads for different controls
        self.steering_head = nn.Linear(256, 1)  # Steering angle [-1, 1]
        self.throttle_head = nn.Linear(256, 1)  # Throttle [0, 1]
        self.brake_head = nn.Linear(256, 1)     # Brake [0, 1]
        
        # Initialize weights using best practices
        self.apply(self._init_weights)
        
        # Calculate and store parameter count
        self.model_info['parameters'] = sum(p.numel() for p in self.parameters())
        
        logger.info(f"Model initialized with {self.model_info['parameters']:,} parameters")
    
    def _build_cnn_backbone(self) -> nn.Sequential:
        """Build CNN backbone following ResNet18 architecture principles"""
        return nn.Sequential(
            # Initial conv block (as per ResNet design)
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet Block 1 (64 channels)
            self._make_layer(64, 64, 2, stride=1),
            
            # ResNet Block 2 (128 channels)
            self._make_layer(64, 128, 2, stride=2),
            
            # ResNet Block 3 (256 channels)
            self._make_layer(128, 256, 2, stride=2),
            
            # ResNet Block 4 (512 channels)
            self._make_layer(256, 512, 2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a ResNet layer with skip connections"""
        layers = []
        
        # First block with potential downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size of convolutional layers dynamically"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.image_height, self.image_width)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.numel()
    
    def _build_attention_module(self) -> nn.Sequential:
        """Build attention mechanism for improved feature focus"""
        return nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),  # Lower dropout for attention
            nn.Linear(512, self.conv_output_size),
            nn.Sigmoid()
        )
    
    def _build_fc_layers(self) -> nn.Sequential:
        """Build fully connected layers with proper regularization"""
        return nn.Sequential(
            nn.Linear(self.conv_output_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),  # Lower dropout before output
        )
    
    def _init_weights(self, m):
        """Initialize model weights using best practices"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 3) containing [steering, throttle, brake]
        """
        # Validate input shape
        if x.shape[1:] != (self.input_channels, self.image_height, self.image_width):
            logger.warning(f"Input shape {x.shape} doesn't match expected {self.model_info['input_shape']}")
        
        # CNN feature extraction
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # Fully connected layers
        fc_output = self.fc_layers(features)
        
        # Generate control outputs with proper activation functions
        steering = torch.tanh(self.steering_head(fc_output))  # [-1, 1]
        throttle = torch.sigmoid(self.throttle_head(fc_output))  # [0, 1]
        brake = torch.sigmoid(self.brake_head(fc_output))  # [0, 1]
        
        # Combine outputs
        outputs = torch.cat([steering, throttle, brake], dim=1)
        
        return outputs
    
    def predict_controls(self, image: torch.Tensor) -> Tuple[float, float, float]:
        """
        Predict controls for a single image (inference mode)
        
        Args:
            image: Single image tensor
            
        Returns:
            Tuple of (steering, throttle, brake) values
        """
        self.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            outputs = self.forward(image)
            
            # Extract and clamp outputs using utils
            steering = clamp(safe_float(outputs[0, 0].cpu().item()), -1.0, 1.0)
            throttle = clamp(safe_float(outputs[0, 1].cpu().item()), 0.0, 1.0)
            brake = clamp(safe_float(outputs[0, 2].cpu().item()), 0.0, 1.0)
            
            return steering, throttle, brake
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            **self.model_info,
            'current_device': str(next(self.parameters()).device),
            'is_training': self.training,
            'dropout_rate': self.dropout_rate,
            'attention_enabled': self.use_attention,
            'steering_weight': self.steering_weight
        }
    
    def save_checkpoint(self, filepath: str, optimizer=None, epoch: int = 0, loss: float = 0.0) -> bool:
        """Save model checkpoint using utils"""
        try:
            return save_pytorch_model(self, optimizer, epoch, loss, filepath)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable,
            'non_trainable': total - trainable,
            'total': total
        }


class BasicBlock(nn.Module):
    """Basic ResNet block with skip connection"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class LightweightBCModel(nn.Module):
    """
    Lightweight version of BC model for faster inference
    Designed for real-time performance in CARLA environment
    """
    
    def __init__(self, input_channels: int = 3, dropout_rate: float = 0.2):
        super(LightweightBCModel, self).__init__()
        
        self.input_channels = safe_int(input_channels, 3)
        self.dropout_rate = clamp(safe_float(dropout_rate), 0.0, 0.5)
        
        logger.info("Initializing LightweightBCModel for fast inference")
        
        self.conv_layers = nn.Sequential(
            # Efficient conv blocks with separable convolutions
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )
        
        # Output heads
        self.output_head = nn.Linear(128, 3)  # Combined output for efficiency
        
        # Initialize weights
        self.apply(self._init_weights)
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"LightweightBCModel initialized with {param_count:,} parameters")
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        fc_output = self.fc_layers(features)
        
        # Single output head with different activations
        raw_output = self.output_head(fc_output)
        
        # Apply appropriate activations
        steering = torch.tanh(raw_output[:, 0:1])  # [-1, 1]
        throttle = torch.sigmoid(raw_output[:, 1:2])  # [0, 1]
        brake = torch.sigmoid(raw_output[:, 2:3])  # [0, 1]
        
        return torch.cat([steering, throttle, brake], dim=1)


class ControlTaskLoss(nn.Module):
    """
    Loss function:
    - Weighted MSE (steering weight = 2.0)
    - Additional regularization for smooth driving
    """
    
    def __init__(self, 
                 steering_weight: float = 2.0,
                 throttle_weight: float = 1.0,
                 brake_weight: float = 1.0,
                 smoothness_weight: float = 0.1):
        super(ControlTaskLoss, self).__init__()
        
        # Weights
        self.steering_weight = safe_float(steering_weight, 2.0)
        self.throttle_weight = safe_float(throttle_weight, 1.0)
        self.brake_weight = safe_float(brake_weight, 1.0)
        self.smoothness_weight = safe_float(smoothness_weight, 0.1)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        
        logger.info(f"ControlTaskLoss initialized with steering_weight={self.steering_weight}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate loss
        
        Args:
            predictions: Model predictions [batch_size, 3] (steering, throttle, brake)
            targets: Ground truth [batch_size, 3] (steering, throttle, brake)
            
        Returns:
            Dictionary with loss components
        """
        # Split predictions and targets
        pred_steering, pred_throttle, pred_brake = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        true_steering, true_throttle, true_brake = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Calculate individual MSE losses
        steering_loss = self.mse_loss(pred_steering, true_steering).mean()
        throttle_loss = self.mse_loss(pred_throttle, true_throttle).mean()
        brake_loss = self.mse_loss(pred_brake, true_brake).mean()
        
        # Smoothness regularization for steering
        steering_smoothness = self.l1_loss(pred_steering, true_steering).mean()
        
        # Weighted total loss
        total_loss = (
            self.steering_weight * steering_loss +
            self.throttle_weight * throttle_loss +
            self.brake_weight * brake_loss +
            self.smoothness_weight * steering_smoothness
        )
        
        return {
            'total_loss': total_loss,
            'steering_loss': steering_loss,
            'throttle_loss': throttle_loss,
            'brake_loss': brake_loss,
            'smoothness_loss': steering_smoothness,
            'weighted_components': {
                'steering': self.steering_weight * steering_loss,
                'throttle': self.throttle_weight * throttle_loss,
                'brake': self.brake_weight * brake_loss,
                'smoothness': self.smoothness_weight * steering_smoothness
            }
        }


class AdaptiveLoss(nn.Module):
    """
    Adaptive control-task loss with learnable weights for advanced training
    """
    
    def __init__(self, num_tasks: int = 3, init_weight: float = 1.0):
        super(AdaptiveLoss, self).__init__()
        
        # Learnable log-variance parameters for automatic task weighting
        self.log_vars = nn.Parameter(torch.ones(num_tasks) * np.log(init_weight))
        self.num_tasks = num_tasks
        
        logger.info(f"AdaptiveLoss initialized with {num_tasks} tasks")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate adaptive loss"""
        # Individual task losses
        mse_losses = []
        for i in range(self.num_tasks):
            loss = F.mse_loss(predictions[:, i], targets[:, i])
            mse_losses.append(loss)
        
        # Adaptive weighting
        total_loss = 0
        for i, loss in enumerate(mse_losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return {
            'total_loss': total_loss,
            'task_losses': mse_losses,
            'task_weights': torch.exp(-self.log_vars).detach(),
            'uncertainties': self.log_vars.detach()
        }


def create_model(model_type: str = 'full', device: str = 'auto', **kwargs) -> nn.Module:
    """
    Factory function to create BC models
    
    Args:
        model_type: 'full', 'lightweight'
        device: Target device or 'auto' for automatic selection
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model on specified device
    """
    # Determine device
    if device == 'auto':
        device = 'cuda' if check_gpu_availability() else 'cpu'
    
    logger.info(f"Creating {model_type} model on device: {device}")
    
    # Create model
    if model_type == 'full':
        model = BehavioralCloningModel(**kwargs)
    elif model_type == 'lightweight':
        model = LightweightBCModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'full' or 'lightweight'")
    
    # Move to device
    model = model.to(device)
    
    logger.info(f"Model created successfully: {model.__class__.__name__}")
    return model


def create_loss_function(loss_type: str = 'control_task', **kwargs) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: 'control_task', 'adaptive'
        **kwargs: Additional arguments for loss initialization
        
    Returns:
        Loss function
    """
    if loss_type == 'control_task':
        return ControlTaskLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_model_transforms(mode: str = 'train', image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get image preprocessing transforms
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size (height, width)
        
    Returns:
        Composed transforms
    """
    if mode == 'train':
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)


def validate_model_output(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Validate model output format and ranges
    
    Args:
        model: The model to validate
        sample_input: Sample input tensor
        
    Returns:
        Validation results
    """
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            
            # Check output shape
            expected_shape = (sample_input.size(0), 3)
            shape_valid = output.shape == expected_shape
            
            # Check output ranges
            steering_range = (output[:, 0].min().item(), output[:, 0].max().item())
            throttle_range = (output[:, 1].min().item(), output[:, 1].max().item())
            brake_range = (output[:, 2].min().item(), output[:, 2].max().item())
            
            # Validate ranges
            steering_valid = -1.0 <= steering_range[0] and steering_range[1] <= 1.0
            throttle_valid = 0.0 <= throttle_range[0] and throttle_range[1] <= 1.0
            brake_valid = 0.0 <= brake_range[0] and brake_range[1] <= 1.0
            
            validation_results = {
                'output_shape_valid': shape_valid,
                'expected_shape': expected_shape,
                'actual_shape': tuple(output.shape),
                'steering_range_valid': steering_valid,
                'throttle_range_valid': throttle_valid,
                'brake_range_valid': brake_valid,
                'steering_range': steering_range,
                'throttle_range': throttle_range,
                'brake_range': brake_range,
                'overall_valid': shape_valid and steering_valid and throttle_valid and brake_valid
            }
            
            if validation_results['overall_valid']:
                logger.info("Model output validation passed")
            else:
                logger.warning(f"Model output validation failed: {validation_results}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed with exception: {e}")
            return {'overall_valid': False, 'error': str(e)}


# Model registry for easy access
MODEL_REGISTRY = {
    'bc_full': BehavioralCloningModel,
    'bc_lightweight': LightweightBCModel
}

LOSS_REGISTRY = {
    'control_task': ControlTaskLoss,
    'adaptive': AdaptiveLoss
}

def get_available_models() -> List[str]:
    """Get list of available model types"""
    return list(MODEL_REGISTRY.keys())

def get_available_losses() -> List[str]:
    """Get list of available loss functions"""
    return list(LOSS_REGISTRY.keys())