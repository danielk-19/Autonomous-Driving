import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict, List, Tuple


class BehavioralCloningModel(nn.Module):
    """
    Behavioral Cloning model for autonomous driving in CARLA.
    Takes RGB camera input and outputs steering, throttle, and brake commands.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 image_height: int = 224,
                 image_width: int = 224,
                 use_attention: bool = True,
                 dropout_rate: float = 0.3):
        super(BehavioralCloningModel, self).__init__()
        
        self.input_channels = input_channels
        self.image_height = image_height
        self.image_width = image_width
        self.use_attention = use_attention
        
        # CNN Feature Extractor (Based on ResNet-like architecture)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Calculate flattened size
        self.conv_output_size = 512 * 7 * 7
        
        # Attention mechanism (optional)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.conv_output_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.conv_output_size),
                nn.Sigmoid()
            )
        
        # Fully connected layers for control prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Output heads for different controls
        self.steering_head = nn.Linear(256, 1)  # Steering angle [-1, 1]
        self.throttle_head = nn.Linear(256, 1)  # Throttle [0, 1]
        self.brake_head = nn.Linear(256, 1)     # Brake [0, 1]
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary containing steering, throttle, and brake predictions
        """
        # CNN feature extraction
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # Fully connected layers
        fc_output = self.fc_layers(features)
        
        # Generate control outputs
        steering = torch.tanh(self.steering_head(fc_output))  # [-1, 1]
        throttle = torch.sigmoid(self.throttle_head(fc_output))  # [0, 1]
        brake = torch.sigmoid(self.brake_head(fc_output))  # [0, 1]
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }
    
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
            
            steering = outputs['steering'].cpu().item()
            throttle = outputs['throttle'].cpu().item()
            brake = outputs['brake'].cpu().item()
            
            return steering, throttle, brake


class LightweightBCModel(nn.Module):
    """
    Lightweight version of BC model for faster inference
    """
    
    def __init__(self, input_channels: int = 3, dropout_rate: float = 0.2):
        super(LightweightBCModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Smaller conv blocks
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Output heads
        self.steering_head = nn.Linear(128, 1)
        self.throttle_head = nn.Linear(128, 1)
        self.brake_head = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        fc_output = self.fc_layers(features)
        
        return {
            'steering': torch.tanh(self.steering_head(fc_output)),
            'throttle': torch.sigmoid(self.throttle_head(fc_output)),
            'brake': torch.sigmoid(self.brake_head(fc_output))
        }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for behavioral cloning
    Combines losses for steering, throttle, and brake with automatic weighting
    """
    
    def __init__(self, 
                 steering_weight: float = 1.0,
                 throttle_weight: float = 0.5,
                 brake_weight: float = 0.5,
                 use_adaptive_weights: bool = True):
        super(MultiTaskLoss, self).__init__()
        
        self.use_adaptive_weights = use_adaptive_weights
        
        if use_adaptive_weights:
            # Learnable parameters for automatic task weighting
            self.steering_log_var = nn.Parameter(torch.zeros(1))
            self.throttle_log_var = nn.Parameter(torch.zeros(1))
            self.brake_log_var = nn.Parameter(torch.zeros(1))
        else:
            # Fixed weights
            self.steering_weight = steering_weight
            self.throttle_weight = throttle_weight
            self.brake_weight = brake_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Calculate individual losses
        steering_loss = self.mse_loss(predictions['steering'], targets['steering'])
        throttle_loss = self.mse_loss(predictions['throttle'], targets['throttle'])
        brake_loss = self.mse_loss(predictions['brake'], targets['brake'])
        
        # Add L1 regularization for steering (smoothness)
        steering_l1 = self.l1_loss(predictions['steering'], targets['steering'])
        
        if self.use_adaptive_weights:
            # Adaptive weighting based on uncertainty
            total_loss = (
                torch.exp(-self.steering_log_var) * steering_loss + self.steering_log_var +
                torch.exp(-self.throttle_log_var) * throttle_loss + self.throttle_log_var +
                torch.exp(-self.brake_log_var) * brake_loss + self.brake_log_var +
                0.1 * steering_l1  # Small L1 term for smoothness
            )
            
            weights = {
                'steering': torch.exp(-self.steering_log_var).item(),
                'throttle': torch.exp(-self.throttle_log_var).item(),
                'brake': torch.exp(-self.brake_log_var).item()
            }
        else:
            # Fixed weighting
            total_loss = (
                self.steering_weight * steering_loss +
                self.throttle_weight * throttle_loss +
                self.brake_weight * brake_loss +
                0.1 * steering_l1
            )
            
            weights = {
                'steering': self.steering_weight,
                'throttle': self.throttle_weight,
                'brake': self.brake_weight
            }
        
        return {
            'total_loss': total_loss,
            'steering_loss': steering_loss,
            'throttle_loss': throttle_loss,
            'brake_loss': brake_loss,
            'steering_l1': steering_l1,
            'weights': weights
        }


def create_model(model_type: str = 'full', **kwargs) -> nn.Module:
    """
    Factory function to create BC models
    
    Args:
        model_type: 'full' or 'lightweight'
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'full':
        return BehavioralCloningModel(**kwargs)
    elif model_type == 'lightweight':
        return LightweightBCModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Image preprocessing transforms
def get_transforms(mode: str = 'train', image_size: Tuple[int, int] = (224, 224)):
    """
    Get image preprocessing transforms
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size (height, width)
        
    Returns:
        torchvision transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])