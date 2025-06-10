"""
A.1: Simple CNN architecture for emotion recognition baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class SimpleCNN(nn.Module):
    """Simple CNN architecture for emotion recognition."""
    
    def __init__(self, 
                 num_classes: int = 1,
                 input_channels: int = 3,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True):
        """
        Args:
            num_classes: Number of output classes (1 for binary, 7 for emotions)
            input_channels: Number of input channels (3 for RGB)
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input shape: (batch_size, 3, 224, 224)
        
        # Convolutional block 1: 224x224 -> 112x112
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Convolutional block 2: 112x112 -> 56x56
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Convolutional block 3: 56x56 -> 28x28
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Convolutional block 4: 28x28 -> 14x14
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Global average pooling: 14x14 -> 1x1
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate feature maps for visualization."""
        features = {}
        
        # Conv block 1
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        features['conv1'] = x.clone()
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        features['conv2'] = x.clone()
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        features['conv3'] = x.clone()
        x = self.pool(x)
        
        # Conv block 4
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = F.relu(x)
        features['conv4'] = x.clone()
        
        return features
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }

class DeepCNN(nn.Module):
    """Deeper CNN variant with residual connections."""
    
    def __init__(self,
                 num_classes: int = 1,
                 input_channels: int = 3,
                 dropout_rate: float = 0.5):
        super(DeepCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 64, 2)
        self.res_block2 = self._make_residual_block(64, 128, 2, stride=2)
        self.res_block3 = self._make_residual_block(128, 256, 2, stride=2)
        self.res_block4 = self._make_residual_block(256, 512, 2, stride=2)
        
        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
        
    def _make_residual_block(self, in_channels: int, out_channels: int, 
                           num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual block."""
        layers = []
        
        # First block might have stride != 1
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Final layers
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

def create_cnn_model(model_type: str = 'simple', 
                    num_classes: int = 1,
                    **kwargs) -> nn.Module:
    """Factory function to create CNN models."""
    
    if model_type == 'simple':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'deep':
        return DeepCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test simple CNN
    model = SimpleCNN(num_classes=1)
    model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_parameters()}")
    
    # Test feature extraction
    features = model.get_feature_maps(dummy_input)
    print(f"Feature map shapes: {[(k, v.shape) for k, v in features.items()]}")
    
    # Test deep CNN
    deep_model = DeepCNN(num_classes=7)
    deep_model.to(device)
    
    output_deep = deep_model(dummy_input)
    print(f"Deep CNN output shape: {output_deep.shape}")
    
    total_params = sum(p.numel() for p in deep_model.parameters())
    print(f"Deep CNN parameters: {total_params:,}")