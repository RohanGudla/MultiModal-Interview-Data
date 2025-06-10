"""
A.3: Pretrained ResNet50 with fine-tuning for emotion recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, List, Any

class PretrainedResNet(nn.Module):
    """ResNet with pretrained ImageNet weights for emotion recognition."""
    
    def __init__(self,
                 num_classes: int = 1,
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 freeze_layers: int = 45,
                 dropout_rate: float = 0.5,
                 use_additional_layers: bool = True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Whether to freeze the backbone initially
            freeze_layers: Number of layers to freeze (0-45 for ResNet50)
            dropout_rate: Dropout probability in classification head
            use_additional_layers: Add extra FC layers in head
        """
        super(PretrainedResNet, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
            
        # Get feature dimension
        in_features = self.backbone.fc.in_features
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classification head
        if use_additional_layers:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone_layers(freeze_layers)
            
        # Initialize new layers
        self._initialize_new_layers()
        
    def freeze_backbone_layers(self, num_layers: int):
        """Freeze specified number of backbone layers."""
        layers = list(self.backbone.children())
        
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        print(f"Frozen first {num_layers} layers of ResNet backbone")
        
    def unfreeze_backbone_layers(self, num_layers: Optional[int] = None):
        """Unfreeze backbone layers for fine-tuning."""
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
        else:
            # Unfreeze last num_layers
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers):
                if i >= len(layers) - num_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            print(f"Unfroze last {num_layers} backbone layers")
            
    def _initialize_new_layers(self):
        """Initialize newly added layers."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Backbone feature extraction
        features = self.backbone(x)  # (batch_size, 2048, 1, 1)
        
        # Classification
        output = self.classifier(features)
        
        return output
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        with torch.no_grad():
            features = self.backbone(x)
            features = features.flatten(1)
        return features
        
    def get_layer_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate features for analysis."""
        features = {}
        
        # Process through backbone layers
        layers = list(self.backbone.children())
        
        for i, layer in enumerate(layers):
            x = layer(x)
            if i in [0, 1, 4, 5, 6, 7]:  # Key feature points
                features[f'layer_{i}'] = x.clone()
                
        return features
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count backbone vs classifier parameters
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'backbone_parameters': backbone_params,
            'classifier_parameters': classifier_params
        }

class EfficientNetPretrained(nn.Module):
    """EfficientNet alternative for comparison."""
    
    def __init__(self,
                 num_classes: int = 1,
                 model_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super(EfficientNetPretrained, self).__init__()
        
        self.num_classes = num_classes
        
        # Load EfficientNet (requires timm library)
        try:
            import timm
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'
            )
            
            # Get feature dimension
            in_features = self.backbone.num_features
            
        except ImportError:
            raise ImportError("timm library required for EfficientNet. Install with: pip install timm")
            
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class MultiTaskResNet(nn.Module):
    """ResNet with multiple output heads for multi-task learning."""
    
    def __init__(self,
                 task_configs: Dict[str, int],
                 pretrained: bool = True,
                 shared_layers: int = 4,
                 dropout_rate: float = 0.5):
        """
        Args:
            task_configs: Dict mapping task names to number of classes
                         e.g., {'attention': 1, 'emotions': 7}
            pretrained: Use pretrained weights
            shared_layers: Number of shared backbone layers
            dropout_rate: Dropout rate
        """
        super(MultiTaskResNet, self).__init__()
        
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Load ResNet backbone
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
            
        # Shared feature extractor
        self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Task-specific heads
        in_features = resnet.fc.in_features
        self.task_heads = nn.ModuleDict()
        
        for task_name, num_classes in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(512, num_classes)
            )
            
        self._initialize_task_heads()
        
    def _initialize_task_heads(self):
        """Initialize task-specific heads."""
        for head in self.task_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all task outputs."""
        # Shared feature extraction
        shared_features = self.shared_backbone(x)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)
            
        return outputs
        
    def forward_single_task(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """Forward pass for a single task."""
        shared_features = self.shared_backbone(x)
        return self.task_heads[task_name](shared_features)

def create_resnet_model(model_type: str = 'resnet50',
                       num_classes: int = 1,
                       pretrained: bool = True,
                       **kwargs) -> nn.Module:
    """Factory function to create ResNet-based models."""
    
    if model_type == 'resnet50':
        return PretrainedResNet(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == 'efficientnet':
        return EfficientNetPretrained(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == 'multitask':
        return MultiTaskResNet(
            task_configs={'attention': 1, 'emotions': 7},
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_learning_rates_for_layers(model: PretrainedResNet, 
                                base_lr: float = 1e-4,
                                backbone_lr_ratio: float = 0.1) -> List[Dict]:
    """Get different learning rates for backbone and classifier."""
    
    params = [
        {
            'params': model.backbone.parameters(),
            'lr': base_lr * backbone_lr_ratio
        },
        {
            'params': model.classifier.parameters(),
            'lr': base_lr
        }
    ]
    
    return params

if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ResNet50
    print("=== Testing PretrainedResNet ===")
    model = PretrainedResNet(num_classes=1, pretrained=True)
    model.to(device)
    
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_parameters()}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    # Test layer features
    layer_features = model.get_layer_features(dummy_input)
    print(f"Layer features: {[(k, v.shape) for k, v in layer_features.items()]}")
    
    # Test multi-task model
    print("\n=== Testing MultiTaskResNet ===")
    multitask_model = MultiTaskResNet(
        task_configs={'attention': 1, 'emotions': 7},
        pretrained=True
    )
    multitask_model.to(device)
    
    multitask_output = multitask_model(dummy_input)
    print(f"Multitask outputs: {[(k, v.shape) for k, v in multitask_output.items()]}")
    
    # Test parameter groups for different learning rates
    param_groups = get_learning_rates_for_layers(model)
    print(f"Parameter groups: {len(param_groups)} groups")
    for i, group in enumerate(param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {num_params:,} parameters, lr={group['lr']}")
        
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")