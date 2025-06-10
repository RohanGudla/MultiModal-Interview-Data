"""
A.4: Pretrained Vision Transformer for emotion recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Tuple
import numpy as np

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not available. Install with: pip install timm")

class PretrainedViT(nn.Module):
    """Pretrained Vision Transformer for emotion recognition."""
    
    def __init__(self,
                 num_classes: int = 1,
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 freeze_layers: int = 10,
                 dropout_rate: float = 0.3,
                 use_additional_layers: bool = True):
        """
        Args:
            num_classes: Number of output classes
            model_name: Name of the ViT model to use
            pretrained: Use pretrained weights
            freeze_backbone: Whether to freeze backbone initially
            freeze_layers: Number of transformer blocks to freeze
            dropout_rate: Dropout rate in classification head
            use_additional_layers: Add extra layers in classification head
        """
        super(PretrainedViT, self).__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm library required. Install with: pip install timm")
            
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_layers = freeze_layers
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='token'  # Use [CLS] token
        )
        
        # Get feature dimension
        self.embed_dim = self.backbone.embed_dim
        
        # Freeze specified layers
        if freeze_backbone:
            self.freeze_backbone_layers(freeze_layers)
            
        # Custom classification head
        if use_additional_layers:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(self.embed_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.7),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(self.embed_dim, num_classes)
            )
            
        self._initialize_classifier()
        
    def freeze_backbone_layers(self, num_layers: int):
        """Freeze specified number of transformer blocks."""
        # Freeze patch embedding and positional embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        for param in self.backbone.pos_embed.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'cls_token'):
            for param in self.backbone.cls_token.parameters():
                param.requires_grad = False
                
        # Freeze transformer blocks
        total_blocks = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
                    
        # Always keep norm layer trainable
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        frozen_params = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.backbone.parameters())
        print(f"Frozen {frozen_params}/{total_params} backbone parameters")
        
    def unfreeze_backbone_layers(self, num_layers: Optional[int] = None):
        """Unfreeze backbone layers for fine-tuning."""
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
        else:
            # Unfreeze last num_layers blocks
            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - num_layers)
            
            for i, block in enumerate(self.backbone.blocks):
                if i >= start_idx:
                    for param in block.parameters():
                        param.requires_grad = True
                        
            print(f"Unfroze last {num_layers} transformer blocks")
            
    def _initialize_classifier(self):
        """Initialize classification head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Backbone feature extraction
        features = self.backbone(x)  # (batch_size, embed_dim)
        
        # Classification
        output = self.classifier(features)
        
        return output
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        with torch.no_grad():
            features = self.backbone(x)
        return features
        
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        attention_weights = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'attn'):
                    # For transformer blocks with attention
                    attention_weights[name] = module.attn.attention_weights
            return hook
            
        # Register hooks
        hooks = []
        for i, block in enumerate(self.backbone.blocks):
            hook = block.register_forward_hook(hook_fn(f'block_{i}'))
            hooks.append(hook)
            
        # Forward pass
        with torch.no_grad():
            _ = self.backbone(x)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return attention_weights
        
    def visualize_attention(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Get attention map for visualization."""
        if not hasattr(self.backbone, 'blocks'):
            raise ValueError("Model doesn't support attention visualization")
            
        # Get attention weights
        attention_weights = self.get_attention_weights(x)
        
        if not attention_weights:
            print("Warning: No attention weights captured")
            return None
            
        # Select layer
        layer_name = f'block_{layer_idx}' if layer_idx >= 0 else list(attention_weights.keys())[layer_idx]
        
        if layer_name not in attention_weights:
            print(f"Layer {layer_name} not found. Available: {list(attention_weights.keys())}")
            return None
            
        attn = attention_weights[layer_name]  # (batch, heads, seq_len, seq_len)
        
        # Average across heads and get attention to CLS token
        attn_map = attn.mean(dim=1)  # (batch, seq_len, seq_len)
        attn_to_cls = attn_map[:, 0, 1:]  # (batch, num_patches)
        
        return attn_to_cls
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'backbone_parameters': backbone_params,
            'classifier_parameters': classifier_params
        }

class MultiTaskViT(nn.Module):
    """ViT with multiple output heads for multi-task learning."""
    
    def __init__(self,
                 task_configs: Dict[str, int],
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 freeze_layers: int = 8,
                 dropout_rate: float = 0.3):
        """
        Args:
            task_configs: Dict mapping task names to number of classes
            model_name: ViT model name
            pretrained: Use pretrained weights
            freeze_layers: Number of layers to freeze
            dropout_rate: Dropout rate
        """
        super(MultiTaskViT, self).__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm library required")
            
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Shared backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='token'
        )
        
        embed_dim = self.backbone.embed_dim
        
        # Freeze layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
            
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
        self._initialize_task_heads()
        
    def _freeze_layers(self, num_layers: int):
        """Freeze specified layers."""
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
            
        for i, block in enumerate(self.backbone.blocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False
                    
    def _initialize_task_heads(self):
        """Initialize task-specific heads."""
        for head in self.task_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all task outputs."""
        # Shared feature extraction
        shared_features = self.backbone(x)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)
            
        return outputs

class ViTWithGradCAM(PretrainedViT):
    """ViT with Grad-CAM visualization support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        """Save gradients for Grad-CAM."""
        self.gradients = grad
        
    def save_activation(self, activation):
        """Save activations for Grad-CAM."""
        self.activations = activation
        
    def forward_with_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with Grad-CAM support."""
        # Register hooks for last attention layer
        last_block = self.backbone.blocks[-1]
        
        def forward_hook(module, input, output):
            self.save_activation(output[0])  # Save attention output
            
        def backward_hook(module, grad_input, grad_output):
            self.save_gradient(grad_output[0])
            
        forward_handle = last_block.register_forward_hook(forward_hook)
        backward_handle = last_block.register_backward_hook(backward_hook)
        
        # Forward pass
        output = self.forward(x)
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return output, self.activations
        
    def generate_cam(self, class_idx: Optional[int] = None) -> torch.Tensor:
        """Generate Grad-CAM visualization."""
        if self.gradients is None or self.activations is None:
            raise ValueError("Must call forward_with_cam first")
            
        # Global average pool gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weight the activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        return cam

def create_vit_model(model_type: str = 'base',
                    num_classes: int = 1,
                    pretrained: bool = True,
                    **kwargs) -> nn.Module:
    """Factory function to create ViT models."""
    
    model_configs = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224', 
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224'
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Model type {model_type} not supported")
        
    model_name = model_configs[model_type]
    
    return PretrainedViT(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        **kwargs
    )

def get_learning_rates_for_vit(model: PretrainedViT,
                              base_lr: float = 1e-5,
                              backbone_lr_ratio: float = 0.1) -> List[Dict]:
    """Get different learning rates for ViT layers."""
    
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
    if not HAS_TIMM:
        print("timm not available, skipping tests")
        exit()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ViT models
    for model_type in ['tiny', 'base']:
        print(f"\n=== Testing ViT-{model_type} ===")
        
        model = create_vit_model(model_type, num_classes=1, pretrained=True)
        model.to(device)
        
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {model.count_parameters()}")
        
        # Test feature extraction
        features = model.get_features(dummy_input)
        print(f"Feature shape: {features.shape}")
        
        # Test attention visualization
        try:
            attention_weights = model.get_attention_weights(dummy_input)
            print(f"Attention layers: {list(attention_weights.keys())}")
        except Exception as e:
            print(f"Attention visualization failed: {e}")
            
    # Test MultiTask ViT
    print("\n=== Testing MultiTaskViT ===")
    multitask_model = MultiTaskViT(
        task_configs={'attention': 1, 'emotions': 7},
        model_name='vit_tiny_patch16_224',
        pretrained=True
    )
    multitask_model.to(device)
    
    multitask_output = multitask_model(dummy_input)
    print(f"Multitask outputs: {[(k, v.shape) for k, v in multitask_output.items()]}")
    
    # Test parameter groups
    param_groups = get_learning_rates_for_vit(model)
    print(f"Parameter groups: {len(param_groups)} groups")
    for i, group in enumerate(param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {num_params:,} parameters, lr={group['lr']}")
        
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")