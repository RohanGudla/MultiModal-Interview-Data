"""
A.2: Vision Transformer (ViT) from scratch for emotion recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding using convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
        Returns:
            (batch_size, n_patches, embed_dim)
        """
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )
        
        # Final projection
        output = self.projection(attention_output)
        
        return output, attention_weights

class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 3072, dropout: float = 0.1):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, 
                 embed_dim: int = 768,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attention_output, attention_weights = self.attention(norm_x)
        x = x + attention_output
        
        # MLP with residual connection
        norm_x = self.norm2(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output
        
        return x, attention_weights

class VisionTransformer(nn.Module):
    """Vision Transformer for emotion recognition."""
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 embed_dim: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_cls_token: bool = True):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embedding.n_patches
        
        # Positional embedding
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
            
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        # Initialize other parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
        Returns:
            (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, embed_dim)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer encoder
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
            
        # Global representation
        x = self.norm(x)
        
        if self.use_cls_token:
            # Use CLS token
            x = x[:, 0]
        else:
            # Global average pooling
            x = x.mean(dim=1)
            
        # Classification
        x = self.head(x)
        
        return x
        
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        batch_size = x.shape[0]
        
        # Forward pass through patch embedding
        x = self.patch_embedding(x)
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Collect attention weights from all layers
        attention_weights = {}
        for i, block in enumerate(self.transformer_blocks):
            x, attn_weights = block(x)
            attention_weights[f'layer_{i}'] = attn_weights
            
        return attention_weights
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }

def create_vit_model(model_size: str = 'small',
                    num_classes: int = 1,
                    img_size: int = 224,
                    **kwargs) -> VisionTransformer:
    """Factory function to create ViT models of different sizes."""
    
    configs = {
        'tiny': {
            'embed_dim': 192,
            'num_layers': 4,
            'num_heads': 3,
            'mlp_ratio': 4.0
        },
        'small': {
            'embed_dim': 384,
            'num_layers': 6,
            'num_heads': 6,
            'mlp_ratio': 4.0
        },
        'base': {
            'embed_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(configs.keys())}")
        
    config = configs[model_size]
    config.update(kwargs)
    
    return VisionTransformer(
        img_size=img_size,
        num_classes=num_classes,
        **config
    )

if __name__ == "__main__":
    # Test the ViT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different model sizes
    for size in ['tiny', 'small']:
        print(f"\n=== Testing ViT-{size} ===")
        
        model = create_vit_model(size, num_classes=1)
        model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {model.count_parameters()}")
        
        # Test attention visualization
        attention_weights = model.get_attention_weights(dummy_input)
        print(f"Attention weights shapes:")
        for layer, weights in attention_weights.items():
            print(f"  {layer}: {weights.shape}")
            
        # Memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")