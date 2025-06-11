"""
B.2: Advanced Fusion ViT
Enhanced ViT with sophisticated cross-modal attention fusion strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for video-annotation fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Cross-modal attention forward pass.
        
        Args:
            query: Video features (B, seq_len, embed_dim)
            key: Annotation features (B, seq_len, embed_dim)  
            value: Annotation features (B, seq_len, embed_dim)
        """
        
        B, seq_len, embed_dim = query.shape
        
        # Save residual
        residual = query
        
        # Layer normalization
        query = self.layer_norm1(query)
        key = self.layer_norm1(key)
        value = self.layer_norm1(value)
        
        # Linear projections
        Q = self.q_proj(query)  # (B, seq_len, embed_dim)
        K = self.k_proj(key)    # (B, seq_len, embed_dim)
        V = self.v_proj(value)  # (B, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (B, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(B, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attended_values)
        
        # Residual connection
        output = output + residual
        
        # Feed-forward network with residual
        ffn_input = self.layer_norm2(output)
        ffn_output = self.ffn(ffn_input)
        output = output + ffn_output
        
        return output


class TemporalTransformer(nn.Module):
    """Transformer for processing temporal annotation sequences."""
    
    def __init__(self, input_dim: int, embed_dim: int, num_layers: int = 3, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)  # Max 100 time steps
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal annotation sequence.
        
        Args:
            x: Annotation features (B, seq_len, input_dim)
        """
        
        B, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (B, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class AdvancedFusionViT(nn.Module):
    """
    B.2: Advanced multimodal ViT with sophisticated fusion strategy.
    
    Architecture:
    - Video Branch: Enhanced ViT with attention mechanisms
    - Annotation Branch: Temporal transformer for sequential data
    - Fusion: Cross-modal attention fusion
    - Target: Predict emotional annotations
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 256,  # Increased for better capacity
                 depth: int = 6,        # Deeper for better learning
                 num_heads: int = 8,    # Even number for better attention
                 mlp_ratio: float = 4.0,
                 physical_dim: int = 33,
                 emotional_dim: int = 17,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.physical_dim = physical_dim
        self.emotional_dim = emotional_dim
        
        # Video processing branch (Enhanced ViT)
        self.video_branch = self._build_video_branch(
            img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout
        )
        
        # Annotation processing branch (Temporal Transformer)
        self.annotation_branch = TemporalTransformer(
            input_dim=physical_dim,
            embed_dim=embed_dim,
            num_layers=3,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal fusion
        self.cross_modal_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion classifier
        self.fusion_classifier = self._build_fusion_classifier(
            embed_dim, emotional_dim, dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_video_branch(self, img_size, patch_size, embed_dim, depth, 
                           num_heads, mlp_ratio, dropout):
        """Build enhanced video processing branch."""
        
        num_patches = (img_size // patch_size) ** 2
        
        # Create individual components
        patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # Learnable parameters
        pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )
        
        norm = nn.LayerNorm(embed_dim)
        
        # Store as modules
        video_branch = nn.ModuleDict({
            'patch_embed': patch_embed,
            'transformer': transformer,
            'norm': norm
        })
        
        # Register parameters
        self.register_parameter('video_pos_embed', pos_embed)
        self.register_parameter('video_cls_token', cls_token)
        
        return video_branch
        
    def _build_fusion_classifier(self, embed_dim, emotional_dim, dropout):
        """Build sophisticated fusion classifier."""
        
        fusion_classifier = nn.Sequential(
            # First fusion layer
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second fusion layer
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Classification layer
            nn.Linear(embed_dim // 4, emotional_dim),
            
            # Sigmoid for multi-label classification
            nn.Sigmoid()
        )
        
        return fusion_classifier
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        """Process video frames through enhanced ViT."""
        
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.video_branch['patch_embed'](x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_token = self.video_cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.video_pos_embed
        
        # Transformer processing
        x = self.video_branch['transformer'](x)
        
        # Layer normalization
        x = self.video_branch['norm'](x)
        
        return x  # Return all tokens for attention fusion
        
    def forward_annotations(self, x: torch.Tensor) -> torch.Tensor:
        """Process physical annotations through temporal transformer."""
        
        # Expand to sequence if single frame
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (B, 1, physical_dim)
            
        # Process through temporal transformer
        annotation_features = self.annotation_branch(x)  # (B, seq_len, embed_dim)
        
        return annotation_features
        
    def forward(self, video: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        """Forward pass through advanced multimodal network."""
        
        # Process video frames
        video_features = self.forward_video(video)  # (B, num_patches + 1, embed_dim)
        
        # Process physical annotations
        annotation_features = self.forward_annotations(physical)  # (B, seq_len, embed_dim)
        
        # Cross-modal attention fusion
        # Repeat annotation features to match video sequence length
        video_seq_len = video_features.shape[1]
        annotation_features_expanded = annotation_features.repeat(1, video_seq_len, 1)  # (B, video_seq_len, embed_dim)
        
        # Use video as query, annotations as key/value
        fused_features = self.cross_modal_attention(
            query=video_features,
            key=annotation_features_expanded, 
            value=annotation_features_expanded
        )  # (B, num_patches + 1, embed_dim)
        
        # Global average pooling for final representation
        global_features = fused_features.mean(dim=1)  # (B, embed_dim)
        
        # Emotional prediction
        emotional_predictions = self.fusion_classifier(global_features)  # (B, emotional_dim)
        
        return emotional_predictions
        
    def get_attention_maps(self, video: torch.Tensor, physical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention maps for visualization."""
        
        with torch.no_grad():
            video_features = self.forward_video(video)
            annotation_features = self.forward_annotations(physical)
            
            # Get attention weights from cross-modal attention
            # This would require modifying CrossModalAttention to return weights
            # For now, return feature representations
            
        return {
            'video_features': video_features,
            'annotation_features': annotation_features,
            'video_cls_token': video_features[:, 0],  # CLS token
            'video_patches': video_features[:, 1:],   # Patch tokens
        }


class AdvancedMultimodalLoss(nn.Module):
    """Advanced loss function with attention regularization."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        
        self.alpha = alpha  # Weight for main emotion loss
        self.beta = beta    # Weight for attention regularization
        
        # Main emotion classification loss
        self.emotion_loss = nn.BCELoss(reduction='mean')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                attention_maps: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Compute advanced multimodal loss."""
        
        # Main emotion classification loss
        main_loss = self.emotion_loss(predictions, targets)
        
        total_loss = self.alpha * main_loss
        
        # Optional attention regularization
        if attention_maps is not None and self.beta > 0:
            # Encourage diverse attention patterns
            video_features = attention_maps.get('video_features')
            if video_features is not None:
                # Diversity loss: encourage different attention heads to focus on different aspects
                attention_diversity = -torch.mean(torch.var(video_features, dim=1))
                total_loss += self.beta * attention_diversity
                
        return total_loss