"""
B.1: Naive Multimodal ViT
Extends A.2 (ViT from scratch) with simple concatenation fusion of video and annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

class NaiveMultimodalViT(nn.Module):
    """
    B.1: Naive multimodal approach extending A.2 ViT with simple fusion.
    
    Architecture:
    - Video Branch: TinyViT from scratch (from A.2)
    - Annotation Branch: MLP encoder for physical features
    - Fusion: Simple concatenation before classifier
    - Target: Predict emotional annotations
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 192,
                 depth: int = 4,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 physical_dim: int = 33,  # 28 physical + 5 enhanced features
                 emotional_dim: int = 17,  # 17 emotional targets
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.physical_dim = physical_dim
        self.emotional_dim = emotional_dim
        
        # Video processing branch (from A.2)
        self.video_branch = self._build_video_branch(
            img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout
        )
        
        # Physical annotation processing branch
        self.annotation_branch = self._build_annotation_branch(
            physical_dim, embed_dim, dropout
        )
        
        # Fusion strategy: Simple concatenation
        self.fusion_classifier = self._build_fusion_classifier(
            embed_dim * 2, emotional_dim, dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_video_branch(self, img_size, patch_size, embed_dim, depth, 
                           num_heads, mlp_ratio, dropout):
        """Build video processing branch (TinyViT from A.2)."""
        
        num_patches = (img_size // patch_size) ** 2
        
        # Create individual components
        patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # Learnable parameters (not modules)
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
        
        # Store as attributes instead of ModuleDict
        video_branch = nn.ModuleDict({
            'patch_embed': patch_embed,
            'transformer': transformer,
            'norm': norm
        })
        
        # Register parameters separately
        self.register_parameter('pos_embed', pos_embed)
        self.register_parameter('cls_token', cls_token)
        
        return video_branch
        
    def _build_annotation_branch(self, physical_dim, embed_dim, dropout):
        """Build physical annotation processing branch."""
        
        annotation_branch = nn.Sequential(
            # Input layer
            nn.Linear(physical_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Hidden layer
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Output normalization
            nn.LayerNorm(embed_dim)
        )
        
        return annotation_branch
        
    def _build_fusion_classifier(self, fused_dim, emotional_dim, dropout):
        """Build fusion classifier for emotional prediction."""
        
        fusion_classifier = nn.Sequential(
            # Fusion layer
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Classification layer
            nn.Linear(fused_dim // 2, emotional_dim),
            
            # Sigmoid for multi-label emotion classification
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
        """Process video frames through ViT."""
        
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.video_branch['patch_embed'](x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer processing
        x = self.video_branch['transformer'](x)
        
        # Layer normalization
        x = self.video_branch['norm'](x)
        
        # Extract CLS token representation
        video_features = x[:, 0]  # (B, embed_dim)
        
        return video_features
        
    def forward_annotations(self, x: torch.Tensor) -> torch.Tensor:
        """Process physical annotations through MLP."""
        
        # Physical feature encoding
        annotation_features = self.annotation_branch(x)  # (B, embed_dim)
        
        return annotation_features
        
    def forward(self, video: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        """Forward pass through multimodal network."""
        
        # Process video frames
        video_features = self.forward_video(video)  # (B, embed_dim)
        
        # Process physical annotations
        annotation_features = self.forward_annotations(physical)  # (B, embed_dim)
        
        # Naive fusion: Simple concatenation
        fused_features = torch.cat([video_features, annotation_features], dim=1)  # (B, embed_dim * 2)
        
        # Emotional prediction
        emotional_predictions = self.fusion_classifier(fused_features)  # (B, emotional_dim)
        
        return emotional_predictions
        
    def get_feature_representations(self, video: torch.Tensor, physical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate feature representations for analysis."""
        
        with torch.no_grad():
            video_features = self.forward_video(video)
            annotation_features = self.forward_annotations(physical)
            fused_features = torch.cat([video_features, annotation_features], dim=1)
            
        return {
            'video_features': video_features,
            'annotation_features': annotation_features,
            'fused_features': fused_features
        }


class MultimodalLoss(nn.Module):
    """Loss function for multimodal emotional prediction."""
    
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        
        # Use BCELoss for multi-label emotion classification
        self.bce_loss = nn.BCELoss(reduction='mean')
        
        # Optional positive weight for class imbalance
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multimodal loss."""
        
        # Multi-label emotion classification loss
        emotion_loss = self.bce_loss(predictions, targets)
        
        return emotion_loss


class MultimodalMetrics:
    """Metrics for multimodal emotional prediction."""
    
    def __init__(self, emotional_dim: int = 17, threshold: float = 0.5):
        self.emotional_dim = emotional_dim
        self.threshold = threshold
        
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        
        # Convert to binary predictions
        pred_binary = (predictions > self.threshold).float()
        
        # Per-emotion accuracy
        per_emotion_acc = []
        for i in range(self.emotional_dim):
            emotion_acc = (pred_binary[:, i] == targets[:, i]).float().mean().item()
            per_emotion_acc.append(emotion_acc)
            
        # Overall metrics
        overall_acc = (pred_binary == targets).all(dim=1).float().mean().item()
        element_acc = (pred_binary == targets).float().mean().item()
        
        # Precision, Recall, F1 (macro-averaged)
        epsilon = 1e-8
        
        tp = (pred_binary * targets).sum(dim=0)
        fp = (pred_binary * (1 - targets)).sum(dim=0)
        fn = ((1 - pred_binary) * targets).sum(dim=0)
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        
        return {
            'overall_accuracy': overall_acc,
            'element_accuracy': element_acc,
            'macro_precision': precision.mean().item(),
            'macro_recall': recall.mean().item(),
            'macro_f1': f1.mean().item(),
            'per_emotion_accuracy': per_emotion_acc
        }