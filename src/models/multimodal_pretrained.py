"""
B.3: Pretrained Multimodal ViT
Extends A.4 (Pretrained ViT) with sophisticated multimodal fusion.
Uses the best performing video model with advanced annotation integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple
import math

class PretrainedMultimodalViT(nn.Module):
    """
    B.3: Pretrained multimodal ViT extending A.4 with advanced fusion.
    
    Architecture:
    - Video Branch: ImageNet pretrained ViT-B/16 (from A.4: 100% accuracy)
    - Annotation Branch: Pretrained transformer for time-series
    - Fusion: Multi-head cross-attention with learned fusion weights
    - Target: Achieve superior performance through transfer learning
    """
    
    def __init__(self, 
                 physical_dim: int = 33,
                 emotional_dim: int = 17,
                 embed_dim: int = 768,  # ViT-B/16 embedding dimension
                 fusion_dim: int = 512, # Reduced dimension for fusion
                 num_fusion_heads: int = 8,
                 dropout: float = 0.3):
        
        super().__init__()
        
        self.physical_dim = physical_dim
        self.emotional_dim = emotional_dim
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim
        
        # Video processing branch: Pretrained ViT-B/16 (from A.4)
        self.video_backbone = self._build_pretrained_vit()
        
        # Annotation processing branch: Advanced temporal encoder
        self.annotation_encoder = self._build_annotation_encoder(
            physical_dim, fusion_dim, dropout
        )
        
        # Multi-modal fusion module
        self.fusion_module = self._build_fusion_module(
            embed_dim, fusion_dim, num_fusion_heads, dropout
        )
        
        # Final classification head
        self.classifier = self._build_classifier(
            fusion_dim, emotional_dim, dropout
        )
        
        # Initialize fusion components (backbone is already pretrained)
        self._init_fusion_weights()
        
    def _build_pretrained_vit(self):
        """Build pretrained ViT backbone (from A.4)."""
        
        # Load ImageNet pretrained ViT-B/16
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        backbone = models.vit_b_16(weights=weights)
        
        # Remove the original classification head
        backbone.heads = nn.Identity()
        
        # Initially freeze backbone for stable training
        for param in backbone.parameters():
            param.requires_grad = False
            
        return backbone
        
    def _build_annotation_encoder(self, input_dim, output_dim, dropout):
        """Build sophisticated annotation encoder."""
        
        annotation_encoder = nn.Sequential(
            # Multi-layer feature extraction
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.LayerNorm(input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Dimensionality reduction to fusion dimension
            nn.Linear(input_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout // 2)
        )
        
        return annotation_encoder
        
    def _build_fusion_module(self, video_dim, annotation_dim, num_heads, dropout):
        """Build multi-modal fusion module with learned attention."""
        
        # Video dimension reduction
        video_projector = nn.Sequential(
            nn.Linear(video_dim, annotation_dim),
            nn.LayerNorm(annotation_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention for fusion
        fusion_attention = nn.MultiheadAttention(
            embed_dim=annotation_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion transformer layer
        fusion_transformer = nn.TransformerEncoderLayer(
            d_model=annotation_dim,
            nhead=num_heads,
            dim_feedforward=annotation_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Learned fusion weights
        fusion_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # Initial: video 60%, annotation 40%
        
        fusion_module = nn.ModuleDict({
            'video_projector': video_projector,
            'fusion_attention': fusion_attention,
            'fusion_transformer': fusion_transformer
        })
        
        # Register fusion weights as parameter
        self.register_parameter('fusion_weights', fusion_weights)
        
        return fusion_module
        
    def _build_classifier(self, input_dim, output_dim, dropout):
        """Build final classification head."""
        
        classifier = nn.Sequential(
            # First classification layer
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second classification layer
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            
            # Output layer (no sigmoid - handled by BCEWithLogitsLoss)
            nn.Linear(input_dim // 4, output_dim)
        )
        
        return classifier
        
    def _init_fusion_weights(self):
        """Initialize fusion components."""
        
        for module in [self.annotation_encoder, self.fusion_module['video_projector'], self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                    
    def unfreeze_backbone(self, layers_to_unfreeze: int = 2):
        """Gradually unfreeze backbone layers for fine-tuning."""
        
        # Unfreeze last N encoder layers
        encoder_layers = self.video_backbone.encoder.layers
        for layer in encoder_layers[-layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        print(f"Unfroze last {layers_to_unfreeze} encoder layers")
        
    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        """Process video through pretrained ViT."""
        
        # Use pretrained ViT backbone
        with torch.cuda.amp.autocast():  # Mixed precision for efficiency
            video_features = self.video_backbone(x)  # (B, embed_dim)
            
        return video_features
        
    def forward_annotations(self, x: torch.Tensor) -> torch.Tensor:
        """Process annotations through advanced encoder."""
        
        # Encode physical features
        annotation_features = self.annotation_encoder(x)  # (B, fusion_dim)
        
        return annotation_features
        
    def forward_fusion(self, video_features: torch.Tensor, 
                      annotation_features: torch.Tensor) -> torch.Tensor:
        """Advanced multi-modal fusion."""
        
        # Project video features to fusion dimension
        video_projected = self.fusion_module['video_projector'](video_features)  # (B, fusion_dim)
        
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        video_weight, annotation_weight = weights[0], weights[1]
        
        # Prepare for attention (add sequence dimension)
        video_seq = video_projected.unsqueeze(1)        # (B, 1, fusion_dim)
        annotation_seq = annotation_features.unsqueeze(1)  # (B, 1, fusion_dim)
        
        # Cross-modal attention: video attends to annotations
        attended_video, attention_weights = self.fusion_module['fusion_attention'](
            query=video_seq,
            key=annotation_seq,
            value=annotation_seq
        )  # (B, 1, fusion_dim)
        
        # Weighted fusion
        fused_features = (video_weight * attended_video.squeeze(1) + 
                         annotation_weight * annotation_features)  # (B, fusion_dim)
        
        # Additional transformer processing
        fused_seq = fused_features.unsqueeze(1)  # (B, 1, fusion_dim)
        enhanced_features = self.fusion_module['fusion_transformer'](fused_seq)  # (B, 1, fusion_dim)
        
        return enhanced_features.squeeze(1), attention_weights
        
    def forward(self, video: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        """Forward pass through pretrained multimodal network."""
        
        # Process video frames
        video_features = self.forward_video(video)  # (B, embed_dim)
        
        # Process physical annotations
        annotation_features = self.forward_annotations(physical)  # (B, fusion_dim)
        
        # Multi-modal fusion
        fused_features, attention_weights = self.forward_fusion(
            video_features, annotation_features
        )  # (B, fusion_dim)
        
        # Final classification
        emotional_predictions = self.classifier(fused_features)  # (B, emotional_dim)
        
        return emotional_predictions
        
    def get_fusion_analysis(self, video: torch.Tensor, physical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed fusion analysis for interpretation."""
        
        with torch.no_grad():
            # Get individual modality features
            video_features = self.forward_video(video)
            annotation_features = self.forward_annotations(physical)
            
            # Get fusion components
            fused_features, attention_weights = self.forward_fusion(
                video_features, annotation_features
            )
            
            # Get fusion weights
            weights = F.softmax(self.fusion_weights, dim=0)
            
            return {
                'video_features': video_features,
                'annotation_features': annotation_features,
                'fused_features': fused_features,
                'attention_weights': attention_weights,
                'fusion_weights': weights,
                'video_weight': weights[0].item(),
                'annotation_weight': weights[1].item()
            }


class PretrainedMultimodalLoss(nn.Module):
    """Advanced loss function for pretrained multimodal learning."""
    
    def __init__(self, main_weight: float = 1.0, 
                 attention_weight: float = 0.1,
                 fusion_weight: float = 0.05):
        super().__init__()
        
        self.main_weight = main_weight
        self.attention_weight = attention_weight
        self.fusion_weight = fusion_weight
        
        # Main emotion classification loss (safer for autocast)
        self.emotion_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                fusion_analysis: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Compute comprehensive multimodal loss."""
        
        # Main emotion classification loss
        main_loss = self.emotion_loss(predictions, targets)
        total_loss = self.main_weight * main_loss
        
        if fusion_analysis is not None:
            # Attention diversity loss: encourage meaningful attention patterns
            if 'attention_weights' in fusion_analysis and self.attention_weight > 0:
                attention_weights = fusion_analysis['attention_weights']
                # Encourage attention weights to be neither too concentrated nor too uniform
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), dim=-1
                ).mean()
                total_loss += self.attention_weight * (-attention_entropy)  # Maximize entropy
                
            # Fusion balance loss: encourage balanced use of modalities
            if 'fusion_weights' in fusion_analysis and self.fusion_weight > 0:
                fusion_weights = fusion_analysis['fusion_weights']
                # Encourage balanced but not necessarily equal fusion weights
                balance_target = torch.tensor([0.6, 0.4], device=fusion_weights.device)
                balance_loss = F.mse_loss(fusion_weights, balance_target)
                total_loss += self.fusion_weight * balance_loss
                
        return total_loss


class PretrainedMultimodalMetrics:
    """Enhanced metrics for pretrained multimodal model."""
    
    def __init__(self, emotional_dim: int = 17, threshold: float = 0.5):
        self.emotional_dim = emotional_dim
        self.threshold = threshold
        
    def compute_comprehensive_metrics(self, predictions: torch.Tensor, 
                                    targets: torch.Tensor,
                                    fusion_analysis: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """Compute comprehensive metrics including fusion analysis."""
        
        # Apply sigmoid since model outputs logits
        pred_probs = torch.sigmoid(predictions)
        pred_binary = (pred_probs > self.threshold).float()
        
        # Per-emotion accuracy
        per_emotion_acc = []
        for i in range(self.emotional_dim):
            emotion_acc = (pred_binary[:, i] == targets[:, i]).float().mean().item()
            per_emotion_acc.append(emotion_acc)
            
        # Overall metrics
        overall_acc = (pred_binary == targets).all(dim=1).float().mean().item()
        element_acc = (pred_binary == targets).float().mean().item()
        
        # Precision, Recall, F1
        epsilon = 1e-8
        tp = (pred_binary * targets).sum(dim=0)
        fp = (pred_binary * (1 - targets)).sum(dim=0)
        fn = ((1 - pred_binary) * targets).sum(dim=0)
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        
        metrics = {
            'overall_accuracy': overall_acc,
            'element_accuracy': element_acc,
            'macro_precision': precision.mean().item(),
            'macro_recall': recall.mean().item(),
            'macro_f1': f1.mean().item(),
            'per_emotion_accuracy': per_emotion_acc
        }
        
        # Add fusion analysis if available
        if fusion_analysis is not None:
            metrics.update({
                'video_weight': fusion_analysis.get('video_weight', 0.0),
                'annotation_weight': fusion_analysis.get('annotation_weight', 0.0),
                'fusion_balance': abs(fusion_analysis.get('video_weight', 0.5) - 0.6)  # How far from target 60/40
            })
            
        return metrics