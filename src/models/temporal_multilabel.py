#!/usr/bin/env python3
"""
Temporal Multi-Label Models for Video Annotation
Handles temporal sequence modeling and boundary detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np

class TemporalMultiLabelViT(nn.Module):
    """
    Vision Transformer with temporal modeling for multi-label prediction
    Predicts all 50 annotation features with temporal understanding
    """
    
    def __init__(self,
                 num_physical_features: int = 33,
                 num_emotional_features: int = 17,
                 sequence_length: int = 10,
                 hidden_dim: int = 768,
                 num_temporal_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_physical_features = num_physical_features
        self.num_emotional_features = num_emotional_features
        self.num_total_features = num_physical_features + num_emotional_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Spatial feature extractor (pretrained ViT)
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.spatial_encoder = models.vit_b_16(weights=weights)
        
        # Remove the classification head
        self.spatial_encoder.heads = nn.Identity()
        
        # Freeze spatial encoder initially
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False
        
        # Temporal modeling layers
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_temporal_layers
        )
        
        # Positional encoding for temporal sequences
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, sequence_length, hidden_dim) * 0.1
        )
        
        # Multi-label prediction heads
        self.physical_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, num_physical_features)
        )
        
        self.emotional_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, num_emotional_features)
        )
        
        # Temporal boundary detection head
        self.boundary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 4, self.num_total_features * 2)  # start/stop for each feature
        )
        
    def extract_spatial_features(self, images):
        """Extract spatial features from image sequences"""
        batch_size, seq_len, c, h, w = images.shape
        
        # Reshape for batch processing
        images_flat = images.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.spatial_encoder(images_flat)  # [batch_size * seq_len, hidden_dim]
        
        # Reshape back to sequences
        features = features.view(batch_size, seq_len, self.hidden_dim)
        
        return features
    
    def forward(self, images, return_temporal_boundaries=False):
        """
        Forward pass for temporal multi-label prediction
        
        Args:
            images: [batch_size, sequence_length, 3, 224, 224]
            return_temporal_boundaries: Whether to return boundary predictions
        
        Returns:
            Dict with predictions for physical, emotional, and optionally boundaries
        """
        batch_size, seq_len = images.shape[:2]
        
        # Extract spatial features
        spatial_features = self.extract_spatial_features(images)
        
        # Add temporal positional encoding
        if seq_len <= self.sequence_length:
            pos_encoding = self.temporal_pos_encoding[:, :seq_len, :]
        else:
            # Interpolate positional encoding for longer sequences
            pos_encoding = F.interpolate(
                self.temporal_pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear'
            ).transpose(1, 2)
        
        temporal_features = spatial_features + pos_encoding
        
        # Temporal modeling
        temporal_output = self.temporal_encoder(temporal_features)
        
        # Use the last timestep for predictions (or pool over all timesteps)
        final_features = temporal_output[:, -1, :]  # [batch_size, hidden_dim]
        
        # Multi-label predictions
        physical_logits = self.physical_head(final_features)
        emotional_logits = self.emotional_head(final_features)
        
        # Apply sigmoid for multi-label classification
        physical_probs = torch.sigmoid(physical_logits)
        emotional_probs = torch.sigmoid(emotional_logits)
        
        results = {
            'physical_logits': physical_logits,
            'emotional_logits': emotional_logits,
            'physical_probs': physical_probs,
            'emotional_probs': emotional_probs,
            'combined_probs': torch.cat([physical_probs, emotional_probs], dim=-1)
        }
        
        # Temporal boundary predictions
        if return_temporal_boundaries:
            boundary_logits = self.boundary_head(final_features)
            boundary_logits = boundary_logits.view(batch_size, self.num_total_features, 2)
            
            # Apply sigmoid to get probabilities for start/stop
            boundary_probs = torch.sigmoid(boundary_logits)
            
            results.update({
                'boundary_logits': boundary_logits,
                'boundary_probs': boundary_probs,
                'start_probs': boundary_probs[:, :, 0],
                'stop_probs': boundary_probs[:, :, 1]
            })
        
        return results
    
    def unfreeze_spatial_encoder(self):
        """Unfreeze spatial encoder for fine-tuning"""
        for param in self.spatial_encoder.parameters():
            param.requires_grad = True

class TemporalMultiLabelResNet(nn.Module):
    """
    ResNet-based temporal multi-label model
    Alternative architecture using CNN backbone
    """
    
    def __init__(self,
                 num_physical_features: int = 33,
                 num_emotional_features: int = 17,
                 sequence_length: int = 10,
                 hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_physical_features = num_physical_features
        self.num_emotional_features = num_emotional_features
        self.num_total_features = num_physical_features + num_emotional_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Spatial feature extractor (pretrained ResNet50)
        self.spatial_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final classification layer
        self.spatial_encoder.fc = nn.Identity()
        spatial_features_dim = 2048
        
        # Project spatial features to hidden dimension
        self.spatial_projection = nn.Linear(spatial_features_dim, hidden_dim)
        
        # Freeze spatial encoder initially
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False
        
        # Temporal modeling with LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Multi-label prediction heads
        self.physical_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_physical_features)
        )
        
        self.emotional_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotional_features)
        )
        
    def extract_spatial_features(self, images):
        """Extract spatial features from image sequences"""
        batch_size, seq_len, c, h, w = images.shape
        
        # Reshape for batch processing
        images_flat = images.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.spatial_encoder(images_flat)  # [batch_size * seq_len, 2048]
        
        # Project to hidden dimension
        features = self.spatial_projection(features)  # [batch_size * seq_len, hidden_dim]
        
        # Reshape back to sequences
        features = features.view(batch_size, seq_len, self.hidden_dim)
        
        return features
    
    def forward(self, images, return_temporal_boundaries=False):
        """Forward pass for temporal multi-label prediction"""
        
        # Extract spatial features
        spatial_features = self.extract_spatial_features(images)
        
        # Temporal modeling with LSTM
        temporal_output, _ = self.temporal_lstm(spatial_features)
        
        # Use the last timestep for predictions
        final_features = temporal_output[:, -1, :]  # [batch_size, lstm_output_dim]
        
        # Multi-label predictions
        physical_logits = self.physical_head(final_features)
        emotional_logits = self.emotional_head(final_features)
        
        # Apply sigmoid for multi-label classification
        physical_probs = torch.sigmoid(physical_logits)
        emotional_probs = torch.sigmoid(emotional_logits)
        
        results = {
            'physical_logits': physical_logits,
            'emotional_logits': emotional_logits,
            'physical_probs': physical_probs,
            'emotional_probs': emotional_probs,
            'combined_probs': torch.cat([physical_probs, emotional_probs], dim=-1)
        }
        
        # Note: Boundary detection not implemented for ResNet model
        if return_temporal_boundaries:
            batch_size = images.shape[0]
            num_total_features = self.num_total_features
            # Return dummy boundary predictions
            dummy_boundaries = torch.zeros(batch_size, num_total_features, 2, device=images.device)
            results.update({
                'boundary_logits': dummy_boundaries,
                'boundary_probs': torch.sigmoid(dummy_boundaries),
                'start_probs': torch.sigmoid(dummy_boundaries[:, :, 0]),
                'stop_probs': torch.sigmoid(dummy_boundaries[:, :, 1])
            })
        
        return results
    
    def unfreeze_spatial_encoder(self):
        """Unfreeze spatial encoder for fine-tuning"""
        for param in self.spatial_encoder.parameters():
            param.requires_grad = True

class MultilabelTemporalLoss(nn.Module):
    """
    Comprehensive loss function for temporal multi-label prediction
    """
    
    def __init__(self,
                 physical_weight: float = 1.0,
                 emotional_weight: float = 1.0,
                 boundary_weight: float = 0.5,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.physical_weight = physical_weight
        self.emotional_weight = emotional_weight
        self.boundary_weight = boundary_weight
        
        # Binary cross entropy for multi-label classification
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, predictions, targets):
        """
        Calculate comprehensive loss
        
        Args:
            predictions: Dict with model outputs
            targets: Dict with ground truth labels
        
        Returns:
            Dict with individual and total losses
        """
        losses = {}
        
        # Physical features loss
        if 'physical_logits' in predictions and 'physical_labels' in targets:
            losses['physical'] = self.bce_loss(
                predictions['physical_logits'],
                targets['physical_labels']
            ) * self.physical_weight
        
        # Emotional features loss
        if 'emotional_logits' in predictions and 'emotional_labels' in targets:
            losses['emotional'] = self.bce_loss(
                predictions['emotional_logits'],
                targets['emotional_labels']
            ) * self.emotional_weight
        
        # Temporal boundary loss (if available)
        if 'boundary_logits' in predictions and 'boundary_labels' in targets:
            losses['boundary'] = self.bce_loss(
                predictions['boundary_logits'].view(-1, predictions['boundary_logits'].shape[-1]),
                targets['boundary_labels'].view(-1, targets['boundary_labels'].shape[-1])
            ) * self.boundary_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

def create_temporal_model(model_type: str = 'vit',
                         num_physical_features: int = 33,
                         num_emotional_features: int = 17,
                         sequence_length: int = 10,
                         **kwargs):
    """
    Factory function to create temporal models
    
    Args:
        model_type: 'vit' or 'resnet'
        num_physical_features: Number of physical annotation features
        num_emotional_features: Number of emotional annotation features
        sequence_length: Length of input sequences
        **kwargs: Additional model parameters
    
    Returns:
        Temporal multi-label model
    """
    
    if model_type.lower() == 'vit':
        return TemporalMultiLabelViT(
            num_physical_features=num_physical_features,
            num_emotional_features=num_emotional_features,
            sequence_length=sequence_length,
            **kwargs
        )
    elif model_type.lower() == 'resnet':
        return TemporalMultiLabelResNet(
            num_physical_features=num_physical_features,
            num_emotional_features=num_emotional_features,
            sequence_length=sequence_length,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Test the temporal models
def test_temporal_models():
    """Test temporal model architectures"""
    
    print("🧪 Testing Temporal Multi-Label Models...")
    
    batch_size = 4
    sequence_length = 8
    num_physical = 33
    num_emotional = 17
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, sequence_length, 3, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test ViT model
    print("\n1️⃣ Testing Temporal ViT:")
    vit_model = create_temporal_model('vit', 
                                     num_physical_features=num_physical,
                                     num_emotional_features=num_emotional,
                                     sequence_length=sequence_length)
    
    with torch.no_grad():
        vit_output = vit_model(dummy_input, return_temporal_boundaries=True)
    
    print(f"   Physical probs shape: {vit_output['physical_probs'].shape}")
    print(f"   Emotional probs shape: {vit_output['emotional_probs'].shape}")
    print(f"   Combined probs shape: {vit_output['combined_probs'].shape}")
    print(f"   Boundary probs shape: {vit_output['boundary_probs'].shape}")
    
    # Test ResNet model
    print("\n2️⃣ Testing Temporal ResNet:")
    resnet_model = create_temporal_model('resnet',
                                        num_physical_features=num_physical,
                                        num_emotional_features=num_emotional,
                                        sequence_length=sequence_length)
    
    with torch.no_grad():
        resnet_output = resnet_model(dummy_input)
    
    print(f"   Physical probs shape: {resnet_output['physical_probs'].shape}")
    print(f"   Emotional probs shape: {resnet_output['emotional_probs'].shape}")
    print(f"   Combined probs shape: {resnet_output['combined_probs'].shape}")
    
    # Test loss function
    print("\n3️⃣ Testing Loss Function:")
    criterion = MultilabelTemporalLoss()
    
    # Create dummy targets
    targets = {
        'physical_labels': torch.randint(0, 2, (batch_size, num_physical), dtype=torch.float32),
        'emotional_labels': torch.randint(0, 2, (batch_size, num_emotional), dtype=torch.float32)
    }
    
    losses = criterion(vit_output, targets)
    print(f"   Physical loss: {losses['physical']:.4f}")
    print(f"   Emotional loss: {losses['emotional']:.4f}")
    print(f"   Total loss: {losses['total']:.4f}")
    
    print("\n✅ All temporal model tests passed!")

if __name__ == "__main__":
    test_temporal_models()