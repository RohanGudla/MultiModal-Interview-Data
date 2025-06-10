#!/usr/bin/env python3
"""
Train pretrained ViT model using real GENEX frame data.
Uses torchvision's ViT since timm is not available.
"""
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import models

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config

class TorchVisionViT(nn.Module):
    """ViT using torchvision's pretrained models."""
    
    def __init__(self, num_classes=1, dropout_rate=0.3, freeze_layers=8):
        super(TorchVisionViT, self).__init__()
        
        # Load pretrained ViT from torchvision
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Get original embedding dimension
        self.embed_dim = self.backbone.heads.head.in_features
        
        # Remove original classification head
        self.backbone.heads = nn.Identity()
        
        # Freeze some layers
        self.freeze_backbone_layers(freeze_layers)
        
        # Custom classification head
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
        
        self._initialize_classifier()
    
    def freeze_backbone_layers(self, num_layers):
        """Freeze specified number of transformer blocks."""
        # Freeze patch embedding
        for param in self.backbone.conv_proj.parameters():
            param.requires_grad = False
        
        # Freeze class token and positional embedding
        self.backbone.class_token.requires_grad = False
        self.backbone.encoder.pos_embedding.requires_grad = False
        
        # Freeze transformer blocks
        total_blocks = len(self.backbone.encoder.layers)
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print(f"Frozen first {num_layers}/{total_blocks} transformer blocks")
    
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
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)  # (batch_size, embed_dim)
        output = self.classifier(features)
        return output
    
    def count_parameters(self):
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

class ImprovedFrameDataset(Dataset):
    """Dataset with data augmentation for ViT training."""
    
    def __init__(self, frame_data, labels, augment=False):
        self.frame_paths = frame_data
        self.labels = labels
        self.augment = augment
        
        # Base transforms
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        # Light augmentation for ViT (they're sensitive to heavy augmentation)
        if augment:
            aug_transforms = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]
            self.transform = transforms.Compose(aug_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def load_real_frame_data():
    """Load real frame data for ViT training."""
    frame_base = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames")
    summary_path = frame_base / "processing_summary.json"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    all_frames = []
    all_labels = []
    all_participants = []
    
    for participant_id, data in summary.items():
        frames = [Path(p) for p in data['frames']]
        
        # Create labels based on participant patterns
        if participant_id in ["CP 0636", "NS 4013", "MP 5114"]:
            labels = [1.0] * len(frames)
        else:
            labels = [0.0] * len(frames)
        
        all_frames.extend(frames)
        all_labels.extend(labels)
        all_participants.extend([participant_id] * len(frames))
    
    return all_frames, all_labels, all_participants

def create_improved_splits(all_frames, all_labels, all_participants):
    """Create stratified train/val splits."""
    train_frames, val_frames, train_labels, val_labels, train_participants, val_participants = train_test_split(
        all_frames, all_labels, all_participants,
        test_size=0.3,
        stratify=all_labels,
        random_state=42
    )
    
    print(f"📈 ViT Pretrained Train split: {len(train_frames)} frames")
    print(f"📊 ViT Pretrained Val split: {len(val_frames)} frames")
    
    train_pos = sum(train_labels)
    val_pos = sum(val_labels)
    print(f"🎯 Train: {train_pos} positive, {len(train_labels)-train_pos} negative")
    print(f"🎯 Val: {val_pos} positive, {len(val_labels)-val_pos} negative")
    
    return train_frames, train_labels, val_frames, val_labels

def get_parameter_groups(model, base_lr=1e-5, backbone_lr_ratio=0.1):
    """Get different learning rates for backbone vs classifier."""
    return [
        {
            'params': model.backbone.parameters(),
            'lr': base_lr * backbone_lr_ratio
        },
        {
            'params': model.classifier.parameters(),
            'lr': base_lr
        }
    ]

def train_vit_pretrained_model():
    """Train pretrained ViT model."""
    config = Config()
    device = config.DEVICE
    
    print(f"🚀 Training PRETRAINED ViT with real data")
    print(f"💻 Device: {device}")
    
    # Load data
    all_frames, all_labels, all_participants = load_real_frame_data()
    
    # Create splits
    train_frames, train_labels, val_frames, val_labels = create_improved_splits(
        all_frames, all_labels, all_participants
    )
    
    # Create datasets with light augmentation
    train_dataset = ImprovedFrameDataset(train_frames, train_labels, augment=True)
    val_dataset = ImprovedFrameDataset(val_frames, val_labels, augment=False)
    
    # Create data loaders - smaller batch size for ViT
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize pretrained ViT
    model = TorchVisionViT(
        num_classes=1,
        dropout_rate=0.3,
        freeze_layers=8  # Freeze first 8 transformer blocks
    )
    model = model.to(device)
    
    # Print model info
    param_count = model.count_parameters()
    print(f"📊 ViT Model: {param_count['total_parameters']:,} total parameters")
    print(f"📊 ViT Trainable: {param_count['trainable_parameters']:,} parameters")
    print(f"📊 ViT Frozen: {param_count['frozen_parameters']:,} parameters")
    
    # Training setup with different learning rates
    criterion = nn.BCEWithLogitsLoss()
    
    # Phase 1: Train only classifier
    phase1_epochs = 20
    phase2_epochs = 30
    
    # Phase 1 optimizer (only classifier)
    param_groups = get_parameter_groups(
        model, 
        base_lr=1e-3,  # Higher LR for classifier
        backbone_lr_ratio=0.0  # Frozen backbone
    )
    optimizer_phase1 = optim.AdamW(param_groups, weight_decay=0.1)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n🎯 PHASE 1: Training classifier only ({phase1_epochs} epochs)")
    
    # Phase 1: Train classifier only
    for epoch in range(phase1_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer_phase1.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping (important for ViT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer_phase1.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                
                val_loss += criterion(output, target).item()
                
                # Calculate accuracy
                predicted = (torch.sigmoid(output) > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"P1-Epoch {epoch+1:2d}/{phase1_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}% | "
              f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Phase 1 early stopping at epoch {epoch+1}")
            break
    
    # Phase 2: Fine-tune with unfrozen layers
    print(f"\n🎯 PHASE 2: Fine-tuning with backbone ({phase2_epochs} epochs)")
    
    # Unfreeze some backbone layers for fine-tuning
    total_blocks = len(model.backbone.encoder.layers)
    unfreeze_blocks = 3  # Unfreeze last 3 blocks
    
    for i, layer in enumerate(model.backbone.encoder.layers):
        if i >= total_blocks - unfreeze_blocks:
            for param in layer.parameters():
                param.requires_grad = True
    
    print(f"Unfroze last {unfreeze_blocks} transformer blocks")
    
    # Phase 2 optimizer with lower learning rates
    param_groups = get_parameter_groups(
        model, 
        base_lr=1e-5,  # Much lower LR for fine-tuning
        backbone_lr_ratio=0.1  # Very low LR for backbone
    )
    optimizer_phase2 = optim.AdamW(param_groups, weight_decay=0.1)
    
    # Learning rate scheduler for phase 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=phase2_epochs, eta_min=1e-7
    )
    
    # Reset patience for phase 2
    best_val_loss = min(val_losses)
    patience_counter = 0
    
    for epoch in range(phase2_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer_phase2.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer_phase2.step()
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                
                val_loss += criterion(output, target).item()
                
                # Calculate accuracy
                predicted = (torch.sigmoid(output) > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"P2-Epoch {epoch+1:2d}/{phase2_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}% | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Phase 2 early stopping at epoch {epoch+1}")
            break
    
    # Calculate final metrics
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_val_accuracy = val_accuracies[-1]
    best_val_accuracy = max(val_accuracies)
    
    # Overfitting detection
    train_val_gap = abs(final_val_accuracy - (100 * (1 - final_train_loss)))
    is_overfitting = train_val_gap > 15
    
    results = {
        'model_name': 'vit_pretrained',
        'model_type': 'Pretrained ViT-B/16 with fine-tuning',
        'backbone_source': 'torchvision (ImageNet pretrained)',
        'total_parameters': param_count['total_parameters'],
        'trainable_parameters': param_count['trainable_parameters'],
        'frozen_parameters': param_count['frozen_parameters'],
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'train_val_gap': train_val_gap,
        'is_overfitting': is_overfitting,
        'epochs_trained': len(train_losses),
        'phase1_epochs': min(phase1_epochs, len([l for l in train_losses])),
        'phase2_epochs': max(0, len(train_losses) - phase1_epochs),
        'train_samples': len(train_frames),
        'val_samples': len(val_frames),
        'using_real_data': True,
        'training_strategy': 'Two-phase: classifier first, then fine-tuning',
        'improvements_applied': [
            'Pretrained ImageNet ViT-B/16 weights',
            'Two-phase training strategy',
            'Different learning rates for backbone vs classifier',
            'Gradual unfreezing of transformer blocks',
            'Light data augmentation (ViT-friendly)',
            'AdamW optimizer with weight decay',
            'Cosine annealing learning rate schedule',
            'Gradient clipping',
            'Early stopping per phase'
        ]
    }
    
    print(f"\n🎯 PRETRAINED ViT RESULTS")
    print(f"✅ Model: {results['model_type']}")
    print(f"✅ Total Parameters: {results['total_parameters']:,}")
    print(f"✅ Trainable: {results['trainable_parameters']:,}")
    print(f"✅ Train Loss: {final_train_loss:.4f}")
    print(f"✅ Val Loss: {final_val_loss:.4f}")
    print(f"✅ Final Val Accuracy: {final_val_accuracy:.1f}%")
    print(f"✅ Best Val Accuracy: {best_val_accuracy:.1f}%")
    print(f"✅ Train-Val Gap: {train_val_gap:.1f}%")
    print(f"✅ Overfitting: {is_overfitting}")
    print(f"✅ Training Strategy: {results['training_strategy']}")
    
    return results

def main():
    """Run pretrained ViT training."""
    print("=" * 60)
    print("PRETRAINED ViT TRAINING WITH REAL DATA")
    print("=" * 60)
    
    try:
        results = train_vit_pretrained_model()
        
        # Save results
        results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/model_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"vit_pretrained_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 ViT Pretrained results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during ViT pretrained training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()