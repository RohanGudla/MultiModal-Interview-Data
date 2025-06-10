#!/usr/bin/env python3
"""
Train pretrained ResNet50 model using real GENEX frame data.
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from models.resnet_pretrained import create_resnet_model, get_learning_rates_for_layers

class ImprovedFrameDataset(Dataset):
    """Dataset with data augmentation for ResNet training."""
    
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
        
        # Augmentation transforms (more aggressive for pretrained models)
        if augment:
            aug_transforms = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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
    """Load real frame data for ResNet training."""
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
    
    print(f"📈 ResNet Train split: {len(train_frames)} frames")
    print(f"📊 ResNet Val split: {len(val_frames)} frames")
    
    train_pos = sum(train_labels)
    val_pos = sum(val_labels)
    print(f"🎯 Train: {train_pos} positive, {len(train_labels)-train_pos} negative")
    print(f"🎯 Val: {val_pos} positive, {len(val_labels)-val_pos} negative")
    
    return train_frames, train_labels, val_frames, val_labels

def train_resnet_model():
    """Train pretrained ResNet50 model."""
    config = Config()
    device = config.DEVICE
    
    print(f"🚀 Training PRETRAINED RESNET50 with real data")
    print(f"💻 Device: {device}")
    
    # Load data
    all_frames, all_labels, all_participants = load_real_frame_data()
    
    # Create splits
    train_frames, train_labels, val_frames, val_labels = create_improved_splits(
        all_frames, all_labels, all_participants
    )
    
    # Create datasets with stronger augmentation
    train_dataset = ImprovedFrameDataset(train_frames, train_labels, augment=True)
    val_dataset = ImprovedFrameDataset(val_frames, val_labels, augment=False)
    
    # Create data loaders - can use larger batch size with ResNet
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize pretrained ResNet50
    model = create_resnet_model(
        model_type='resnet50',
        num_classes=1,
        pretrained=True,
        freeze_backbone=True,
        freeze_layers=40,  # Freeze most layers initially
        dropout_rate=0.5,
        use_additional_layers=True
    )
    model = model.to(device)
    
    # Print model info
    param_count = model.count_parameters()
    print(f"📊 ResNet Model: {param_count['total_parameters']:,} total parameters")
    print(f"📊 ResNet Trainable: {param_count['trainable_parameters']:,} parameters")
    print(f"📊 ResNet Frozen: {param_count['frozen_parameters']:,} parameters")
    
    # Training setup with different learning rates for backbone vs classifier
    criterion = nn.BCEWithLogitsLoss()
    
    # Phase 1: Train only classifier
    phase1_epochs = 15
    phase2_epochs = 25
    
    # Phase 1 optimizer (only classifier)
    param_groups = get_learning_rates_for_layers(
        model, 
        base_lr=1e-3,  # Higher LR for classifier
        backbone_lr_ratio=0.0  # Frozen backbone
    )
    optimizer_phase1 = optim.Adam(param_groups, weight_decay=1e-4)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 8
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
            
            # Gradient clipping
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
    
    # Phase 2: Fine-tune backbone
    print(f"\n🎯 PHASE 2: Fine-tuning backbone ({phase2_epochs} epochs)")
    
    # Unfreeze some backbone layers for fine-tuning
    model.unfreeze_backbone_layers(5)  # Unfreeze last 5 layers
    
    # Phase 2 optimizer with lower learning rates
    param_groups = get_learning_rates_for_layers(
        model, 
        base_lr=1e-4,  # Lower LR for fine-tuning
        backbone_lr_ratio=0.1  # Much lower LR for backbone
    )
    optimizer_phase2 = optim.Adam(param_groups, weight_decay=1e-4)
    
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
        'model_name': 'resnet50_pretrained',
        'model_type': 'Pretrained ResNet50 with fine-tuning',
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
        'phase1_epochs': min(phase1_epochs, len(train_losses)),
        'phase2_epochs': max(0, len(train_losses) - phase1_epochs),
        'train_samples': len(train_frames),
        'val_samples': len(val_frames),
        'using_real_data': True,
        'training_strategy': 'Two-phase: classifier first, then fine-tuning',
        'improvements_applied': [
            'Pretrained ImageNet weights',
            'Two-phase training strategy',
            'Different learning rates for backbone vs classifier',
            'Gradual unfreezing of backbone layers',
            'Strong data augmentation',
            'Gradient clipping',
            'Early stopping per phase'
        ]
    }
    
    print(f"\n🎯 RESNET50 RESULTS")
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
    """Run pretrained ResNet50 training."""
    print("=" * 60)
    print("PRETRAINED RESNET50 TRAINING WITH REAL DATA")
    print("=" * 60)
    
    try:
        results = train_resnet_model()
        
        # Save results
        results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/model_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"resnet50_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 ResNet50 results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during ResNet50 training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()