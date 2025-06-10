#!/usr/bin/env python3
"""
Train ViT from scratch model using real GENEX frame data.
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
from models.vit_simple import create_vit_model

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
        
        # Augmentation transforms
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
    
    print(f"📈 ViT Train split: {len(train_frames)} frames")
    print(f"📊 ViT Val split: {len(val_frames)} frames")
    
    train_pos = sum(train_labels)
    val_pos = sum(val_labels)
    print(f"🎯 Train: {train_pos} positive, {len(train_labels)-train_pos} negative")
    print(f"🎯 Val: {val_pos} positive, {len(val_labels)-val_pos} negative")
    
    return train_frames, train_labels, val_frames, val_labels

def train_vit_model():
    """Train ViT from scratch model."""
    config = Config()
    device = config.DEVICE
    
    print(f"🚀 Training ViT FROM SCRATCH with real data")
    print(f"💻 Device: {device}")
    
    # Load data
    all_frames, all_labels, all_participants = load_real_frame_data()
    
    # Create splits
    train_frames, train_labels, val_frames, val_labels = create_improved_splits(
        all_frames, all_labels, all_participants
    )
    
    # Create datasets with augmentation
    train_dataset = ImprovedFrameDataset(train_frames, train_labels, augment=True)
    val_dataset = ImprovedFrameDataset(val_frames, val_labels, augment=False)
    
    # Create data loaders - smaller batch size for ViT
    batch_size = 2  # Very small due to ViT memory requirements
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize ViT model (tiny version for our small dataset)
    model = create_vit_model(
        model_size='tiny',  # Use tiny ViT for small dataset
        num_classes=1,
        dropout=0.3
    )
    model = model.to(device)
    
    # Print model info
    param_count = model.count_parameters()
    print(f"📊 ViT Model: {param_count['total_parameters']:,} total parameters")
    print(f"📊 ViT Trainable: {param_count['trainable_parameters']:,} parameters")
    
    # Training setup with careful hyperparameters for ViT
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(  # AdamW is better for transformers
        model.parameters(), 
        lr=3e-4,  # Standard ViT learning rate
        weight_decay=0.1,  # Higher weight decay for ViT
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    # Training loop
    num_epochs = 50
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n🎯 Starting ViT training for up to {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping (important for ViT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
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
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}% | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 ViT Early stopping at epoch {epoch+1}")
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
        'model_name': 'vit_scratch',
        'model_type': 'Vision Transformer from scratch',
        'model_size': 'tiny',
        'total_parameters': param_count['total_parameters'],
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'train_val_gap': train_val_gap,
        'is_overfitting': is_overfitting,
        'epochs_trained': len(train_losses),
        'early_stopped': len(train_losses) < num_epochs,
        'train_samples': len(train_frames),
        'val_samples': len(val_frames),
        'using_real_data': True,
        'improvements_applied': [
            'Tiny ViT architecture for small dataset',
            'AdamW optimizer with cosine annealing',
            'Gradient clipping',
            'Light data augmentation',
            'Early stopping',
            'Proper weight decay for transformers'
        ]
    }
    
    print(f"\n🎯 ViT RESULTS")
    print(f"✅ Model: {results['model_type']}")
    print(f"✅ Parameters: {results['total_parameters']:,}")
    print(f"✅ Train Loss: {final_train_loss:.4f}")
    print(f"✅ Val Loss: {final_val_loss:.4f}")
    print(f"✅ Final Val Accuracy: {final_val_accuracy:.1f}%")
    print(f"✅ Best Val Accuracy: {best_val_accuracy:.1f}%")
    print(f"✅ Train-Val Gap: {train_val_gap:.1f}%")
    print(f"✅ Overfitting: {is_overfitting}")
    print(f"✅ Early Stopped: {results['early_stopped']}")
    
    return results

def main():
    """Run ViT from scratch training."""
    print("=" * 60)
    print("ViT FROM SCRATCH TRAINING WITH REAL DATA")
    print("=" * 60)
    
    try:
        results = train_vit_model()
        
        # Save results
        results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/model_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"vit_scratch_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 ViT results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during ViT training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()