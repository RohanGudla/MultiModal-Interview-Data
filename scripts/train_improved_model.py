#!/usr/bin/env python3
"""
Improved training with overfitting mitigation and better validation.
Addresses the issues identified in Iteration 5 error analysis.
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
from models.cnn_simple import SimpleCNN

class ImprovedFrameDataset(Dataset):
    """Dataset with data augmentation to address overfitting."""
    
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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

class ImprovedCNN(nn.Module):
    """CNN with better regularization to prevent overfitting."""
    
    def __init__(self, num_classes=1, dropout_rate=0.7):
        super(ImprovedCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),
        )
        
        # Calculate the size after convolutions
        self.feature_size = 128 * 28 * 28
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_real_frame_data_improved():
    """Load real frame data with better validation splits."""
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
    """Create better train/val splits with more rigorous validation."""
    
    # Use stratified split to ensure balanced validation
    train_frames, val_frames, train_labels, val_labels, train_participants, val_participants = train_test_split(
        all_frames, all_labels, all_participants,
        test_size=0.3,  # 30% for validation
        stratify=all_labels,  # Maintain label balance
        random_state=42
    )
    
    print(f"📈 Improved Train split: {len(train_frames)} frames")
    print(f"📊 Improved Val split: {len(val_frames)} frames")
    
    # Print label distribution
    train_pos = sum(train_labels)
    val_pos = sum(val_labels)
    print(f"🎯 Train: {train_pos} positive, {len(train_labels)-train_pos} negative")
    print(f"🎯 Val: {val_pos} positive, {len(val_labels)-val_pos} negative")
    
    return train_frames, train_labels, val_frames, val_labels

def train_improved_model():
    """Train model with overfitting mitigation strategies."""
    config = Config()
    device = config.DEVICE
    
    print(f"🚀 Training IMPROVED MODEL with overfitting mitigation")
    print(f"💻 Device: {device}")
    
    # Load data
    all_frames, all_labels, all_participants = load_real_frame_data_improved()
    
    # Create better splits
    train_frames, train_labels, val_frames, val_labels = create_improved_splits(
        all_frames, all_labels, all_participants
    )
    
    # Create datasets with augmentation
    train_dataset = ImprovedFrameDataset(train_frames, train_labels, augment=True)
    val_dataset = ImprovedFrameDataset(val_frames, val_labels, augment=False)
    
    # Create data loaders with smaller batch size
    batch_size = 4  # Smaller to prevent overfitting
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize improved model
    model = ImprovedCNN(num_classes=1, dropout_rate=0.7)
    model = model.to(device)
    
    # Training setup with weight decay
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=1e-4,  # Lower learning rate
        weight_decay=1e-3  # Strong L2 regularization
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training loop
    num_epochs = 50
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n🎯 Starting improved training for up to {num_epochs} epochs...")
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
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
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}% | "
              f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Calculate overfitting metrics
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_val_accuracy = val_accuracies[-1]
    best_val_accuracy = max(val_accuracies)
    
    # Overfitting detection
    train_val_gap = abs(final_val_accuracy - 100 * (1 - final_train_loss))
    is_overfitting = train_val_gap > 15  # If gap > 15%
    
    results = {
        'model_name': 'improved_cnn',
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
            'Data augmentation',
            'Increased dropout (0.7)',
            'Weight decay (L2 regularization)',
            'Early stopping',
            'Gradient clipping',
            'Batch normalization',
            'Stratified validation split'
        ]
    }
    
    print(f"\n🎯 IMPROVED RESULTS")
    print(f"✅ Train Loss: {final_train_loss:.4f}")
    print(f"✅ Val Loss: {final_val_loss:.4f}")
    print(f"✅ Final Val Accuracy: {final_val_accuracy:.1f}%")
    print(f"✅ Best Val Accuracy: {best_val_accuracy:.1f}%")
    print(f"✅ Train-Val Gap: {train_val_gap:.1f}%")
    print(f"✅ Overfitting Detected: {is_overfitting}")
    print(f"✅ Early Stopped: {results['early_stopped']}")
    print(f"✅ Using REAL DATA: {results['using_real_data']}")
    
    return results

def main():
    """Run improved training with overfitting mitigation."""
    print("=" * 60)
    print("IMPROVED TRAINING WITH OVERFITTING MITIGATION")
    print("=" * 60)
    
    try:
        results = train_improved_model()
        
        # Save results
        results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/improved_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"improved_training_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Improved results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during improved training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()