#!/usr/bin/env python3
"""
Train models using real GENEX frame data.
This replaces the synthetic data training with actual video frames.
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from models.cnn_simple import SimpleCNN

class RealFrameDataset(Dataset):
    """Dataset that loads real extracted frames."""
    
    def __init__(self, frame_data, labels, transform=None):
        self.frame_paths = frame_data
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
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
    """Load real frame data for all participants."""
    frame_base = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames")
    summary_path = frame_base / "processing_summary.json"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    all_frames = []
    all_labels = []
    participant_info = {}
    
    for participant_id, data in summary.items():
        frames = [Path(p) for p in data['frames']]
        
        # Create simple binary labels based on participant
        # CP 0636, NS 4013, MP 5114 -> label 1 (attention)
        # JM 9684, LE 3299 -> label 0 (no attention)
        if participant_id in ["CP 0636", "NS 4013", "MP 5114"]:
            labels = [1.0] * len(frames)
        else:
            labels = [0.0] * len(frames)
        
        all_frames.extend(frames)
        all_labels.extend(labels)
        
        participant_info[participant_id] = {
            'frames': len(frames),
            'label': labels[0],
            'start_idx': len(all_frames) - len(frames),
            'end_idx': len(all_frames)
        }
    
    print(f"📊 Loaded {len(all_frames)} real frames from {len(summary)} participants")
    for pid, info in participant_info.items():
        print(f"  {pid}: {info['frames']} frames, label={info['label']}")
    
    return all_frames, all_labels, participant_info

def create_train_val_splits(all_frames, all_labels, participant_info):
    """Create train/validation splits using real data."""
    config = Config()
    
    train_frames, train_labels = [], []
    val_frames, val_labels = [], []
    
    for participant_id, info in participant_info.items():
        start_idx = info['start_idx']
        end_idx = info['end_idx']
        
        frames = all_frames[start_idx:end_idx]
        labels = all_labels[start_idx:end_idx]
        
        if participant_id in config.TRAIN_PARTICIPANTS:
            train_frames.extend(frames)
            train_labels.extend(labels)
        elif participant_id in config.VAL_PARTICIPANTS:
            val_frames.extend(frames)
            val_labels.extend(labels)
    
    print(f"📈 Train split: {len(train_frames)} frames")
    print(f"📊 Val split: {len(val_frames)} frames")
    
    return train_frames, train_labels, val_frames, val_labels

def train_model_real_data(model_name="cnn_simple"):
    """Train a model using real frame data."""
    config = Config()
    device = config.DEVICE
    
    print(f"🚀 Training {model_name} with REAL DATA")
    print(f"💻 Device: {device}")
    
    # Load real data
    all_frames, all_labels, participant_info = load_real_frame_data()
    
    # Create splits
    train_frames, train_labels, val_frames, val_labels = create_train_val_splits(
        all_frames, all_labels, participant_info
    )
    
    # Create datasets
    train_dataset = RealFrameDataset(train_frames, train_labels)
    val_dataset = RealFrameDataset(val_frames, val_labels)
    
    # Create data loaders
    batch_size = config.get_model_config(model_name).get('batch_size', 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    if model_name == "cnn_simple":
        model = SimpleCNN(num_classes=1)
    else:
        raise ValueError(f"Model {model_name} not implemented yet")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get_model_config(model_name).get('learning_rate', 1e-3),
        weight_decay=config.get_model_config(model_name).get('weight_decay', 1e-4)
    )
    
    # Training loop
    num_epochs = 20  # Reduced for testing
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    
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
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}%")
    
    # Final results
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_val_accuracy = val_accuracies[-1]
    best_val_accuracy = max(val_accuracies)
    
    results = {
        'model_name': model_name,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'num_epochs': num_epochs,
        'train_samples': len(train_frames),
        'val_samples': len(val_frames),
        'participant_info': participant_info,
        'using_real_data': True
    }
    
    print(f"\n🎯 FINAL RESULTS - {model_name.upper()}")
    print(f"✅ Train Loss: {final_train_loss:.4f}")
    print(f"✅ Val Loss: {final_val_loss:.4f}")
    print(f"✅ Final Val Accuracy: {final_val_accuracy:.1f}%")
    print(f"✅ Best Val Accuracy: {best_val_accuracy:.1f}%")
    print(f"✅ Using REAL DATA: {results['using_real_data']}")
    
    return results

def main():
    """Run real data training."""
    print("=" * 60)
    print("TRAINING WITH REAL GENEX FRAME DATA")
    print("=" * 60)
    
    try:
        results = train_model_real_data("cnn_simple")
        
        # Save results
        results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/real_data_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"real_data_training_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during real data training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()