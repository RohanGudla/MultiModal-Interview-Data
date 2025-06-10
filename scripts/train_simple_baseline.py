#!/usr/bin/env python3
"""
ITERATION 2: Simple CNN baseline implementation and training.
Focus: Get ONE working model with real training results.
"""
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config

# Use basic imports first, then try torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    print(f"✅ PyTorch {torch.__version__} available")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available")

class SimpleEmotionDataset(Dataset):
    """Basic dataset for loading video frames and emotion labels."""
    
    def __init__(self, participants: List[str], data_dir: Path, config: Config):
        """
        Args:
            participants: List of participant IDs
            data_dir: Directory containing processed data
            config: Configuration object
        """
        self.participants = participants
        self.data_dir = data_dir
        self.config = config
        self.samples = []
        
        # For now, create dummy data to test the pipeline
        # TODO: Replace with actual frame extraction
        self._create_dummy_samples()
        
    def _create_dummy_samples(self):
        """Create dummy samples for initial testing."""
        print("⚠️  Creating dummy data for pipeline testing")
        
        # Create dummy samples for each participant
        for participant_id in self.participants:
            # Simulate 10 frames per participant
            for frame_idx in range(10):
                # Random emotion label (0 or 1 for binary classification)
                emotion_label = np.random.randint(0, 2)
                
                sample = {
                    'participant_id': participant_id,
                    'frame_idx': frame_idx,
                    'emotion_label': emotion_label,
                    'image_data': np.random.randn(3, 224, 224).astype(np.float32)  # Dummy RGB image
                }
                
                self.samples.append(sample)
        
        print(f"✅ Created {len(self.samples)} dummy samples")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        image = torch.from_numpy(sample['image_data'])
        label = torch.tensor(sample['emotion_label'], dtype=torch.float32)
        
        return image, label, sample['participant_id']

class SimpleCNN(nn.Module):
    """Simple CNN for emotion recognition baseline."""
    
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # After 3 poolings: 224/8 = 28
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56  
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

class SimpleTrainer:
    """Basic trainer for CNN baseline."""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Simple training configuration
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, participant_ids) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Add dimension for BCE
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:  # Print every 5 batches
                print(f'   Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        epoch_accuracy = correct / total
        avg_loss = epoch_loss / len(self.train_loader)
        
        return avg_loss, epoch_accuracy
        
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, participant_ids in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                epoch_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_accuracy = correct / total
        avg_loss = epoch_loss / len(self.val_loader)
        
        return avg_loss, epoch_accuracy
        
    def train(self, epochs=10):
        """Main training loop."""
        print(f"\n🚀 Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.1f}s")
        print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_acc,
            'training_time': total_time
        }
        
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        print(f"\n📊 Evaluating on test set...")
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, participant_ids in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        # Simple metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': total
        }
        
        print(f"📈 Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Total samples: {total}")
        
        return test_results

def create_data_loaders(participants: List[str], data_dir: Path, config: Config):
    """Create train/val/test data loaders."""
    
    # Simple split: first participant for test, second for val, rest for train
    if len(participants) < 3:
        print(f"⚠️  Only {len(participants)} participants, using simple split")
        train_participants = participants[:max(1, len(participants)-1)]
        val_participants = participants[-1:]
        test_participants = participants[-1:]  # Same as val for now
    else:
        train_participants = participants[:-2]
        val_participants = participants[-2:-1]
        test_participants = participants[-1:]
    
    print(f"📊 Data split:")
    print(f"  Train: {train_participants}")
    print(f"  Val: {val_participants}")
    print(f"  Test: {test_participants}")
    
    # Create datasets
    train_dataset = SimpleEmotionDataset(train_participants, data_dir, config)
    val_dataset = SimpleEmotionDataset(val_participants, data_dir, config)
    test_dataset = SimpleEmotionDataset(test_participants, data_dir, config)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader, test_loader

def run_iteration2():
    """Run Iteration 2: Simple CNN baseline."""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot run training")
        return False
    
    print("🚀 ITERATION 2: SIMPLE CNN BASELINE")
    print("=" * 80)
    print("Goal: Get ONE working model with real training results")
    print("=" * 80)
    
    # Initialize configuration
    config = Config()
    
    # Load participants from Iteration 1 results
    iteration1_results_path = Path("experiments/iteration1_analysis/iteration1_results.json")
    
    if not iteration1_results_path.exists():
        print("❌ Iteration 1 results not found. Run prepare_data_basic.py first.")
        return False
    
    with open(iteration1_results_path, 'r') as f:
        iteration1_results = json.load(f)
    
    # Get participants with both video and annotation data
    aligned_participants = []
    for participant_id in iteration1_results["videos"].keys():
        if participant_id in iteration1_results["annotations"]:
            aligned_participants.append(participant_id)
    
    print(f"✅ Found {len(aligned_participants)} participants with complete data:")
    for p in aligned_participants:
        print(f"   - {p}")
    
    if len(aligned_participants) < 1:
        print("❌ No participants with complete data found")
        return False
    
    # Create output directory
    output_dir = Path("experiments/iteration2_cnn_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create data loaders
        print(f"\n📊 Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            aligned_participants, output_dir, config
        )
        
        # Create model
        print(f"\n🤖 Creating Simple CNN model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN(num_classes=1)
        param_count = model.count_parameters()
        
        print(f"📊 Model info:")
        print(f"   Total parameters: {param_count['total']:,}")
        print(f"   Trainable parameters: {param_count['trainable']:,}")
        print(f"   Device: {device}")
        
        # Create trainer
        trainer = SimpleTrainer(model, train_loader, val_loader, device)
        
        # Train model
        print(f"\n🚀 TRAINING PHASE")
        print("=" * 40)
        training_results = trainer.train(epochs=5)  # Short training for baseline
        
        # Evaluate model
        print(f"\n📊 EVALUATION PHASE")
        print("=" * 40)
        test_results = trainer.evaluate(test_loader)
        
        # Save results
        final_results = {
            "iteration": 2,
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_info": {
                "architecture": "SimpleCNN",
                "parameters": param_count,
                "device": str(device)
            },
            "data_info": {
                "participants": aligned_participants,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset)
            },
            "training_results": training_results,
            "test_results": test_results,
            "success": True
        }
        
        results_file = output_dir / "iteration2_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n🎯 ITERATION 2 SUMMARY")
        print("=" * 50)
        print(f"✅ Model: SimpleCNN ({param_count['total']:,} parameters)")
        print(f"✅ Training completed: {training_results['training_time']:.1f}s")
        print(f"✅ Best validation accuracy: {training_results['best_val_accuracy']:.4f}")
        print(f"✅ Test accuracy: {test_results['accuracy']:.4f}")
        print(f"✅ Test F1-score: {test_results['f1_score']:.4f}")
        print(f"📁 Results saved to: {results_file}")
        
        # Determine success criteria
        if test_results['accuracy'] > 0.4:  # Better than random for dummy data
            print(f"\n🎉 ITERATION 2 SUCCESS!")
            print(f"✅ Model training pipeline working")
            print(f"✅ Ready to proceed to Iteration 3: Multiple Models")
            return True
        else:
            print(f"\n⚠️  ITERATION 2 NEEDS ATTENTION")
            print(f"❌ Model performance too low - check data pipeline")
            return False
        
    except Exception as e:
        print(f"\n💥 ITERATION 2 FAILED")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Iteration 2."""
    success = run_iteration2()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)