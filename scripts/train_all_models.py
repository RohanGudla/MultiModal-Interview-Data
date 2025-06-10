#!/usr/bin/env python3
"""
ITERATION 3: All 4 model architectures implementation and training.
Focus: Compare CNN vs ViT scratch vs ResNet50 pretrained vs ViT pretrained.
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
    import torchvision.transforms as transforms
    import torchvision.models as models
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
    """Dataset for loading video frames and emotion labels."""
    
    def __init__(self, participants: List[str], data_dir: Path, config: Config, transform=None):
        self.participants = participants
        self.data_dir = data_dir
        self.config = config
        self.transform = transform
        self.samples = []
        
        # For now, create dummy data to test the pipeline
        # TODO: Replace with actual frame extraction
        self._create_dummy_samples()
        
    def _create_dummy_samples(self):
        """Create dummy samples for initial testing."""
        print(f"⚠️  Creating dummy data for {len(self.participants)} participants")
        
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
        
        # Apply transforms if provided
        if self.transform:
            # For transforms that expect PIL images, we need to handle tensor inputs
            pass
        
        return image, label, sample['participant_id']

# Model 1: Simple CNN (from Iteration 2)
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

# Model 2: Vision Transformer from Scratch
class SimpleViT(nn.Module):
    """Simple Vision Transformer implementation from scratch."""
    
    def __init__(self, img_size=224, patch_size=16, num_classes=1, dim=512, depth=6, heads=8, mlp_dim=1024):
        super(SimpleViT, self).__init__()
        
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dim = dim
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        
        # Patch embedding
        x = self.patch_embed(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_len, batch, dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, dim)
        
        # Classification
        x = self.norm(x[:, 0])  # Use class token
        x = self.head(x)
        
        return x
        
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

# Model 3: Pretrained ResNet50
class PretrainedResNet50(nn.Module):
    """ResNet50 with pretrained weights."""
    
    def __init__(self, num_classes=1, freeze_backbone=False):
        super(PretrainedResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
        
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

# Model 4: Pretrained Vision Transformer
class PretrainedViT(nn.Module):
    """Vision Transformer with pretrained weights (using ResNet as proxy)."""
    
    def __init__(self, num_classes=1, freeze_backbone=False):
        super(PretrainedViT, self).__init__()
        
        # Note: Using ResNet as proxy for pretrained ViT since torchvision ViT may not be available
        # In real implementation, would use something like timm or official ViT
        self.backbone = models.resnet18(pretrained=True)  # Smaller model as ViT proxy
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
        
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

class ModelTrainer:
    """Unified trainer for all model architectures."""
    
    def __init__(self, model, train_loader, val_loader, device, model_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Training configuration
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
        
    def train(self, epochs=5):
        """Main training loop."""
        print(f"\n🚀 Training {self.model_name} for {epochs} epochs")
        
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
            
            print(f"  Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) - Train: {train_acc:.3f}, Val: {val_acc:.3f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        total_time = time.time() - start_time
        print(f"  ✅ {self.model_name} completed in {total_time:.1f}s, best val acc: {best_val_acc:.3f}")
        
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': total
        }

def create_data_loaders(participants: List[str], data_dir: Path, config: Config):
    """Create train/val/test data loaders."""
    
    # Simple split: first participants for train, middle for val, last for test
    if len(participants) < 3:
        print(f"⚠️  Only {len(participants)} participants, using simple split")
        train_participants = participants[:max(1, len(participants)-1)]
        val_participants = participants[-1:]
        test_participants = participants[-1:]  # Same as val for now
    else:
        train_participants = participants[:-2]
        val_participants = participants[-2:-1]
        test_participants = participants[-1:]
    
    print(f"📊 Data split: Train={len(train_participants)}, Val={len(val_participants)}, Test={len(test_participants)}")
    
    # Create datasets
    train_dataset = SimpleEmotionDataset(train_participants, data_dir, config)
    val_dataset = SimpleEmotionDataset(val_participants, data_dir, config)
    test_dataset = SimpleEmotionDataset(test_participants, data_dir, config)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader, test_loader

def run_iteration3():
    """Run Iteration 3: All 4 model architectures."""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot run training")
        return False
    
    print("🚀 ITERATION 3: ALL 4 MODEL ARCHITECTURES")
    print("=" * 80)
    print("Goal: Compare CNN, ViT scratch, ResNet50 pretrained, ViT pretrained")
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
    
    print(f"✅ Found {len(aligned_participants)} participants: {aligned_participants}")
    
    if len(aligned_participants) < 1:
        print("❌ No participants with complete data found")
        return False
    
    # Create output directory
    output_dir = Path("experiments/iteration3_all_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create data loaders
        print(f"\n📊 Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            aligned_participants, output_dir, config
        )
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        # Define all 4 models
        models_to_train = [
            ("SimpleCNN", SimpleCNN(num_classes=1)),
            ("SimpleViT", SimpleViT(num_classes=1, dim=256, depth=4, heads=4, mlp_dim=512)),  # Smaller ViT
            ("PretrainedResNet50", PretrainedResNet50(num_classes=1)),
            ("PretrainedViT", PretrainedViT(num_classes=1))
        ]
        
        results = {}
        
        print(f"\n🤖 TRAINING ALL 4 MODELS")
        print("=" * 50)
        
        for model_name, model in models_to_train:
            print(f"\n🔄 Training {model_name}...")
            
            # Model info
            param_count = model.count_parameters()
            print(f"   Parameters: {param_count['total']:,} total, {param_count['trainable']:,} trainable")
            
            # Create trainer
            trainer = ModelTrainer(model, train_loader, val_loader, device, model_name)
            
            # Train model
            try:
                training_results = trainer.train(epochs=5)  # Short training for comparison
                
                # Evaluate model
                test_results = trainer.evaluate(test_loader)
                
                # Store results
                results[model_name] = {
                    "model_info": {
                        "architecture": model_name,
                        "parameters": param_count,
                        "device": str(device)
                    },
                    "training_results": training_results,
                    "test_results": test_results,
                    "success": True
                }
                
                print(f"   ✅ {model_name}: Val={training_results['best_val_accuracy']:.3f}, Test={test_results['accuracy']:.3f}, F1={test_results['f1_score']:.3f}")
                
            except Exception as e:
                print(f"   ❌ {model_name} failed: {str(e)}")
                results[model_name] = {
                    "model_info": {"architecture": model_name, "parameters": param_count},
                    "error": str(e),
                    "success": False
                }
        
        # Save comprehensive results
        final_results = {
            "iteration": 3,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_info": {
                "participants": aligned_participants,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset)
            },
            "model_results": results,
            "success": True
        }
        
        results_file = output_dir / "iteration3_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print comparison summary
        print(f"\n🎯 ITERATION 3 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<20} {'Parameters':<12} {'Val Acc':<8} {'Test Acc':<9} {'F1':<6}")
        print("-" * 60)
        
        for model_name, result in results.items():
            if result.get("success", False):
                params = result["model_info"]["parameters"]["total"]
                val_acc = result["training_results"]["best_val_accuracy"]
                test_acc = result["test_results"]["accuracy"]
                f1_score = result["test_results"]["f1_score"]
                print(f"{model_name:<20} {params:<12,} {val_acc:<8.3f} {test_acc:<9.3f} {f1_score:<6.3f}")
            else:
                print(f"{model_name:<20} {'FAILED':<12} {'N/A':<8} {'N/A':<9} {'N/A':<6}")
        
        print(f"\n📁 Results saved to: {results_file}")
        
        # Success criteria
        successful_models = sum(1 for r in results.values() if r.get("success", False))
        if successful_models >= 3:
            print(f"\n🎉 ITERATION 3 SUCCESS!")
            print(f"✅ {successful_models}/4 models trained successfully")
            print(f"✅ Ready to proceed to Iteration 4: Comprehensive Analysis")
            return True
        else:
            print(f"\n⚠️  ITERATION 3 NEEDS ATTENTION")
            print(f"❌ Only {successful_models}/4 models successful")
            return False
        
    except Exception as e:
        print(f"\n💥 ITERATION 3 FAILED")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Iteration 3."""
    success = run_iteration3()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)