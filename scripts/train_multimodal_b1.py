#!/usr/bin/env python3
"""
Training Script for B.1: Naive Multimodal ViT
Extends A.2 (ViT from scratch) with simple concatenation fusion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from data.multimodal_dataset import MultimodalDataModule
from models.multimodal_naive import NaiveMultimodalViT, MultimodalLoss, MultimodalMetrics

class MultimodalTrainer:
    """Trainer for B.1: Naive Multimodal ViT."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = MultimodalLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Metrics
        self.metrics = MultimodalMetrics()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        # Early stopping
        self.best_val_f1 = 0.0
        self.patience = 10
        self.patience_counter = 0
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            video = batch['video'].to(self.device)
            physical = batch['physical'].to(self.device)
            emotional = batch['emotional'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(video, physical)
            
            # Compute loss
            loss = self.criterion(predictions, emotional)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def validate(self) -> dict:
        """Validate model."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                video = batch['video'].to(self.device)
                physical = batch['physical'].to(self.device)
                emotional = batch['emotional'].to(self.device)
                
                # Forward pass
                predictions = self.model(video, physical)
                
                # Compute loss
                loss = self.criterion(predictions, emotional)
                total_loss += loss.item()
                
                # Collect predictions and targets
                all_predictions.append(predictions.cpu())
                all_targets.append(emotional.cpu())
                
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        val_metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        val_metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return val_metrics
        
    def train(self, num_epochs: int = 50) -> dict:
        """Full training loop."""
        
        print(f"🚀 Starting B.1 Naive Multimodal ViT Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_accuracy'].append(val_metrics['element_accuracy'])
            self.history['val_f1'].append(val_metrics['macro_f1'])
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['element_accuracy']:.1%} | "
                  f"Val F1: {val_metrics['macro_f1']:.3f}")
            
            # Early stopping
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print("\n✅ Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.3f}")
        
        # Final validation with best model
        self.load_checkpoint('best_model.pth')
        final_metrics = self.validate()
        
        return {
            'model_name': 'naive_multimodal_vit',
            'model_type': 'B.1: Naive ViT + Simple Fusion',
            'epochs_trained': epoch + 1,
            'best_val_f1': self.best_val_f1,
            'final_metrics': final_metrics,
            'training_history': self.history
        }
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        
        checkpoint_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        
        checkpoint_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
        checkpoint = torch.load(checkpoint_dir / filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']


def main():
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data module
    print("📊 Loading multimodal data...")
    data_module = MultimodalDataModule(batch_size=8, num_workers=4)  # Smaller batch for multimodal
    train_loader, val_loader = data_module.get_dataloaders()
    
    # Get a sample to determine dimensions
    sample_batch = next(iter(train_loader))
    physical_dim = sample_batch['physical'].shape[1]
    emotional_dim = sample_batch['emotional'].shape[1]
    
    print(f"Physical features: {physical_dim}")
    print(f"Emotional targets: {emotional_dim}")
    
    # Create model
    print("🧠 Creating B.1: Naive Multimodal ViT...")
    model = NaiveMultimodalViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        physical_dim=physical_dim,
        emotional_dim=emotional_dim,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    # Train model
    results = trainer.train(num_epochs=50)
    
    # Save results
    results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"b1_naive_multimodal_{timestamp}.json"
    
    # Add metadata
    results.update({
        'timestamp': timestamp,
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'physical_dim': physical_dim,
        'emotional_dim': emotional_dim,
        'using_real_data': True,
        'data_source': 'GENEX Interview Annotations + Video Frames'
    })
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved: {results_file}")
    
    # Print final summary
    print("\n🎯 B.1 Naive Multimodal ViT Results:")
    print(f"Best F1 Score: {results['best_val_f1']:.3f}")
    print(f"Final Accuracy: {results['final_metrics']['element_accuracy']:.1%}")
    print(f"Final Precision: {results['final_metrics']['macro_precision']:.3f}")
    print(f"Final Recall: {results['final_metrics']['macro_recall']:.3f}")


if __name__ == "__main__":
    main()