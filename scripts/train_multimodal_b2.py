#!/usr/bin/env python3
"""
Training Script for B.2: Advanced Fusion ViT
Enhanced ViT with sophisticated cross-modal attention fusion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from data.multimodal_dataset import MultimodalDataModule
from models.multimodal_advanced import AdvancedFusionViT, AdvancedMultimodalLoss
from models.multimodal_naive import MultimodalMetrics  # Reuse metrics

class AdvancedMultimodalTrainer:
    """Trainer for B.2: Advanced Fusion ViT."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 5e-5,  # Lower LR for larger model
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Advanced loss with attention regularization
        self.criterion = AdvancedMultimodalLoss(alpha=1.0, beta=0.1)
        
        # Optimizer with different learning rates for different components
        video_params = list(self.model.video_branch.parameters()) + \
                      [self.model.video_pos_embed, self.model.video_cls_token]
        annotation_params = list(self.model.annotation_branch.parameters())
        fusion_params = list(self.model.cross_modal_attention.parameters()) + \
                       list(self.model.fusion_classifier.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': video_params, 'lr': learning_rate},
            {'params': annotation_params, 'lr': learning_rate * 1.5},  # Higher LR for annotation branch
            {'params': fusion_params, 'lr': learning_rate * 2.0}       # Highest LR for fusion
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler with warm-up
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=[learning_rate, learning_rate * 1.5, learning_rate * 2.0],
            epochs=50,
            steps_per_epoch=len(train_loader),
            pct_start=0.1  # 10% warm-up
        )
        
        # Metrics
        self.metrics = MultimodalMetrics()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_f1 = 0.0
        self.patience = 15  # More patience for larger model
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
            
            # Get attention maps for regularization
            attention_maps = self.model.get_attention_maps(video, physical)
            
            # Compute loss with attention regularization
            loss = self.criterion(predictions, emotional, attention_maps)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (important for transformers)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()  # Step per batch for OneCycleLR
            
            total_loss += loss.item()
            num_batches += 1
            
        # Store learning rates
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.history['learning_rates'].append(current_lrs)
            
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
                
                # Compute loss (without attention regularization for validation)
                loss = self.criterion.emotion_loss(predictions, emotional)
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
        
        print(f"🚀 Starting B.2 Advanced Fusion ViT Training")
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
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_accuracy'].append(val_metrics['element_accuracy'])
            self.history['val_f1'].append(val_metrics['macro_f1'])
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['element_accuracy']:.1%} | "
                  f"Val F1: {val_metrics['macro_f1']:.3f} | "
                  f"LR: {current_lr:.2e}")
            
            # Early stopping
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model_b2.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print("\n✅ Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.3f}")
        
        # Final validation with best model
        self.load_checkpoint('best_model_b2.pth')
        final_metrics = self.validate()
        
        return {
            'model_name': 'advanced_fusion_vit',
            'model_type': 'B.2: Advanced ViT + Cross-Modal Attention',
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
            'scheduler_state_dict': self.scheduler.state_dict(),
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
    data_module = MultimodalDataModule(batch_size=6, num_workers=4)  # Smaller batch for larger model
    train_loader, val_loader = data_module.get_dataloaders()
    
    # Get dimensions
    sample_batch = next(iter(train_loader))
    physical_dim = sample_batch['physical'].shape[1]
    emotional_dim = sample_batch['emotional'].shape[1]
    
    print(f"Physical features: {physical_dim}")
    print(f"Emotional targets: {emotional_dim}")
    
    # Create advanced model
    print("🧠 Creating B.2: Advanced Fusion ViT...")
    model = AdvancedFusionViT(
        img_size=224,
        patch_size=16,
        embed_dim=256,         # Larger embedding dimension
        depth=6,               # Deeper transformer
        num_heads=8,           # More attention heads
        physical_dim=physical_dim,
        emotional_dim=emotional_dim,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = AdvancedMultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=5e-5,    # Lower learning rate for larger model
        weight_decay=0.01
    )
    
    # Train model
    results = trainer.train(num_epochs=50)
    
    # Save results
    results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"b2_advanced_fusion_{timestamp}.json"
    
    # Add metadata
    results.update({
        'timestamp': timestamp,
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'physical_dim': physical_dim,
        'emotional_dim': emotional_dim,
        'using_real_data': True,
        'data_source': 'GENEX Interview Annotations + Video Frames',
        'fusion_strategy': 'Cross-modal attention with temporal transformer',
        'model_improvements': [
            'Larger embedding dimension (256 vs 192)',
            'Deeper transformer (6 vs 4 layers)',
            'Cross-modal attention fusion',
            'Temporal annotation processing',
            'Advanced learning rate scheduling',
            'Attention regularization'
        ]
    })
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved: {results_file}")
    
    # Print final summary
    print("\n🎯 B.2 Advanced Fusion ViT Results:")
    print(f"Best F1 Score: {results['best_val_f1']:.3f}")
    print(f"Final Accuracy: {results['final_metrics']['element_accuracy']:.1%}")
    print(f"Final Precision: {results['final_metrics']['macro_precision']:.3f}")
    print(f"Final Recall: {results['final_metrics']['macro_recall']:.3f}")
    
    # Compare with B.1
    b1_results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
    b1_files = list(b1_results_dir.glob("b1_naive_multimodal_*.json"))
    
    if b1_files:
        with open(b1_files[-1], 'r') as f:
            b1_results = json.load(f)
            
        b1_f1 = b1_results['best_val_f1']
        b2_f1 = results['best_val_f1']
        improvement = ((b2_f1 - b1_f1) / b1_f1) * 100
        
        print(f"\n📈 Improvement over B.1:")
        print(f"B.1 F1: {b1_f1:.3f} → B.2 F1: {b2_f1:.3f}")
        print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()