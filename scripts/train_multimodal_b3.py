#!/usr/bin/env python3
"""
Training Script for B.3: Pretrained Multimodal ViT
Extends A.4 (Pretrained ViT: 100% accuracy) with advanced multimodal fusion.
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
from models.multimodal_pretrained import (
    PretrainedMultimodalViT, 
    PretrainedMultimodalLoss, 
    PretrainedMultimodalMetrics
)

class PretrainedMultimodalTrainer:
    """Trainer for B.3: Pretrained Multimodal ViT."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-5,  # Very low LR for pretrained model
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.weight_decay = weight_decay
        
        # Advanced loss with fusion regularization
        self.criterion = PretrainedMultimodalLoss(
            main_weight=1.0, 
            attention_weight=0.1, 
            fusion_weight=0.05
        )
        
        # Two-phase training strategy
        self.phase = 1
        self.phase1_epochs = 15  # Train fusion components first
        self.phase2_epochs = 35  # Then fine-tune backbone
        
        # Phase 1: Only fusion components (backbone frozen)
        fusion_params = (
            list(self.model.annotation_encoder.parameters()) + 
            list(self.model.fusion_module.parameters()) + 
            list(self.model.classifier.parameters()) +
            [self.model.fusion_weights]
        )
        
        self.optimizer_phase1 = optim.AdamW(
            fusion_params, 
            lr=learning_rate * 10,  # Higher LR for fusion components
            weight_decay=weight_decay
        )
        
        # Phase 1 scheduler
        self.scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_phase1, T_max=self.phase1_epochs, eta_min=1e-7
        )
        
        # Enhanced metrics
        self.metrics = PretrainedMultimodalMetrics()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'fusion_weights': [],
            'phase_transitions': []
        }
        
        # Early stopping
        self.best_val_f1 = 0.0
        self.patience = 20  # More patience for two-phase training
        self.patience_counter = 0
        
    def setup_phase2(self):
        """Setup phase 2: Fine-tune backbone."""
        
        print(f"\n🔄 Transitioning to Phase 2: Fine-tuning pretrained backbone")
        
        # Unfreeze last layers of backbone
        self.model.unfreeze_backbone(layers_to_unfreeze=2)
        
        # Phase 2: All parameters with different learning rates
        backbone_params = []
        for name, param in self.model.video_backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
                
        fusion_params = (
            list(self.model.annotation_encoder.parameters()) + 
            list(self.model.fusion_module.parameters()) + 
            list(self.model.classifier.parameters()) +
            [self.model.fusion_weights]
        )
        
        self.optimizer_phase2 = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-6},    # Very conservative for pretrained
            {'params': fusion_params, 'lr': 5e-5}       # More aggressive for fusion
        ], weight_decay=self.weight_decay)
        
        # Phase 2 scheduler
        self.scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_phase2, T_max=self.phase2_epochs, eta_min=1e-8
        )
        
        self.phase = 2
        self.history['phase_transitions'].append(len(self.history['train_loss']))
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Select optimizer based on phase
        optimizer = self.optimizer_phase1 if self.phase == 1 else self.optimizer_phase2
        scheduler = self.scheduler_phase1 if self.phase == 1 else self.scheduler_phase2
        
        for batch in self.train_loader:
            # Move to device
            video = batch['video'].to(self.device)
            physical = batch['physical'].to(self.device)
            emotional = batch['emotional'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Mixed precision
                predictions = self.model(video, physical)
                
                # Get fusion analysis for regularization
                fusion_analysis = self.model.get_fusion_analysis(video, physical)
                
                # Compute loss with fusion regularization
                loss = self.criterion(predictions, emotional, fusion_analysis)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (crucial for pretrained models)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        # Step scheduler (per epoch for CosineAnnealing)
        scheduler.step()
            
        return total_loss / num_batches
        
    def validate(self) -> dict:
        """Validate model with fusion analysis."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_fusion_analyses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                video = batch['video'].to(self.device)
                physical = batch['physical'].to(self.device)
                emotional = batch['emotional'].to(self.device)
                
                # Forward pass
                predictions = self.model(video, physical)
                
                # Get fusion analysis
                fusion_analysis = self.model.get_fusion_analysis(video, physical)
                
                # Compute loss
                loss = self.criterion.emotion_loss(predictions, emotional)  # Main loss only for validation
                total_loss += loss.item()
                
                # Collect data
                all_predictions.append(predictions.cpu())
                all_targets.append(emotional.cpu())
                all_fusion_analyses.append({
                    k: v.cpu() if torch.is_tensor(v) else v 
                    for k, v in fusion_analysis.items()
                })
                
        # Aggregate predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Aggregate fusion analysis
        avg_fusion_analysis = {
            'video_weight': np.mean([fa['video_weight'] for fa in all_fusion_analyses]),
            'annotation_weight': np.mean([fa['annotation_weight'] for fa in all_fusion_analyses])
        }
        
        # Compute comprehensive metrics
        val_metrics = self.metrics.compute_comprehensive_metrics(
            all_predictions, all_targets, avg_fusion_analysis
        )
        val_metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return val_metrics
        
    def train(self, num_epochs: int = 50) -> dict:
        """Full two-phase training loop."""
        
        print(f"🚀 Starting B.3 Pretrained Multimodal ViT Training")
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Phase 1 (fusion): {self.phase1_epochs} epochs")
        print(f"Phase 2 (fine-tune): {self.phase2_epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Transition to phase 2 if needed
            if epoch == self.phase1_epochs:
                self.setup_phase2()
                
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_accuracy'].append(val_metrics['element_accuracy'])
            self.history['val_f1'].append(val_metrics['macro_f1'])
            self.history['fusion_weights'].append([
                val_metrics.get('video_weight', 0.0),
                val_metrics.get('annotation_weight', 0.0)
            ])
            
            # Print progress
            phase_str = f"P{self.phase}"
            optimizer = self.optimizer_phase1 if self.phase == 1 else self.optimizer_phase2
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} ({phase_str}) | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['element_accuracy']:.1%} | "
                  f"Val F1: {val_metrics['macro_f1']:.3f} | "
                  f"Fusion: {val_metrics.get('video_weight', 0):.2f}/{val_metrics.get('annotation_weight', 0):.2f} | "
                  f"LR: {current_lr:.2e}")
            
            # Early stopping
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model_b3.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print("\n✅ Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.3f}")
        
        # Final validation with best model
        self.load_checkpoint('best_model_b3.pth')
        final_metrics = self.validate()
        
        return {
            'model_name': 'pretrained_multimodal_vit',
            'model_type': 'B.3: Pretrained ViT-B/16 + Advanced Fusion',
            'epochs_trained': epoch + 1,
            'phase1_epochs': self.phase1_epochs,
            'phase2_epochs': epoch + 1 - self.phase1_epochs,
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
            'best_val_f1': self.best_val_f1,
            'phase': self.phase,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        
        checkpoint_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
        checkpoint = torch.load(checkpoint_dir / filename, weights_only=False)
        
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
    data_module = MultimodalDataModule(batch_size=4, num_workers=4)  # Smaller batch for larger model
    train_loader, val_loader = data_module.get_dataloaders()
    
    # Get dimensions
    sample_batch = next(iter(train_loader))
    physical_dim = sample_batch['physical'].shape[1]
    emotional_dim = sample_batch['emotional'].shape[1]
    
    print(f"Physical features: {physical_dim}")
    print(f"Emotional targets: {emotional_dim}")
    
    # Create pretrained multimodal model
    print("🧠 Creating B.3: Pretrained Multimodal ViT...")
    model = PretrainedMultimodalViT(
        physical_dim=physical_dim,
        emotional_dim=emotional_dim,
        embed_dim=768,           # ViT-B/16 dimension
        fusion_dim=512,          # Fusion dimension
        num_fusion_heads=8,      # Multi-head attention
        dropout=0.3              # Regularization
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Frozen percentage: {(frozen_params/total_params)*100:.1f}%")
    
    # Create trainer
    trainer = PretrainedMultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,     # Conservative LR for pretrained model
        weight_decay=0.01
    )
    
    # Train model
    results = trainer.train(num_epochs=50)
    
    # Save results
    results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"b3_pretrained_multimodal_{timestamp}.json"
    
    # Add metadata
    results.update({
        'timestamp': timestamp,
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'physical_dim': physical_dim,
        'emotional_dim': emotional_dim,
        'using_real_data': True,
        'data_source': 'GENEX Interview Annotations + Video Frames',
        'base_model': 'ImageNet Pretrained ViT-B/16',
        'fusion_strategy': 'Multi-head cross-attention with learned fusion weights',
        'training_strategy': 'Two-phase: fusion first, then backbone fine-tuning',
        'model_improvements': [
            'ImageNet pretrained ViT-B/16 backbone',
            'Advanced annotation encoder',
            'Learned fusion weights',
            'Multi-head cross-attention fusion',
            'Two-phase training strategy',
            'Mixed precision training',
            'Comprehensive fusion regularization'
        ]
    })
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # Handle tensor serialization
        
    print(f"\n💾 Results saved: {results_file}")
    
    # Print final summary
    print("\n🎯 B.3 Pretrained Multimodal ViT Results:")
    print(f"Best F1 Score: {results['best_val_f1']:.3f}")
    print(f"Final Accuracy: {results['final_metrics']['element_accuracy']:.1%}")
    print(f"Final Precision: {results['final_metrics']['macro_precision']:.3f}")
    print(f"Final Recall: {results['final_metrics']['macro_recall']:.3f}")
    
    # Compare with previous models
    results_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/experiments/multimodal_results")
    
    print(f"\n📈 Comparison with other multimodal models:")
    
    # B.1 comparison
    b1_files = list(results_dir.glob("b1_naive_multimodal_*.json"))
    if b1_files:
        with open(b1_files[-1], 'r') as f:
            b1_results = json.load(f)
        b1_f1 = b1_results['best_val_f1']
        improvement_b1 = ((results['best_val_f1'] - b1_f1) / b1_f1) * 100
        print(f"vs B.1 (Naive): {b1_f1:.3f} → {results['best_val_f1']:.3f} ({improvement_b1:+.1f}%)")
    
    # B.2 comparison
    b2_files = list(results_dir.glob("b2_advanced_fusion_*.json"))
    if b2_files:
        with open(b2_files[-1], 'r') as f:
            b2_results = json.load(f)
        b2_f1 = b2_results['best_val_f1']
        improvement_b2 = ((results['best_val_f1'] - b2_f1) / b2_f1) * 100
        print(f"vs B.2 (Advanced): {b2_f1:.3f} → {results['best_val_f1']:.3f} ({improvement_b2:+.1f}%)")


if __name__ == "__main__":
    main()