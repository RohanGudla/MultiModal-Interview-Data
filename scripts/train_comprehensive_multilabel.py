#!/usr/bin/env python3
"""
Comprehensive Multi-Label Training Pipeline
Trains temporal models on all 50 annotation features with verification
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

# Import our modules
from data.multilabel_dataset import MultiLabelAnnotationDataset, create_dataloaders
from models.temporal_multilabel import create_temporal_model, MultilabelTemporalLoss
from utils.output_verification import OutputVerificationSystem

class ComprehensiveTrainer:
    """
    Comprehensive trainer for temporal multi-label models
    """
    
    def __init__(self,
                 frames_dir: str,
                 annotations_dir: str,
                 output_dir: str,
                 model_type: str = 'vit',
                 sequence_length: int = 5,
                 batch_size: int = 8,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-4,
                 device: str = 'auto'):
        
        self.frames_dir = frames_dir
        self.annotations_dir = annotations_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Trainer initialized:")
        print(f"   Model type: {model_type}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        
        # Will be initialized in setup()
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.verifier = None
        
    def setup(self):
        """Setup datasets, model, and training components"""
        
        print("📚 Setting up datasets...")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader, self.dataset = create_dataloaders(
            frames_dir=self.frames_dir,
            annotations_dir=self.annotations_dir,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            num_workers=4
        )
        
        # Get feature information
        feature_info = self.dataset.get_feature_names()
        num_physical = len(feature_info['physical'])
        num_emotional = len(feature_info['emotional'])
        
        print(f"   Dataset size: {len(self.dataset)}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
        print(f"   Physical features: {num_physical}")
        print(f"   Emotional features: {num_emotional}")
        
        # Create model
        print(f"🧠 Creating {self.model_type} model...")
        self.model = create_temporal_model(
            model_type=self.model_type,
            num_physical_features=num_physical,
            num_emotional_features=num_emotional,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Create loss function
        print("⚖️ Setting up loss function...")
        self.criterion = MultilabelTemporalLoss(
            physical_weight=1.0,
            emotional_weight=1.5,  # Slightly higher weight for emotional features
            boundary_weight=0.5
        )
        
        # Create optimizer
        print("🎯 Setting up optimizer...")
        
        if hasattr(self.model, 'spatial_encoder'):
            # Different learning rates for spatial encoder and other components
            spatial_params = list(self.model.spatial_encoder.parameters())
            other_params = [p for p in self.model.parameters() if p not in spatial_params]
            
            self.optimizer = optim.AdamW([
                {'params': spatial_params, 'lr': self.learning_rate * 0.1, 'weight_decay': 0.01},
                {'params': other_params, 'lr': self.learning_rate, 'weight_decay': 0.01}
            ])
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Setup output verification
        print("📋 Setting up output verification...")
        verification_dir = self.output_dir / "verification"
        self.verifier = OutputVerificationSystem(
            output_dir=verification_dir,
            feature_names=feature_info,
            save_individual_predictions=True,
            save_aggregated_results=True,
            create_visualizations=True
        )
        
        print("✅ Setup complete!")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        physical_loss = 0.0
        emotional_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if self.sequence_length > 1:
                images = batch['images'].to(self.device)
            else:
                images = batch['image'].unsqueeze(1).to(self.device)  # Add sequence dimension
            
            physical_labels = batch['physical_labels'].to(self.device)
            emotional_labels = batch['emotional_labels'].to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            
            # Calculate loss
            targets = {
                'physical_labels': physical_labels,
                'emotional_labels': emotional_labels
            }
            
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += losses['total'].item()
            physical_loss += losses.get('physical', torch.tensor(0)).item()
            emotional_loss += losses.get('emotional', torch.tensor(0)).item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'P': f"{losses.get('physical', torch.tensor(0)).item():.3f}",
                'E': f"{losses.get('emotional', torch.tensor(0)).item():.3f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Update scheduler
        self.scheduler.step()
        
        avg_total_loss = total_loss / len(self.train_loader)
        avg_physical_loss = physical_loss / len(self.train_loader)
        avg_emotional_loss = emotional_loss / len(self.train_loader)
        
        return {
            'total_loss': avg_total_loss,
            'physical_loss': avg_physical_loss,
            'emotional_loss': avg_emotional_loss
        }
    
    def validate_epoch(self, epoch, save_predictions=False):
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        physical_loss = 0.0
        emotional_loss = 0.0
        
        # Clear previous verification data if saving predictions
        if save_predictions:
            self.verifier.clear_batch_data()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                if self.sequence_length > 1:
                    images = batch['images'].to(self.device)
                else:
                    images = batch['image'].unsqueeze(1).to(self.device)
                
                physical_labels = batch['physical_labels'].to(self.device)
                emotional_labels = batch['emotional_labels'].to(self.device)
                all_labels = batch['all_labels'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                targets = {
                    'physical_labels': physical_labels,
                    'emotional_labels': emotional_labels
                }
                
                losses = self.criterion(predictions, targets)
                
                # Update statistics
                total_loss += losses['total'].item()
                physical_loss += losses.get('physical', torch.tensor(0)).item()
                emotional_loss += losses.get('emotional', torch.tensor(0)).item()
                
                # Save predictions for verification
                if save_predictions:
                    gt_dict = {
                        'physical_labels': physical_labels,
                        'emotional_labels': emotional_labels,
                        'all_labels': all_labels
                    }
                    
                    meta_dict = {
                        'participant_id': batch['participant_id'],
                        'frame_id': batch['frame_id'],
                        'temporal_info': batch.get('temporal_info', [])
                    }
                    
                    self.verifier.add_batch_predictions(predictions, gt_dict, meta_dict)
        
        avg_total_loss = total_loss / len(self.val_loader)
        avg_physical_loss = physical_loss / len(self.val_loader)
        avg_emotional_loss = emotional_loss / len(self.val_loader)
        
        results = {
            'total_loss': avg_total_loss,
            'physical_loss': avg_physical_loss,
            'emotional_loss': avg_emotional_loss
        }
        
        # Generate verification report if saving predictions
        if save_predictions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report, report_file = self.verifier.generate_verification_report(f"epoch_{epoch}_{timestamp}")
            results['verification_report'] = report
            results['verification_file'] = str(report_file)
        
        return results
    
    def test_model(self):
        """Test the final model"""
        
        print("🧪 Testing final model...")
        
        self.model.eval()
        self.verifier.clear_batch_data()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                if self.sequence_length > 1:
                    images = batch['images'].to(self.device)
                else:
                    images = batch['image'].unsqueeze(1).to(self.device)
                
                physical_labels = batch['physical_labels'].to(self.device)
                emotional_labels = batch['emotional_labels'].to(self.device)
                all_labels = batch['all_labels'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Save predictions for verification
                gt_dict = {
                    'physical_labels': physical_labels,
                    'emotional_labels': emotional_labels,
                    'all_labels': all_labels
                }
                
                meta_dict = {
                    'participant_id': batch['participant_id'],
                    'frame_id': batch['frame_id'],
                    'temporal_info': batch.get('temporal_info', [])
                }
                
                self.verifier.add_batch_predictions(predictions, gt_dict, meta_dict)
        
        # Generate final verification report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report, report_file = self.verifier.generate_verification_report(f"final_test_{timestamp}")
        
        print(f"✅ Final test complete!")
        print(f"   Test accuracy: {report['performance']['overall_accuracy']:.3f}")
        print(f"   Physical accuracy: {report['performance']['physical_mean_accuracy']:.3f}")
        print(f"   Emotional accuracy: {report['performance']['emotional_mean_accuracy']:.3f}")
        print(f"   Report: {report_file}")
        
        return report, report_file
    
    def save_model(self, epoch, metrics, best=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'model_config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'num_physical_features': self.model.num_physical_features,
                'num_emotional_features': self.model.num_emotional_features
            }
        }
        
        if best:
            checkpoint_path = self.output_dir / f"best_model_{self.model_type}_seq{self.sequence_length}.pth"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_{self.model_type}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def train(self):
        """Main training loop"""
        
        print("🚀 Starting comprehensive training...")
        
        # Setup everything
        self.setup()
        
        # Training history
        train_history = []
        val_history = []
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Phase 1: Train with frozen spatial encoder
        print("\n🔒 Phase 1: Training with frozen spatial encoder...")
        phase1_epochs = min(10, self.num_epochs // 3)
        
        for epoch in range(phase1_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate (save predictions every 5 epochs)
            save_preds = (epoch + 1) % 5 == 0
            val_metrics = self.validate_epoch(epoch, save_predictions=save_preds)
            
            # Save metrics
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{phase1_epochs}:")
            print(f"  Train - Total: {train_metrics['total_loss']:.4f}, Physical: {train_metrics['physical_loss']:.4f}, Emotional: {train_metrics['emotional_loss']:.4f}")
            print(f"  Val   - Total: {val_metrics['total_loss']:.4f}, Physical: {val_metrics['physical_loss']:.4f}, Emotional: {val_metrics['emotional_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_epoch = epoch
                self.save_model(epoch, val_metrics, best=True)
                print(f"  ⭐ New best model saved!")
        
        # Phase 2: Unfreeze and fine-tune
        if hasattr(self.model, 'unfreeze_spatial_encoder'):
            print("\n🔓 Phase 2: Fine-tuning with unfrozen spatial encoder...")
            self.model.unfreeze_spatial_encoder()
            
            # Lower learning rate for fine-tuning
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
            
            for epoch in range(phase1_epochs, self.num_epochs):
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Validate (save predictions every 5 epochs)
                save_preds = (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1
                val_metrics = self.validate_epoch(epoch, save_predictions=save_preds)
                
                # Save metrics
                train_history.append(train_metrics)
                val_history.append(val_metrics)
                
                # Print progress
                print(f"Epoch {epoch+1}/{self.num_epochs}:")
                print(f"  Train - Total: {train_metrics['total_loss']:.4f}, Physical: {train_metrics['physical_loss']:.4f}, Emotional: {train_metrics['emotional_loss']:.4f}")
                print(f"  Val   - Total: {val_metrics['total_loss']:.4f}, Physical: {val_metrics['physical_loss']:.4f}, Emotional: {val_metrics['emotional_loss']:.4f}")
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    best_epoch = epoch
                    self.save_model(epoch, val_metrics, best=True)
                    print(f"  ⭐ New best model saved!")
                
                # Save regular checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_model(epoch, val_metrics, best=False)
        
        # Final testing
        print(f"\n🎯 Training complete! Best epoch: {best_epoch+1}")
        
        # Load best model for testing
        best_model_path = self.output_dir / f"best_model_{self.model_type}_seq{self.sequence_length}.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Test final model
        test_report, test_report_file = self.test_model()
        
        # Save training history
        history = {
            'train_history': train_history,
            'val_history': val_history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_test_report': test_report,
            'config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate
            }
        }
        
        history_file = self.output_dir / f"training_history_{self.model_type}_seq{self.sequence_length}.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n🎉 Comprehensive training complete!")
        print(f"   Best model: {best_model_path}")
        print(f"   Test report: {test_report_file}")
        print(f"   Training history: {history_file}")
        
        return {
            'best_model_path': best_model_path,
            'test_report': test_report,
            'test_report_file': test_report_file,
            'history_file': history_file,
            'final_accuracy': test_report['performance']['overall_accuracy']
        }

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Comprehensive Multi-Label Training')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'resnet'],
                       help='Model architecture to use')
    parser.add_argument('--sequence_length', type=int, default=5,
                       help='Length of input sequences')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./training_outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup paths
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    
    print("🎯 Comprehensive Multi-Label Training Pipeline")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 60)
    
    # Create trainer
    trainer = ComprehensiveTrainer(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Train model
    results = trainer.train()
    
    print(f"\n🏆 Training Results:")
    print(f"   Final Accuracy: {results['final_accuracy']:.3f}")
    print(f"   Model saved: {results['best_model_path']}")

if __name__ == "__main__":
    main()