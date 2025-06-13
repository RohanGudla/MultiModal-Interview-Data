#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Multi-Participant Multi-Label Video Annotation
Handles all 50 annotation features with temporal modeling and verification outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from data.multilabel_dataset import create_dataloaders
from models.temporal_multilabel import TemporalMultiLabelViT, TemporalMultiLabelResNet
from utils.enhanced_verification import EnhancedOutputVerifier

class EnhancedMultiLabelTrainer:
    """Enhanced trainer for multi-participant multi-label video annotation"""
    
    def __init__(self,
                 frames_dir: str,
                 annotations_dir: str,
                 output_dir: str,
                 model_type: str = 'vit',
                 sequence_length: int = 10,
                 batch_size: int = 16,
                 learning_rate: float = 0.001,
                 num_epochs: int = 50,
                 device: str = 'auto'):
        """
        Initialize enhanced trainer
        
        Args:
            frames_dir: Directory containing extracted frames
            annotations_dir: Directory containing annotation CSV files
            output_dir: Directory for saving outputs
            model_type: 'vit' or 'resnet'
            sequence_length: Number of frames in temporal sequences
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        
        self.frames_dir = frames_dir
        self.annotations_dir = annotations_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Enhanced Multi-Label Trainer Initialized")
        print(f"   Model: {model_type}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        print(f"   Output dir: {output_dir}")
        
        # Initialize components
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Setup output verification
        self.verifier = EnhancedOutputVerifier(str(self.output_dir))
        
    def setup_data(self):
        """Setup data loaders with participant-based splitting"""
        
        print(f"\n📊 Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.test_loader, self.dataset = create_dataloaders(
            frames_dir=self.frames_dir,
            annotations_dir=self.annotations_dir,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            split_by_participant=True,
            num_workers=0  # Disable multiprocessing to avoid tensor resize issues
        )
        
        # Get feature information
        self.feature_info = self.dataset.get_feature_names()
        self.num_physical_features = len(self.feature_info['physical'])
        self.num_emotional_features = len(self.feature_info['emotional'])
        self.num_total_features = len(self.feature_info['all'])
        
        print(f"✅ Data setup complete:")
        print(f"   Physical features: {self.num_physical_features}")
        print(f"   Emotional features: {self.num_emotional_features}")
        print(f"   Total features: {self.num_total_features}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
        
    def setup_model(self):
        """Setup model, optimizer, and loss function"""
        
        print(f"\n🧠 Setting up model...")
        
        # Create model
        if self.model_type == 'vit':
            self.model = TemporalMultiLabelViT(
                num_physical_features=self.num_physical_features,
                num_emotional_features=self.num_emotional_features,
                sequence_length=self.sequence_length
            )
        else:
            self.model = TemporalMultiLabelResNet(
                num_physical_features=self.num_physical_features,
                num_emotional_features=self.num_emotional_features,
                sequence_length=self.sequence_length
            )
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Setup loss function with class weighting
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model setup complete:")
        print(f"   Architecture: {self.model_type}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            if self.sequence_length > 1:
                images = batch['images'].to(self.device)
            else:
                images = batch['image'].to(self.device)
            
            physical_labels = batch['physical_labels'].to(self.device)
            emotional_labels = batch['emotional_labels'].to(self.device)
            all_labels = batch['all_labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.sequence_length > 1:
                results = self.model(images)
            else:
                # For single frame, add sequence dimension
                images = images.unsqueeze(1)
                results = self.model(images)
            
            # Calculate losses
            physical_loss = self.criterion(results['physical_logits'], physical_labels)
            emotional_loss = self.criterion(results['emotional_logits'], emotional_labels)
            total_loss_batch = physical_loss + emotional_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            
            # Calculate accuracy (threshold at 0.5)
            all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
            predicted = (torch.sigmoid(all_pred) > 0.5).float()
            correct_predictions += (predicted == all_labels).sum().item()
            total_predictions += all_labels.numel()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                if self.sequence_length > 1:
                    images = batch['images'].to(self.device)
                else:
                    images = batch['image'].to(self.device)
                
                physical_labels = batch['physical_labels'].to(self.device)
                emotional_labels = batch['emotional_labels'].to(self.device)
                all_labels = batch['all_labels'].to(self.device)
                
                # Forward pass
                if self.sequence_length > 1:
                    results = self.model(images)
                else:
                    images = images.unsqueeze(1)
                    results = self.model(images)
                
                # Calculate losses
                physical_loss = self.criterion(results['physical_logits'], physical_labels)
                emotional_loss = self.criterion(results['emotional_logits'], emotional_labels)
                total_loss_batch = physical_loss + emotional_loss
                
                total_loss += total_loss_batch.item()
                
                # Calculate accuracy
                all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
                predicted = (torch.sigmoid(all_pred) > 0.5).float()
                correct_predictions += (predicted == all_labels).sum().item()
                total_predictions += all_labels.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop"""
        
        print(f"\n🎯 Starting training for {self.num_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        
        # Load best model for evaluation
        self.load_model('best_model.pth')
        
    def evaluate(self) -> Dict:
        """Comprehensive evaluation on test set"""
        
        print(f"\n📈 Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_participants = []
        all_frame_ids = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move to device
                if self.sequence_length > 1:
                    images = batch['images'].to(self.device)
                else:
                    images = batch['image'].to(self.device)
                
                physical_labels = batch['physical_labels'].to(self.device)
                emotional_labels = batch['emotional_labels'].to(self.device)
                all_labels_batch = batch['all_labels'].to(self.device)
                
                # Forward pass
                if self.sequence_length > 1:
                    results = self.model(images)
                else:
                    images = images.unsqueeze(1)
                    results = self.model(images)
                
                # Combine predictions
                all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
                
                # Apply sigmoid and collect results
                predictions = torch.sigmoid(all_pred).cpu().numpy()
                labels = all_labels_batch.cpu().numpy()
                
                all_predictions.append(predictions)
                all_labels.append(labels)
                all_participants.extend(batch['participant_id'])
                all_frame_ids.extend(batch['frame_id'].tolist())
        
        # Concatenate all results
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Calculate metrics
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = labels.astype(int)
        
        # Overall metrics (simplified calculation for multi-label)
        accuracy = (binary_predictions == binary_labels).mean()
        
        # Calculate per-feature metrics first, then average
        precisions, recalls, f1s = [], [], []
        for i in range(binary_labels.shape[1]):
            try:
                p, r, f, _ = precision_recall_fscore_support(
                    binary_labels[:, i], binary_predictions[:, i], 
                    average='binary', zero_division=0
                )
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            except:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
        
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
        
        # Per-feature metrics
        feature_metrics = {}
        for i, feature_name in enumerate(self.feature_info['all']):
            try:
                # Skip features with no positive samples
                if binary_labels[:, i].sum() == 0:
                    feature_metrics[feature_name] = {
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0
                    }
                else:
                    p, r, f, _ = precision_recall_fscore_support(
                        binary_labels[:, i], binary_predictions[:, i], 
                        average='binary', zero_division=0
                    )
                    try:
                        auc = roc_auc_score(binary_labels[:, i], predictions[:, i])
                    except ValueError:
                        auc = 0.0
                    
                    feature_metrics[feature_name] = {
                        'precision': p, 'recall': r, 'f1': f, 'auc': auc
                    }
            except Exception as e:
                feature_metrics[feature_name] = {
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0
                }
        
        results = {
            'overall_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            },
            'feature_metrics': feature_metrics,
            'predictions': predictions,
            'labels': labels,
            'participants': all_participants,
            'frame_ids': all_frame_ids
        }
        
        print(f"✅ Evaluation complete:")
        print(f"   Overall F1: {f1:.4f}")
        print(f"   Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        print(f"   Test samples: {len(predictions)}")
        
        return results
    
    def evaluate_all_participants(self) -> Dict:
        """Comprehensive evaluation on ALL participants (train + val + test)"""
        
        print(f"\n📈 Evaluating on ALL participants...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_participants = []
        all_frame_ids = []
        
        # Evaluate on all data loaders
        data_loaders = [
            ("train", self.train_loader),
            ("val", self.val_loader), 
            ("test", self.test_loader)
        ]
        
        with torch.no_grad():
            for split_name, data_loader in data_loaders:
                print(f"   Processing {split_name} set...")
                
                for batch in data_loader:
                    # Move to device
                    if self.sequence_length > 1:
                        images = batch['images'].to(self.device)
                    else:
                        images = batch['image'].to(self.device)
                    
                    physical_labels = batch['physical_labels'].to(self.device)
                    emotional_labels = batch['emotional_labels'].to(self.device)
                    all_labels_batch = batch['all_labels'].to(self.device)
                    
                    # Forward pass
                    if self.sequence_length > 1:
                        results = self.model(images)
                    else:
                        images = images.unsqueeze(1)
                        results = self.model(images)
                    
                    # Combine predictions
                    all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
                    
                    # Apply sigmoid and collect results
                    predictions = torch.sigmoid(all_pred).cpu().numpy()
                    labels = all_labels_batch.cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_labels.append(labels)
                    all_participants.extend(batch['participant_id'])
                    all_frame_ids.extend(batch['frame_id'].tolist())
        
        # Concatenate all results
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Calculate metrics
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = labels.astype(int)
        
        # Overall metrics (simplified calculation for multi-label)
        accuracy = (binary_predictions == binary_labels).mean()
        
        # Calculate per-feature metrics first, then average
        precisions, recalls, f1s = [], [], []
        for i in range(binary_labels.shape[1]):
            try:
                p, r, f, _ = precision_recall_fscore_support(
                    binary_labels[:, i], binary_predictions[:, i], 
                    average='binary', zero_division=0
                )
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            except:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
        
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
        
        # Per-feature metrics
        feature_metrics = {}
        for i, feature_name in enumerate(self.feature_info['all']):
            try:
                # Skip features with no positive samples
                if binary_labels[:, i].sum() == 0:
                    feature_metrics[feature_name] = {
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0
                    }
                else:
                    p, r, f, _ = precision_recall_fscore_support(
                        binary_labels[:, i], binary_predictions[:, i], 
                        average='binary', zero_division=0
                    )
                    try:
                        auc = roc_auc_score(binary_labels[:, i], predictions[:, i])
                    except ValueError:
                        auc = 0.0
                    
                    feature_metrics[feature_name] = {
                        'precision': p, 'recall': r, 'f1': f, 'auc': auc
                    }
            except Exception as e:
                feature_metrics[feature_name] = {
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0
                }
        
        results = {
            'overall_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            },
            'feature_metrics': feature_metrics,
            'predictions': predictions,
            'labels': labels,
            'participants': all_participants,
            'frame_ids': all_frame_ids
        }
        
        print(f"✅ All participants evaluation complete:")
        print(f"   Overall F1: {f1:.4f}")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Total samples: {len(predictions)}")
        print(f"   Unique participants: {len(set(all_participants))}")
        
        return results
    
    def save_model(self, filename: str):
        """Save model state"""
        save_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_info': self.feature_info,
            'history': self.history,
            'config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'num_physical_features': self.num_physical_features,
                'num_emotional_features': self.num_emotional_features
            }
        }, save_path)
    
    def load_model(self, filename: str):
        """Load model state"""
        load_path = self.output_dir / filename
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(self.history['val_acc'], label='Val Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_training(self):
        """Run complete training pipeline"""
        
        print(f"🚀 Starting Complete Training Pipeline")
        print(f"=" * 60)
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training
        self.train()
        
        # Evaluation on all participants (addressing colleague requirement)
        results = self.evaluate_all_participants()
        
        # Save outputs
        self.plot_training_history()
        
        # Generate verification outputs
        self.verifier.create_verification_outputs(
            results['predictions'],
            results['labels'],
            results['participants'],
            results['frame_ids'],
            self.feature_info['all']
        )
        
        # Save results
        results_file = self.output_dir / 'evaluation_results.json'
        results_to_save = {
            'overall_metrics': results['overall_metrics'],
            'feature_metrics': results['feature_metrics'],
            'config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n🎉 Complete training pipeline finished!")
        print(f"   Results saved to: {self.output_dir}")
        print(f"   Overall F1 Score: {results['overall_metrics']['f1']:.4f}")
        
        return results

def main():
    """Main training function"""
    
    # Configuration
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    output_dir = "/home/rohan/Multimodal/multimodal_video_ml/outputs/enhanced_training"
    
    # Create trainer
    trainer = EnhancedMultiLabelTrainer(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        model_type='vit',
        sequence_length=10,
        batch_size=8,  # Smaller batch for memory efficiency
        learning_rate=0.0001,
        num_epochs=30
    )
    
    # Run training
    results = trainer.run_complete_training()
    
    return results

if __name__ == "__main__":
    main()