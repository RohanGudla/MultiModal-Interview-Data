#!/usr/bin/env python3
"""
Complete Enhanced Training Pipeline for ALL Participants
Implements real temporal boundary prediction and comprehensive verification
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from data.complete_dataset import create_complete_dataloaders
from models.temporal_multilabel import TemporalMultiLabelViT, TemporalMultiLabelResNet
from utils.enhanced_verification import EnhancedOutputVerifier

class CompleteMultiLabelTrainer:
    """Complete trainer for ALL participants with real temporal modeling"""
    
    def __init__(self,
                 frames_dir: str,
                 annotations_dir: str,
                 output_dir: str,
                 model_type: str = 'vit',
                 sequence_length: int = 1,
                 batch_size: int = 16,
                 learning_rate: float = 0.001,
                 num_epochs: int = 50,
                 device: str = 'auto',
                 temporal_boundary_weight: float = 0.3):
        """
        Initialize complete trainer
        
        Args:
            temporal_boundary_weight: Weight for temporal boundary prediction loss
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
        self.temporal_boundary_weight = temporal_boundary_weight
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Complete Multi-Label Trainer Initialized")
        print(f"   Model: {model_type}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        print(f"   Temporal boundary weight: {temporal_boundary_weight}")
        print(f"   Output dir: {output_dir}")
        
        # Initialize components
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.boundary_criterion = None
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'train_boundary_loss': [], 'val_boundary_loss': []
        }
        
        # Setup output verification
        self.verifier = EnhancedOutputVerifier(str(self.output_dir))
        
    def setup_data(self):
        """Setup data loaders for all participants"""
        
        print(f"\n📊 Setting up data loaders for ALL participants...")
        
        self.train_loader, self.val_loader, self.test_loader, self.dataset = create_complete_dataloaders(
            frames_dir=self.frames_dir,
            annotations_dir=self.annotations_dir,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            split_by_participant=True,
            num_workers=0  # Disable multiprocessing for stability
        )
        
        # Get feature information
        self.feature_info = self.dataset.get_feature_names()
        self.num_physical_features = len(self.feature_info['physical'])
        self.num_emotional_features = len(self.feature_info['emotional'])
        self.num_total_features = len(self.feature_info['all'])
        
        participants = self.dataset.get_participants_with_data()
        
        print(f"✅ Data setup complete:")
        print(f"   Participants: {len(participants)} {participants}")
        print(f"   Total samples: {len(self.dataset)}")
        print(f"   Physical features: {self.num_physical_features}")
        print(f"   Emotional features: {self.num_emotional_features}")
        print(f"   Total features: {self.num_total_features}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
        
    def setup_model(self):
        """Setup model, optimizer, and loss functions"""
        
        print(f"\n🧠 Setting up model with temporal boundary prediction...")
        
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
        
        # Setup loss functions
        self.criterion = nn.BCEWithLogitsLoss()  # For feature prediction
        self.boundary_criterion = nn.BCEWithLogitsLoss()  # For temporal boundaries
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model setup complete:")
        print(f"   Architecture: {self.model_type}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Feature prediction loss: BCEWithLogitsLoss")
        print(f"   Temporal boundary loss: BCEWithLogitsLoss (weight: {self.temporal_boundary_weight})")
        
    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch with temporal boundary prediction"""
        
        self.model.train()
        total_loss = 0.0
        total_feature_loss = 0.0
        total_boundary_loss = 0.0
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
            
            # Temporal boundary labels
            start_boundaries = batch['start_boundaries'].to(self.device)
            stop_boundaries = batch['stop_boundaries'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.sequence_length > 1:
                results = self.model(images, return_temporal_boundaries=True)
            else:
                # For single frame, add sequence dimension
                images = images.unsqueeze(1)
                results = self.model(images, return_temporal_boundaries=True)
            
            # Feature prediction losses
            physical_loss = self.criterion(results['physical_logits'], physical_labels)
            emotional_loss = self.criterion(results['emotional_logits'], emotional_labels)
            feature_loss = physical_loss + emotional_loss
            
            # Temporal boundary losses (if model supports it)
            boundary_loss = 0.0
            if 'boundary_logits' in results:
                start_loss = self.boundary_criterion(
                    results['boundary_logits'][:, :, 0], start_boundaries
                )
                stop_loss = self.boundary_criterion(
                    results['boundary_logits'][:, :, 1], stop_boundaries
                )
                boundary_loss = (start_loss + stop_loss) / 2
            
            # Total loss
            total_loss_batch = feature_loss + self.temporal_boundary_weight * boundary_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_feature_loss += feature_loss.item()
            total_boundary_loss += boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss
            
            # Calculate accuracy (threshold at 0.5)
            all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
            predicted = (torch.sigmoid(all_pred) > 0.5).float()
            correct_predictions += (predicted == all_labels).sum().item()
            total_predictions += all_labels.numel()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_feature_loss = total_feature_loss / len(self.train_loader)
        avg_boundary_loss = total_boundary_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy, avg_boundary_loss
    
    def validate_epoch(self) -> Tuple[float, float, float]:
        """Validate for one epoch with temporal boundary evaluation"""
        
        self.model.eval()
        total_loss = 0.0
        total_feature_loss = 0.0
        total_boundary_loss = 0.0
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
                
                # Temporal boundary labels
                start_boundaries = batch['start_boundaries'].to(self.device)
                stop_boundaries = batch['stop_boundaries'].to(self.device)
                
                # Forward pass
                if self.sequence_length > 1:
                    results = self.model(images, return_temporal_boundaries=True)
                else:
                    images = images.unsqueeze(1)
                    results = self.model(images, return_temporal_boundaries=True)
                
                # Feature prediction losses
                physical_loss = self.criterion(results['physical_logits'], physical_labels)
                emotional_loss = self.criterion(results['emotional_logits'], emotional_labels)
                feature_loss = physical_loss + emotional_loss
                
                # Temporal boundary losses
                boundary_loss = 0.0
                if 'boundary_logits' in results:
                    start_loss = self.boundary_criterion(
                        results['boundary_logits'][:, :, 0], start_boundaries
                    )
                    stop_loss = self.boundary_criterion(
                        results['boundary_logits'][:, :, 1], stop_boundaries
                    )
                    boundary_loss = (start_loss + stop_loss) / 2
                
                # Total loss
                total_loss_batch = feature_loss + self.temporal_boundary_weight * boundary_loss
                
                total_loss += total_loss_batch.item()
                total_feature_loss += feature_loss.item()
                total_boundary_loss += boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss
                
                # Calculate accuracy
                all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
                predicted = (torch.sigmoid(all_pred) > 0.5).float()
                correct_predictions += (predicted == all_labels).sum().item()
                total_predictions += all_labels.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_feature_loss = total_feature_loss / len(self.val_loader)
        avg_boundary_loss = total_boundary_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy, avg_boundary_loss
    
    def train(self):
        """Main training loop with temporal boundary learning"""
        
        print(f"\n🎯 Starting training for {self.num_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc, train_boundary_loss = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_boundary_loss = self.validate_epoch()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_boundary_loss'].append(train_boundary_loss)
            self.history['val_boundary_loss'].append(val_boundary_loss)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Boundary Loss: {train_boundary_loss:.4f} | "
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
    
    def evaluate_all_participants(self) -> Dict:
        """Comprehensive evaluation on ALL participants with temporal analysis"""
        
        print(f"\n📈 Evaluating on ALL participants with temporal analysis...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_participants = []
        all_frame_ids = []
        all_temporal_predictions = []
        
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
                        results = self.model(images, return_temporal_boundaries=True)
                    else:
                        images = images.unsqueeze(1)
                        results = self.model(images, return_temporal_boundaries=True)
                    
                    # Combine predictions
                    all_pred = torch.cat([results['physical_logits'], results['emotional_logits']], dim=1)
                    
                    # Apply sigmoid and collect results
                    predictions = torch.sigmoid(all_pred).cpu().numpy()
                    labels = all_labels_batch.cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_labels.append(labels)
                    all_participants.extend(batch['participant_id'])
                    all_frame_ids.extend(batch['frame_id'].tolist())
                    
                    # Collect temporal boundary predictions
                    if 'boundary_probs' in results:
                        temporal_preds = results['boundary_probs'].cpu().numpy()
                        all_temporal_predictions.append(temporal_preds)
        
        # Concatenate all results
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Calculate metrics
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = labels.astype(int)
        
        # Overall metrics
        accuracy = (binary_predictions == binary_labels).mean()
        
        # Calculate per-feature metrics
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
            'frame_ids': all_frame_ids,
            'temporal_predictions': all_temporal_predictions if all_temporal_predictions else None
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
                'num_emotional_features': self.num_emotional_features,
                'temporal_boundary_weight': self.temporal_boundary_weight
            }
        }, save_path)
    
    def load_model(self, filename: str):
        """Load model state"""
        load_path = self.output_dir / filename
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def run_complete_training(self):
        """Run complete training pipeline for ALL participants"""
        
        print(f"🚀 Starting Complete Training Pipeline for ALL Participants")
        print(f"=" * 70)
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training
        self.train()
        
        # Evaluation on ALL participants
        results = self.evaluate_all_participants()
        
        # Generate verification outputs
        self.verifier.create_verification_outputs(
            results['predictions'],
            results['labels'],
            results['participants'],
            results['frame_ids'],
            self.feature_info['all']
        )
        
        # Save results
        results_file = self.output_dir / 'complete_evaluation_results.json'
        results_to_save = {
            'overall_metrics': results['overall_metrics'],
            'feature_metrics': results['feature_metrics'],
            'total_participants': len(set(results['participants'])),
            'total_samples': len(results['predictions']),
            'config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'temporal_boundary_weight': self.temporal_boundary_weight
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n🎉 Complete training pipeline finished!")
        print(f"   Results saved to: {self.output_dir}")
        print(f"   Participants processed: {len(set(results['participants']))}")
        print(f"   Total samples: {len(results['predictions'])}")
        print(f"   Overall F1 Score: {results['overall_metrics']['f1']:.4f}")
        print(f"   Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        
        return results

def main():
    """Test the complete trainer"""
    
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_annotations"
    output_dir = "/home/rohan/Multimodal/multimodal_video_ml/outputs/complete_training"
    
    # Create trainer
    trainer = CompleteMultiLabelTrainer(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        model_type='vit',
        sequence_length=1,
        batch_size=4,
        learning_rate=0.0001,
        num_epochs=10,
        temporal_boundary_weight=0.3
    )
    
    # Run training
    results = trainer.run_complete_training()
    
    return results

if __name__ == "__main__":
    main()