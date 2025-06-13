#!/usr/bin/env python3
"""
Output Verification System
Saves model predictions and ground truth for manual verification
Handles all annotation types with temporal information
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

class OutputVerificationSystem:
    """
    System to save, compare, and verify model outputs against ground truth
    """
    
    def __init__(self, 
                 output_dir: str,
                 feature_names: Dict[str, List[str]],
                 save_individual_predictions: bool = True,
                 save_aggregated_results: bool = True,
                 create_visualizations: bool = True):
        """
        Initialize verification system
        
        Args:
            output_dir: Directory to save verification outputs
            feature_names: Dict with 'physical', 'emotional', 'all' feature lists
            save_individual_predictions: Save per-sample predictions
            save_aggregated_results: Save aggregated performance metrics
            create_visualizations: Create comparison visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_names = feature_names
        self.save_individual = save_individual_predictions
        self.save_aggregated = save_aggregated_results
        self.create_visualizations = create_visualizations
        
        # Create subdirectories
        self.predictions_dir = self.output_dir / "predictions"
        self.comparisons_dir = self.output_dir / "comparisons"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        if self.save_individual:
            self.predictions_dir.mkdir(exist_ok=True)
        if self.save_aggregated:
            self.comparisons_dir.mkdir(exist_ok=True)
        if self.create_visualizations:
            self.visualizations_dir.mkdir(exist_ok=True)
        
        # Storage for batch processing
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_metadata = []
        
    def add_batch_predictions(self,
                            predictions: Dict[str, torch.Tensor],
                            ground_truth: Dict[str, torch.Tensor],
                            metadata: Dict[str, Union[List, torch.Tensor]]):
        """
        Add a batch of predictions for later processing
        
        Args:
            predictions: Model predictions (probabilities)
            ground_truth: True labels
            metadata: Additional information (participant_id, frame_id, temporal_info, etc.)
        """
        
        batch_size = predictions['combined_probs'].shape[0]
        
        for i in range(batch_size):
            # Extract predictions for this sample
            sample_pred = {
                'physical_probs': predictions['physical_probs'][i].cpu().numpy(),
                'emotional_probs': predictions['emotional_probs'][i].cpu().numpy(),
                'combined_probs': predictions['combined_probs'][i].cpu().numpy()
            }
            
            # Add boundary predictions if available
            if 'boundary_probs' in predictions:
                sample_pred['boundary_probs'] = predictions['boundary_probs'][i].cpu().numpy()
                sample_pred['start_probs'] = predictions['start_probs'][i].cpu().numpy()
                sample_pred['stop_probs'] = predictions['stop_probs'][i].cpu().numpy()
            
            # Extract ground truth for this sample
            sample_gt = {
                'physical_labels': ground_truth['physical_labels'][i].cpu().numpy(),
                'emotional_labels': ground_truth['emotional_labels'][i].cpu().numpy(),
                'all_labels': ground_truth['all_labels'][i].cpu().numpy()
            }
            
            # Extract metadata for this sample
            sample_meta = {}
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    sample_meta[key] = value[i] if i < len(value) else None
                elif isinstance(value, torch.Tensor):
                    sample_meta[key] = value[i].item() if value.ndim > 0 else value.item()
                else:
                    sample_meta[key] = value
            
            self.all_predictions.append(sample_pred)
            self.all_ground_truth.append(sample_gt)
            self.all_metadata.append(sample_meta)
    
    def save_individual_predictions(self, timestamp: str = None):
        """Save individual sample predictions to CSV"""
        
        if not self.save_individual or not self.all_predictions:
            return None
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for CSV
        rows = []
        
        for pred, gt, meta in zip(self.all_predictions, self.all_ground_truth, self.all_metadata):
            row = {
                'participant_id': meta.get('participant_id', 'unknown'),
                'frame_id': meta.get('frame_id', -1),
                'timestamp_seconds': meta.get('temporal_info', {}).get('timestamp_seconds', -1)
            }
            
            # Add physical features
            for i, feature in enumerate(self.feature_names['physical']):
                row[f'pred_physical_{feature}'] = pred['physical_probs'][i]
                row[f'true_physical_{feature}'] = gt['physical_labels'][i]
                row[f'correct_physical_{feature}'] = int((pred['physical_probs'][i] > 0.5) == gt['physical_labels'][i])
            
            # Add emotional features
            for i, feature in enumerate(self.feature_names['emotional']):
                row[f'pred_emotional_{feature}'] = pred['emotional_probs'][i]
                row[f'true_emotional_{feature}'] = gt['emotional_labels'][i]
                row[f'correct_emotional_{feature}'] = int((pred['emotional_probs'][i] > 0.5) == gt['emotional_labels'][i])
            
            # Add boundary predictions if available
            if 'boundary_probs' in pred:
                for i, feature in enumerate(self.feature_names['all']):
                    row[f'pred_start_{feature}'] = pred['start_probs'][i]
                    row[f'pred_stop_{feature}'] = pred['stop_probs'][i]
            
            rows.append(row)
        
        # Save to CSV
        df = pd.DataFrame(rows)
        output_file = self.predictions_dir / f"individual_predictions_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"📄 Saved individual predictions: {output_file}")
        return output_file
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        if not self.all_predictions:
            return {}
        
        # Aggregate predictions and ground truth
        all_pred_physical = np.array([p['physical_probs'] for p in self.all_predictions])
        all_pred_emotional = np.array([p['emotional_probs'] for p in self.all_predictions])
        all_pred_combined = np.array([p['combined_probs'] for p in self.all_predictions])
        
        all_gt_physical = np.array([gt['physical_labels'] for gt in self.all_ground_truth])
        all_gt_emotional = np.array([gt['emotional_labels'] for gt in self.all_ground_truth])
        all_gt_combined = np.array([gt['all_labels'] for gt in self.all_ground_truth])
        
        # Convert probabilities to binary predictions
        pred_physical_binary = (all_pred_physical > 0.5).astype(int)
        pred_emotional_binary = (all_pred_emotional > 0.5).astype(int)
        pred_combined_binary = (all_pred_combined > 0.5).astype(int)
        
        metrics = {}
        
        # Physical feature metrics
        physical_accuracy = (pred_physical_binary == all_gt_physical).mean(axis=0)
        physical_precision = []
        physical_recall = []
        physical_f1 = []
        
        for i in range(len(self.feature_names['physical'])):
            tp = ((pred_physical_binary[:, i] == 1) & (all_gt_physical[:, i] == 1)).sum()
            fp = ((pred_physical_binary[:, i] == 1) & (all_gt_physical[:, i] == 0)).sum()
            fn = ((pred_physical_binary[:, i] == 0) & (all_gt_physical[:, i] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            physical_precision.append(precision)
            physical_recall.append(recall)
            physical_f1.append(f1)
        
        # Emotional feature metrics
        emotional_accuracy = (pred_emotional_binary == all_gt_emotional).mean(axis=0)
        emotional_precision = []
        emotional_recall = []
        emotional_f1 = []
        
        for i in range(len(self.feature_names['emotional'])):
            tp = ((pred_emotional_binary[:, i] == 1) & (all_gt_emotional[:, i] == 1)).sum()
            fp = ((pred_emotional_binary[:, i] == 1) & (all_gt_emotional[:, i] == 0)).sum()
            fn = ((pred_emotional_binary[:, i] == 0) & (all_gt_emotional[:, i] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            emotional_precision.append(precision)
            emotional_recall.append(recall)
            emotional_f1.append(f1)
        
        # Overall metrics
        overall_accuracy = (pred_combined_binary == all_gt_combined).mean()
        
        metrics = {
            'physical': {
                'features': self.feature_names['physical'],
                'accuracy': physical_accuracy.tolist(),
                'precision': physical_precision,
                'recall': physical_recall,
                'f1_score': physical_f1,
                'mean_accuracy': float(physical_accuracy.mean()),
                'mean_precision': float(np.mean(physical_precision)),
                'mean_recall': float(np.mean(physical_recall)),
                'mean_f1': float(np.mean(physical_f1))
            },
            'emotional': {
                'features': self.feature_names['emotional'],
                'accuracy': emotional_accuracy.tolist(),
                'precision': emotional_precision,
                'recall': emotional_recall,
                'f1_score': emotional_f1,
                'mean_accuracy': float(emotional_accuracy.mean()),
                'mean_precision': float(np.mean(emotional_precision)),
                'mean_recall': float(np.mean(emotional_recall)),
                'mean_f1': float(np.mean(emotional_f1))
            },
            'overall': {
                'accuracy': float(overall_accuracy),
                'total_samples': len(self.all_predictions),
                'total_features': len(self.feature_names['all'])
            }
        }
        
        return metrics
    
    def save_performance_summary(self, metrics: Dict, timestamp: str = None):
        """Save performance metrics summary"""
        
        if not self.save_aggregated:
            return None
            
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed metrics as JSON
        json_file = self.comparisons_dir / f"performance_metrics_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create summary CSV
        summary_rows = []
        
        # Physical features
        for i, feature in enumerate(metrics['physical']['features']):
            summary_rows.append({
                'feature_type': 'physical',
                'feature_name': feature,
                'accuracy': metrics['physical']['accuracy'][i],
                'precision': metrics['physical']['precision'][i],
                'recall': metrics['physical']['recall'][i],
                'f1_score': metrics['physical']['f1_score'][i]
            })
        
        # Emotional features
        for i, feature in enumerate(metrics['emotional']['features']):
            summary_rows.append({
                'feature_type': 'emotional',
                'feature_name': feature,
                'accuracy': metrics['emotional']['accuracy'][i],
                'precision': metrics['emotional']['precision'][i],
                'recall': metrics['emotional']['recall'][i],
                'f1_score': metrics['emotional']['f1_score'][i]
            })
        
        summary_df = pd.DataFrame(summary_rows)
        csv_file = self.comparisons_dir / f"performance_summary_{timestamp}.csv"
        summary_df.to_csv(csv_file, index=False)
        
        print(f"📊 Saved performance metrics: {json_file}")
        print(f"📊 Saved performance summary: {csv_file}")
        
        return json_file, csv_file
    
    def create_verification_visualizations(self, metrics: Dict, timestamp: str = None):
        """Create visualizations for verification"""
        
        if not self.create_visualizations:
            return []
            
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        visualization_files = []
        
        # 1. Feature accuracy comparison
        plt.figure(figsize=(15, 8))
        
        # Physical features
        plt.subplot(2, 1, 1)
        physical_acc = metrics['physical']['accuracy']
        physical_features = metrics['physical']['features']
        
        bars1 = plt.bar(range(len(physical_features)), physical_acc, alpha=0.7, color='blue')
        plt.title('Physical Feature Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(physical_features)), physical_features, rotation=45, ha='right')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
        plt.ylim(0, 1)
        plt.legend()
        
        # Emotional features
        plt.subplot(2, 1, 2)
        emotional_acc = metrics['emotional']['accuracy']
        emotional_features = metrics['emotional']['features']
        
        bars2 = plt.bar(range(len(emotional_features)), emotional_acc, alpha=0.7, color='green')
        plt.title('Emotional Feature Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Features')
        plt.xticks(range(len(emotional_features)), emotional_features, rotation=45, ha='right')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        acc_file = self.visualizations_dir / f"feature_accuracy_{timestamp}.png"
        plt.savefig(acc_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(acc_file)
        
        # 2. Performance metrics heatmap
        plt.figure(figsize=(12, 8))
        
        # Combine all metrics for heatmap
        all_features = metrics['physical']['features'] + metrics['emotional']['features']
        all_accuracy = metrics['physical']['accuracy'] + metrics['emotional']['accuracy']
        all_precision = metrics['physical']['precision'] + metrics['emotional']['precision']
        all_recall = metrics['physical']['recall'] + metrics['emotional']['recall']
        all_f1 = metrics['physical']['f1_score'] + metrics['emotional']['f1_score']
        
        heatmap_data = np.array([all_accuracy, all_precision, all_recall, all_f1])
        
        sns.heatmap(heatmap_data, 
                   xticklabels=all_features,
                   yticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Score'})
        
        plt.title('Performance Metrics Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        heatmap_file = self.visualizations_dir / f"metrics_heatmap_{timestamp}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(heatmap_file)
        
        print(f"🎨 Created visualizations: {len(visualization_files)} files")
        for file in visualization_files:
            print(f"   {file}")
        
        return visualization_files
    
    def generate_verification_report(self, timestamp: str = None):
        """Generate comprehensive verification report"""
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"📋 Generating verification report ({timestamp})...")
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        # Save individual predictions
        pred_file = None
        if self.save_individual:
            pred_file = self.save_individual_predictions(timestamp)
        
        # Save performance summary
        json_file, csv_file = None, None
        if self.save_aggregated:
            json_file, csv_file = self.save_performance_summary(metrics, timestamp)
        
        # Create visualizations
        viz_files = []
        if self.create_visualizations:
            viz_files = self.create_verification_visualizations(metrics, timestamp)
        
        # Create summary report
        report = {
            'timestamp': timestamp,
            'summary': {
                'total_samples': len(self.all_predictions),
                'total_features': len(self.feature_names['all']),
                'physical_features': len(self.feature_names['physical']),
                'emotional_features': len(self.feature_names['emotional'])
            },
            'performance': {
                'overall_accuracy': metrics['overall']['accuracy'],
                'physical_mean_accuracy': metrics['physical']['mean_accuracy'],
                'emotional_mean_accuracy': metrics['emotional']['mean_accuracy'],
                'physical_mean_f1': metrics['physical']['mean_f1'],
                'emotional_mean_f1': metrics['emotional']['mean_f1']
            },
            'files': {
                'individual_predictions': str(pred_file) if pred_file else None,
                'performance_metrics': str(json_file) if json_file else None,
                'performance_summary': str(csv_file) if csv_file else None,
                'visualizations': [str(f) for f in viz_files]
            }
        }
        
        report_file = self.output_dir / f"verification_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Verification report complete:")
        print(f"   Overall accuracy: {metrics['overall']['accuracy']:.3f}")
        print(f"   Physical mean accuracy: {metrics['physical']['mean_accuracy']:.3f}")
        print(f"   Emotional mean accuracy: {metrics['emotional']['mean_accuracy']:.3f}")
        print(f"   Report saved: {report_file}")
        
        return report, report_file
    
    def clear_batch_data(self):
        """Clear stored batch data"""
        self.all_predictions.clear()
        self.all_ground_truth.clear()
        self.all_metadata.clear()

# Test the verification system
def test_verification_system():
    """Test the output verification system"""
    
    print("🧪 Testing Output Verification System...")
    
    # Create dummy feature names
    feature_names = {
        'physical': [f'Physical_Feature_{i}' for i in range(5)],
        'emotional': [f'Emotional_Feature_{i}' for i in range(3)],
        'all': [f'Feature_{i}' for i in range(8)]
    }
    
    # Initialize verification system
    verifier = OutputVerificationSystem(
        output_dir="/tmp/test_verification",
        feature_names=feature_names
    )
    
    # Add some dummy predictions
    batch_size = 10
    for batch in range(3):
        predictions = {
            'physical_probs': torch.rand(batch_size, 5),
            'emotional_probs': torch.rand(batch_size, 3),
            'combined_probs': torch.rand(batch_size, 8)
        }
        
        ground_truth = {
            'physical_labels': torch.randint(0, 2, (batch_size, 5), dtype=torch.float32),
            'emotional_labels': torch.randint(0, 2, (batch_size, 3), dtype=torch.float32),
            'all_labels': torch.randint(0, 2, (batch_size, 8), dtype=torch.float32)
        }
        
        metadata = {
            'participant_id': [f'P{i}' for i in range(batch_size)],
            'frame_id': torch.arange(batch * batch_size, (batch + 1) * batch_size),
            'temporal_info': [{'timestamp_seconds': i * 1.0} for i in range(batch_size)]
        }
        
        verifier.add_batch_predictions(predictions, ground_truth, metadata)
    
    # Generate verification report
    report, report_file = verifier.generate_verification_report()
    
    print(f"✅ Verification system test complete!")
    print(f"   Report file: {report_file}")

if __name__ == "__main__":
    test_verification_system()