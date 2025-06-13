#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework
Evaluates trained models with temporal metrics and detailed analysis
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm
import argparse

# Import our modules
from data.multilabel_dataset import MultiLabelAnnotationDataset, create_dataloaders
from models.temporal_multilabel import create_temporal_model
from utils.output_verification import OutputVerificationSystem

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for temporal multi-label models
    """
    
    def __init__(self,
                 model_path: str,
                 frames_dir: str,
                 annotations_dir: str,
                 output_dir: str,
                 device: str = 'auto'):
        
        self.model_path = Path(model_path)
        self.frames_dir = frames_dir
        self.annotations_dir = annotations_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔍 Evaluator initialized:")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        
        # Will be initialized in setup()
        self.model = None
        self.model_config = None
        self.test_loader = None
        self.dataset = None
        self.feature_names = None
        
    def load_model(self):
        """Load trained model from checkpoint"""
        
        print("📥 Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = checkpoint['model_config']
        
        print(f"   Model type: {self.model_config['model_type']}")
        print(f"   Sequence length: {self.model_config['sequence_length']}")
        print(f"   Physical features: {self.model_config['num_physical_features']}")
        print(f"   Emotional features: {self.model_config['num_emotional_features']}")
        
        # Create model
        self.model = create_temporal_model(
            model_type=self.model_config['model_type'],
            num_physical_features=self.model_config['num_physical_features'],
            num_emotional_features=self.model_config['num_emotional_features'],
            sequence_length=self.model_config['sequence_length']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from epoch {checkpoint['epoch']+1}")
        
    def setup_data(self):
        """Setup data loaders"""
        
        print("📚 Setting up data...")
        
        # Create dataset (use test split only)
        _, _, self.test_loader, self.dataset = create_dataloaders(
            frames_dir=self.frames_dir,
            annotations_dir=self.annotations_dir,
            batch_size=16,  # Larger batch for evaluation
            sequence_length=self.model_config['sequence_length'],
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            num_workers=4
        )
        
        # Get feature information
        self.feature_names = self.dataset.get_feature_names()
        
        print(f"   Test samples: {len(self.test_loader.dataset)}")
        print(f"   Test batches: {len(self.test_loader)}")
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        
        print("🧪 Evaluating model...")
        
        all_predictions = []
        all_ground_truth = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move data to device
                if self.model_config['sequence_length'] > 1:
                    images = batch['images'].to(self.device)
                else:
                    images = batch['image'].unsqueeze(1).to(self.device)
                
                physical_labels = batch['physical_labels'].to(self.device)
                emotional_labels = batch['emotional_labels'].to(self.device)
                all_labels = batch['all_labels'].to(self.device)
                
                # Forward pass
                predictions = self.model(images, return_temporal_boundaries=True)
                
                # Store results
                batch_size = physical_labels.shape[0]
                for i in range(batch_size):
                    # Predictions
                    pred_dict = {
                        'physical_probs': predictions['physical_probs'][i].cpu().numpy(),
                        'emotional_probs': predictions['emotional_probs'][i].cpu().numpy(),
                        'combined_probs': predictions['combined_probs'][i].cpu().numpy()
                    }
                    
                    if 'boundary_probs' in predictions:
                        pred_dict.update({
                            'boundary_probs': predictions['boundary_probs'][i].cpu().numpy(),
                            'start_probs': predictions['start_probs'][i].cpu().numpy(),
                            'stop_probs': predictions['stop_probs'][i].cpu().numpy()
                        })
                    
                    # Ground truth
                    gt_dict = {
                        'physical_labels': physical_labels[i].cpu().numpy(),
                        'emotional_labels': emotional_labels[i].cpu().numpy(),
                        'all_labels': all_labels[i].cpu().numpy()
                    }
                    
                    # Metadata
                    meta_dict = {
                        'participant_id': batch['participant_id'][i] if isinstance(batch['participant_id'], list) else batch['participant_id'][i].item(),
                        'frame_id': batch['frame_id'][i].item(),
                        'temporal_info': batch.get('temporal_info', [{}])[i] if batch.get('temporal_info') else {}
                    }
                    
                    all_predictions.append(pred_dict)
                    all_ground_truth.append(gt_dict)
                    all_metadata.append(meta_dict)
        
        print(f"✅ Evaluated {len(all_predictions)} samples")
        
        return all_predictions, all_ground_truth, all_metadata
    
    def calculate_detailed_metrics(self, predictions, ground_truth):
        """Calculate detailed performance metrics"""
        
        print("📊 Calculating detailed metrics...")
        
        # Convert to arrays
        all_pred_physical = np.array([p['physical_probs'] for p in predictions])
        all_pred_emotional = np.array([p['emotional_probs'] for p in predictions])
        all_pred_combined = np.array([p['combined_probs'] for p in predictions])
        
        all_gt_physical = np.array([gt['physical_labels'] for gt in ground_truth])
        all_gt_emotional = np.array([gt['emotional_labels'] for gt in ground_truth])
        all_gt_combined = np.array([gt['all_labels'] for gt in ground_truth])
        
        # Binary predictions (threshold = 0.5)
        pred_physical_binary = (all_pred_physical > 0.5).astype(int)
        pred_emotional_binary = (all_pred_emotional > 0.5).astype(int)
        pred_combined_binary = (all_pred_combined > 0.5).astype(int)
        
        metrics = {}
        
        # Physical features
        metrics['physical'] = self._calculate_feature_metrics(
            all_pred_physical, all_gt_physical, pred_physical_binary,
            self.feature_names['physical'], 'Physical'
        )
        
        # Emotional features
        metrics['emotional'] = self._calculate_feature_metrics(
            all_pred_emotional, all_gt_emotional, pred_emotional_binary,
            self.feature_names['emotional'], 'Emotional'
        )
        
        # Overall metrics
        metrics['overall'] = {
            'accuracy': float((pred_combined_binary == all_gt_combined).mean()),
            'samples': len(predictions),
            'features': len(self.feature_names['all'])
        }
        
        # Temporal analysis
        metrics['temporal'] = self._analyze_temporal_patterns(predictions, ground_truth)
        
        return metrics
    
    def _calculate_feature_metrics(self, pred_probs, gt_labels, pred_binary, feature_names, category):
        """Calculate metrics for a specific feature category"""
        
        metrics = {
            'features': feature_names,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_roc': [],
            'auc_pr': []
        }
        
        for i, feature in enumerate(feature_names):
            # Basic metrics
            tp = ((pred_binary[:, i] == 1) & (gt_labels[:, i] == 1)).sum()
            fp = ((pred_binary[:, i] == 1) & (gt_labels[:, i] == 0)).sum()
            fn = ((pred_binary[:, i] == 0) & (gt_labels[:, i] == 1)).sum()
            tn = ((pred_binary[:, i] == 0) & (gt_labels[:, i] == 0)).sum()
            
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)
            
            # AUC metrics (if there are positive and negative samples)\n            try:\n                if len(np.unique(gt_labels[:, i])) > 1:\n                    # ROC AUC\n                    fpr, tpr, _ = roc_curve(gt_labels[:, i], pred_probs[:, i])\n                    auc_roc = auc(fpr, tpr)\n                    \n                    # PR AUC\n                    precision_curve, recall_curve, _ = precision_recall_curve(gt_labels[:, i], pred_probs[:, i])\n                    auc_pr = auc(recall_curve, precision_curve)\n                else:\n                    auc_roc = 0.5  # Random performance for constant labels\n                    auc_pr = gt_labels[:, i].mean()  # Baseline for constant labels\n            except Exception:\n                auc_roc = 0.0\n                auc_pr = 0.0\n            \n            metrics['auc_roc'].append(auc_roc)\n            metrics['auc_pr'].append(auc_pr)\n        \n        # Summary statistics\n        metrics['summary'] = {\n            'mean_accuracy': float(np.mean(metrics['accuracy'])),\n            'mean_precision': float(np.mean(metrics['precision'])),\n            'mean_recall': float(np.mean(metrics['recall'])),\n            'mean_f1': float(np.mean(metrics['f1_score'])),\n            'mean_auc_roc': float(np.mean(metrics['auc_roc'])),\n            'mean_auc_pr': float(np.mean(metrics['auc_pr']))\n        }\n        \n        return metrics\n    \n    def _analyze_temporal_patterns(self, predictions, ground_truth):\n        \"\"\"Analyze temporal patterns in predictions\"\"\"\n        \n        # Group by participant\n        participant_data = {}\n        \n        for pred, gt, meta in zip(predictions, ground_truth, self.all_metadata):\n            participant = meta['participant_id']\n            if participant not in participant_data:\n                participant_data[participant] = []\n            \n            participant_data[participant].append({\n                'frame_id': meta['frame_id'],\n                'timestamp': meta.get('temporal_info', {}).get('timestamp_seconds', meta['frame_id']),\n                'predictions': pred,\n                'ground_truth': gt\n            })\n        \n        # Sort by timestamp\n        for participant in participant_data:\n            participant_data[participant].sort(key=lambda x: x['timestamp'])\n        \n        temporal_metrics = {\n            'participants': list(participant_data.keys()),\n            'num_participants': len(participant_data),\n            'avg_sequence_length': np.mean([len(data) for data in participant_data.values()]),\n            'temporal_consistency': self._calculate_temporal_consistency(participant_data)\n        }\n        \n        return temporal_metrics\n    \n    def _calculate_temporal_consistency(self, participant_data):\n        \"\"\"Calculate temporal consistency of predictions\"\"\"\n        \n        consistencies = []\n        \n        for participant, data in participant_data.items():\n            if len(data) < 2:\n                continue\n            \n            # Calculate frame-to-frame consistency\n            frame_consistencies = []\n            \n            for i in range(len(data) - 1):\n                pred1 = data[i]['predictions']['combined_probs']\n                pred2 = data[i + 1]['predictions']['combined_probs']\n                \n                # Calculate similarity (1 - mean absolute difference)\n                similarity = 1 - np.mean(np.abs(pred1 - pred2))\n                frame_consistencies.append(similarity)\n            \n            if frame_consistencies:\n                consistencies.append(np.mean(frame_consistencies))\n        \n        return {\n            'per_participant': consistencies,\n            'mean_consistency': float(np.mean(consistencies)) if consistencies else 0.0,\n            'std_consistency': float(np.std(consistencies)) if consistencies else 0.0\n        }\n    \n    def create_evaluation_visualizations(self, metrics, timestamp):\n        \"\"\"Create comprehensive evaluation visualizations\"\"\"\n        \n        print(\"🎨 Creating evaluation visualizations...\")\n        \n        viz_dir = self.output_dir / \"visualizations\"\n        viz_dir.mkdir(exist_ok=True)\n        \n        visualization_files = []\n        \n        # 1. Performance comparison chart\n        plt.figure(figsize=(16, 10))\n        \n        # Physical features\n        plt.subplot(2, 2, 1)\n        physical_metrics = ['accuracy', 'precision', 'recall', 'f1_score']\n        physical_values = [metrics['physical']['summary'][f'mean_{m}'] for m in physical_metrics]\n        \n        bars = plt.bar(physical_metrics, physical_values, alpha=0.7, color='blue')\n        plt.title('Physical Features - Average Performance')\n        plt.ylabel('Score')\n        plt.ylim(0, 1)\n        \n        # Add value labels on bars\n        for bar, value in zip(bars, physical_values):\n            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, \n                    f'{value:.3f}', ha='center', va='bottom')\n        \n        # Emotional features\n        plt.subplot(2, 2, 2)\n        emotional_values = [metrics['emotional']['summary'][f'mean_{m}'] for m in physical_metrics]\n        \n        bars = plt.bar(physical_metrics, emotional_values, alpha=0.7, color='green')\n        plt.title('Emotional Features - Average Performance')\n        plt.ylabel('Score')\n        plt.ylim(0, 1)\n        \n        for bar, value in zip(bars, emotional_values):\n            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, \n                    f'{value:.3f}', ha='center', va='bottom')\n        \n        # Feature-wise accuracy comparison\n        plt.subplot(2, 1, 2)\n        all_features = metrics['physical']['features'] + metrics['emotional']['features']\n        all_accuracies = metrics['physical']['accuracy'] + metrics['emotional']['accuracy']\n        feature_colors = ['blue'] * len(metrics['physical']['features']) + ['green'] * len(metrics['emotional']['features'])\n        \n        bars = plt.bar(range(len(all_features)), all_accuracies, color=feature_colors, alpha=0.7)\n        plt.title('Per-Feature Accuracy')\n        plt.xlabel('Features')\n        plt.ylabel('Accuracy')\n        plt.xticks(range(len(all_features)), all_features, rotation=45, ha='right')\n        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')\n        plt.ylim(0, 1)\n        plt.legend()\n        \n        plt.tight_layout()\n        performance_file = viz_dir / f\"performance_overview_{timestamp}.png\"\n        plt.savefig(performance_file, dpi=300, bbox_inches='tight')\n        plt.close()\n        visualization_files.append(performance_file)\n        \n        # 2. ROC and PR curves for top features\n        fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n        \n        # Top 3 physical features by F1 score\n        top_physical_indices = np.argsort(metrics['physical']['f1_score'])[-3:]\n        for i, idx in enumerate(top_physical_indices):\n            ax = axes[0, i]\n            feature_name = metrics['physical']['features'][idx]\n            auc_roc = metrics['physical']['auc_roc'][idx]\n            \n            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)\n            ax.set_xlim([0, 1])\n            ax.set_ylim([0, 1])\n            ax.set_xlabel('False Positive Rate')\n            ax.set_ylabel('True Positive Rate')\n            ax.set_title(f'{feature_name}\\nAUC = {auc_roc:.3f}')\n            ax.grid(True, alpha=0.3)\n        \n        # Top 3 emotional features by F1 score\n        top_emotional_indices = np.argsort(metrics['emotional']['f1_score'])[-3:]\n        for i, idx in enumerate(top_emotional_indices):\n            ax = axes[1, i]\n            feature_name = metrics['emotional']['features'][idx]\n            auc_roc = metrics['emotional']['auc_roc'][idx]\n            \n            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)\n            ax.set_xlim([0, 1])\n            ax.set_ylim([0, 1])\n            ax.set_xlabel('False Positive Rate')\n            ax.set_ylabel('True Positive Rate')\n            ax.set_title(f'{feature_name}\\nAUC = {auc_roc:.3f}')\n            ax.grid(True, alpha=0.3)\n        \n        plt.suptitle('ROC Curves for Top Features', fontsize=16)\n        plt.tight_layout()\n        roc_file = viz_dir / f\"roc_curves_{timestamp}.png\"\n        plt.savefig(roc_file, dpi=300, bbox_inches='tight')\n        plt.close()\n        visualization_files.append(roc_file)\n        \n        # 3. Temporal consistency visualization\n        if metrics['temporal']['temporal_consistency']['per_participant']:\n            plt.figure(figsize=(12, 6))\n            \n            consistencies = metrics['temporal']['temporal_consistency']['per_participant']\n            participants = metrics['temporal']['participants']\n            \n            plt.subplot(1, 2, 1)\n            plt.bar(range(len(consistencies)), consistencies, alpha=0.7)\n            plt.title('Temporal Consistency by Participant')\n            plt.xlabel('Participant')\n            plt.ylabel('Consistency Score')\n            plt.xticks(range(len(participants)), participants, rotation=45)\n            plt.ylim(0, 1)\n            \n            plt.subplot(1, 2, 2)\n            plt.hist(consistencies, bins=10, alpha=0.7, edgecolor='black')\n            plt.title('Distribution of Temporal Consistency')\n            plt.xlabel('Consistency Score')\n            plt.ylabel('Frequency')\n            plt.axvline(np.mean(consistencies), color='red', linestyle='--', \n                       label=f'Mean: {np.mean(consistencies):.3f}')\n            plt.legend()\n            \n            plt.tight_layout()\n            temporal_file = viz_dir / f\"temporal_consistency_{timestamp}.png\"\n            plt.savefig(temporal_file, dpi=300, bbox_inches='tight')\n            plt.close()\n            visualization_files.append(temporal_file)\n        \n        print(f\"   Created {len(visualization_files)} visualization files\")\n        return visualization_files\n    \n    def save_detailed_results(self, metrics, predictions, ground_truth, timestamp):\n        \"\"\"Save detailed evaluation results\"\"\"\n        \n        print(\"💾 Saving detailed results...\")\n        \n        results_dir = self.output_dir / \"detailed_results\"\n        results_dir.mkdir(exist_ok=True)\n        \n        # Save metrics as JSON\n        metrics_file = results_dir / f\"evaluation_metrics_{timestamp}.json\"\n        with open(metrics_file, 'w') as f:\n            json.dump(metrics, f, indent=2)\n        \n        # Save per-sample predictions\n        sample_results = []\n        for i, (pred, gt, meta) in enumerate(zip(predictions, ground_truth, self.all_metadata)):\n            sample_data = {\n                'sample_id': i,\n                'participant_id': meta['participant_id'],\n                'frame_id': meta['frame_id'],\n                'timestamp': meta.get('temporal_info', {}).get('timestamp_seconds', meta['frame_id'])\n            }\n            \n            # Add predictions and ground truth for each feature\n            for j, feature in enumerate(self.feature_names['physical']):\n                sample_data[f'pred_physical_{feature}'] = float(pred['physical_probs'][j])\n                sample_data[f'true_physical_{feature}'] = int(gt['physical_labels'][j])\n                sample_data[f'correct_physical_{feature}'] = int((pred['physical_probs'][j] > 0.5) == gt['physical_labels'][j])\n            \n            for j, feature in enumerate(self.feature_names['emotional']):\n                sample_data[f'pred_emotional_{feature}'] = float(pred['emotional_probs'][j])\n                sample_data[f'true_emotional_{feature}'] = int(gt['emotional_labels'][j])\n                sample_data[f'correct_emotional_{feature}'] = int((pred['emotional_probs'][j] > 0.5) == gt['emotional_labels'][j])\n            \n            sample_results.append(sample_data)\n        \n        # Save as CSV\n        samples_df = pd.DataFrame(sample_results)\n        samples_file = results_dir / f\"sample_predictions_{timestamp}.csv\"\n        samples_df.to_csv(samples_file, index=False)\n        \n        # Save feature performance summary\n        feature_summary = []\n        \n        for i, feature in enumerate(self.feature_names['physical']):\n            feature_summary.append({\n                'feature_type': 'physical',\n                'feature_name': feature,\n                'accuracy': metrics['physical']['accuracy'][i],\n                'precision': metrics['physical']['precision'][i],\n                'recall': metrics['physical']['recall'][i],\n                'f1_score': metrics['physical']['f1_score'][i],\n                'auc_roc': metrics['physical']['auc_roc'][i],\n                'auc_pr': metrics['physical']['auc_pr'][i]\n            })\n        \n        for i, feature in enumerate(self.feature_names['emotional']):\n            feature_summary.append({\n                'feature_type': 'emotional',\n                'feature_name': feature,\n                'accuracy': metrics['emotional']['accuracy'][i],\n                'precision': metrics['emotional']['precision'][i],\n                'recall': metrics['emotional']['recall'][i],\n                'f1_score': metrics['emotional']['f1_score'][i],\n                'auc_roc': metrics['emotional']['auc_roc'][i],\n                'auc_pr': metrics['emotional']['auc_pr'][i]\n            })\n        \n        feature_df = pd.DataFrame(feature_summary)\n        feature_file = results_dir / f\"feature_performance_{timestamp}.csv\"\n        feature_df.to_csv(feature_file, index=False)\n        \n        return {\n            'metrics_file': metrics_file,\n            'samples_file': samples_file,\n            'feature_file': feature_file\n        }\n    \n    def run_comprehensive_evaluation(self):\n        \"\"\"Run complete evaluation pipeline\"\"\"\n        \n        print(\"🚀 Starting comprehensive evaluation...\")\n        \n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        \n        # Setup\n        self.load_model()\n        self.setup_data()\n        \n        # Evaluate\n        predictions, ground_truth, metadata = self.evaluate_model()\n        self.all_metadata = metadata  # Store for later use\n        \n        # Calculate metrics\n        metrics = self.calculate_detailed_metrics(predictions, ground_truth)\n        \n        # Create visualizations\n        viz_files = self.create_evaluation_visualizations(metrics, timestamp)\n        \n        # Save detailed results\n        result_files = self.save_detailed_results(metrics, predictions, ground_truth, timestamp)\n        \n        # Create summary report\n        summary_report = {\n            'timestamp': timestamp,\n            'model_info': {\n                'model_path': str(self.model_path),\n                'model_type': self.model_config['model_type'],\n                'sequence_length': self.model_config['sequence_length'],\n                'num_features': self.model_config['num_physical_features'] + self.model_config['num_emotional_features']\n            },\n            'evaluation_summary': {\n                'total_samples': len(predictions),\n                'overall_accuracy': metrics['overall']['accuracy'],\n                'physical_mean_accuracy': metrics['physical']['summary']['mean_accuracy'],\n                'emotional_mean_accuracy': metrics['emotional']['summary']['mean_accuracy'],\n                'physical_mean_f1': metrics['physical']['summary']['mean_f1'],\n                'emotional_mean_f1': metrics['emotional']['summary']['mean_f1'],\n                'temporal_consistency': metrics['temporal']['temporal_consistency']['mean_consistency']\n            },\n            'files': {\n                'metrics': str(result_files['metrics_file']),\n                'samples': str(result_files['samples_file']),\n                'features': str(result_files['feature_file']),\n                'visualizations': [str(f) for f in viz_files]\n            }\n        }\n        \n        # Save summary report\n        summary_file = self.output_dir / f\"evaluation_summary_{timestamp}.json\"\n        with open(summary_file, 'w') as f:\n            json.dump(summary_report, f, indent=2)\n        \n        print(\"\\n✅ Comprehensive evaluation complete!\")\n        print(f\"   Overall accuracy: {metrics['overall']['accuracy']:.3f}\")\n        print(f\"   Physical mean F1: {metrics['physical']['summary']['mean_f1']:.3f}\")\n        print(f\"   Emotional mean F1: {metrics['emotional']['summary']['mean_f1']:.3f}\")\n        print(f\"   Temporal consistency: {metrics['temporal']['temporal_consistency']['mean_consistency']:.3f}\")\n        print(f\"   Summary report: {summary_file}\")\n        \n        return summary_report, summary_file\n\ndef main():\n    \"\"\"Main evaluation function\"\"\"\n    \n    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')\n    parser.add_argument('--model_path', type=str, required=True,\n                       help='Path to trained model checkpoint')\n    parser.add_argument('--output_dir', type=str, default='./evaluation_outputs',\n                       help='Output directory for evaluation results')\n    \n    args = parser.parse_args()\n    \n    # Setup paths\n    frames_dir = \"/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames\"\n    annotations_dir = \"/home/rohan/Multimodal/multimodal_video_ml/data/annotations\"\n    \n    print(\"🔍 Comprehensive Model Evaluation\")\n    print(\"=\" * 60)\n    print(f\"Model: {args.model_path}\")\n    print(f\"Output: {args.output_dir}\")\n    print(\"=\" * 60)\n    \n    # Create evaluator\n    evaluator = ComprehensiveEvaluator(\n        model_path=args.model_path,\n        frames_dir=frames_dir,\n        annotations_dir=annotations_dir,\n        output_dir=args.output_dir\n    )\n    \n    # Run evaluation\n    summary_report, summary_file = evaluator.run_comprehensive_evaluation()\n    \n    print(f\"\\n🏆 Evaluation Results:\")\n    print(f\"   Overall Accuracy: {summary_report['evaluation_summary']['overall_accuracy']:.3f}\")\n    print(f\"   Physical F1: {summary_report['evaluation_summary']['physical_mean_f1']:.3f}\")\n    print(f\"   Emotional F1: {summary_report['evaluation_summary']['emotional_mean_f1']:.3f}\")\n    print(f\"   Report saved: {summary_file}\")\n\nif __name__ == \"__main__\":\n    main()