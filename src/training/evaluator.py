"""
Evaluation metrics and validation utilities for emotion recognition.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EmotionEvaluator:
    """Comprehensive evaluation for emotion recognition models."""
    
    def __init__(self, 
                 task_type: str = 'binary_classification',
                 num_classes: int = 1,
                 class_names: Optional[List[str]] = None,
                 threshold: float = 0.5):
        """
        Args:
            task_type: 'binary_classification', 'multiclass', 'regression', 'multilabel'
            num_classes: Number of classes/outputs
            class_names: Names of classes for reporting
            threshold: Decision threshold for binary classification
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.threshold = threshold
        
        # Storage for predictions and targets
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.all_predictions.clear()
        self.all_targets.clear()
        self.all_probabilities.clear()
        
    def update(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None):
        """Update with batch predictions and targets."""
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
            
        self.all_predictions.append(predictions)
        self.all_targets.append(targets)
        
        if probabilities is not None:
            self.all_probabilities.append(probabilities)
        elif self.task_type in ['binary_classification', 'multiclass']:
            # Convert logits to probabilities
            if self.task_type == 'binary_classification':
                probs = 1 / (1 + np.exp(-predictions))  # Sigmoid
            else:
                # Softmax
                exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            self.all_probabilities.append(probs)
            
    def compute_metrics(self) -> Dict[str, Union[float, np.ndarray]]:
        """Compute comprehensive evaluation metrics."""
        if not self.all_predictions:
            return {}
            
        # Concatenate all predictions and targets
        predictions = np.concatenate(self.all_predictions, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        
        if self.all_probabilities:
            probabilities = np.concatenate(self.all_probabilities, axis=0)
        else:
            probabilities = None
            
        metrics = {}
        
        if self.task_type == 'binary_classification':
            metrics.update(self._compute_binary_metrics(predictions, targets, probabilities))
        elif self.task_type == 'multiclass':
            metrics.update(self._compute_multiclass_metrics(predictions, targets, probabilities))
        elif self.task_type == 'regression':
            metrics.update(self._compute_regression_metrics(predictions, targets))
        elif self.task_type == 'multilabel':
            metrics.update(self._compute_multilabel_metrics(predictions, targets, probabilities))
            
        return metrics
        
    def _compute_binary_metrics(self, 
                               predictions: np.ndarray, 
                               targets: np.ndarray,
                               probabilities: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute binary classification metrics."""
        # Handle different input shapes
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()
            
        # Convert probabilities to binary predictions
        if probabilities is not None:
            if probabilities.ndim > 1:
                probabilities = probabilities.flatten()
            binary_preds = (probabilities >= self.threshold).astype(int)
        else:
            binary_preds = (predictions >= self.threshold).astype(int)
            
        metrics = {
            'accuracy': accuracy_score(targets, binary_preds),
            'precision': precision_score(targets, binary_preds, zero_division=0),
            'recall': recall_score(targets, binary_preds, zero_division=0),
            'f1_score': f1_score(targets, binary_preds, zero_division=0),
            'specificity': self._compute_specificity(targets, binary_preds),
        }
        
        # AUC-ROC (needs probabilities)
        if probabilities is not None and len(np.unique(targets)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(targets, probabilities)
            except:
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
            
        # Balanced accuracy
        tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
        
        return metrics
        
    def _compute_multiclass_metrics(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   probabilities: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute multiclass classification metrics."""
        # Convert predictions to class indices
        if predictions.ndim > 1:
            predicted_classes = np.argmax(predictions, axis=1)
        else:
            predicted_classes = predictions.astype(int)
            
        if targets.ndim > 1:
            targets = np.argmax(targets, axis=1)
        targets = targets.astype(int)
        
        metrics = {
            'accuracy': accuracy_score(targets, predicted_classes),
            'precision_macro': precision_score(targets, predicted_classes, average='macro', zero_division=0),
            'precision_micro': precision_score(targets, predicted_classes, average='micro', zero_division=0),
            'recall_macro': recall_score(targets, predicted_classes, average='macro', zero_division=0),
            'recall_micro': recall_score(targets, predicted_classes, average='micro', zero_division=0),
            'f1_macro': f1_score(targets, predicted_classes, average='macro', zero_division=0),
            'f1_micro': f1_score(targets, predicted_classes, average='micro', zero_division=0),
        }
        
        # Per-class metrics
        if len(self.class_names) == self.num_classes:
            precision_per_class = precision_score(targets, predicted_classes, average=None, zero_division=0)
            recall_per_class = recall_score(targets, predicted_classes, average=None, zero_division=0)
            f1_per_class = f1_score(targets, predicted_classes, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.class_names[:len(precision_per_class)]):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
                
        # AUC-ROC for multiclass (one-vs-rest)
        if probabilities is not None and len(np.unique(targets)) > 1:
            try:
                metrics['auc_roc_macro'] = roc_auc_score(targets, probabilities, 
                                                        multi_class='ovr', average='macro')
            except:
                metrics['auc_roc_macro'] = 0.0
                
        return metrics
        
    def _compute_regression_metrics(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()
            
        metrics = {
            'mae': mean_absolute_error(targets, predictions),
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2_score': r2_score(targets, predictions),
        }
        
        # Mean absolute percentage error
        mask = targets != 0
        if np.any(mask):
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = float('inf')
            
        # Explained variance
        metrics['explained_variance'] = 1 - np.var(targets - predictions) / np.var(targets)
        
        return metrics
        
    def _compute_multilabel_metrics(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   probabilities: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute multilabel classification metrics."""
        # Convert to binary predictions
        if probabilities is not None:
            binary_preds = (probabilities >= self.threshold).astype(int)
        else:
            binary_preds = (predictions >= self.threshold).astype(int)
            
        metrics = {
            'hamming_loss': np.mean(binary_preds != targets),
            'subset_accuracy': accuracy_score(targets, binary_preds),
            'precision_micro': precision_score(targets, binary_preds, average='micro', zero_division=0),
            'precision_macro': precision_score(targets, binary_preds, average='macro', zero_division=0),
            'recall_micro': recall_score(targets, binary_preds, average='micro', zero_division=0),
            'recall_macro': recall_score(targets, binary_preds, average='macro', zero_division=0),
            'f1_micro': f1_score(targets, binary_preds, average='micro', zero_division=0),
            'f1_macro': f1_score(targets, binary_preds, average='macro', zero_division=0),
        }
        
        return metrics
        
    def _compute_specificity(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        cm = confusion_matrix(targets, predictions, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            return 0.0
            
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for the accumulated predictions."""
        if not self.all_predictions:
            return np.array([])
            
        predictions = np.concatenate(self.all_predictions, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        
        if self.task_type == 'binary_classification':
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            if targets.ndim > 1:
                targets = targets.flatten()
            binary_preds = (predictions >= self.threshold).astype(int)
            return confusion_matrix(targets, binary_preds, labels=[0, 1])
        elif self.task_type == 'multiclass':
            if predictions.ndim > 1:
                predicted_classes = np.argmax(predictions, axis=1)
            else:
                predicted_classes = predictions.astype(int)
            if targets.ndim > 1:
                targets = np.argmax(targets, axis=1)
            return confusion_matrix(targets, predicted_classes)
        else:
            return np.array([])
            
    def plot_confusion_matrix(self, save_path: Optional[Path] = None, normalize: bool = True):
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix()
        if cm.size == 0:
            print("No confusion matrix available for this task type")
            return
            
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names[:cm.shape[1]],
                   yticklabels=self.class_names[:cm.shape[0]])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.all_predictions:
            return "No predictions available"
            
        predictions = np.concatenate(self.all_predictions, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        
        if self.task_type == 'binary_classification':
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            if targets.ndim > 1:
                targets = targets.flatten()
            binary_preds = (predictions >= self.threshold).astype(int)
            return classification_report(targets, binary_preds, 
                                       target_names=self.class_names[:2])
        elif self.task_type == 'multiclass':
            if predictions.ndim > 1:
                predicted_classes = np.argmax(predictions, axis=1)
            else:
                predicted_classes = predictions.astype(int)
            if targets.ndim > 1:
                targets = np.argmax(targets, axis=1)
            return classification_report(targets, predicted_classes,
                                       target_names=self.class_names)
        else:
            return "Classification report not available for this task type"
            
    def compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute model calibration metrics."""
        if not self.all_probabilities or self.task_type not in ['binary_classification', 'multiclass']:
            return {}
            
        probabilities = np.concatenate(self.all_probabilities, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        
        if self.task_type == 'binary_classification':
            if probabilities.ndim > 1:
                probabilities = probabilities.flatten()
            if targets.ndim > 1:
                targets = targets.flatten()
                
            # Expected Calibration Error (ECE)
            ece = self._compute_ece(probabilities, targets)
            
            # Maximum Calibration Error (MCE)
            mce = self._compute_mce(probabilities, targets)
            
            return {'ece': ece, 'mce': mce}
        else:
            return {}
            
    def _compute_ece(self, probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
        
    def _compute_mce(self, probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
        return mce

class MultiTaskEvaluator:
    """Evaluator for multi-task learning scenarios."""
    
    def __init__(self, task_configs: Dict[str, Dict[str, Any]]):
        """
        Args:
            task_configs: Dict mapping task names to their configurations
                         e.g., {'attention': {'type': 'binary_classification', 'num_classes': 1}}
        """
        self.task_configs = task_configs
        self.task_evaluators = {}
        
        for task_name, config in task_configs.items():
            self.task_evaluators[task_name] = EmotionEvaluator(**config)
            
    def reset(self):
        """Reset all task evaluators."""
        for evaluator in self.task_evaluators.values():
            evaluator.reset()
            
    def update(self, 
               predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]):
        """Update all task evaluators."""
        for task_name, task_predictions in predictions.items():
            if task_name in self.task_evaluators and task_name in targets:
                self.task_evaluators[task_name].update(
                    task_predictions, targets[task_name]
                )
                
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for all tasks."""
        all_metrics = {}
        
        for task_name, evaluator in self.task_evaluators.items():
            all_metrics[task_name] = evaluator.compute_metrics()
            
        # Compute aggregate metrics
        if len(all_metrics) > 1:
            all_metrics['aggregate'] = self._compute_aggregate_metrics(all_metrics)
            
        return all_metrics
        
    def _compute_aggregate_metrics(self, task_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute aggregate metrics across tasks."""
        aggregate = {}
        
        # Common metrics to aggregate
        common_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        for metric in common_metrics:
            values = []
            for task_name, metrics in task_metrics.items():
                if task_name != 'aggregate' and metric in metrics:
                    values.append(metrics[metric])
                    
            if values:
                aggregate[f'mean_{metric}'] = np.mean(values)
                aggregate[f'std_{metric}'] = np.std(values)
                
        return aggregate

if __name__ == "__main__":
    # Test the evaluator
    print("=== Testing EmotionEvaluator ===")
    
    # Test binary classification
    evaluator = EmotionEvaluator(
        task_type='binary_classification',
        num_classes=1,
        class_names=['Low Attention', 'High Attention']
    )
    
    # Simulate some predictions
    for _ in range(5):
        predictions = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()
        probabilities = torch.sigmoid(predictions)
        
        evaluator.update(predictions, targets, probabilities)
        
    # Compute metrics
    metrics = evaluator.compute_metrics()
    print("Binary Classification Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
        
    # Test confusion matrix
    cm = evaluator.get_confusion_matrix()
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Test classification report
    report = evaluator.get_classification_report()
    print(f"\nClassification Report:\n{report}")
    
    # Test calibration
    calibration = evaluator.compute_calibration_metrics()
    print(f"\nCalibration Metrics: {calibration}")
    
    print("\nAll evaluator tests completed successfully!")