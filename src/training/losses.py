"""
Custom loss functions for emotion recognition training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class (default 1.0)
            gamma: Focusing parameter (default 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, 1) for binary or (N, C) for multiclass
            targets: Ground truth of shape (N,) or (N, 1)
        """
        # Handle binary classification
        if inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            
            # Convert to probabilities
            p = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            
            # Calculate focal weight
            p_t = p * targets + (1 - p) * (1 - targets)
            focal_weight = self.alpha * (1 - p_t) ** self.gamma
            
        else:
            # Multiclass case
            ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
            p = F.softmax(inputs, dim=-1)
            p_t = p.gather(1, targets.long().unsqueeze(1)).squeeze(1)
            focal_weight = self.alpha * (1 - p_t) ** self.gamma
            
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Ground truth of shape (N,)
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot encoding with smoothing
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.long().unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy for imbalanced datasets."""
    
    def __init__(self, pos_weight: Optional[float] = None):
        """
        Args:
            pos_weight: Weight for positive class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross entropy with optional positive weighting."""
        if inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
            
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets)

class MultiTaskLoss(nn.Module):
    """Multi-task loss with automatic weighting."""
    
    def __init__(self, 
                 task_names: list,
                 loss_types: Dict[str, str],
                 task_weights: Optional[Dict[str, float]] = None,
                 auto_weighting: bool = True):
        """
        Args:
            task_names: List of task names
            loss_types: Dict mapping task names to loss types
            task_weights: Optional manual weights for tasks
            auto_weighting: Use automatic task weighting
        """
        super(MultiTaskLoss, self).__init__()
        
        self.task_names = task_names
        self.loss_types = loss_types
        self.auto_weighting = auto_weighting
        
        # Initialize loss functions
        self.loss_functions = nn.ModuleDict()
        for task_name, loss_type in loss_types.items():
            if loss_type == 'bce':
                self.loss_functions[task_name] = nn.BCEWithLogitsLoss()
            elif loss_type == 'mse':
                self.loss_functions[task_name] = nn.MSELoss()
            elif loss_type == 'focal':
                self.loss_functions[task_name] = FocalLoss()
            elif loss_type == 'ce':
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
                
        # Task weights
        if task_weights is None:
            task_weights = {name: 1.0 for name in task_names}
        self.task_weights = task_weights
        
        # Learnable weights for auto-weighting
        if auto_weighting:
            self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict of task predictions
            targets: Dict of task targets
        Returns:
            Dict containing individual losses and total loss
        """
        losses = {}
        total_loss = 0
        
        for i, task_name in enumerate(self.task_names):
            if task_name not in predictions or task_name not in targets:
                continue
                
            # Calculate task loss
            task_loss = self.loss_functions[task_name](
                predictions[task_name], targets[task_name]
            )
            losses[f'{task_name}_loss'] = task_loss
            
            # Apply weighting
            if self.auto_weighting:
                # Automatic uncertainty weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
            else:
                # Manual weighting
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = weight * task_loss
                
            total_loss += weighted_loss
            
        losses['total_loss'] = total_loss
        
        # Add task weights to output for monitoring
        if self.auto_weighting:
            for i, task_name in enumerate(self.task_names):
                losses[f'{task_name}_weight'] = torch.exp(-self.log_vars[i])
                
        return losses

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better representations."""
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, 
                output1: torch.Tensor, 
                output2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output1: First set of features
            output2: Second set of features  
            label: 1 if same class, 0 if different class
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

class TemporalConsistencyLoss(nn.Module):
    """Loss to encourage temporal consistency in video predictions."""
    
    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Weight for temporal consistency term
        """
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Tensor of shape (batch_size, sequence_length, num_classes)
        """
        if predictions.dim() != 3:
            return torch.tensor(0.0, device=predictions.device)
            
        # Calculate differences between consecutive predictions
        diff = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 norm of differences
        temporal_loss = torch.mean(torch.norm(diff, dim=-1))
        
        return self.weight * temporal_loss

def create_loss_function(loss_type: str, 
                        num_classes: int = 1,
                        class_weights: Optional[torch.Tensor] = None,
                        **kwargs) -> nn.Module:
    """Factory function to create loss functions."""
    
    if loss_type == 'bce':
        pos_weight = kwargs.get('pos_weight', None)
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'ce':
        weight = class_weights
        return nn.CrossEntropyLoss(weight=weight)
    
    elif loss_type == 'mse':
        return nn.MSELoss()
    
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    
    elif loss_type == 'multitask':
        task_names = kwargs.get('task_names', ['main'])
        loss_types = kwargs.get('loss_types', {'main': 'bce'})
        return MultiTaskLoss(task_names=task_names, loss_types=loss_types)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Utility functions for loss computation
def compute_class_weights(labels: np.ndarray, method: str = 'balanced') -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    
    if method == 'balanced':
        weights = compute_class_weight(
            'balanced', 
            classes=unique_labels, 
            y=labels
        )
    elif method == 'inv_freq':
        # Inverse frequency weighting
        counts = np.bincount(labels.astype(int))
        weights = 1.0 / counts[unique_labels]
        weights = weights / weights.sum() * len(unique_labels)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return torch.tensor(weights, dtype=torch.float32)

def get_loss_weights_from_dataset(dataset, label_type: str = 'binary') -> Dict[str, Any]:
    """Get appropriate loss weights from dataset statistics."""
    if hasattr(dataset, 'get_label_statistics'):
        stats = dataset.get_label_statistics()
        
        if label_type == 'binary':
            # For binary classification
            pos_ratio = stats.get('positive_ratio', 0.5)
            neg_ratio = 1 - pos_ratio
            
            # Inverse frequency weighting
            pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0
            
            return {
                'pos_weight': pos_weight,
                'class_balance': min(pos_ratio, neg_ratio)
            }
        else:
            # For multiclass/regression
            return {
                'mean': stats.get('mean', 0),
                'std': stats.get('std', 1),
                'class_balance': 1.0  # Default for non-binary
            }
    else:
        return {'pos_weight': 1.0, 'class_balance': 0.5}

if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test binary classification losses
    print("=== Testing Binary Classification Losses ===")
    logits = torch.randn(32, 1).to(device)
    targets = torch.randint(0, 2, (32, 1)).float().to(device)
    
    # BCE Loss
    bce_loss = WeightedBCELoss(pos_weight=2.0)
    bce_result = bce_loss(logits, targets)
    print(f"BCE Loss: {bce_result.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    focal_result = focal_loss(logits, targets)
    print(f"Focal Loss: {focal_result.item():.4f}")
    
    # Test multi-task loss
    print("\n=== Testing Multi-Task Loss ===")
    predictions = {
        'attention': torch.randn(32, 1).to(device),
        'emotions': torch.randn(32, 7).to(device)
    }
    targets_mt = {
        'attention': torch.randint(0, 2, (32, 1)).float().to(device),
        'emotions': torch.randint(0, 7, (32,)).long().to(device)
    }
    
    multitask_loss = MultiTaskLoss(
        task_names=['attention', 'emotions'],
        loss_types={'attention': 'bce', 'emotions': 'ce'},
        auto_weighting=True
    )
    
    mt_result = multitask_loss(predictions, targets_mt)
    print(f"Multi-task losses: {[(k, v.item()) for k, v in mt_result.items()]}")
    
    # Test temporal consistency loss
    print("\n=== Testing Temporal Consistency Loss ===")
    seq_predictions = torch.randn(16, 10, 7).to(device)  # 16 sequences, 10 frames, 7 classes
    temporal_loss = TemporalConsistencyLoss(weight=0.1)
    temp_result = temporal_loss(seq_predictions)
    print(f"Temporal Consistency Loss: {temp_result.item():.4f}")
    
    print("\nAll loss tests completed successfully!")