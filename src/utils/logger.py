"""
Logging utilities for experiment tracking and monitoring.
"""
import logging
import wandb
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Logger:
    """Comprehensive logging for training and evaluation."""
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # Setup file logging
        self.setup_file_logging()
        
        # Initialize wandb
        self.wandb_run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            save_code=True
        )
        
        self.step = 0
        
    def setup_file_logging(self):
        """Setup file-based logging."""
        log_dir = Path("experiments") / self.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb and console."""
        if step is not None:
            self.step = step
            
        # Log to wandb
        wandb.log(metrics, step=self.step)
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {self.step}: {metrics_str}")
        
    def log_model_summary(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        self.logger.info(f"Model Summary: {summary}")
        wandb.log(summary)
        
        # Try to log model graph (if input provided)
        try:
            dummy_input = torch.randn(1, *input_shape).to(next(model.parameters()).device)
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            self.logger.warning(f"Could not log model graph: {e}")
            
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: list, title: str = "Confusion Matrix"):
        """Log confusion matrix visualization."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        wandb.log({title.lower().replace(' ', '_'): wandb.Image(plt)})
        plt.close()
        
    def log_attention_maps(self, attention_weights: torch.Tensor, 
                          input_image: torch.Tensor, title: str = "Attention Map"):
        """Log attention visualization for ViT models."""
        # Average attention across heads and layers
        if attention_weights.dim() == 4:  # [layers, heads, seq_len, seq_len]
            attention = attention_weights.mean(dim=(0, 1))  # Average across layers and heads
        else:
            attention = attention_weights
            
        # Get attention to [CLS] token or average
        attention_map = attention[0, 1:]  # Skip CLS token
        
        # Reshape to spatial dimensions
        grid_size = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map.view(grid_size, grid_size)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        if input_image.dim() == 4:
            img = input_image[0]
        else:
            img = input_image
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize
        ax1.imshow(img)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Attention map
        ax2.imshow(attention_map.cpu().numpy(), cmap='hot', interpolation='bilinear')
        ax2.set_title("Attention Map")
        ax2.axis('off')
        
        plt.tight_layout()
        wandb.log({title.lower().replace(' ', '_'): wandb.Image(plt)})
        plt.close()
        
    def log_training_curves(self, train_metrics: Dict[str, list], 
                           val_metrics: Dict[str, list]):
        """Log training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['loss', 'accuracy', 'f1_score', 'auc']
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes) and metric in train_metrics:
                ax = axes[i]
                ax.plot(train_metrics[metric], label=f'Train {metric}', alpha=0.8)
                if metric in val_metrics:
                    ax.plot(val_metrics[metric], label=f'Val {metric}', alpha=0.8)
                ax.set_title(f'{metric.title()} Curves')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.title())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"training_curves": wandb.Image(plt)})
        plt.close()
        
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path("experiments") / self.experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': dict(wandb.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with metrics: {metrics}")
            
        # Save to wandb
        wandb.save(str(checkpoint_path))
        if is_best:
            wandb.save(str(best_path))
            
    def finish(self):
        """Clean up logging."""
        if self.wandb_run:
            self.wandb_run.finish()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()