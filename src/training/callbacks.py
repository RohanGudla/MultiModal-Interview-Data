"""
Training callbacks for monitoring and controlling training process.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

class Callback:
    """Base class for training callbacks."""
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Called at the end of each epoch.
        
        Returns:
            bool: True to stop training, False to continue
        """
        return False
        
    def on_train_end(self, logs: Dict[str, Any]):
        """Called at the end of training."""
        pass

class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 min_delta: float = 0.0,
                 patience: int = 10,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement after which training stops
            mode: 'min' for minimizing metric, 'max' for maximizing
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")
            
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Check for early stopping condition."""
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose:
                warnings.warn(f"Early stopping conditioned on metric `{self.monitor}` "
                            f"which is not available. Available metrics: {list(logs.keys())}")
            return False
            
        # Check if current metric is better than best
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            
            # Save best weights if requested
            if self.restore_best_weights and 'model' in kwargs:
                self.best_weights = {
                    name: param.clone() 
                    for name, param in kwargs['model'].named_parameters()
                }
        else:
            self.wait += 1
            
        # Check if patience exceeded
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: early stopping")
                print(f"Best {self.monitor}: {self.best:.6f}")
                
            # Restore best weights
            if self.restore_best_weights and self.best_weights and 'model' in kwargs:
                if self.verbose:
                    print("Restoring model weights from the end of the best epoch")
                    
                model = kwargs['model']
                for name, param in model.named_parameters():
                    if name in self.best_weights:
                        param.data.copy_(self.best_weights[name])
                        
            return True
            
        return False
        
    def on_train_end(self, logs: Dict[str, Any]):
        """Print early stopping summary."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped at epoch {self.stopped_epoch + 1}")

class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_last: bool = True,
                 verbose: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for the monitored metric
            save_best_only: Only save when metric improves
            save_last: Always save the last epoch
            verbose: Print save messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} not supported")
            
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Save checkpoint if conditions are met."""
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose:
                warnings.warn(f"Checkpoint callback conditioned on metric `{self.monitor}` "
                            f"which is not available. Available: {list(logs.keys())}")
            return False
            
        # Check if current is better than best
        if self.monitor_op(current, self.best):
            self.best = current
            
            if 'model' in kwargs and 'optimizer' in kwargs:
                self._save_checkpoint(
                    epoch=epoch,
                    model=kwargs['model'],
                    optimizer=kwargs['optimizer'],
                    metrics=logs,
                    is_best=True
                )
        elif not self.save_best_only:
            # Save regular checkpoint
            if 'model' in kwargs and 'optimizer' in kwargs:
                self._save_checkpoint(
                    epoch=epoch,
                    model=kwargs['model'],
                    optimizer=kwargs['optimizer'],
                    metrics=logs,
                    is_best=False
                )
                
        return False
        
    def _save_checkpoint(self,
                        epoch: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        metrics: Dict[str, float],
                        is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            
            if self.verbose:
                print(f"\nSaved best model (epoch {epoch + 1}) - "
                      f"{self.monitor}: {self.best:.6f}")
                
        # Last model checkpoint (always save most recent)
        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pth"
            torch.save(checkpoint, last_path)

class LearningRateScheduler(Callback):
    """Learning rate scheduling callback."""
    
    def __init__(self,
                 schedule_type: str = 'reduce_on_plateau',
                 monitor: str = 'val_loss',
                 factor: float = 0.5,
                 patience: int = 5,
                 min_lr: float = 1e-7,
                 verbose: bool = True):
        """
        Args:
            schedule_type: Type of scheduling ('reduce_on_plateau', 'step', 'cosine')
            monitor: Metric to monitor for plateau detection
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement before reducing LR
            min_lr: Minimum learning rate
            verbose: Print LR changes
        """
        self.schedule_type = schedule_type
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        self.best = np.inf
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Update learning rate based on schedule."""
        if self.schedule_type == 'reduce_on_plateau':
            return self._reduce_on_plateau(epoch, logs, **kwargs)
        else:
            return False
            
    def _reduce_on_plateau(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Reduce learning rate when metric plateaus."""
        current = logs.get(self.monitor)
        
        if current is None:
            return False
            
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            if 'optimizer' in kwargs:
                optimizer = kwargs['optimizer']
                
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    
                    if new_lr < old_lr:
                        param_group['lr'] = new_lr
                        
                        if self.verbose:
                            print(f"\nReducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")
                            
                self.wait = 0
                
        return False

class MetricLogger(Callback):
    """Log metrics to various backends."""
    
    def __init__(self, 
                 log_dir: Union[str, Path],
                 metrics_to_log: Optional[list] = None):
        """
        Args:
            log_dir: Directory to save logs
            metrics_to_log: List of metrics to log (None for all)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_to_log = metrics_to_log
        self.history = []
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Log metrics for this epoch."""
        epoch_logs = {'epoch': epoch}
        
        if self.metrics_to_log is None:
            epoch_logs.update(logs)
        else:
            for metric in self.metrics_to_log:
                if metric in logs:
                    epoch_logs[metric] = logs[metric]
                    
        self.history.append(epoch_logs)
        return False
        
    def on_train_end(self, logs: Dict[str, Any]):
        """Save training history to file."""
        import json
        
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        print(f"Training history saved to {history_path}")

class GradientClipping(Callback):
    """Gradient clipping callback."""
    
    def __init__(self, 
                 max_norm: float = 1.0,
                 norm_type: float = 2.0):
        """
        Args:
            max_norm: Maximum norm for gradients
            norm_type: Type of norm to use
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def on_batch_end(self, model: torch.nn.Module, **kwargs):
        """Clip gradients after backward pass."""
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

class MemoryLogger(Callback):
    """Log GPU memory usage during training."""
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Print memory usage
        """
        self.verbose = verbose
        self.memory_history = []
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """Log memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            
            memory_info = {
                'epoch': epoch,
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved
            }
            
            self.memory_history.append(memory_info)
            
            if self.verbose:
                print(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, "
                      f"Reserved: {memory_reserved:.1f}MB")
                
        return False

def create_callbacks(config: Dict[str, Any], 
                    checkpoint_dir: Union[str, Path]) -> list:
    """Factory function to create standard callbacks."""
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', True):
        early_stopping = EarlyStopping(
            monitor=config.get('monitor_metric', 'val_f1_score'),
            patience=config.get('patience', 15),
            mode=config.get('monitor_mode', 'max'),
            verbose=True
        )
        callbacks.append(early_stopping)
        
    # Model checkpointing
    if config.get('save_checkpoints', True):
        model_checkpoint = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=config.get('monitor_metric', 'val_f1_score'),
            mode=config.get('monitor_mode', 'max'),
            save_best_only=config.get('save_best_only', True),
            verbose=True
        )
        callbacks.append(model_checkpoint)
        
    # Learning rate scheduling
    if config.get('lr_scheduling', False):
        lr_scheduler = LearningRateScheduler(
            monitor=config.get('lr_monitor', 'val_loss'),
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 5),
            verbose=True
        )
        callbacks.append(lr_scheduler)
        
    # Memory logging
    if config.get('log_memory', False):
        memory_logger = MemoryLogger(verbose=True)
        callbacks.append(memory_logger)
        
    return callbacks

if __name__ == "__main__":
    # Test callbacks
    print("=== Testing Callbacks ===")
    
    # Test early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True)
    
    # Simulate training epochs
    for epoch in range(10):
        # Simulate metrics (loss decreasing then increasing)
        if epoch < 5:
            val_loss = 1.0 - epoch * 0.1
        else:
            val_loss = 0.5 + (epoch - 5) * 0.05
            
        logs = {'val_loss': val_loss, 'val_accuracy': 0.8 + epoch * 0.02}
        
        stop = early_stopping.on_epoch_end(epoch, logs)
        print(f"Epoch {epoch}: val_loss={val_loss:.3f}, stop={stop}")
        
        if stop:
            break
            
    print("\nCallback tests completed!")