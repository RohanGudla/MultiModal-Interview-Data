"""
Main training framework for emotion recognition models.
Optimized for RTX 4080 16GB with mixed precision training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from ..utils.config import Config
from ..utils.logger import Logger
from .losses import create_loss_function, get_loss_weights_from_dataset
from .evaluator import EmotionEvaluator, MultiTaskEvaluator
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

class EmotionTrainer:
    """Main trainer class for emotion recognition models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Config,
                 model_name: str = "emotion_model",
                 loss_type: str = "bce",
                 task_type: str = "binary_classification"):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            model_name: Name for logging and checkpoints
            loss_type: Type of loss function to use
            task_type: Type of task (binary_classification, multiclass, regression)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.task_type = task_type
        
        # Setup device
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup mixed precision training for RTX 4080
        self.use_amp = config.MIXED_PRECISION
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup loss function
        self.loss_weights = get_loss_weights_from_dataset(train_loader.dataset, task_type)
        self.criterion = create_loss_function(
            loss_type=loss_type,
            num_classes=getattr(model, 'num_classes', 1),
            **self.loss_weights
        )
        self.criterion.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup evaluator
        self.evaluator = EmotionEvaluator(
            task_type=task_type,
            num_classes=getattr(model, 'num_classes', 1)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Setup logging
        self.logger = None
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with model-specific parameters."""
        model_config = self.config.get_model_config(self.model_name)
        base_lr = model_config.get('learning_rate', 1e-4)
        weight_decay = model_config.get('weight_decay', 1e-4)
        
        # Check if model has custom parameter groups (for pretrained models)
        if hasattr(self.model, 'get_parameter_groups'):
            param_groups = self.model.get_parameter_groups(base_lr)
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
        return optimizer
        
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.EPOCHS
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 4,  # Restart every 1/4 of training
            T_mult=2,
            eta_min=1e-7
        )
        
        return scheduler
        
    def _setup_callbacks(self) -> List:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=1e-4,
            mode='max',
            monitor='val_f1_score'
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        checkpoint_dir = Path("experiments") / self.model_name / "checkpoints"
        model_checkpoint = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            save_last=True
        )
        callbacks.append(model_checkpoint)
        
        return callbacks
        
    def setup_logging(self, experiment_name: str, wandb_project: str = "emotion_recognition"):
        """Setup experiment logging."""
        config_dict = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'batch_size': self.train_loader.batch_size,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epochs': self.config.EPOCHS,
            'device': str(self.device),
            'mixed_precision': self.use_amp,
            'loss_weights': self.loss_weights
        }
        
        self.logger = Logger(
            project_name=wandb_project,
            experiment_name=experiment_name,
            config=config_dict
        )
        
        # Log model summary
        dummy_input = next(iter(self.train_loader))[0][:1]  # Single sample
        self.logger.log_model_summary(self.model, dummy_input.shape[1:])
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Reset evaluator
        self.evaluator.reset()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 3:
                inputs, targets, participant_ids = batch
            else:
                inputs, targets = batch
                
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.GRADIENT_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.GRADIENT_CLIP_NORM
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if self.config.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_NORM
                    )
                
                self.optimizer.step()
                
            # Update scheduler (if using step-based scheduling)
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update evaluator
            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        pred_probs = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else outputs
                else:
                    pred_probs = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else outputs
                    
                self.evaluator.update(outputs, targets, pred_probs)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log batch metrics
            if self.logger and batch_idx % self.config.LOG_INTERVAL == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.logger.log_metrics({
                    'train_batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=step)
                
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.evaluator.compute_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
        
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        
        # Reset evaluator
        self.evaluator.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Handle different batch formats
                if len(batch) == 3:
                    inputs, targets, participant_ids = batch
                else:
                    inputs, targets = batch
                    
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        pred_probs = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else outputs
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    pred_probs = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else outputs
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update evaluator
                self.evaluator.update(outputs, targets, pred_probs)
                
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.evaluator.compute_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
        
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Main training loop."""
        if epochs is None:
            epochs = self.config.EPOCHS
            
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Task: {self.task_type}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        training_start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_metrics.append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics.get('accuracy', 0):.4f}, "
                  f"F1: {train_metrics.get('f1_score', 0):.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics.get('accuracy', 0):.4f}, "
                  f"F1: {val_metrics.get('f1_score', 0):.4f}")
            
            # Log metrics
            if self.logger:
                combined_metrics = {
                    f'train_{k}': v for k, v in train_metrics.items()
                }
                combined_metrics.update({
                    f'val_{k}': v for k, v in val_metrics.items()
                })
                combined_metrics['epoch_time'] = epoch_time
                
                self.logger.log_metrics(combined_metrics, step=epoch)
                
            # Update best metric
            monitor_metric = val_metrics.get('f1_score', val_metrics.get('accuracy', 0))
            if monitor_metric > self.best_metric:
                self.best_metric = monitor_metric
                
            # Run callbacks
            stop_training = False
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    stop = callback.on_epoch_end(
                        epoch=epoch,
                        logs=val_metrics,
                        model=self.model,
                        optimizer=self.optimizer
                    )
                    if stop:
                        stop_training = True
                        print(f"Early stopping triggered by {callback.__class__.__name__}")
                        
            if stop_training:
                break
                
        training_time = time.time() - training_start_time
        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"Best validation metric: {self.best_metric:.4f}")
        
        # Log training curves
        if self.logger:
            train_metrics_dict = {
                metric: [m.get(metric, 0) for m in self.train_metrics]
                for metric in ['loss', 'accuracy', 'f1_score']
            }
            val_metrics_dict = {
                metric: [m.get(metric, 0) for m in self.val_metrics]
                for metric in ['loss', 'accuracy', 'f1_score']
            }
            
            self.logger.log_training_curves(train_metrics_dict, val_metrics_dict)
            
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        print("Evaluating on test set...")
        
        self.model.eval()
        test_evaluator = EmotionEvaluator(
            task_type=self.task_type,
            num_classes=getattr(self.model, 'num_classes', 1)
        )
        
        test_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                if len(batch) == 3:
                    inputs, targets, participant_ids = batch
                else:
                    inputs, targets = batch
                    
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        pred_probs = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else outputs
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    pred_probs = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else outputs
                
                test_loss += loss.item()
                num_batches += 1
                
                test_evaluator.update(outputs, targets, pred_probs)
                
        # Compute test metrics
        test_metrics = test_evaluator.compute_metrics()
        test_metrics['loss'] = test_loss / num_batches
        
        print("\nTest Results:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        # Log test metrics
        if self.logger:
            test_metrics_logged = {f'test_{k}': v for k, v in test_metrics.items()}
            self.logger.log_metrics(test_metrics_logged)
            
            # Plot confusion matrix
            test_evaluator.plot_confusion_matrix()
            
        return test_metrics
        
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
        
    def __del__(self):
        """Cleanup logging on deletion."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.finish()