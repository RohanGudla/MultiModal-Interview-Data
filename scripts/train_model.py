#!/usr/bin/env python3
"""
Main training script for emotion recognition models.
Supports all 4 model architectures with optimized configurations.
"""
import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from data.dataset import create_data_loaders
from models.cnn_simple import create_cnn_model
from models.vit_simple import create_vit_model
from models.resnet_pretrained import create_resnet_model
from models.vit_pretrained import create_vit_model as create_pretrained_vit_model
from training.trainer import EmotionTrainer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train emotion recognition models")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["cnn_simple", "vit_scratch", "resnet_pretrained", "vit_pretrained"],
        help="Model architecture to train"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="binary_attention",
        choices=["binary_attention", "attention_regression", "emotion_multilabel"],
        help="Type of task to train"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (uses config default if not specified)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (uses model-specific default if not specified)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (uses model-specific default if not specified)"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for experiment (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation on test set"
    )
    
    return parser.parse_args()

def create_model(model_name: str, num_classes: int, config: Config):
    """Create model based on name."""
    print(f"Creating {model_name} model...")
    
    if model_name == "cnn_simple":
        model = create_cnn_model(
            model_type="simple",
            num_classes=num_classes,
            dropout_rate=0.5
        )
        
    elif model_name == "vit_scratch":
        # Use small ViT for faster training
        model = create_vit_model(
            model_size="small",
            num_classes=num_classes,
            dropout=0.1
        )
        
    elif model_name == "resnet_pretrained":
        model = create_resnet_model(
            model_type="resnet50",
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True,
            freeze_layers=45
        )
        
    elif model_name == "vit_pretrained":
        try:
            model = create_pretrained_vit_model(
                model_type="base",
                num_classes=num_classes,
                pretrained=True,
                freeze_backbone=True,
                freeze_layers=10
            )
        except ImportError:
            print("❌ timm library required for pretrained ViT. Install with: pip install timm")
            sys.exit(1)
            
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    # Print model summary
    if hasattr(model, 'count_parameters'):
        param_info = model.count_parameters()
        print(f"Model parameters: {param_info}")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    return model

def get_task_config(task_type: str):
    """Get task configuration."""
    if task_type == "binary_attention":
        return {
            "num_classes": 1,
            "loss_type": "bce",
            "task_type": "binary_classification"
        }
    elif task_type == "attention_regression":
        return {
            "num_classes": 1,
            "loss_type": "mse",
            "task_type": "regression"
        }
    elif task_type == "emotion_multilabel":
        return {
            "num_classes": 7,  # 7 emotions
            "loss_type": "bce",
            "task_type": "multilabel"
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def main():
    """Main training function."""
    args = parse_arguments()
    
    print("=" * 80)
    print("MULTIMODAL EMOTION RECOGNITION - MODEL TRAINING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize configuration
    config = Config()
    
    # Get task configuration
    task_config = get_task_config(args.task)
    num_classes = task_config["num_classes"]
    loss_type = task_config["loss_type"]
    task_type = task_config["task_type"]
    
    # Get model-specific batch size
    model_config = config.get_model_config(args.model)
    batch_size = args.batch_size or model_config.get("batch_size", 32)
    
    print(f"Batch size: {batch_size} (optimized for RTX 4080 16GB)")
    
    # Create data loaders
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            label_type=args.task,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("Did you run 'python scripts/prepare_data.py' first?")
        sys.exit(1)
    
    # Create model
    model = create_model(args.model, num_classes, config)
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name=args.model,
        loss_type=loss_type,
        task_type=task_type
    )
    
    # Setup experiment name
    if args.experiment_name is None:
        experiment_name = f"{args.model}_{args.task}_{batch_size}bs"
    else:
        experiment_name = args.experiment_name
        
    # Setup logging
    if not args.no_wandb:
        try:
            trainer.setup_logging(
                experiment_name=experiment_name,
                wandb_project="multimodal_emotion_recognition"
            )
            print(f"✅ Weights & Biases logging enabled: {experiment_name}")
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")
            print("Continuing without W&B logging...")
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            trainer.load_checkpoint(resume_path)
            print(f"✅ Resumed from checkpoint: {resume_path}")
        else:
            print(f"❌ Checkpoint not found: {resume_path}")
            sys.exit(1)
    
    # Evaluation only mode
    if args.eval_only:
        print("\n" + "=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        
        if args.resume is None:
            print("❌ --resume required for evaluation mode")
            sys.exit(1)
            
        test_metrics = trainer.evaluate(test_loader)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        return
    
    # Training mode
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    try:
        # Train model
        training_results = trainer.train(epochs=args.epochs)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED - RUNNING FINAL EVALUATION")
        print("=" * 60)
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(test_loader)
        
        # Save final results
        results_dir = Path("experiments") / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        final_results = {
            "model": args.model,
            "task": args.task,
            "batch_size": batch_size,
            "best_val_metric": trainer.best_metric,
            "test_metrics": test_metrics,
            "training_history": {
                "train_losses": training_results["train_losses"],
                "val_losses": training_results["val_losses"]
            }
        }
        
        with open(results_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
            
        print(f"\n✅ Results saved to: {results_dir}")
        print(f"✅ Best validation metric: {trainer.best_metric:.4f}")
        print(f"✅ Test F1 score: {test_metrics.get('f1_score', 'N/A')}")
        
        # Memory usage summary
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"📊 GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        
        # Save current state
        checkpoint_path = Path("experiments") / experiment_name / "interrupted_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        print(f"💾 Saved checkpoint to: {checkpoint_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup
        if hasattr(trainer, 'logger') and trainer.logger:
            trainer.logger.finish()

if __name__ == "__main__":
    main()