#!/usr/bin/env python3
"""
Comprehensive evaluation script for all trained models.
Compares performance across all 4 architectures.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from data.dataset import create_data_loaders
from training.trainer import EmotionTrainer
from training.evaluator import EmotionEvaluator

def load_model_and_trainer(model_name: str, checkpoint_path: Path, config: Config):
    """Load a trained model and create trainer for evaluation."""
    # Import model creation functions
    if model_name == "cnn_simple":
        from models.cnn_simple import create_cnn_model
        model = create_cnn_model(model_type="simple", num_classes=1)
    elif model_name == "vit_scratch":
        from models.vit_simple import create_vit_model
        model = create_vit_model(model_size="small", num_classes=1)
    elif model_name == "resnet_pretrained":
        from models.resnet_pretrained import create_resnet_model
        model = create_resnet_model(model_type="resnet50", num_classes=1, pretrained=True)
    elif model_name == "vit_pretrained":
        from models.vit_pretrained import create_vit_model as create_pretrained_vit_model
        model = create_pretrained_vit_model(model_type="base", num_classes=1, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create dummy data loaders (we'll only use test loader)
    _, val_loader, test_loader = create_data_loaders(
        config=config,
        label_type="binary_attention",
        batch_size=16  # Smaller batch for evaluation
    )
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=val_loader,  # Dummy
        val_loader=val_loader,
        config=config,
        model_name=model_name,
        loss_type="bce",
        task_type="binary_classification"
    )
    
    # Load checkpoint
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
        print(f"✅ Loaded checkpoint for {model_name}")
        return trainer, test_loader
    else:
        print(f"❌ Checkpoint not found for {model_name}: {checkpoint_path}")
        return None, None

def evaluate_model(model_name: str, trainer: EmotionTrainer, test_loader, save_dir: Path):
    """Evaluate a single model and save detailed results."""
    print(f"\nEvaluating {model_name}...")
    
    # Run evaluation
    test_metrics = trainer.evaluate(test_loader)
    
    # Save individual results
    model_results = {
        "model_name": model_name,
        "test_metrics": test_metrics,
        "model_parameters": trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else None
    }
    
    with open(save_dir / f"{model_name}_results.json", "w") as f:
        json.dump(model_results, f, indent=2)
    
    return test_metrics

def create_comparison_plots(results: dict, save_dir: Path):
    """Create comparison plots across all models."""
    # Prepare data for plotting
    models = list(results.keys())
    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc']
    
    plot_data = []
    for model, metrics in results.items():
        for metric in metrics_to_plot:
            if metric in metrics:
                plot_data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metrics[metric]
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Individual metric plots
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            metric_data = df[df['Metric'] == metric.replace('_', ' ').title()]
            
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Model', y='Value', ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].set_xlabel('Model Architecture')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt='%.3f')
    
    # Overall performance radar chart
    if len(axes) > len(metrics_to_plot):
        ax_radar = axes[len(metrics_to_plot)]
        ax_radar.remove()  # Remove the extra subplot
    
    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create performance summary table
    summary_data = []
    for model, metrics in results.items():
        summary_data.append({
            'Model': model,
            'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'F1 Score': f"{metrics.get('f1_score', 0):.3f}",
            'Precision': f"{metrics.get('precision', 0):.3f}",
            'Recall': f"{metrics.get('recall', 0):.3f}",
            'AUC-ROC': f"{metrics.get('auc_roc', 0):.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_dir / "model_comparison_summary.csv", index=False)
    
    print(f"\n📊 Performance Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def create_detailed_analysis(results: dict, save_dir: Path):
    """Create detailed analysis and insights."""
    analysis = {
        "experiment_summary": {
            "total_models_evaluated": len(results),
            "models": list(results.keys()),
            "evaluation_date": pd.Timestamp.now().isoformat()
        },
        "performance_ranking": {},
        "insights": {}
    }
    
    # Rank models by different metrics
    ranking_metrics = ['accuracy', 'f1_score', 'auc_roc']
    
    for metric in ranking_metrics:
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        analysis["performance_ranking"][metric] = [
            {"model": model, "score": metrics.get(metric, 0)}
            for model, metrics in ranked
        ]
    
    # Generate insights
    best_overall = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
    worst_overall = min(results.items(), key=lambda x: x[1].get('f1_score', 0))
    
    analysis["insights"] = {
        "best_model": {
            "name": best_overall[0],
            "f1_score": best_overall[1].get('f1_score', 0),
            "strengths": "Highest F1 score indicates best balance of precision and recall"
        },
        "worst_model": {
            "name": worst_overall[0],
            "f1_score": worst_overall[1].get('f1_score', 0),
            "areas_for_improvement": "Consider hyperparameter tuning or architecture modifications"
        },
        "performance_spread": {
            "f1_score_range": best_overall[1].get('f1_score', 0) - worst_overall[1].get('f1_score', 0),
            "interpretation": "Larger spread indicates significant differences between architectures"
        }
    }
    
    # Model complexity analysis (if parameter counts available)
    param_counts = {}
    for model, metrics in results.items():
        # Try to get parameter count from saved results
        results_file = save_dir / f"{model}_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                model_data = json.load(f)
                if model_data.get('model_parameters'):
                    param_counts[model] = model_data['model_parameters']['total_parameters']
    
    if param_counts:
        analysis["model_complexity"] = {
            "parameter_counts": param_counts,
            "efficiency_analysis": {
                model: {
                    "parameters": params,
                    "f1_per_million_params": results[model].get('f1_score', 0) / (params / 1e6)
                }
                for model, params in param_counts.items()
            }
        }
    
    # Save detailed analysis
    with open(save_dir / "detailed_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    return analysis

def main():
    """Main evaluation function."""
    print("=" * 80)
    print("MULTIMODAL EMOTION RECOGNITION - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    # Initialize configuration
    config = Config()
    
    # Models to evaluate
    models_to_evaluate = [
        "cnn_simple",
        "vit_scratch", 
        "resnet_pretrained",
        "vit_pretrained"
    ]
    
    # Results storage
    all_results = {}
    save_dir = Path("experiments/model_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {save_dir}")
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        # Look for checkpoint
        checkpoint_patterns = [
            Path("experiments") / f"{model_name}_binary_attention_*bs" / "checkpoints" / "best_model.pth",
            Path("experiments") / model_name / "checkpoints" / "best_model.pth",
            Path("experiments") / model_name / "best_model.pth"
        ]
        
        checkpoint_path = None
        for pattern in checkpoint_patterns:
            # Handle glob patterns
            if '*' in str(pattern):
                matches = list(pattern.parent.parent.glob(pattern.name.replace('*bs', '*')))
                if matches:
                    checkpoint_path = matches[0] / "checkpoints" / "best_model.pth"
                    break
            elif pattern.exists():
                checkpoint_path = pattern
                break
        
        if checkpoint_path is None:
            print(f"⚠️  No checkpoint found for {model_name}, skipping...")
            continue
        
        try:
            # Load model and evaluate
            trainer, test_loader = load_model_and_trainer(model_name, checkpoint_path, config)
            
            if trainer is not None and test_loader is not None:
                test_metrics = evaluate_model(model_name, trainer, test_loader, save_dir)
                all_results[model_name] = test_metrics
                
                print(f"✅ {model_name} evaluation completed")
                print(f"   F1 Score: {test_metrics.get('f1_score', 0):.3f}")
                print(f"   Accuracy: {test_metrics.get('accuracy', 0):.3f}")
            else:
                print(f"❌ Failed to load {model_name}")
                
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comprehensive analysis
    if all_results:
        print(f"\n{'='*80}")
        print("CREATING COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        # Create comparison plots
        summary_df = create_comparison_plots(all_results, save_dir)
        
        # Create detailed analysis
        analysis = create_detailed_analysis(all_results, save_dir)
        
        # Print final summary
        print(f"\n🎯 FINAL RESULTS SUMMARY:")
        print(f"📁 All results saved to: {save_dir}")
        print(f"📊 {len(all_results)} models evaluated")
        
        if analysis["insights"]["best_model"]:
            best = analysis["insights"]["best_model"]
            print(f"🏆 Best model: {best['name']} (F1: {best['f1_score']:.3f})")
        
        print(f"\n📋 Files created:")
        print(f"   - model_comparison.png (performance charts)")
        print(f"   - model_comparison_summary.csv (summary table)")
        print(f"   - detailed_analysis.json (comprehensive analysis)")
        print(f"   - [model]_results.json (individual results)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if len(all_results) >= 2:
            f1_scores = [(model, metrics.get('f1_score', 0)) for model, metrics in all_results.items()]
            f1_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   1. Deploy {f1_scores[0][0]} for production (best performance)")
            if len(f1_scores) > 1:
                print(f"   2. Consider {f1_scores[1][0]} as backup option")
            print(f"   3. Fine-tune hyperparameters for lower-performing models")
            print(f"   4. Collect more data if all models show low performance")
        
    else:
        print("❌ No models were successfully evaluated")
        print("Make sure you have trained models with checkpoints available")
        sys.exit(1)

if __name__ == "__main__":
    main()