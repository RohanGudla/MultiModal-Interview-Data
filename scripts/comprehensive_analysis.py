#!/usr/bin/env python3
"""
ITERATION 4: Comprehensive comparison and analysis.
Focus: Detailed analysis of all 4 model performances with statistical comparisons.
"""
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_results():
    """Load all iteration results."""
    results = {}
    
    # Load Iteration 1 results
    iter1_path = Path("experiments/iteration1_analysis/iteration1_results.json")
    if iter1_path.exists():
        with open(iter1_path, 'r') as f:
            results['iteration1'] = json.load(f)
    
    # Load Iteration 2 results
    iter2_path = Path("experiments/iteration2_cnn_baseline/iteration2_results.json")
    if iter2_path.exists():
        with open(iter2_path, 'r') as f:
            results['iteration2'] = json.load(f)
    
    # Load Iteration 3 results
    iter3_path = Path("experiments/iteration3_all_models/iteration3_results.json")
    if iter3_path.exists():
        with open(iter3_path, 'r') as f:
            results['iteration3'] = json.load(f)
    
    return results

def create_model_comparison_table(results):
    """Create comprehensive model comparison table."""
    if 'iteration3' not in results:
        print("❌ Iteration 3 results not found")
        return None
    
    models_data = []
    
    for model_name, model_result in results['iteration3']['model_results'].items():
        if model_result.get('success', False):
            model_info = {
                'Model': model_name,
                'Parameters': model_result['model_info']['parameters']['total'],
                'Trainable_Params': model_result['model_info']['parameters']['trainable'],
                'Training_Time': model_result['training_results']['training_time'],
                'Best_Val_Acc': model_result['training_results']['best_val_accuracy'],
                'Test_Accuracy': model_result['test_results']['accuracy'],
                'Test_Precision': model_result['test_results']['precision'],
                'Test_Recall': model_result['test_results']['recall'],
                'Test_F1': model_result['test_results']['f1_score'],
                'Final_Train_Acc': model_result['training_results']['train_accuracies'][-1],
                'Final_Val_Acc': model_result['training_results']['val_accuracies'][-1]
            }
            models_data.append(model_info)
    
    df = pd.DataFrame(models_data)
    return df

def analyze_training_dynamics(results):
    """Analyze training curves and convergence patterns."""
    if 'iteration3' not in results:
        return None
    
    analysis = {
        'convergence_analysis': {},
        'overfitting_analysis': {},
        'efficiency_analysis': {}
    }
    
    for model_name, model_result in results['iteration3']['model_results'].items():
        if not model_result.get('success', False):
            continue
            
        train_accs = model_result['training_results']['train_accuracies']
        val_accs = model_result['training_results']['val_accuracies']
        train_losses = model_result['training_results']['train_losses']
        val_losses = model_result['training_results']['val_losses']
        
        # Convergence analysis
        train_improvement = train_accs[-1] - train_accs[0]
        val_improvement = val_accs[-1] - val_accs[0]
        
        analysis['convergence_analysis'][model_name] = {
            'train_improvement': train_improvement,
            'val_improvement': val_improvement,
            'final_gap': train_accs[-1] - val_accs[-1],
            'loss_reduction': train_losses[0] - train_losses[-1]
        }
        
        # Overfitting analysis (train vs val performance)
        avg_train_val_gap = np.mean([t - v for t, v in zip(train_accs, val_accs)])
        analysis['overfitting_analysis'][model_name] = {
            'avg_train_val_gap': avg_train_val_gap,
            'final_train_val_gap': train_accs[-1] - val_accs[-1],
            'overfitting_score': max(0, avg_train_val_gap)  # Higher = more overfitting
        }
        
        # Efficiency analysis (performance per parameter)
        params = model_result['model_info']['parameters']['total']
        training_time = model_result['training_results']['training_time']
        
        analysis['efficiency_analysis'][model_name] = {
            'params_millions': params / 1e6,
            'training_time': training_time,
            'accuracy_per_param': model_result['test_results']['accuracy'] / (params / 1e6),
            'accuracy_per_second': model_result['test_results']['accuracy'] / training_time
        }
    
    return analysis

def create_ranking_system(df, analysis):
    """Create comprehensive ranking system for models."""
    if df is None or analysis is None:
        return None
    
    rankings = {}
    
    # Performance ranking (40% weight)
    performance_metrics = ['Test_Accuracy', 'Test_F1', 'Best_Val_Acc']
    performance_scores = df[performance_metrics].mean(axis=1)
    
    # Efficiency ranking (30% weight)
    # Higher accuracy per parameter is better, normalize by max
    efficiency_scores = []
    for _, row in df.iterrows():
        model_name = row['Model']
        if model_name in analysis['efficiency_analysis']:
            acc_per_param = analysis['efficiency_analysis'][model_name]['accuracy_per_param']
            acc_per_sec = analysis['efficiency_analysis'][model_name]['accuracy_per_second']
            efficiency_scores.append((acc_per_param + acc_per_sec) / 2)
        else:
            efficiency_scores.append(0)
    
    efficiency_scores = np.array(efficiency_scores)
    if efficiency_scores.max() > 0:
        efficiency_scores = efficiency_scores / efficiency_scores.max()
    
    # Stability ranking (20% weight)
    stability_scores = []
    for _, row in df.iterrows():
        model_name = row['Model']
        if model_name in analysis['overfitting_analysis']:
            # Lower overfitting score is better
            overfitting = analysis['overfitting_analysis'][model_name]['overfitting_score']
            stability_scores.append(max(0, 1 - overfitting))  # Invert so higher is better
        else:
            stability_scores.append(0)
    
    stability_scores = np.array(stability_scores)
    
    # Training efficiency (10% weight)
    training_eff_scores = 1 / (df['Training_Time'] + 0.1)  # Lower time is better
    training_eff_scores = training_eff_scores / training_eff_scores.max()
    
    # Combined ranking
    final_scores = (
        0.4 * performance_scores + 
        0.3 * efficiency_scores + 
        0.2 * stability_scores + 
        0.1 * training_eff_scores
    )
    
    # Create ranking dataframe
    ranking_df = df.copy()
    ranking_df['Performance_Score'] = performance_scores
    ranking_df['Efficiency_Score'] = efficiency_scores
    ranking_df['Stability_Score'] = stability_scores
    ranking_df['Training_Eff_Score'] = training_eff_scores
    ranking_df['Final_Score'] = final_scores
    ranking_df['Rank'] = ranking_df['Final_Score'].rank(ascending=False, method='dense')
    
    return ranking_df.sort_values('Rank')

def generate_insights(df, analysis, rankings):
    """Generate key insights from the analysis."""
    insights = []
    
    if df is None or analysis is None:
        return ["❌ Insufficient data for insights generation"]
    
    # Best performer
    if len(df) > 0:
        best_model = df.loc[df['Test_Accuracy'].idxmax(), 'Model']
        best_acc = df.loc[df['Test_Accuracy'].idxmax(), 'Test_Accuracy']
        insights.append(f"🏆 Best performer: {best_model} ({best_acc:.1%} test accuracy)")
    
    # Most efficient
    min_params_model = df.loc[df['Parameters'].idxmin(), 'Model']
    min_params = df.loc[df['Parameters'].idxmin(), 'Parameters']
    insights.append(f"⚡ Most efficient: {min_params_model} ({min_params/1e6:.1f}M parameters)")
    
    # Fastest training
    fastest_model = df.loc[df['Training_Time'].idxmin(), 'Model']
    fastest_time = df.loc[df['Training_Time'].idxmin(), 'Training_Time']
    insights.append(f"🚀 Fastest training: {fastest_model} ({fastest_time:.1f}s)")
    
    # Architecture insights
    cnn_models = [m for m in df['Model'] if 'CNN' in m]
    vit_models = [m for m in df['Model'] if 'ViT' in m]
    pretrained_models = [m for m in df['Model'] if 'Pretrained' in m]
    
    if cnn_models and vit_models:
        cnn_avg_acc = df[df['Model'].isin(cnn_models)]['Test_Accuracy'].mean()
        vit_avg_acc = df[df['Model'].isin(vit_models)]['Test_Accuracy'].mean()
        
        if vit_avg_acc > cnn_avg_acc:
            insights.append(f"🔍 Vision Transformers outperform CNNs ({vit_avg_acc:.1%} vs {cnn_avg_acc:.1%})")
        else:
            insights.append(f"🔍 CNNs outperform Vision Transformers ({cnn_avg_acc:.1%} vs {vit_avg_acc:.1%})")
    
    if pretrained_models:
        pretrained_avg_acc = df[df['Model'].isin(pretrained_models)]['Test_Accuracy'].mean()
        scratch_models = [m for m in df['Model'] if m not in pretrained_models]
        if scratch_models:
            scratch_avg_acc = df[df['Model'].isin(scratch_models)]['Test_Accuracy'].mean()
            
            if pretrained_avg_acc > scratch_avg_acc:
                insights.append(f"📚 Pretrained models show advantage ({pretrained_avg_acc:.1%} vs {scratch_avg_acc:.1%})")
    
    # Performance consistency
    acc_std = df['Test_Accuracy'].std()
    if acc_std < 0.1:
        insights.append("📊 Model performances are quite consistent across architectures")
    else:
        insights.append("📊 Significant performance variation between architectures")
    
    # Training stability
    for model_name, overfitting_data in analysis['overfitting_analysis'].items():
        if overfitting_data['overfitting_score'] > 0.2:
            insights.append(f"⚠️  {model_name} shows signs of overfitting (train-val gap: {overfitting_data['avg_train_val_gap']:.1%})")
    
    return insights

def save_analysis_report(results, df, analysis, rankings, insights, output_dir):
    """Save comprehensive analysis report."""
    
    # Create summary report
    summary_report = {
        "iteration": 4,
        "timestamp": pd.Timestamp.now().isoformat(),
        "analysis_summary": {
            "total_models_analyzed": len(df) if df is not None else 0,
            "best_model": df.loc[df['Test_Accuracy'].idxmax(), 'Model'] if df is not None and len(df) > 0 else "N/A",
            "best_accuracy": float(df['Test_Accuracy'].max()) if df is not None and len(df) > 0 else 0.0,
            "avg_accuracy": float(df['Test_Accuracy'].mean()) if df is not None and len(df) > 0 else 0.0,
            "total_training_time": float(df['Training_Time'].sum()) if df is not None and len(df) > 0 else 0.0
        },
        "key_insights": insights,
        "detailed_analysis": analysis,
        "model_rankings": rankings.to_dict('records') if rankings is not None else [],
        "raw_comparison": df.to_dict('records') if df is not None else []
    }
    
    # Save JSON report
    report_file = output_dir / "iteration4_comprehensive_analysis.json"
    with open(report_file, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    # Save CSV comparison
    if df is not None:
        csv_file = output_dir / "model_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        if rankings is not None:
            ranking_file = output_dir / "model_rankings.csv"
            rankings.to_csv(ranking_file, index=False)
    
    return summary_report

def run_iteration4():
    """Run Iteration 4: Comprehensive analysis."""
    
    print("🚀 ITERATION 4: COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print("Goal: Detailed comparison and analysis of all model architectures")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("experiments/iteration4_comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load all results
        print("📊 Loading results from all iterations...")
        results = load_results()
        
        if not results:
            print("❌ No results found. Run previous iterations first.")
            return False
        
        print(f"✅ Loaded {len(results)} iteration results")
        
        # Create model comparison table
        print("\n🔍 Creating model comparison table...")
        comparison_df = create_model_comparison_table(results)
        
        if comparison_df is None:
            print("❌ Failed to create comparison table")
            return False
        
        print(f"✅ Analyzed {len(comparison_df)} models")
        
        # Analyze training dynamics
        print("📈 Analyzing training dynamics...")
        training_analysis = analyze_training_dynamics(results)
        
        # Create ranking system
        print("🏆 Creating model ranking system...")
        rankings = create_ranking_system(comparison_df, training_analysis)
        
        # Generate insights
        print("💡 Generating insights...")
        insights = generate_insights(comparison_df, training_analysis, rankings)
        
        # Save comprehensive report
        print("💾 Saving analysis report...")
        report = save_analysis_report(
            results, comparison_df, training_analysis, rankings, insights, output_dir
        )
        
        # Display results
        print(f"\n🎯 ITERATION 4 ANALYSIS RESULTS")
        print("=" * 60)
        
        # Model comparison table
        if len(comparison_df) > 0:
            print(f"\n📊 MODEL COMPARISON:")
            print(comparison_df[['Model', 'Parameters', 'Test_Accuracy', 'Test_F1', 'Training_Time']].to_string(index=False))
        
        # Rankings
        if rankings is not None and len(rankings) > 0:
            print(f"\n🏆 MODEL RANKINGS:")
            print(rankings[['Rank', 'Model', 'Final_Score', 'Test_Accuracy']].round(3).to_string(index=False))
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Models analyzed: {report['analysis_summary']['total_models_analyzed']}")
        print(f"   Best model: {report['analysis_summary']['best_model']}")
        print(f"   Best accuracy: {report['analysis_summary']['best_accuracy']:.1%}")
        print(f"   Average accuracy: {report['analysis_summary']['avg_accuracy']:.1%}")
        print(f"   Total training time: {report['analysis_summary']['total_training_time']:.1f}s")
        
        print(f"\n📁 Reports saved to: {output_dir}")
        
        # Success criteria
        if len(comparison_df) >= 3 and report['analysis_summary']['best_accuracy'] > 0.2:
            print(f"\n🎉 ITERATION 4 SUCCESS!")
            print(f"✅ Comprehensive analysis completed")
            print(f"✅ {len(insights)} insights generated")
            print(f"✅ Ready to proceed to Iteration 5: Error Analysis")
            return True
        else:
            print(f"\n⚠️  ITERATION 4 NEEDS ATTENTION")
            print(f"❌ Insufficient models or poor performance detected")
            return False
        
    except Exception as e:
        print(f"\n💥 ITERATION 4 FAILED")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Iteration 4."""
    success = run_iteration4()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)