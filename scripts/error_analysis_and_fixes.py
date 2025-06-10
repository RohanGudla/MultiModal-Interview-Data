#!/usr/bin/env python3
"""
ITERATION 5: Error analysis and implementation fixes.
Focus: Identify and address key issues found in previous iterations.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_all_results():
    """Load results from all previous iterations."""
    results = {}
    
    # Define iteration paths
    iteration_paths = {
        'iteration1': 'experiments/iteration1_analysis/iteration1_results.json',
        'iteration2': 'experiments/iteration2_cnn_baseline/iteration2_results.json', 
        'iteration3': 'experiments/iteration3_all_models/iteration3_results.json',
        'iteration4': 'experiments/iteration4_comprehensive_analysis/iteration4_comprehensive_analysis.json'
    }
    
    for iteration, path in iteration_paths.items():
        if Path(path).exists():
            with open(path, 'r') as f:
                results[iteration] = json.load(f)
    
    return results

def identify_critical_issues(results):
    """Identify critical issues that need addressing."""
    issues = []
    
    # Issue 1: Dummy data usage
    if 'iteration3' in results:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Using dummy/synthetic data instead of real video frames',
            'description': 'All models trained on random numpy arrays, not actual video data',
            'impact': 'Results are meaningless for real-world emotion recognition',
            'fix_priority': 1
        })
    
    # Issue 2: Overfitting detected
    if 'iteration4' in results:
        overfitting_models = []
        for insight in results['iteration4'].get('key_insights', []):
            if 'overfitting' in insight:
                model_name = insight.split()[1]
                overfitting_models.append(model_name)
        
        if overfitting_models:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Overfitting detected in multiple models',
                'description': f'Models {overfitting_models} show high train-validation gaps',
                'impact': 'Poor generalization to unseen data',
                'fix_priority': 2
            })
    
    # Issue 3: Low overall performance
    if 'iteration4' in results:
        best_acc = results['iteration4']['analysis_summary'].get('best_accuracy', 0)
        if best_acc < 0.5:
            issues.append({
                'severity': 'HIGH', 
                'issue': 'Poor model performance across all architectures',
                'description': f'Best accuracy only {best_acc:.1%}, all models at random chance level',
                'impact': 'Models are not learning meaningful patterns',
                'fix_priority': 2
            })
    
    # Issue 4: Data quality issues from Iteration 1
    if 'iteration1' in results:
        errors = results['iteration1'].get('errors', [])
        if errors:
            issues.append({
                'severity': 'MEDIUM',
                'issue': 'Data alignment issues',
                'description': f'{len(errors)} data errors found: {errors[:2]}...',
                'impact': 'Some participants missing complete data',
                'fix_priority': 3
            })
    
    # Issue 5: Small dataset size
    if 'iteration3' in results:
        total_samples = results['iteration3']['data_info'].get('train_samples', 0)
        if total_samples < 100:
            issues.append({
                'severity': 'MEDIUM',
                'issue': 'Very small dataset size',
                'description': f'Only {total_samples} training samples total',
                'impact': 'Insufficient data for robust model training',
                'fix_priority': 3
            })
    
    return sorted(issues, key=lambda x: x['fix_priority'])

def create_fix_recommendations(issues):
    """Create specific fix recommendations for identified issues."""
    fixes = []
    
    for issue in issues:
        if 'dummy data' in issue['issue'].lower():
            fixes.append({
                'issue_id': 'dummy_data',
                'title': 'Implement Real Video Frame Extraction',
                'description': 'Replace dummy data with actual video frame extraction',
                'steps': [
                    '1. Install OpenCV: pip install opencv-python',
                    '2. Create VideoFrameExtractor class to load actual MP4 files',
                    '3. Extract frames at fixed intervals (e.g., every 30 frames)',
                    '4. Resize frames to 224x224 for consistency',
                    '5. Align frames with corresponding emotion annotations',
                    '6. Create proper train/val/test splits by participant'
                ],
                'expected_improvement': 'Enable training on real data, meaningful results',
                'effort': 'HIGH',
                'timeline': '2-3 days'
            })
        
        elif 'overfitting' in issue['issue'].lower():
            fixes.append({
                'issue_id': 'overfitting',
                'title': 'Implement Overfitting Mitigation Strategies',
                'description': 'Add regularization and data augmentation',
                'steps': [
                    '1. Increase dropout rates (0.5 → 0.7)',
                    '2. Add weight decay (L2 regularization)',
                    '3. Implement early stopping based on validation loss',
                    '4. Add data augmentation (rotation, brightness, contrast)',
                    '5. Reduce model complexity if needed',
                    '6. Implement cross-validation for better evaluation'
                ],
                'expected_improvement': 'Better generalization, stable training',
                'effort': 'MEDIUM', 
                'timeline': '1-2 days'
            })
        
        elif 'poor performance' in issue['issue'].lower():
            fixes.append({
                'issue_id': 'poor_performance',
                'title': 'Optimize Model Training and Architecture',
                'description': 'Improve training process and model design',
                'steps': [
                    '1. Implement proper learning rate scheduling',
                    '2. Add batch normalization layers',
                    '3. Use better loss functions (focal loss for imbalanced data)',
                    '4. Increase training epochs with proper monitoring',
                    '5. Fine-tune hyperparameters (learning rate, batch size)',
                    '6. Consider ensemble methods'
                ],
                'expected_improvement': 'Higher accuracy and better convergence',
                'effort': 'MEDIUM',
                'timeline': '1-2 days'
            })
        
        elif 'data alignment' in issue['issue'].lower():
            fixes.append({
                'issue_id': 'data_alignment',
                'title': 'Fix Data Loading and Alignment Issues',
                'description': 'Ensure all participants have complete video+annotation data',
                'steps': [
                    '1. Audit all video files for corruption/accessibility',
                    '2. Verify annotation file format consistency',
                    '3. Implement robust error handling in data loading',
                    '4. Create data validation pipeline',
                    '5. Add missing data imputation strategies',
                    '6. Document data quality requirements'
                ],
                'expected_improvement': 'More reliable data loading, fewer errors',
                'effort': 'LOW',
                'timeline': '0.5-1 day'
            })
        
        elif 'small dataset' in issue['issue'].lower():
            fixes.append({
                'issue_id': 'small_dataset',
                'title': 'Implement Data Augmentation and Expansion',
                'description': 'Increase effective dataset size through augmentation',
                'steps': [
                    '1. Extract more frames per video (every 15 frames instead of sparse)',
                    '2. Implement temporal augmentation (frame sampling strategies)',
                    '3. Add spatial augmentations (rotation, flip, crop, color jitter)',
                    '4. Consider synthetic data generation if appropriate',
                    '5. Implement sliding window approach for temporal sequences',
                    '6. Use transfer learning more effectively'
                ],
                'expected_improvement': 'More training data, better model robustness',
                'effort': 'MEDIUM',
                'timeline': '1-2 days'
            })
    
    return fixes

def create_implementation_plan(fixes):
    """Create prioritized implementation plan."""
    
    # Sort by effort and expected impact
    priority_order = ['dummy_data', 'overfitting', 'data_alignment', 'poor_performance', 'small_dataset']
    
    plan = {
        'phase_1_critical': [],
        'phase_2_important': [],
        'phase_3_optimization': []
    }
    
    for fix in fixes:
        if fix['issue_id'] == 'dummy_data':
            plan['phase_1_critical'].append(fix)
        elif fix['issue_id'] in ['overfitting', 'data_alignment']:
            plan['phase_2_important'].append(fix)
        else:
            plan['phase_3_optimization'].append(fix)
    
    return plan

def estimate_project_timeline(plan):
    """Estimate total timeline for implementing all fixes."""
    
    timeline_map = {
        '0.5-1 day': 0.75,
        '1-2 days': 1.5, 
        '2-3 days': 2.5,
        '3-5 days': 4.0
    }
    
    total_days = 0
    phase_estimates = {}
    
    for phase, fixes in plan.items():
        phase_days = 0
        for fix in fixes:
            timeline_str = fix['timeline']
            phase_days += timeline_map.get(timeline_str, 2.0)
        
        phase_estimates[phase] = phase_days
        total_days += phase_days
    
    return {
        'total_days': total_days,
        'phase_estimates': phase_estimates,
        'total_weeks': total_days / 5,  # Working days
        'recommended_approach': 'Implement phases sequentially for best results'
    }

def generate_success_metrics(fixes):
    """Define success metrics for measuring fix effectiveness."""
    
    metrics = {
        'data_quality_metrics': [
            'Real video frames successfully extracted and processed',
            'Zero data loading errors',
            'All participants have complete aligned data',
            'Minimum 100 samples per participant'
        ],
        'model_performance_metrics': [
            'Test accuracy > 60% (above random chance)',
            'F1-score > 0.5',
            'Validation accuracy within 10% of training accuracy',
            'Training loss consistently decreasing'
        ],
        'training_stability_metrics': [
            'No overfitting (train-val gap < 15%)',
            'Consistent convergence across multiple runs',
            'Early stopping triggered appropriately',
            'Learning curves show proper progression'
        ],
        'system_reliability_metrics': [
            'Zero crashes during training',
            'Reproducible results with fixed random seeds',
            'Memory usage within acceptable limits',
            'Training time < 10 minutes per model'
        ]
    }
    
    return metrics

def save_error_analysis_report(issues, fixes, plan, timeline, metrics, output_dir):
    """Save comprehensive error analysis and fix plan."""
    
    report = {
        'iteration': 5,
        'timestamp': pd.Timestamp.now().isoformat(),
        'analysis_summary': {
            'total_issues_identified': len(issues),
            'critical_issues': len([i for i in issues if i['severity'] == 'CRITICAL']),
            'high_priority_issues': len([i for i in issues if i['severity'] == 'HIGH']),
            'total_fixes_proposed': len(fixes)
        },
        'identified_issues': issues,
        'proposed_fixes': fixes,
        'implementation_plan': plan,
        'timeline_estimate': timeline,
        'success_metrics': metrics,
        'recommendations': [
            'Prioritize Phase 1 (real data implementation) before proceeding',
            'Implement fixes incrementally and test after each phase',
            'Monitor success metrics continuously during implementation',
            'Consider this as foundation for future multimodal emotion recognition work',
            'Document all changes for reproducibility'
        ]
    }
    
    # Save main report
    report_file = output_dir / 'iteration5_error_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create implementation checklist
    checklist = []
    for phase_name, phase_fixes in plan.items():
        checklist.append(f"\n## {phase_name.upper().replace('_', ' ')}")
        for fix in phase_fixes:
            checklist.append(f"\n### {fix['title']}")
            for step in fix['steps']:
                checklist.append(f"- [ ] {step}")
    
    checklist_file = output_dir / 'implementation_checklist.md'
    with open(checklist_file, 'w') as f:
        f.write('# Multimodal Video ML Implementation Checklist\n')
        f.write('\n'.join(checklist))
    
    return report

def run_iteration5():
    """Run Iteration 5: Error analysis and fixes."""
    
    print("🚀 ITERATION 5: ERROR ANALYSIS & FIXES")
    print("=" * 80)
    print("Goal: Identify critical issues and create comprehensive fix plan")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("experiments/iteration5_error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load all results
        print("📊 Loading results from all iterations...")
        results = load_all_results()
        
        if not results:
            print("❌ No iteration results found")
            return False
        
        print(f"✅ Loaded {len(results)} iteration results")
        
        # Identify critical issues
        print("\n🔍 Identifying critical issues...")
        issues = identify_critical_issues(results)
        
        print(f"✅ Identified {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. [{issue['severity']}] {issue['issue']}")
        
        # Create fix recommendations
        print("\n🔧 Creating fix recommendations...")
        fixes = create_fix_recommendations(issues)
        
        print(f"✅ Created {len(fixes)} fix recommendations")
        
        # Create implementation plan
        print("📋 Creating implementation plan...")
        plan = create_implementation_plan(fixes)
        
        print(f"✅ Plan created with {len(plan)} phases")
        
        # Estimate timeline
        print("⏱️  Estimating implementation timeline...")
        timeline = estimate_project_timeline(plan)
        
        print(f"✅ Estimated {timeline['total_days']:.1f} days ({timeline['total_weeks']:.1f} weeks)")
        
        # Generate success metrics
        print("📊 Defining success metrics...")
        metrics = generate_success_metrics(fixes)
        
        print(f"✅ Defined {sum(len(v) for v in metrics.values())} success metrics")
        
        # Save comprehensive report
        print("💾 Saving error analysis report...")
        report = save_error_analysis_report(issues, fixes, plan, timeline, metrics, output_dir)
        
        # Display key results
        print(f"\n🎯 ITERATION 5 ERROR ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"\n❌ CRITICAL ISSUES IDENTIFIED:")
        for issue in issues:
            if issue['severity'] == 'CRITICAL':
                print(f"   🔥 {issue['issue']}")
                print(f"      Impact: {issue['impact']}")
        
        print(f"\n🔧 IMPLEMENTATION PHASES:")
        for phase_name, phase_fixes in plan.items():
            phase_days = timeline['phase_estimates'][phase_name]
            print(f"   📋 {phase_name.replace('_', ' ').title()}: {len(phase_fixes)} fixes ({phase_days:.1f} days)")
            for fix in phase_fixes:
                print(f"      - {fix['title']}")
        
        print(f"\n⏱️  TIMELINE ESTIMATE:")
        print(f"   Total implementation time: {timeline['total_days']:.1f} days ({timeline['total_weeks']:.1f} weeks)")
        print(f"   Recommended approach: {timeline['recommended_approach']}")
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print(f"   Phase 1: Real video data successfully implemented")
        print(f"   Phase 2: Overfitting resolved, stable training achieved")
        print(f"   Phase 3: Model performance > 60% accuracy")
        
        print(f"\n📁 Complete analysis saved to: {output_dir}")
        print(f"📋 Implementation checklist: {output_dir}/implementation_checklist.md")
        
        # Final assessment
        critical_count = len([i for i in issues if i['severity'] == 'CRITICAL'])
        if critical_count > 0:
            print(f"\n⚠️  ITERATION 5 COMPLETE - ACTION REQUIRED")
            print(f"❌ {critical_count} critical issues must be addressed")
            print(f"🔧 Follow implementation plan to resolve issues")
            print(f"🎯 Expected timeline: {timeline['total_weeks']:.1f} weeks for full implementation")
        else:
            print(f"\n🎉 ITERATION 5 SUCCESS!")
            print(f"✅ No critical issues found - project ready for optimization")
        
        return True
        
    except Exception as e:
        print(f"\n💥 ITERATION 5 FAILED")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Iteration 5."""
    success = run_iteration5()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)