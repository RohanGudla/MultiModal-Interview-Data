#!/usr/bin/env python3
"""
Enhanced Output Verification System
Creates comprehensive CSV files comparing predictions vs ground truth as requested by colleague
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class EnhancedOutputVerifier:
    """Enhanced comprehensive output verification and analysis system"""
    
    def __init__(self, output_dir: str):
        """
        Initialize enhanced output verifier
        
        Args:
            output_dir: Directory to save verification outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.csv_dir = self.output_dir / "verification_csvs"
        self.plots_dir = self.output_dir / "verification_plots"
        self.reports_dir = self.output_dir / "verification_reports"
        
        for dir_path in [self.csv_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📋 Enhanced Output Verifier initialized: {output_dir}")
    
    def create_verification_outputs(self,
                                  predictions: np.ndarray,
                                  ground_truth: np.ndarray,
                                  participants: List[str],
                                  frame_ids: List[int],
                                  feature_names: List[str]):
        """
        Create comprehensive verification outputs as requested by colleague
        
        Args:
            predictions: Model predictions (N x num_features)
            ground_truth: Ground truth labels (N x num_features)
            participants: List of participant IDs
            frame_ids: List of frame IDs
            feature_names: List of feature names
        """
        
        print(f"📊 Creating comprehensive verification outputs...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Create main comparison CSV
        comparison_df = self._create_main_comparison_csv(
            predictions, ground_truth, participants, frame_ids, feature_names, timestamp
        )
        
        # 2. Create per-participant CSVs
        self._create_per_participant_csvs(comparison_df, timestamp)
        
        # 3. Create feature-wise analysis
        self._create_feature_analysis(comparison_df, feature_names, timestamp)
        
        # 4. Create temporal analysis (start/stop times) - As requested by colleague
        self._create_temporal_analysis(comparison_df, feature_names, timestamp)
        
        # 5. Create visualization plots
        self._create_verification_plots(comparison_df, feature_names, timestamp)
        
        # 6. Create summary report
        self._create_summary_report(comparison_df, feature_names, timestamp)
        
        print(f"✅ Verification outputs created in: {self.output_dir}")
        
        return comparison_df
        
    def _create_main_comparison_csv(self,
                                   predictions: np.ndarray,
                                   ground_truth: np.ndarray,
                                   participants: List[str],
                                   frame_ids: List[int],
                                   feature_names: List[str],
                                   timestamp: str) -> pd.DataFrame:
        """Create main comparison CSV with predictions vs ground truth"""
        
        print("   Creating main comparison CSV...")
        
        # Create base dataframe
        data = {
            'participant_id': participants,
            'frame_id': frame_ids,
            'timestamp_seconds': [fid * 1.0 for fid in frame_ids]  # Assuming 1 FPS
        }
        
        # Add predictions and ground truth for each feature
        for i, feature in enumerate(feature_names):
            data[f'{feature}_prediction'] = predictions[:, i]
            data[f'{feature}_ground_truth'] = ground_truth[:, i]
            data[f'{feature}_binary_prediction'] = (predictions[:, i] > 0.5).astype(int)
            data[f'{feature}_match'] = (
                (predictions[:, i] > 0.5).astype(int) == ground_truth[:, i].astype(int)
            ).astype(int)
        
        df = pd.DataFrame(data)
        
        # Save main CSV - This is the key output requested by colleague
        main_csv_path = self.csv_dir / f"predictions_vs_ground_truth_{timestamp}.csv"
        df.to_csv(main_csv_path, index=False)
        
        print(f"     Main CSV saved: {main_csv_path}")
        return df
    
    def _create_per_participant_csvs(self, df: pd.DataFrame, timestamp: str):
        """Create separate CSV files for each participant"""
        
        print("   Creating per-participant CSVs...")
        
        participant_dir = self.csv_dir / "per_participant"
        participant_dir.mkdir(exist_ok=True)
        
        for participant in df['participant_id'].unique():
            participant_df = df[df['participant_id'] == participant].copy()
            
            # Sort by frame_id
            participant_df = participant_df.sort_values('frame_id')
            
            # Save participant CSV
            participant_csv = participant_dir / f"{participant}_predictions_{timestamp}.csv"
            participant_df.to_csv(participant_csv, index=False)
        
        print(f"     Per-participant CSVs saved: {participant_dir}")
    
    def _create_feature_analysis(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Create feature-wise analysis and statistics"""
        
        print("   Creating feature analysis...")
        
        feature_stats = []
        
        for feature in feature_names:
            pred_col = f'{feature}_binary_prediction'
            truth_col = f'{feature}_ground_truth'
            match_col = f'{feature}_match'
            
            if pred_col in df.columns and truth_col in df.columns:
                stats = {
                    'feature_name': feature,
                    'total_samples': len(df),
                    'true_positive': ((df[pred_col] == 1) & (df[truth_col] == 1)).sum(),
                    'true_negative': ((df[pred_col] == 0) & (df[truth_col] == 0)).sum(),
                    'false_positive': ((df[pred_col] == 1) & (df[truth_col] == 0)).sum(),
                    'false_negative': ((df[pred_col] == 0) & (df[truth_col] == 1)).sum(),
                    'accuracy': df[match_col].mean(),
                    'ground_truth_positive_rate': df[truth_col].mean(),
                    'prediction_positive_rate': df[pred_col].mean()
                }
                
                # Calculate precision, recall, F1
                tp = stats['true_positive']
                fp = stats['false_positive']
                fn = stats['false_negative']
                
                stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                stats['f1_score'] = (
                    2 * stats['precision'] * stats['recall'] / 
                    (stats['precision'] + stats['recall'])
                    if (stats['precision'] + stats['recall']) > 0 else 0.0
                )
                
                feature_stats.append(stats)
        
        # Save feature analysis
        feature_df = pd.DataFrame(feature_stats)
        feature_csv = self.csv_dir / f"feature_analysis_{timestamp}.csv"
        feature_df.to_csv(feature_csv, index=False)
        
        print(f"     Feature analysis saved: {feature_csv}")
    
    def _create_temporal_analysis(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Create temporal analysis with start/stop times as specifically requested by colleague"""
        
        print("   Creating temporal analysis (start/stop times)...")
        
        temporal_events = []
        
        # Group by participant
        for participant in df['participant_id'].unique():
            participant_df = df[df['participant_id'] == participant].sort_values('frame_id')
            
            for feature in feature_names:
                pred_col = f'{feature}_binary_prediction'
                truth_col = f'{feature}_ground_truth'
                
                if pred_col in participant_df.columns:
                    # Find start/stop events for predictions
                    pred_events = self._find_start_stop_events(
                        participant_df, pred_col, 'timestamp_seconds', participant, feature, 'prediction'
                    )
                    temporal_events.extend(pred_events)
                    
                    # Find start/stop events for ground truth
                    truth_events = self._find_start_stop_events(
                        participant_df, truth_col, 'timestamp_seconds', participant, feature, 'ground_truth'
                    )
                    temporal_events.extend(truth_events)
        
        # Save temporal analysis - This addresses the colleague's request for start/stop times
        if temporal_events:
            temporal_df = pd.DataFrame(temporal_events)
            temporal_csv = self.csv_dir / f"temporal_start_stop_events_{timestamp}.csv"
            temporal_df.to_csv(temporal_csv, index=False)
            
            print(f"     Temporal analysis saved: {temporal_csv}")
        else:
            print("     No temporal events found")
    
    def _find_start_stop_events(self, df: pd.DataFrame, value_col: str, time_col: str, 
                               participant: str, feature: str, data_type: str) -> List[Dict]:
        """Find start and stop events for a binary feature"""
        
        events = []
        values = df[value_col].values
        times = df[time_col].values
        
        if len(values) == 0:
            return events
        
        # Find transitions
        in_event = False
        start_time = None
        
        for i, (value, time) in enumerate(zip(values, times)):
            if value == 1 and not in_event:
                # Start of event
                in_event = True
                start_time = time
            elif value == 0 and in_event:
                # End of event
                in_event = False
                events.append({
                    'participant_id': participant,
                    'feature_name': feature,
                    'data_type': data_type,
                    'event_type': 'complete',
                    'start_time': start_time,
                    'stop_time': time,
                    'duration': time - start_time
                })
        
        # Handle case where event continues to end
        if in_event and start_time is not None:
            events.append({
                'participant_id': participant,
                'feature_name': feature,
                'data_type': data_type,
                'event_type': 'ongoing',
                'start_time': start_time,
                'stop_time': times[-1],
                'duration': times[-1] - start_time
            })
        
        return events
    
    def _create_verification_plots(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Create visualization plots for verification"""
        
        print("   Creating verification plots...")
        
        # 1. Overall accuracy heatmap
        self._plot_accuracy_heatmap(df, feature_names, timestamp)
        
        # 2. Feature distribution plots
        self._plot_feature_distributions(df, feature_names, timestamp)
        
        # 3. Participant comparison plots
        self._plot_participant_comparison(df, feature_names, timestamp)
        
        print(f"     Plots saved in: {self.plots_dir}")
    
    def _plot_accuracy_heatmap(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Create accuracy heatmap"""
        
        # Calculate accuracy per feature per participant
        accuracy_data = []
        
        for participant in df['participant_id'].unique():
            participant_df = df[df['participant_id'] == participant]
            participant_accuracy = []
            
            for feature in feature_names:
                match_col = f'{feature}_match'
                if match_col in participant_df.columns:
                    accuracy = participant_df[match_col].mean()
                    participant_accuracy.append(accuracy)
                else:
                    participant_accuracy.append(0.0)
            
            accuracy_data.append(participant_accuracy)
        
        # Create heatmap
        accuracy_matrix = np.array(accuracy_data)
        
        plt.figure(figsize=(20, 8))
        sns.heatmap(
            accuracy_matrix,
            xticklabels=feature_names,
            yticklabels=df['participant_id'].unique(),
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1
        )
        plt.title('Prediction Accuracy by Participant and Feature')
        plt.xlabel('Features')
        plt.ylabel('Participants')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'accuracy_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distributions(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Plot feature distributions"""
        
        # Select subset of features for plotting (to avoid overcrowding)
        plot_features = feature_names[:12]  # First 12 features
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(plot_features):
            if i >= len(axes):
                break
                
            pred_col = f'{feature}_prediction'
            truth_col = f'{feature}_ground_truth'
            
            if pred_col in df.columns and truth_col in df.columns:
                # Plot distributions
                axes[i].hist(df[df[truth_col] == 0][pred_col], 
                           alpha=0.5, label='Ground Truth: 0', bins=20)
                axes[i].hist(df[df[truth_col] == 1][pred_col], 
                           alpha=0.5, label='Ground Truth: 1', bins=20)
                axes[i].axvline(0.5, color='red', linestyle='--', label='Threshold')
                axes[i].set_title(f'{feature}')
                axes[i].set_xlabel('Prediction Score')
                axes[i].set_ylabel('Count')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'feature_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_participant_comparison(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Plot participant comparison"""
        
        # Calculate overall accuracy per participant
        participant_accuracies = []
        participants = []
        
        for participant in df['participant_id'].unique():
            participant_df = df[df['participant_id'] == participant]
            
            # Calculate average accuracy across all features
            match_cols = [f'{feature}_match' for feature in feature_names if f'{feature}_match' in df.columns]
            if match_cols:
                avg_accuracy = participant_df[match_cols].mean().mean()
                participant_accuracies.append(avg_accuracy)
                participants.append(participant)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(participants, participant_accuracies, color='skyblue', edgecolor='navy')
        plt.title('Average Prediction Accuracy by Participant')
        plt.xlabel('Participant ID')
        plt.ylabel('Average Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, participant_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'participant_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_report(self, df: pd.DataFrame, feature_names: List[str], timestamp: str):
        """Create comprehensive summary report"""
        
        print("   Creating summary report...")
        
        report = {
            'timestamp': timestamp,
            'total_samples': len(df),
            'unique_participants': df['participant_id'].nunique(),
            'participants': df['participant_id'].unique().tolist(),
            'total_features': len(feature_names),
            'feature_categories': {
                'physical_features': [f for f in feature_names if any(
                    keyword in f.lower() for keyword in ['head', 'eye', 'mouth', 'lip', 'brow', 'chin', 'nose']
                )],
                'emotional_features': [f for f in feature_names if any(
                    keyword in f.lower() for keyword in ['joy', 'anger', 'fear', 'sadness', 'surprise', 'valence', 'attention']
                )]
            }
        }
        
        # Overall statistics
        match_cols = [f'{feature}_match' for feature in feature_names if f'{feature}_match' in df.columns]
        if match_cols:
            report['overall_accuracy'] = df[match_cols].mean().mean()
            report['feature_accuracies'] = {}
            
            for feature in feature_names:
                match_col = f'{feature}_match'
                if match_col in df.columns:
                    report['feature_accuracies'][feature] = df[match_col].mean()
        
        # Participant statistics
        report['participant_stats'] = {}
        for participant in df['participant_id'].unique():
            participant_df = df[df['participant_id'] == participant]
            if match_cols:
                participant_accuracy = participant_df[match_cols].mean().mean()
                report['participant_stats'][participant] = {
                    'samples': len(participant_df),
                    'accuracy': participant_accuracy
                }
        
        # Save report
        report_file = self.reports_dir / f"verification_summary_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"     Summary report saved: {report_file}")
        
        # Create human-readable text report
        self._create_text_report(report, timestamp)
    
    def _create_text_report(self, report: Dict[str, Any], timestamp: str):
        """Create human-readable text report"""
        
        text_report = []
        text_report.append("MULTIMODAL VIDEO ANNOTATION - VERIFICATION REPORT")
        text_report.append("=" * 60)
        text_report.append(f"Generated: {timestamp}")
        text_report.append("")
        
        text_report.append("DATASET SUMMARY")
        text_report.append("-" * 30)
        text_report.append(f"Total samples: {report['total_samples']:,}")
        text_report.append(f"Unique participants: {report['unique_participants']}")
        text_report.append(f"Total features: {report['total_features']}")
        text_report.append("")
        
        text_report.append("PARTICIPANTS")
        text_report.append("-" * 30)
        for participant in report['participants']:
            stats = report['participant_stats'].get(participant, {})
            samples = stats.get('samples', 0)
            accuracy = stats.get('accuracy', 0.0)
            text_report.append(f"{participant}: {samples} samples, {accuracy:.3f} accuracy")
        text_report.append("")
        
        text_report.append("OVERALL PERFORMANCE")
        text_report.append("-" * 30)
        if 'overall_accuracy' in report:
            text_report.append(f"Overall accuracy: {report['overall_accuracy']:.4f}")
        text_report.append("")
        
        text_report.append("TOP PERFORMING FEATURES")
        text_report.append("-" * 30)
        if 'feature_accuracies' in report:
            sorted_features = sorted(
                report['feature_accuracies'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, accuracy in sorted_features[:10]:
                text_report.append(f"{feature}: {accuracy:.3f}")
        
        # Save text report
        text_file = self.reports_dir / f"verification_report_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write('\n'.join(text_report))
        
        print(f"     Text report saved: {text_file}")

def main():
    """Test the enhanced output verification system"""
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    predictions = np.random.random((n_samples, n_features))
    ground_truth = np.random.randint(0, 2, (n_samples, n_features))
    participants = [f"TEST_{i//20:02d}" for i in range(n_samples)]
    frame_ids = list(range(n_samples))
    feature_names = [f"Feature_{i:02d}" for i in range(n_features)]
    
    # Create verifier
    verifier = EnhancedOutputVerifier("/home/rohan/Multimodal/multimodal_video_ml/outputs/verification_test")
    
    # Generate verification outputs
    verifier.create_verification_outputs(
        predictions, ground_truth, participants, frame_ids, feature_names
    )

if __name__ == "__main__":
    main()