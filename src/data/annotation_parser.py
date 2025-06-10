"""
Annotation parsing utilities for extracting emotion labels from CSV files.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AnnotationParser:
    """Parses multimodal annotation data from CSV files."""
    
    def __init__(self, annotation_path: Path, expression_table_path: Optional[Path] = None):
        self.annotation_path = annotation_path
        self.expression_table_path = expression_table_path
        
        # Load annotation data
        self.annotations_df = self.load_annotations()
        
        # Load expression table if available
        if expression_table_path and expression_table_path.exists():
            self.expression_df = self.load_expression_table()
        else:
            self.expression_df = None
            
        # Define emotion mappings
        self.emotion_columns = self._get_emotion_columns()
        self.attention_columns = self._get_attention_columns()
        
    def load_annotations(self) -> pd.DataFrame:
        """Load and clean the main annotations CSV."""
        # Read CSV, skipping metadata rows
        df = pd.read_csv(self.annotation_path, skiprows=10)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Filter for valid participants with video data
        video_participants = ["AM 1355", "AW 8961", "BU 6095", "CP 0636", "JM 9684"]
        df = df[df['Respondent Name'].isin(video_participants)]
        
        print(f"Loaded annotations for {len(df)} records from {len(df['Respondent Name'].unique())} participants")
        return df
        
    def load_expression_table(self) -> pd.DataFrame:
        """Load the expression table for temporal data."""
        df = pd.read_csv(self.expression_table_path, skiprows=6)
        df.columns = df.columns.str.strip()
        return df
        
    def _get_emotion_columns(self) -> Dict[str, str]:
        """Map emotion names to column names in the CSV."""
        emotion_mapping = {}
        
        # Core emotions from FEA threshold metrics
        emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
        
        for emotion in emotions:
            time_col = f'FEA_ThresholdedMetrics_{emotion}TimePercentage'
            if time_col in self.annotations_df.columns:
                emotion_mapping[emotion] = time_col
                
        return emotion_mapping
        
    def _get_attention_columns(self) -> Dict[str, str]:
        """Map attention/engagement metrics to column names."""
        attention_mapping = {}
        
        attention_metrics = [
            'Attention', 'Engagement', 'PositiveValence', 
            'NegativeValence', 'NeutralValence', 'Confusion'
        ]
        
        for metric in attention_metrics:
            time_col = f'FEA_ThresholdedMetrics_{metric}TimePercentage'
            if time_col in self.annotations_df.columns:
                attention_mapping[metric] = time_col
                
        return attention_mapping
        
    def get_participant_annotations(self, participant_id: str) -> Dict[str, float]:
        """Get emotion annotations for a specific participant."""
        participant_data = self.annotations_df[
            self.annotations_df['Respondent Name'] == participant_id
        ]
        
        if participant_data.empty:
            print(f"No annotations found for participant: {participant_id}")
            return {}
            
        # Get the first (should be only) record for this participant
        record = participant_data.iloc[0]
        
        annotations = {}
        
        # Extract emotion percentages
        for emotion, column in self.emotion_columns.items():
            if column in record and pd.notna(record[column]):
                annotations[emotion] = float(record[column])
            else:
                annotations[emotion] = 0.0
                
        # Extract attention/engagement metrics
        for metric, column in self.attention_columns.items():
            if column in record and pd.notna(record[column]):
                annotations[metric] = float(record[column])
            else:
                annotations[metric] = 0.0
                
        # Add derived metrics
        annotations['TotalEmotionScore'] = sum([
            annotations.get(emotion, 0) for emotion in self.emotion_columns.keys()
        ])
        
        # Binary attention label (>50% attention)
        annotations['HighAttention'] = 1.0 if annotations.get('Attention', 0) > 50.0 else 0.0
        
        # Emotional intensity (non-neutral emotions)
        neutral_score = annotations.get('NeutralValence', 0)
        annotations['EmotionalIntensity'] = max(0, 100 - neutral_score)
        
        return annotations
        
    def get_all_participant_annotations(self) -> Dict[str, Dict[str, float]]:
        """Get annotations for all participants."""
        all_annotations = {}
        
        for participant in self.annotations_df['Respondent Name'].unique():
            all_annotations[participant] = self.get_participant_annotations(participant)
            
        return all_annotations
        
    def get_temporal_annotations(self, participant_id: str) -> Optional[pd.DataFrame]:
        """Get temporal expression data if available."""
        if self.expression_df is None:
            return None
            
        participant_expressions = self.expression_df[
            self.expression_df['Respondent Name'] == participant_id
        ]
        
        return participant_expressions
        
    def create_frame_level_labels(self, participant_id: str, timestamps: List[float], 
                                fps: int = 30) -> np.ndarray:
        """Create frame-level labels by mapping timestamps to annotations."""
        # Get participant annotations
        annotations = self.get_participant_annotations(participant_id)
        
        if not annotations:
            return np.zeros((len(timestamps), len(self.emotion_columns)))
            
        # For now, use static labels for all frames (could be enhanced with temporal data)
        emotion_labels = []
        for emotion in self.emotion_columns.keys():
            emotion_labels.append(annotations.get(emotion, 0.0) / 100.0)  # Normalize to [0,1]
            
        # Repeat for all timestamps
        frame_labels = np.tile(emotion_labels, (len(timestamps), 1))
        
        return frame_labels
        
    def get_binary_attention_labels(self, participant_id: str, timestamps: List[float]) -> np.ndarray:
        """Get binary attention labels for frames."""
        annotations = self.get_participant_annotations(participant_id)
        attention_score = annotations.get('Attention', 0.0)
        
        # Binary classification: high attention (>50%) vs low attention
        binary_label = 1.0 if attention_score > 50.0 else 0.0
        
        return np.full(len(timestamps), binary_label)
        
    def get_regression_labels(self, participant_id: str, timestamps: List[float], 
                            target: str = 'Attention') -> np.ndarray:
        """Get regression labels for continuous prediction."""
        annotations = self.get_participant_annotations(participant_id)
        target_score = annotations.get(target, 0.0) / 100.0  # Normalize to [0,1]
        
        return np.full(len(timestamps), target_score)
        
    def print_annotation_summary(self):
        """Print summary of available annotations."""
        print("=== Annotation Summary ===")
        print(f"Total participants: {len(self.annotations_df['Respondent Name'].unique())}")
        print(f"Participants: {list(self.annotations_df['Respondent Name'].unique())}")
        
        print("\nEmotion Columns:")
        for emotion, column in self.emotion_columns.items():
            print(f"  {emotion}: {column}")
            
        print("\nAttention/Engagement Columns:")
        for metric, column in self.attention_columns.items():
            print(f"  {metric}: {column}")
            
        # Show sample statistics
        print("\nSample Statistics:")
        for emotion in self.emotion_columns.keys():
            column = self.emotion_columns[emotion]
            if column in self.annotations_df.columns:
                values = self.annotations_df[column].dropna()
                if len(values) > 0:
                    print(f"  {emotion}: Mean={values.mean():.2f}, Std={values.std():.2f}, "
                          f"Min={values.min():.2f}, Max={values.max():.2f}")
                          
    def get_label_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive label statistics."""
        stats = {}
        
        # Emotion statistics
        for emotion, column in self.emotion_columns.items():
            if column in self.annotations_df.columns:
                values = self.annotations_df[column].dropna()
                if len(values) > 0:
                    stats[emotion] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': len(values)
                    }
                    
        # Attention statistics
        for metric, column in self.attention_columns.items():
            if column in self.annotations_df.columns:
                values = self.annotations_df[column].dropna()
                if len(values) > 0:
                    stats[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': len(values)
                    }
                    
        return stats