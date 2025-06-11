#!/usr/bin/env python3
"""
GENEX Annotation Data Processor
Processes real GENEX interview annotation data for multimodal learning.
Extracts physical features and emotional labels from CSV files.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class GENEXAnnotationProcessor:
    """Process GENEX interview annotation data for multimodal learning."""
    
    def __init__(self, base_path: str = "/home/rohan/Multimodal/GENEX Intreview/Analysis"):
        self.base_path = Path(base_path)
        self.output_path = Path("/home/rohan/Multimodal/multimodal_video_ml/data/annotations")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define our 5 participants from video analysis
        self.participants = ["LE 3299", "CP 0636", "NS 4013", "MP 5114", "JM 9684"]
        
        # Physical features (input) - behavioral and physiological data
        self.physical_features = [
            # Head pose and movement
            "Head Leaning Forward", "Head Leaning Backward", "Head Not Tilted", 
            "Head Pointing Forward", "Head Pointing Up", "Head Down",
            "Head Tilted Left", "Head Tilted Right", 
            "Head Turned Forward", "Head Turned Left", "Head Turned Right",
            
            # Eye behavior
            "Eye Closure", "Eye Widen", "Brow Furrow", "Brow Raise",
            
            # Mouth/Speech
            "Mouth Open", "Jaw Drop", "Speaking",
            "Lip Press", "Lip Pucker", "Lip Stretch", "Lip Suck", "Lip Tighten",
            
            # Physical actions
            "Cheek Raise", "Chin Raise", "Dimpler", "Nose Wrinkle", "Upper Lip Raise"
        ]
        
        # Emotional targets (output) - what we want to predict
        self.emotional_targets = [
            # Core emotions
            "Joy", "Anger", "Fear", "Disgust", "Sadness", "Surprise", "Contempt",
            
            # Valence and engagement
            "Positive Valence", "Negative Valence", "Neutral Valence",
            "Attention", "Adaptive Engagement", "Confusion",
            
            # Complex emotional states
            "Sentimentality", "Smile", "Smirk", "Neutral"
        ]
        
    def load_facial_expression_data(self) -> pd.DataFrame:
        """Load and clean facial expression data from FEAExpressionTable.csv"""
        fea_path = self.base_path / "Facial Coding" / "FEAExpressionTable.csv"
        
        print(f"Loading facial expression data from: {fea_path}")
        
        # Read CSV, skipping metadata rows
        df = pd.read_csv(fea_path, skiprows=6)
        
        print(f"Loaded {len(df)} facial expression records")
        print(f"Participants in data: {df['Respondent Name'].unique()}")
        print(f"Expression types: {len(df['Expression Type'].unique())} unique types")
        
        # Filter for our 5 participants
        df_filtered = df[df['Respondent Name'].isin(self.participants)]
        print(f"Filtered to {len(df_filtered)} records for our 5 participants")
        
        return df_filtered
        
    def load_eye_tracking_data(self) -> pd.DataFrame:
        """Load and clean eye tracking data from FixationTable.csv"""
        et_path = self.base_path / "Eye Tracking" / "FixationTable.csv"
        
        print(f"Loading eye tracking data from: {et_path}")
        
        # Read CSV, skipping metadata rows
        df = pd.read_csv(et_path, skiprows=6)
        
        print(f"Loaded {len(df)} eye tracking records")
        
        # Filter for our 5 participants
        df_filtered = df[df['Respondent Name'].isin(self.participants)]
        print(f"Filtered to {len(df_filtered)} eye tracking records for our participants")
        
        return df_filtered
        
    def load_gsr_data(self) -> pd.DataFrame:
        """Load and clean GSR data from GSRPeakTable.csv"""
        gsr_path = self.base_path / "GSR" / "GSRPeakTable.csv"
        
        print(f"Loading GSR data from: {gsr_path}")
        
        # Read CSV, skipping metadata rows
        df = pd.read_csv(gsr_path, skiprows=6)
        
        print(f"Loaded {len(df)} GSR records")
        
        # Filter for our 5 participants
        df_filtered = df[df['Respondent Name'].isin(self.participants)]
        print(f"Filtered to {len(df_filtered)} GSR records for our participants")
        
        return df_filtered
        
    def extract_physical_features(self, fea_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract physical/behavioral features for each participant"""
        physical_data = {}
        
        for participant in self.participants:
            participant_data = fea_df[fea_df['Respondent Name'] == participant].copy()
            
            # Filter for physical features only
            physical_records = participant_data[
                participant_data['Expression Type'].isin(self.physical_features)
            ].copy()
            
            print(f"{participant}: {len(physical_records)} physical feature records")
            
            # Create time-based feature vectors
            if len(physical_records) > 0:
                # Sort by onset time
                physical_records = physical_records.sort_values('Expression Onset')
                
                # Create feature timeline
                features_timeline = self._create_feature_timeline(
                    physical_records, self.physical_features
                )
                
                physical_data[participant] = features_timeline
            else:
                print(f"WARNING: No physical features found for {participant}")
                
        return physical_data
        
    def extract_emotional_targets(self, fea_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract emotional target labels for each participant"""
        emotional_data = {}
        
        for participant in self.participants:
            participant_data = fea_df[fea_df['Respondent Name'] == participant].copy()
            
            # Filter for emotional targets only
            emotional_records = participant_data[
                participant_data['Expression Type'].isin(self.emotional_targets)
            ].copy()
            
            print(f"{participant}: {len(emotional_records)} emotional target records")
            
            # Create time-based target vectors
            if len(emotional_records) > 0:
                # Sort by onset time
                emotional_records = emotional_records.sort_values('Expression Onset')
                
                # Create target timeline
                targets_timeline = self._create_feature_timeline(
                    emotional_records, self.emotional_targets
                )
                
                emotional_data[participant] = targets_timeline
            else:
                print(f"WARNING: No emotional targets found for {participant}")
                
        return emotional_data
        
    def _create_feature_timeline(self, records: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """Create time-based feature vectors from expression records"""
        
        # Create timeline with 1-second intervals (1000ms)
        if len(records) == 0:
            return pd.DataFrame()
            
        start_time = records['Expression Onset'].min()
        end_time = records['Expression Offset'].max()
        
        # Create timeline in 1-second intervals
        timeline = np.arange(start_time, end_time + 1000, 1000)
        
        # Initialize feature matrix
        feature_matrix = np.zeros((len(timeline), len(feature_list)))
        feature_df = pd.DataFrame(
            feature_matrix, 
            columns=feature_list,
            index=timeline
        )
        feature_df.index.name = 'timestamp_ms'
        
        # Fill in features based on expression durations
        for _, record in records.iterrows():
            expression_type = record['Expression Type']
            onset = record['Expression Onset']
            offset = record['Expression Offset']
            
            if expression_type in feature_list:
                # Mark time intervals where this expression is active
                active_times = (timeline >= onset) & (timeline <= offset)
                feature_df.loc[timeline[active_times], expression_type] = 1.0
                
        return feature_df
        
    def enhance_with_eye_tracking(self, physical_data: Dict[str, pd.DataFrame], 
                                  et_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Enhance physical features with eye tracking data"""
        
        for participant in self.participants:
            if participant not in physical_data:
                continue
                
            participant_et = et_df[et_df['Respondent Name'] == participant].copy()
            
            if len(participant_et) == 0:
                print(f"No eye tracking data for {participant}")
                continue
                
            print(f"Adding eye tracking features for {participant}: {len(participant_et)} fixations")
            
            # Get timeline from existing physical data
            timeline = physical_data[participant].index
            
            # Add eye tracking features
            fixation_density = np.zeros(len(timeline))
            avg_fixation_duration = np.zeros(len(timeline))
            gaze_dispersion = np.zeros(len(timeline))
            
            for i, timestamp in enumerate(timeline):
                # Find fixations within 1-second window
                window_fixations = participant_et[
                    (participant_et['Fixation Start'] <= timestamp) & 
                    (participant_et['Fixation End'] >= timestamp - 1000)
                ]
                
                if len(window_fixations) > 0:
                    fixation_density[i] = len(window_fixations)
                    avg_fixation_duration[i] = window_fixations['Fixation Duration'].mean()
                    gaze_dispersion[i] = window_fixations['Fixation Dispersion'].mean()
                    
            # Add to dataframe
            physical_data[participant]['fixation_density'] = fixation_density
            physical_data[participant]['avg_fixation_duration'] = avg_fixation_duration
            physical_data[participant]['gaze_dispersion'] = gaze_dispersion
            
        return physical_data
        
    def enhance_with_gsr(self, physical_data: Dict[str, pd.DataFrame], 
                         gsr_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Enhance physical features with GSR (arousal) data"""
        
        for participant in self.participants:
            if participant not in physical_data:
                continue
                
            participant_gsr = gsr_df[gsr_df['Respondent Name'] == participant].copy()
            
            if len(participant_gsr) == 0:
                print(f"No GSR data for {participant}")
                continue
                
            print(f"Adding GSR features for {participant}: {len(participant_gsr)} peaks")
            
            # Get timeline from existing physical data
            timeline = physical_data[participant].index
            
            # Add GSR features
            gsr_peak_count = np.zeros(len(timeline))
            gsr_avg_amplitude = np.zeros(len(timeline))
            
            for i, timestamp in enumerate(timeline):
                # Find GSR peaks within 1-second window
                window_peaks = participant_gsr[
                    (participant_gsr['Peak Time'] <= timestamp) & 
                    (participant_gsr['Peak Time'] >= timestamp - 1000)
                ]
                
                if len(window_peaks) > 0:
                    gsr_peak_count[i] = len(window_peaks)
                    gsr_avg_amplitude[i] = window_peaks['Peak Amplitude'].mean()
                    
            # Add to dataframe
            physical_data[participant]['gsr_peak_count'] = gsr_peak_count
            physical_data[participant]['gsr_avg_amplitude'] = gsr_avg_amplitude
            
        return physical_data
        
    def align_with_video_frames(self, annotation_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align annotation data with video frame timestamps"""
        
        # Load video frame metadata
        frame_summary_path = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames/processing_summary.json")
        
        if not frame_summary_path.exists():
            print("WARNING: Video frame metadata not found, using synthetic alignment")
            return annotation_data
            
        with open(frame_summary_path, 'r') as f:
            frame_metadata = json.load(f)
            
        aligned_data = {}
        
        for participant in self.participants:
            if participant not in annotation_data:
                continue
                
            if participant not in frame_metadata:
                print(f"No video frames for {participant}, skipping alignment")
                continue
                
            # For each participant, we have 20 frames
            # Create frame-aligned annotation vectors
            num_frames = frame_metadata[participant]['num_frames']
            
            # Sample annotation data to match frame count
            annotation_df = annotation_data[participant]
            
            if len(annotation_df) == 0:
                continue
                
            # Create frame-aligned samples by downsampling annotations
            frame_indices = np.linspace(0, len(annotation_df) - 1, num_frames).astype(int)
            frame_aligned = annotation_df.iloc[frame_indices].copy()
            
            # Reset index to frame numbers
            frame_aligned.index = range(num_frames)
            frame_aligned.index.name = 'frame_id'
            
            aligned_data[participant] = frame_aligned
            
            print(f"{participant}: Aligned {len(annotation_df)} annotations to {num_frames} frames")
            
        return aligned_data
        
    def save_processed_data(self, physical_data: Dict[str, pd.DataFrame], 
                           emotional_data: Dict[str, pd.DataFrame]):
        """Save processed annotation data for training"""
        
        # Save physical features
        physical_dir = self.output_path / "physical_features"
        physical_dir.mkdir(exist_ok=True)
        
        for participant, data in physical_data.items():
            output_file = physical_dir / f"{participant.replace(' ', '_')}_physical.csv"
            data.to_csv(output_file)
            print(f"Saved physical features for {participant}: {output_file}")
            
        # Save emotional targets
        emotional_dir = self.output_path / "emotional_targets"
        emotional_dir.mkdir(exist_ok=True)
        
        for participant, data in emotional_data.items():
            output_file = emotional_dir / f"{participant.replace(' ', '_')}_emotional.csv"
            data.to_csv(output_file)
            print(f"Saved emotional targets for {participant}: {output_file}")
            
        # Save processing summary
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "participants": self.participants,
            "physical_features": self.physical_features,
            "emotional_targets": self.emotional_targets,
            "physical_feature_count": len(self.physical_features),
            "emotional_target_count": len(self.emotional_targets),
            "participants_processed": {
                "physical": list(physical_data.keys()),
                "emotional": list(emotional_data.keys())
            }
        }
        
        summary_file = self.output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Saved processing summary: {summary_file}")

def main():
    """Main processing pipeline"""
    print("🎯 GENEX Annotation Processing Pipeline")
    print("=" * 50)
    
    processor = GENEXAnnotationProcessor()
    
    # Load raw annotation data
    print("\n📊 Loading GENEX annotation datasets...")
    fea_df = processor.load_facial_expression_data()
    et_df = processor.load_eye_tracking_data()
    gsr_df = processor.load_gsr_data()
    
    # Extract features and targets
    print("\n🔧 Extracting physical features and emotional targets...")
    physical_data = processor.extract_physical_features(fea_df)
    emotional_data = processor.extract_emotional_targets(fea_df)
    
    # Enhance with multi-modal data
    print("\n👁️ Enhancing with eye tracking data...")
    physical_data = processor.enhance_with_eye_tracking(physical_data, et_df)
    
    print("\n📈 Enhancing with GSR data...")
    physical_data = processor.enhance_with_gsr(physical_data, gsr_df)
    
    # Align with video frames
    print("\n🎬 Aligning with video frame timestamps...")
    physical_data = processor.align_with_video_frames(physical_data)
    emotional_data = processor.align_with_video_frames(emotional_data)
    
    # Save processed data
    print("\n💾 Saving processed annotation data...")
    processor.save_processed_data(physical_data, emotional_data)
    
    # Summary
    print("\n✅ GENEX Annotation Processing Complete!")
    print(f"Participants processed: {len(physical_data)}")
    print(f"Physical features: {len(processor.physical_features)} + eye tracking + GSR")
    print(f"Emotional targets: {len(processor.emotional_targets)}")
    print("Ready for multimodal training!")

if __name__ == "__main__":
    main()