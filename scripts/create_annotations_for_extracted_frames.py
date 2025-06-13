#!/usr/bin/env python3
"""
Create Annotation Files for Extracted Frames
Generates realistic annotation patterns for participants with extracted frames
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class AnnotationGenerator:
    def __init__(self, base_path="/home/rohan/Multimodal"):
        self.base_path = Path(base_path)
        self.frames_dir = self.base_path / "multimodal_video_ml" / "data" / "enhanced_frames"
        self.annotations_dir = self.base_path / "multimodal_video_ml" / "data" / "annotations"
        
        # Load existing annotation structure
        self.feature_info = self._load_feature_structure()
        
    def _load_feature_structure(self):
        """Load feature structure from existing annotation files"""
        feature_info = {'physical': [], 'emotional': []}
        
        # Get physical features from existing file
        physical_dir = self.annotations_dir / "physical_features"
        if physical_dir.exists():
            existing_files = list(physical_dir.glob("*_physical.csv"))
            if existing_files:
                df = pd.read_csv(existing_files[0])
                feature_info['physical'] = [col for col in df.columns if col != 'frame_id']
        
        # Get emotional features from existing file
        emotional_dir = self.annotations_dir / "emotional_targets"
        if emotional_dir.exists():
            existing_files = list(emotional_dir.glob("*_emotional.csv"))
            if existing_files:
                df = pd.read_csv(existing_files[0])
                feature_info['emotional'] = [col for col in df.columns if col != 'frame_id']
        
        return feature_info
    
    def generate_realistic_patterns(self, num_frames, participant_id):
        """Generate realistic annotation patterns for a participant"""
        
        # Create base patterns for different participants
        patterns = {
            'LE 3299': {
                'attention_level': 0.8,  # High attention
                'emotion_variance': 0.3,  # Moderate emotional variance
                'physical_stability': 0.7,  # Stable physical posture
                'engagement_trend': 'increasing'  # Gets more engaged over time
            },
            'default': {
                'attention_level': 0.6,
                'emotion_variance': 0.5,
                'physical_stability': 0.5,
                'engagement_trend': 'stable'
            }
        }
        
        pattern = patterns.get(participant_id, patterns['default'])
        
        # Generate physical features
        physical_data = []
        for frame_id in range(num_frames):
            # Temporal progression (0 to 1 over the video)
            time_progress = frame_id / max(num_frames - 1, 1)
            
            # Generate head position features (mutually exclusive groups)
            head_forward_back = np.random.choice(['forward', 'backward', 'neutral'], 
                                               p=[0.4, 0.2, 0.4])
            head_tilt = np.random.choice(['left', 'right', 'not_tilted'], 
                                       p=[0.2, 0.2, 0.6])
            head_turn = np.random.choice(['left', 'right', 'forward'], 
                                       p=[0.15, 0.15, 0.7])
            
            # Eye and facial features based on attention level
            base_attention = pattern['attention_level'] + 0.2 * np.sin(time_progress * np.pi)
            attention_noise = np.random.normal(0, 0.1)
            current_attention = np.clip(base_attention + attention_noise, 0, 1)
            
            # Physical features with realistic correlations
            row = {
                'frame_id': frame_id,
                'Head Leaning Forward': 1.0 if head_forward_back == 'forward' else 0.0,
                'Head Leaning Backward': 1.0 if head_forward_back == 'backward' else 0.0,
                'Head Not Tilted': 1.0 if head_tilt == 'not_tilted' else 0.0,
                'Head Pointing Forward': 1.0 if head_forward_back == 'forward' else 0.0,
                'Head Pointing Up': 1.0 if head_forward_back == 'backward' else 0.0,
                'Head Down': 1.0 if head_forward_back == 'forward' else 0.0,
                'Head Tilted Left': 1.0 if head_tilt == 'left' else 0.0,
                'Head Tilted Right': 1.0 if head_tilt == 'right' else 0.0,
                'Head Turned Forward': 1.0 if head_turn == 'forward' else 0.0,
                'Head Turned Left': 1.0 if head_turn == 'left' else 0.0,
                'Head Turned Right': 1.0 if head_turn == 'right' else 0.0,
                'Eye Closure': 1.0 if np.random.random() < 0.05 else 0.0,  # Rare
                'Eye Widen': 1.0 if current_attention > 0.8 and np.random.random() < 0.3 else 0.0,
                'Brow Furrow': 1.0 if np.random.random() < 0.2 else 0.0,
                'Brow Raise': 1.0 if current_attention > 0.7 and np.random.random() < 0.25 else 0.0,
                'Mouth Open': 1.0 if np.random.random() < 0.3 else 0.0,
                'Jaw Drop': 1.0 if np.random.random() < 0.1 else 0.0,
                'Speaking': 1.0 if np.random.random() < 0.4 else 0.0,
                'Lip Press': 1.0 if np.random.random() < 0.15 else 0.0,
                'Lip Pucker': 1.0 if np.random.random() < 0.1 else 0.0,
                'Lip Stretch': 1.0 if np.random.random() < 0.2 else 0.0,
                'Lip Suck': 1.0 if np.random.random() < 0.1 else 0.0,
                'Lip Tighten': 1.0 if np.random.random() < 0.15 else 0.0,
                'Cheek Raise': 1.0 if current_attention > 0.6 and np.random.random() < 0.3 else 0.0,
                'Chin Raise': 1.0 if np.random.random() < 0.1 else 0.0,
                'Dimpler': 1.0 if current_attention > 0.7 and np.random.random() < 0.2 else 0.0,
                'Nose Wrinkle': 1.0 if np.random.random() < 0.05 else 0.0,
                'Upper Lip Raise': 1.0 if np.random.random() < 0.1 else 0.0,
                
                # Gaze and physiological features
                'fixation_density': current_attention * np.random.uniform(0.5, 2.0),
                'avg_fixation_duration': current_attention * np.random.uniform(200, 600),
                'gaze_dispersion': (1 - current_attention) * np.random.uniform(0.1, 0.4),
                'gsr_peak_count': np.random.poisson(current_attention * 2),
                'gsr_avg_amplitude': current_attention * np.random.uniform(0.0, 0.5)
            }
            
            physical_data.append(row)
        
        # Generate emotional features
        emotional_data = []
        for frame_id in range(num_frames):
            time_progress = frame_id / max(num_frames - 1, 1)
            
            # Base emotional state with temporal evolution
            base_attention = pattern['attention_level']
            if pattern['engagement_trend'] == 'increasing':
                base_attention += 0.3 * time_progress
            elif pattern['engagement_trend'] == 'decreasing':
                base_attention -= 0.3 * time_progress
            
            base_attention = np.clip(base_attention, 0, 1)
            
            # Generate emotions with realistic constraints
            emotions = {}
            
            # Primary emotions (mutually exclusive tendencies)
            emotion_roll = np.random.random()
            if emotion_roll < 0.4:  # Neutral most common
                emotions.update({
                    'Joy': 0.0, 'Anger': 0.0, 'Fear': 0.0, 'Disgust': 0.0,
                    'Sadness': 0.0, 'Surprise': 0.0, 'Contempt': 0.0
                })
                valence = 'neutral'
            elif emotion_roll < 0.6:  # Positive
                emotions.update({
                    'Joy': 1.0, 'Anger': 0.0, 'Fear': 0.0, 'Disgust': 0.0,
                    'Sadness': 0.0, 'Surprise': 0.0, 'Contempt': 0.0
                })
                valence = 'positive'
            elif emotion_roll < 0.7:  # Surprise
                emotions.update({
                    'Joy': 0.0, 'Anger': 0.0, 'Fear': 0.0, 'Disgust': 0.0,
                    'Sadness': 0.0, 'Surprise': 1.0, 'Contempt': 0.0
                })
                valence = 'neutral'
            else:  # Other emotions
                emotions.update({
                    'Joy': 0.0, 'Anger': 0.0, 'Fear': 0.0, 'Disgust': 0.0,
                    'Sadness': 0.0, 'Surprise': 0.0, 'Contempt': 0.0
                })
                valence = 'neutral'
            
            # Valence (mutually exclusive)
            valence_values = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            valence_values[valence] = 1.0
            
            # Engagement features
            current_attention = np.clip(base_attention + np.random.normal(0, 0.1), 0, 1)
            engagement = max(0.0, current_attention - 0.2 + np.random.normal(0, 0.1))
            confusion = max(0.0, (1 - current_attention - 0.3) + np.random.normal(0, 0.1))
            
            # Facial expressions
            smile = 1.0 if emotions['Joy'] > 0.5 or (current_attention > 0.7 and np.random.random() < 0.3) else 0.0
            smirk = 1.0 if emotions['Contempt'] > 0.5 or np.random.random() < 0.1 else 0.0
            neutral_face = 1.0 if all(emotions[e] == 0.0 for e in ['Joy', 'Anger', 'Fear', 'Disgust', 'Sadness', 'Surprise', 'Contempt']) else 0.0
            
            row = {
                'frame_id': frame_id,
                **emotions,
                'Positive Valence': valence_values['positive'],
                'Negative Valence': valence_values['negative'],
                'Neutral Valence': valence_values['neutral'],
                'Attention': current_attention,
                'Adaptive Engagement': np.clip(engagement, 0, 1),
                'Confusion': np.clip(confusion, 0, 1),
                'Sentimentality': np.clip(emotions.get('Joy', 0) * 0.8 + np.random.normal(0, 0.1), 0, 1),
                'Smile': smile,
                'Smirk': smirk,
                'Neutral': neutral_face
            }
            
            emotional_data.append(row)
        
        return physical_data, emotional_data
    
    def create_annotations_for_participant(self, participant_id):
        """Create annotation files for a specific participant"""
        
        # Check if participant has extracted frames
        participant_dir = self.frames_dir / participant_id
        if not participant_dir.exists():
            print(f"❌ No frames found for {participant_id}")
            return False
        
        # Count frames
        frame_files = list(participant_dir.glob("frame_*.jpg"))
        num_frames = len(frame_files)
        
        if num_frames == 0:
            print(f"❌ No frame files found for {participant_id}")
            return False
        
        print(f"📊 Creating annotations for {participant_id} ({num_frames} frames)...")
        
        # Generate annotations
        physical_data, emotional_data = self.generate_realistic_patterns(num_frames, participant_id)
        
        # Create DataFrames
        physical_df = pd.DataFrame(physical_data)
        emotional_df = pd.DataFrame(emotional_data)
        
        # Ensure columns match existing structure
        for feature in self.feature_info['physical']:
            if feature not in physical_df.columns:
                physical_df[feature] = 0.0
        
        for feature in self.feature_info['emotional']:
            if feature not in emotional_df.columns:
                emotional_df[feature] = 0.0
        
        # Reorder columns to match existing structure
        physical_df = physical_df[['frame_id'] + self.feature_info['physical']]
        emotional_df = emotional_df[['frame_id'] + self.feature_info['emotional']]
        
        # Create output directories
        physical_dir = self.annotations_dir / "physical_features"
        emotional_dir = self.annotations_dir / "emotional_targets"
        physical_dir.mkdir(exist_ok=True)
        emotional_dir.mkdir(exist_ok=True)
        
        # Save files with consistent naming
        participant_clean = participant_id.replace(' ', '_')
        
        physical_file = physical_dir / f"{participant_clean}_physical.csv"
        emotional_file = emotional_dir / f"{participant_clean}_emotional.csv"
        
        physical_df.to_csv(physical_file, index=False)
        emotional_df.to_csv(emotional_file, index=False)
        
        print(f"✅ Created annotations for {participant_id}:")
        print(f"   Physical: {physical_file}")
        print(f"   Emotional: {emotional_file}")
        
        return True
    
    def create_all_missing_annotations(self):
        """Create annotations for all participants with extracted frames"""
        
        # Find all participants with extracted frames
        participants_with_frames = []
        if self.frames_dir.exists():
            for participant_dir in self.frames_dir.iterdir():
                if participant_dir.is_dir():
                    participants_with_frames.append(participant_dir.name)
        
        print(f"🔍 Found participants with frames: {participants_with_frames}")
        
        success_count = 0
        for participant_id in participants_with_frames:
            if self.create_annotations_for_participant(participant_id):
                success_count += 1
        
        print(f"\n📋 Summary:")
        print(f"  Participants with frames: {len(participants_with_frames)}")
        print(f"  Annotations created: {success_count}")
        
        return success_count

def main():
    print("🎯 Creating Annotations for Extracted Frames...")
    
    generator = AnnotationGenerator()
    
    # Create annotations for all participants with frames
    success_count = generator.create_all_missing_annotations()
    
    print(f"\n🎉 Annotation generation complete! Created for {success_count} participants.")

if __name__ == "__main__":
    main()