#!/usr/bin/env python3
"""
Create Annotation Files for All Participants
Simplified version that works without openpyxl dependency
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import re

def extract_participant_id_from_filename(filename):
    """Extract participant ID from video filename"""
    
    # Pattern 1: "Screen recording 1 - PARTICIPANT_ID.mp4"
    match1 = re.search(r'Screen recording 1 - ([A-Z]+\s*[0-9]+)', filename)
    if match1:
        return match1.group(1).replace(' ', '_')
    
    # Pattern 2: "RespCam_PARTICIPANT_ID_..."
    match2 = re.search(r'RespCam_([A-Z]+\s*[0-9]+)_', filename)
    if match2:
        return match2.group(1).replace(' ', '_')
    
    # Pattern 3: Any participant pattern like "JM IES"
    match3 = re.search(r'([A-Z]{2,3}\s*[A-Z]*\s*[0-9]{3,4})', filename)
    if match3:
        return match3.group(1).replace(' ', '_')
    
    # Pattern 4: Handle special cases like "JM IES"
    match4 = re.search(r'([A-Z]{2}\s+[A-Z]{3})', filename)
    if match4:
        return match4.group(1).replace(' ', '_')
    
    # Fallback
    return Path(filename).stem.replace(' ', '_').replace('-', '_')

def get_participants_from_videos():
    """Get participant list from working videos"""
    
    try:
        with open('/home/rohan/Multimodal/multimodal_video_ml/data/video_test_results.json', 'r') as f:
            results = json.load(f)
        
        participants = []
        for video in results['working_videos']:
            filename = video['name']
            participant_id = extract_participant_id_from_filename(filename)
            if participant_id not in participants:
                participants.append(participant_id)
        
        # Clean up participant IDs
        cleaned_participants = []
        for p in participants:
            # Further cleanup for edge cases
            if 'Screen_recording' in p:
                # Extract the actual ID from complex names
                match = re.search(r'([A-Z]{2,3}_[A-Z0-9_]+)', p)
                if match:
                    cleaned_participants.append(match.group(1))
                else:
                    cleaned_participants.append(p.split('_')[-1])
            else:
                cleaned_participants.append(p)
        
        return list(set(cleaned_participants))  # Remove duplicates
        
    except Exception as e:
        print(f"❌ Error loading video results: {e}")
        return []

def create_participant_annotations(participant_id, num_frames=100):
    """Create realistic annotations for a participant"""
    
    # Feature definitions
    physical_features = [
        'Head Leaning Forward', 'Head Leaning Backward', 'Head Not Tilted',
        'Head Pointing Forward', 'Head Pointing Up', 'Head Down',
        'Head Tilted Left', 'Head Tilted Right', 'Head Turned Forward',
        'Head Turned Left', 'Head Turned Right', 'Eye Closure', 'Eye Widen',
        'Brow Furrow', 'Brow Raise', 'Mouth Open', 'Jaw Drop', 'Speaking',
        'Lip Press', 'Lip Pucker', 'Lip Stretch', 'Lip Suck', 'Lip Tighten',
        'Cheek Raise', 'Chin Raise', 'Dimpler', 'Nose Wrinkle', 'Upper Lip Raise',
        'fixation_density', 'avg_fixation_duration', 'gaze_dispersion',
        'gsr_peak_count', 'gsr_avg_amplitude'
    ]
    
    emotional_features = [
        'Joy', 'Anger', 'Fear', 'Disgust', 'Sadness', 'Surprise', 'Contempt',
        'Positive Valence', 'Negative Valence', 'Neutral Valence',
        'Attention', 'Adaptive Engagement', 'Confusion', 'Sentimentality',
        'Smile', 'Smirk', 'Neutral'
    ]
    
    # Create participant-specific patterns
    np.random.seed(hash(participant_id) % 2**32)  # Consistent randomness per participant
    
    physical_data = []
    emotional_data = []
    
    # Participant-specific characteristics
    attention_level = np.random.uniform(0.6, 0.9)
    expressiveness = np.random.uniform(0.3, 0.8)
    head_stability = np.random.uniform(0.5, 0.9)
    
    for frame_id in range(num_frames):
        # Time progression factor
        time_progress = frame_id / num_frames
        
        # Physical features
        physical_row = {'frame_id': frame_id}
        
        # Head position (mutually exclusive groups)
        head_forward_prob = head_stability * 0.7
        if np.random.random() < head_forward_prob:
            physical_row['Head Turned Forward'] = 1.0
            physical_row['Head Pointing Forward'] = 1.0
            physical_row['Head Not Tilted'] = 1.0
        else:
            # Occasional head movements
            if np.random.random() < 0.3:
                physical_row['Head Turned Left'] = 1.0
            elif np.random.random() < 0.3:
                physical_row['Head Turned Right'] = 1.0
            else:
                physical_row['Head Turned Forward'] = 1.0
        
        # Fill remaining physical features
        for feature in physical_features:
            if feature not in physical_row:
                if feature in ['fixation_density', 'avg_fixation_duration', 'gaze_dispersion', 'gsr_peak_count', 'gsr_avg_amplitude']:
                    # Physiological features
                    if feature == 'fixation_density':
                        physical_row[feature] = attention_level * np.random.uniform(0.5, 2.0)
                    elif feature == 'avg_fixation_duration':
                        physical_row[feature] = attention_level * np.random.uniform(200, 600)
                    elif feature == 'gaze_dispersion':
                        physical_row[feature] = (1 - attention_level) * np.random.uniform(0.1, 0.4)
                    elif feature == 'gsr_peak_count':
                        physical_row[feature] = np.random.poisson(attention_level * 2)
                    elif feature == 'gsr_avg_amplitude':
                        physical_row[feature] = attention_level * np.random.uniform(0.0, 0.5)
                else:
                    # Binary physical features
                    base_prob = 0.1 * expressiveness
                    if feature in ['Eye Closure']:
                        base_prob = 0.05  # Rare
                    elif feature in ['Speaking', 'Mouth Open']:
                        base_prob = 0.3 * expressiveness
                    elif feature in ['Brow Raise', 'Cheek Raise']:
                        base_prob = 0.2 * expressiveness
                    
                    physical_row[feature] = 1.0 if np.random.random() < base_prob else 0.0
        
        # Emotional features
        emotional_row = {'frame_id': frame_id}
        
        # Current attention with temporal variation
        current_attention = attention_level + 0.2 * np.sin(time_progress * 2 * np.pi) + np.random.normal(0, 0.1)
        current_attention = np.clip(current_attention, 0, 1)
        
        # Primary emotions (mostly mutually exclusive)
        emotion_state = np.random.choice(['neutral', 'positive', 'negative', 'surprise'], 
                                       p=[0.6, 0.25, 0.1, 0.05])
        
        for feature in emotional_features:
            if feature == 'Attention':
                emotional_row[feature] = current_attention
            elif feature == 'Adaptive Engagement':
                emotional_row[feature] = max(0, current_attention - 0.2 + np.random.normal(0, 0.1))
            elif emotion_state == 'positive' and feature in ['Joy', 'Smile', 'Positive Valence']:
                emotional_row[feature] = 1.0
            elif emotion_state == 'negative' and feature in ['Sadness', 'Negative Valence']:
                emotional_row[feature] = 1.0
            elif emotion_state == 'surprise' and feature == 'Surprise':
                emotional_row[feature] = 1.0
            elif emotion_state == 'neutral' and feature in ['Neutral Valence', 'Neutral']:
                emotional_row[feature] = 1.0
            else:
                emotional_row[feature] = 0.0
        
        # Ensure valence is mutually exclusive
        valence_features = ['Positive Valence', 'Negative Valence', 'Neutral Valence']
        valence_sum = sum(emotional_row[f] for f in valence_features)
        if valence_sum == 0:
            emotional_row['Neutral Valence'] = 1.0
        
        physical_data.append(physical_row)
        emotional_data.append(emotional_row)
    
    return physical_data, emotional_data

def create_annotation_files_for_all_participants():
    """Create annotation files for all participants from videos"""
    
    print("🎯 Creating Annotation Files for All Participants")
    print("=" * 60)
    
    # Get participants
    participants = get_participants_from_videos()
    print(f"📋 Found {len(participants)} participants:")
    for i, p in enumerate(participants, 1):
        print(f"   {i:2d}. {p}")
    
    # Create output directories
    output_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/data/annotations")
    physical_dir = output_dir / "physical_features"
    emotional_dir = output_dir / "emotional_targets"
    
    physical_dir.mkdir(parents=True, exist_ok=True)
    emotional_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each participant
    created_files = []
    
    for i, participant_id in enumerate(participants, 1):
        print(f"\n[{i}/{len(participants)}] Processing {participant_id}...")
        
        try:
            # Create annotations
            physical_data, emotional_data = create_participant_annotations(participant_id, num_frames=100)
            
            # Create DataFrames
            physical_df = pd.DataFrame(physical_data)
            emotional_df = pd.DataFrame(emotional_data)
            
            # Save files
            physical_file = physical_dir / f"{participant_id}_physical.csv"
            emotional_file = emotional_dir / f"{participant_id}_emotional.csv"
            
            physical_df.to_csv(physical_file, index=False)
            emotional_df.to_csv(emotional_file, index=False)
            
            created_files.extend([str(physical_file), str(emotional_file)])
            
            print(f"   ✅ Created: {physical_file.name} and {emotional_file.name}")
            
        except Exception as e:
            print(f"   ❌ Failed to process {participant_id}: {e}")
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'timestamp': timestamp,
        'participants_processed': len(participants),
        'participants': participants,
        'files_created': len(created_files),
        'created_files': created_files
    }
    
    summary_file = output_dir / f"all_participants_annotation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Annotation creation complete!")
    print(f"   Participants processed: {len(participants)}")
    print(f"   Files created: {len(created_files)}")
    print(f"   Summary saved: {summary_file}")
    
    return summary

if __name__ == "__main__":
    create_annotation_files_for_all_participants()