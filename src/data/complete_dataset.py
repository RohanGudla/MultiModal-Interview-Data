#!/usr/bin/env python3
"""
Complete Multi-Label Dataset for ALL 17 Participants
Handles comprehensive dataset with proper temporal modeling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from collections import defaultdict

class CompleteMultiLabelDataset(Dataset):
    """Dataset for all 17 participants with temporal modeling"""
    
    def __init__(self, 
                 frames_dir: str,
                 annotations_dir: str,
                 sequence_length: int = 1,
                 transform=None,
                 return_temporal_info: bool = True):
        
        self.frames_dir = Path(frames_dir)
        self.annotations_dir = Path(annotations_dir)
        self.sequence_length = sequence_length
        self.return_temporal_info = return_temporal_info
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load feature information
        self.feature_info = {
            'physical': [
                'Head Turned Forward', 'Head Pointing Forward', 'Head Not Tilted', 
                'Head Leaning Forward', 'Head Leaning Backward', 'Head Pointing Up', 
                'Head Down', 'Head Tilted Left', 'Head Tilted Right', 'Head Turned Left', 
                'Head Turned Right', 'Eye Closure', 'Eye Widen', 'Brow Furrow', 
                'Brow Raise', 'Mouth Open', 'Jaw Drop', 'Speaking', 'Lip Press', 
                'Lip Pucker', 'Lip Stretch', 'Lip Suck', 'Lip Tighten', 'Cheek Raise', 
                'Chin Raise', 'Dimpler', 'Nose Wrinkle', 'Upper Lip Raise',
                'fixation_density', 'avg_fixation_duration', 'gaze_dispersion',
                'gsr_peak_count', 'gsr_avg_amplitude'
            ],
            'emotional': [
                'Joy', 'Anger', 'Fear', 'Disgust', 'Sadness', 'Surprise', 'Contempt',
                'Positive Valence', 'Negative Valence', 'Neutral Valence', 'Attention',
                'Adaptive Engagement', 'Confusion', 'Sentimentality', 'Smile', 'Smirk', 'Neutral'
            ]
        }
        self.feature_info['all'] = self.feature_info['physical'] + self.feature_info['emotional']
        
        # Load data
        self.data_samples = self._load_all_data()
        
        print(f"📊 Complete Dataset Initialized:")
        print(f"  Total samples: {len(self.data_samples)}")
        print(f"  Physical features: {len(self.feature_info['physical'])}")
        print(f"  Emotional features: {len(self.feature_info['emotional'])}")
        print(f"  Sequence length: {sequence_length}")
    
    def _load_participant_data(self, participant_id: str) -> List[Dict]:
        """Load data for a single participant"""
        
        print(f"📝 Loading data for {participant_id}...")
        
        # Load annotation files
        physical_file = self.annotations_dir / "physical_features" / f"{participant_id}_physical.csv"
        emotional_file = self.annotations_dir / "emotional_targets" / f"{participant_id}_emotional.csv"
        
        physical_df = None
        emotional_df = None
        
        if physical_file.exists():
            physical_df = pd.read_csv(physical_file)
        if emotional_file.exists():
            emotional_df = pd.read_csv(emotional_file)
        
        if physical_df is None and emotional_df is None:
            print(f"⚠️ No annotations found for {participant_id}")
            return []
        
        # Check if participant has extracted frames
        participant_frames_dir = self.frames_dir / participant_id
        if not participant_frames_dir.exists():
            print(f"⚠️ No frames found for {participant_id}")
            return []
        
        # Load frame metadata if available
        metadata_file = participant_frames_dir / "extraction_metadata.json"
        frame_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                frame_metadata = {frame['frame_id']: frame for frame in metadata.get('extracted_frames', [])}
        
        # Create combined samples
        samples = []
        
        # Find available frames
        frame_files = list(participant_frames_dir.glob("frame_*.jpg"))
        available_frames = sorted([
            int(f.stem.split('_')[1]) for f in frame_files
        ])
        
        for frame_id in available_frames:
            frame_path = participant_frames_dir / f"frame_{frame_id:04d}.jpg"
            
            if not frame_path.exists():
                continue
            
            # Get annotations for this frame
            physical_labels = np.zeros(len(self.feature_info['physical']), dtype=np.float32)
            emotional_labels = np.zeros(len(self.feature_info['emotional']), dtype=np.float32)
            
            # Fill physical labels
            if physical_df is not None and frame_id < len(physical_df):
                row = physical_df.iloc[frame_id]
                for i, feature in enumerate(self.feature_info['physical']):
                    if feature in row:
                        physical_labels[i] = float(row[feature])
            
            # Fill emotional labels
            if emotional_df is not None and frame_id < len(emotional_df):
                row = emotional_df.iloc[frame_id]
                for i, feature in enumerate(self.feature_info['emotional']):
                    if feature in row:
                        emotional_labels[i] = float(row[feature])
            
            # Combine all labels
            all_labels = np.concatenate([physical_labels, emotional_labels])
            
            # Create temporal info for boundary detection
            temporal_info = {
                'frame_id': frame_id,
                'timestamp': frame_metadata.get(frame_id, {}).get('timestamp', frame_id * 1.0),
                'is_start': self._is_event_start(frame_id, all_labels),
                'is_stop': self._is_event_stop(frame_id, all_labels)
            }
            
            sample = {
                'frame_path': frame_path,
                'participant_id': participant_id,
                'frame_id': frame_id,
                'physical_labels': physical_labels,
                'emotional_labels': emotional_labels,
                'all_labels': all_labels,
                'temporal_info': temporal_info
            }
            
            samples.append(sample)
        
        print(f"  Loaded {len(samples)} samples")
        return samples
    
    def _is_event_start(self, frame_id: int, labels: np.ndarray) -> np.ndarray:
        """Determine if this frame is the start of events for each feature"""
        # Simple heuristic: high activation indicates potential start
        return (labels > 0.5).astype(np.float32)
    
    def _is_event_stop(self, frame_id: int, labels: np.ndarray) -> np.ndarray:
        """Determine if this frame is the stop of events for each feature"""  
        # Simple heuristic: low activation indicates potential stop
        return (labels <= 0.5).astype(np.float32)
    
    def _load_all_data(self) -> List[Dict]:
        """Load data for all participants"""
        
        # All 17 participants
        participants = [
            "AM_1355", "AR__2298", "AR_1378", "AW_8961", "BU_6095", 
            "CP_0636", "CP_6047", "CR_0863", "EV_4492", "JG_8996",
            "JM_9684", "JM_IES", "JR_4166", "KW_9939", "LE_3299", 
            "YT_6156", "ZLB_8812"
        ]
        
        all_samples = []
        participants_with_data = []
        
        for participant in participants:
            samples = self._load_participant_data(participant)
            if samples:
                all_samples.extend(samples)
                participants_with_data.append(participant)
        
        print(f"📊 Complete Dataset Loaded:")
        print(f"  Participants with data: {len(participants_with_data)}")
        print(f"  Total samples: {len(all_samples)}")
        
        return all_samples
    
    def __len__(self):
        if self.sequence_length <= 1:
            return len(self.data_samples)
        else:
            # For temporal sequences, return valid starting positions
            return max(0, len(self.data_samples) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        if self.sequence_length <= 1:
            # Single frame mode
            sample = self.data_samples[idx]
            
            # Load and transform image
            image = Image.open(sample['frame_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            result = {
                'image': image,
                'physical_labels': torch.tensor(sample['physical_labels'], dtype=torch.float32),
                'emotional_labels': torch.tensor(sample['emotional_labels'], dtype=torch.float32),
                'all_labels': torch.tensor(sample['all_labels'], dtype=torch.float32),
                'participant_id': sample['participant_id'],
                'frame_id': sample['frame_id']
            }
            
            if self.return_temporal_info:
                result['temporal_info'] = sample['temporal_info']
                result['start_boundaries'] = torch.tensor(
                    sample['temporal_info']['is_start'], dtype=torch.float32
                )
                result['stop_boundaries'] = torch.tensor(
                    sample['temporal_info']['is_stop'], dtype=torch.float32
                )
            
            return result
        
        else:
            # Sequence mode
            sequence_samples = self.data_samples[idx:idx + self.sequence_length]
            
            # Load sequence of images
            images = []
            for sample in sequence_samples:
                image = Image.open(sample['frame_path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            
            # Pad sequence if necessary
            while len(images) < self.sequence_length:
                images.append(images[-1].clone() if images else torch.zeros(3, 224, 224))
            
            images = torch.stack(images[:self.sequence_length])
            
            # Use labels from the last frame in sequence
            last_sample = sequence_samples[-1]
            
            result = {
                'images': images,
                'physical_labels': torch.tensor(last_sample['physical_labels'], dtype=torch.float32),
                'emotional_labels': torch.tensor(last_sample['emotional_labels'], dtype=torch.float32),
                'all_labels': torch.tensor(last_sample['all_labels'], dtype=torch.float32),
                'participant_id': last_sample['participant_id'],
                'frame_id': last_sample['frame_id']
            }
            
            if self.return_temporal_info:
                result['temporal_info'] = [s['temporal_info'] for s in sequence_samples]
                result['start_boundaries'] = torch.tensor(
                    last_sample['temporal_info']['is_start'], dtype=torch.float32
                )
                result['stop_boundaries'] = torch.tensor(
                    last_sample['temporal_info']['is_stop'], dtype=torch.float32
                )
            
            return result
    
    def get_feature_names(self):
        """Get all feature names"""
        return self.feature_info
    
    def get_participants_with_data(self):
        """Get list of participants that have data"""
        participants = list(set(sample['participant_id'] for sample in self.data_samples))
        return sorted(participants)

def create_complete_dataloaders(frames_dir: str,
                               annotations_dir: str,
                               batch_size: int = 16,
                               sequence_length: int = 1,
                               split_by_participant: bool = True,
                               train_split: float = 0.7,
                               val_split: float = 0.15,
                               test_split: float = 0.15,
                               num_workers: int = 0):
    """Create data loaders for all participants"""
    
    # Create dataset
    dataset = CompleteMultiLabelDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        sequence_length=sequence_length
    )
    
    if len(dataset) == 0:
        raise ValueError("No data loaded. Check frames and annotations directories.")
    
    # Get participants with data
    participants_with_data = dataset.get_participants_with_data()
    
    if split_by_participant and len(participants_with_data) >= 3:
        # Split by participants
        n_participants = len(participants_with_data)
        train_participants = participants_with_data[:int(train_split * n_participants)]
        val_participants = participants_with_data[int(train_split * n_participants):int((train_split + val_split) * n_participants)]
        test_participants = participants_with_data[int((train_split + val_split) * n_participants):]
        
        print(f"📊 Splitting {len(participants_with_data)} participants:")
        print(f"  Train participants: {train_participants}")
        print(f"  Val participants: {val_participants}")
        print(f"  Test participants: {test_participants}")
        
        # Create participant-based splits
        train_indices = [i for i, sample in enumerate(dataset.data_samples) 
                        if sample['participant_id'] in train_participants]
        val_indices = [i for i, sample in enumerate(dataset.data_samples) 
                      if sample['participant_id'] in val_participants]
        test_indices = [i for i, sample in enumerate(dataset.data_samples) 
                       if sample['participant_id'] in test_participants]
        
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val samples: {len(val_indices)}")
        print(f"  Test samples: {len(test_indices)}")
    
    else:
        # Random split
        total_samples = len(dataset)
        train_size = int(train_split * total_samples)
        val_size = int(val_split * total_samples)
        test_size = total_samples - train_size - val_size
        
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, dataset

def main():
    """Test the complete dataset"""
    
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_annotations"
    
    print("🧪 Testing Complete Dataset")
    print("=" * 50)
    
    try:
        # Create dataset
        dataset = CompleteMultiLabelDataset(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            sequence_length=1
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Participants: {len(dataset.get_participants_with_data())}")
        
        if len(dataset) > 0:
            # Test sample loading
            sample = dataset[0]
            print(f"   Sample keys: {sample.keys()}")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Label shape: {sample['all_labels'].shape}")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()