#!/usr/bin/env python3
"""
Multi-Label Dataset for All Annotation Features
Handles all 50 annotation features with temporal information
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Subset

class MultiLabelAnnotationDataset(Dataset):
    """
    Dataset for multi-label emotion and physical feature prediction
    Handles all 50 annotation features with temporal alignment
    """
    
    def __init__(self, 
                 frames_dir: str,
                 annotations_dir: str,
                 participants: List[str] = None,
                 transform=None,
                 sequence_length: int = 1,
                 return_temporal_info: bool = True):
        """
        Initialize multi-label dataset
        
        Args:
            frames_dir: Directory containing extracted frames
            annotations_dir: Directory containing annotation CSV files  
            participants: List of participant IDs to include
            transform: Image transformations
            sequence_length: Number of consecutive frames to use (1 = single frame, >1 = sequence)
            return_temporal_info: Whether to return temporal metadata
        """
        self.frames_dir = Path(frames_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform or self._default_transform()
        self.sequence_length = sequence_length
        self.return_temporal_info = return_temporal_info
        
        # Load annotation feature definitions
        self.feature_info = self._load_feature_definitions()
        
        # Load data for specified participants
        if participants is None:
            participants = self._discover_participants()
        
        self.data_samples = self._load_all_data(participants)
        
        print(f"📊 MultiLabel Dataset Initialized:")
        print(f"  Participants: {len(participants)} ({participants})")
        print(f"  Total samples: {len(self.data_samples)}")
        print(f"  Physical features: {len(self.feature_info['physical'])}")
        print(f"  Emotional features: {len(self.feature_info['emotional'])}")
        print(f"  Total features: {len(self.feature_info['all'])}")
        print(f"  Sequence length: {sequence_length}")
    
    def _default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _discover_participants(self):
        """Discover available participants from annotation files"""
        participants = set()
        
        # Check physical features
        physical_dir = self.annotations_dir / "physical_features"
        if physical_dir.exists():
            for file in physical_dir.glob("*_physical.csv"):
                participant = file.stem.replace("_physical", "")
                participants.add(participant)
        
        # Check emotional targets
        emotional_dir = self.annotations_dir / "emotional_targets"
        if emotional_dir.exists():
            for file in emotional_dir.glob("*_emotional.csv"):
                participant = file.stem.replace("_emotional", "")
                participants.add(participant)
        
        return list(participants)
    
    def _load_feature_definitions(self):
        """Load feature definitions from annotation files"""
        feature_info = {
            'physical': [],
            'emotional': [],
            'all': [],
            'physical_indices': {},
            'emotional_indices': {},
            'all_indices': {}
        }
        
        # Load physical features
        physical_dir = self.annotations_dir / "physical_features"
        if physical_dir.exists():
            physical_files = list(physical_dir.glob("*_physical.csv"))
            if physical_files:
                df = pd.read_csv(physical_files[0])
                feature_info['physical'] = [col for col in df.columns if col != 'frame_id']
        
        # Load emotional features
        emotional_dir = self.annotations_dir / "emotional_targets"
        if emotional_dir.exists():
            emotional_files = list(emotional_dir.glob("*_emotional.csv"))
            if emotional_files:
                df = pd.read_csv(emotional_files[0])
                feature_info['emotional'] = [col for col in df.columns if col != 'frame_id']
        
        # Combine all features
        feature_info['all'] = feature_info['physical'] + feature_info['emotional']
        
        # Create index mappings
        for i, feat in enumerate(feature_info['physical']):
            feature_info['physical_indices'][feat] = i
        
        for i, feat in enumerate(feature_info['emotional']):
            feature_info['emotional_indices'][feat] = i
            
        for i, feat in enumerate(feature_info['all']):
            feature_info['all_indices'][feat] = i
        
        return feature_info
    
    def _load_participant_data(self, participant_id: str):
        """Load data for a specific participant"""
        
        # Load physical annotations
        physical_file = self.annotations_dir / "physical_features" / f"{participant_id}_physical.csv"
        physical_df = None
        if physical_file.exists():
            physical_df = pd.read_csv(physical_file)
        
        # Load emotional annotations
        emotional_file = self.annotations_dir / "emotional_targets" / f"{participant_id}_emotional.csv"
        emotional_df = None
        if emotional_file.exists():
            emotional_df = pd.read_csv(emotional_file)
        
        # Check if participant has extracted frames
        participant_frames_dir = self.frames_dir / participant_id
        if not participant_frames_dir.exists():
            print(f"⚠️ No frames found for {participant_id}")
            return []
        
        # Load frame metadata
        metadata_file = participant_frames_dir / "extraction_metadata.json"
        frame_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                frame_metadata = {frame['frame_id']: frame for frame in metadata['extracted_frames']}
        
        # Create combined samples
        samples = []
        
        # Find the range of available frames
        if physical_df is not None:
            max_frame = min(physical_df['frame_id'].max(), len(frame_metadata) - 1)
        elif emotional_df is not None:
            max_frame = min(emotional_df['frame_id'].max(), len(frame_metadata) - 1)
        else:
            max_frame = len(frame_metadata) - 1
        
        for frame_id in range(max_frame + 1):
            frame_path = participant_frames_dir / f"frame_{frame_id:04d}.jpg"
            
            if not frame_path.exists():
                continue
            
            # Get annotations for this frame
            physical_labels = np.zeros(len(self.feature_info['physical']), dtype=np.float32)
            emotional_labels = np.zeros(len(self.feature_info['emotional']), dtype=np.float32)
            
            # Fill physical labels
            if physical_df is not None and frame_id in physical_df['frame_id'].values:
                row = physical_df[physical_df['frame_id'] == frame_id].iloc[0]
                for i, feature in enumerate(self.feature_info['physical']):
                    if feature in row:
                        physical_labels[i] = float(row[feature])
            
            # Fill emotional labels
            if emotional_df is not None and frame_id in emotional_df['frame_id'].values:
                row = emotional_df[emotional_df['frame_id'] == frame_id].iloc[0]
                for i, feature in enumerate(self.feature_info['emotional']):
                    if feature in row:
                        emotional_labels[i] = float(row[feature])
            
            # Combine all labels
            all_labels = np.concatenate([physical_labels, emotional_labels])
            
            # Get temporal info
            temporal_info = frame_metadata.get(frame_id, {
                'timestamp_seconds': frame_id * 1.0,  # Assume 1 FPS
                'original_frame_number': frame_id * 30  # Assume 30 FPS source
            })
            
            sample = {
                'participant_id': participant_id,
                'frame_id': frame_id,
                'frame_path': str(frame_path),
                'physical_labels': physical_labels,
                'emotional_labels': emotional_labels,
                'all_labels': all_labels,
                'temporal_info': temporal_info
            }
            
            samples.append(sample)
        
        return samples
    
    def _load_all_data(self, participants: List[str]):
        """Load data for all participants"""
        all_samples = []
        
        for participant_id in participants:
            print(f"📝 Loading data for {participant_id}...")
            participant_samples = self._load_participant_data(participant_id)
            all_samples.extend(participant_samples)
            print(f"  Loaded {len(participant_samples)} samples")
        
        return all_samples
    
    def __len__(self):
        """Return dataset size"""
        # For sequence models, we need to account for sequence length
        if self.sequence_length > 1:
            return max(0, len(self.data_samples) - self.sequence_length + 1)
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """Get a data sample"""
        if self.sequence_length == 1:
            # Single frame mode
            sample = self.data_samples[idx]
            
            # Load image
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
                # Repeat last image if sequence is too short
                images.append(images[-1].clone() if images else torch.zeros(3, 224, 224))
            
            # Stack images into tensor
            images = torch.stack(images[:self.sequence_length])
            
            # Use labels from the last frame in sequence
            last_sample = sequence_samples[-1]
            
            result = {
                'images': images,  # Shape: [sequence_length, 3, 224, 224]
                'physical_labels': torch.tensor(last_sample['physical_labels'], dtype=torch.float32),
                'emotional_labels': torch.tensor(last_sample['emotional_labels'], dtype=torch.float32),
                'all_labels': torch.tensor(last_sample['all_labels'], dtype=torch.float32),
                'participant_id': last_sample['participant_id'],
                'frame_id': last_sample['frame_id']
            }
            
            if self.return_temporal_info:
                result['temporal_info'] = [s['temporal_info'] for s in sequence_samples]
            
            return result
    
    def get_feature_names(self):
        """Get all feature names"""
        return self.feature_info
    
    def get_label_statistics(self):
        """Calculate label distribution statistics"""
        if not self.data_samples:
            return {}
        
        # Collect all labels
        all_physical = np.array([s['physical_labels'] for s in self.data_samples])
        all_emotional = np.array([s['emotional_labels'] for s in self.data_samples])
        all_combined = np.array([s['all_labels'] for s in self.data_samples])
        
        stats = {
            'physical': {
                'mean': all_physical.mean(axis=0),
                'std': all_physical.std(axis=0),
                'positive_rate': (all_physical > 0.5).mean(axis=0)
            },
            'emotional': {
                'mean': all_emotional.mean(axis=0),
                'std': all_emotional.std(axis=0),
                'positive_rate': (all_emotional > 0.5).mean(axis=0)
            },
            'combined': {
                'mean': all_combined.mean(axis=0),
                'std': all_combined.std(axis=0),
                'positive_rate': (all_combined > 0.5).mean(axis=0)
            }
        }
        
        return stats

    def get_participant_summary(self):
        """Get summary of data per participant"""
        participant_counts = {}
        for sample in self.data_samples:
            pid = sample['participant_id']
            if pid not in participant_counts:
                participant_counts[pid] = 0
            participant_counts[pid] += 1
        
        return participant_counts

class ParticipantSubsetDataset:
    """Dataset wrapper for participant-based subsets"""
    
    def __init__(self, base_dataset: MultiLabelAnnotationDataset, samples: List[Dict]):
        self.base_dataset = base_dataset
        self.samples = samples
        
        # Create mapping from sample to original index
        self.sample_to_index = {}
        for i, original_sample in enumerate(base_dataset.data_samples):
            key = (original_sample['participant_id'], original_sample['frame_id'])
            self.sample_to_index[key] = i
        
        # Map subset samples to original indices
        self.indices = []
        for sample in samples:
            key = (sample['participant_id'], sample['frame_id'])
            if key in self.sample_to_index:
                self.indices.append(self.sample_to_index[key])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]

def create_dataloaders(frames_dir: str,
                      annotations_dir: str,
                      participants: List[str] = None,
                      batch_size: int = 32,
                      sequence_length: int = 1,
                      train_split: float = 0.8,
                      val_split: float = 0.1,
                      test_split: float = 0.1,
                      num_workers: int = 4,
                      split_by_participant: bool = True):
    """
    Create train/val/test dataloaders for multi-label dataset
    
    Args:
        split_by_participant: If True, ensure participants are not shared across splits
    """
    
    dataset = MultiLabelAnnotationDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        participants=participants,
        sequence_length=sequence_length
    )
    
    if split_by_participant and len(dataset.data_samples) > 0:
        # Get unique participants who have data
        participants_with_data = list(set(s['participant_id'] for s in dataset.data_samples))
        print(f"📊 Splitting {len(participants_with_data)} participants: {participants_with_data}")
        
        # Split participants across train/val/test
        np.random.seed(42)
        np.random.shuffle(participants_with_data)
        
        n_participants = len(participants_with_data)
        train_participants = participants_with_data[:int(train_split * n_participants)]
        val_participants = participants_with_data[int(train_split * n_participants):int((train_split + val_split) * n_participants)]
        test_participants = participants_with_data[int((train_split + val_split) * n_participants):]
        
        print(f"  Train participants: {train_participants}")
        print(f"  Val participants: {val_participants}")
        print(f"  Test participants: {test_participants}")
        
        # Create participant-based splits
        train_samples = [s for s in dataset.data_samples if s['participant_id'] in train_participants]
        val_samples = [s for s in dataset.data_samples if s['participant_id'] in val_participants]
        test_samples = [s for s in dataset.data_samples if s['participant_id'] in test_participants]
        
        # Create sub-datasets
        train_dataset = ParticipantSubsetDataset(dataset, train_samples)
        val_dataset = ParticipantSubsetDataset(dataset, val_samples)
        test_dataset = ParticipantSubsetDataset(dataset, test_samples)
        
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")
        print(f"  Test samples: {len(test_samples)}")
        
    else:
        # Random split (original behavior)
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset

# Test the dataset
def test_dataset():
    """Test the multi-label dataset"""
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    
    # Test single frame mode
    print("Testing single frame mode...")
    dataset = MultiLabelAnnotationDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        sequence_length=1
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"All labels shape: {sample['all_labels'].shape}")
        print(f"Physical labels shape: {sample['physical_labels'].shape}")
        print(f"Emotional labels shape: {sample['emotional_labels'].shape}")
    
    # Test sequence mode
    print("\nTesting sequence mode...")
    dataset_seq = MultiLabelAnnotationDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        sequence_length=5
    )
    
    if len(dataset_seq) > 0:
        sample = dataset_seq[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"All labels shape: {sample['all_labels'].shape}")
    
    # Get statistics
    stats = dataset.get_label_statistics()
    print(f"\nDataset statistics calculated for {len(stats)} label types")

if __name__ == "__main__":
    test_dataset()