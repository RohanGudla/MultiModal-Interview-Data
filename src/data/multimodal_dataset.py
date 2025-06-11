"""
Multimodal Dataset for GENEX Video + Annotation Data
Combines video frames with physical/behavioral annotations for multimodal learning.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class MultimodalGENEXDataset(Dataset):
    """Dataset combining GENEX video frames with annotation data."""
    
    def __init__(self, 
                 video_frames_path: str = "/home/rohan/Multimodal/multimodal_video_ml/data/real_frames",
                 annotations_path: str = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations",
                 transform=None,
                 participants: Optional[List[str]] = None):
        
        self.video_frames_path = Path(video_frames_path)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        
        # Load processing summary to get participant info
        summary_path = self.annotations_path / "processing_summary.json"
        with open(summary_path, 'r') as f:
            self.processing_summary = json.load(f)
        
        # Use participants with both video and annotation data
        if participants is None:
            self.participants = self.processing_summary["participants_processed"]["physical"]
        else:
            self.participants = participants
            
        # Load annotation data
        self.physical_features, self.emotional_targets = self._load_annotation_data()
        
        # Create sample index mapping
        self.samples = self._create_sample_index()
        
        # Feature dimensions
        self.physical_dim = len(self.processing_summary["physical_features"]) + 5  # +5 for eye tracking + GSR
        self.emotional_dim = len(self.processing_summary["emotional_targets"])
        
        print(f"MultimodalGENEXDataset initialized:")
        print(f"  Participants: {len(self.participants)}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Physical features: {self.physical_dim}")
        print(f"  Emotional targets: {self.emotional_dim}")
        
    def _load_annotation_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Load physical and emotional annotation data for all participants."""
        
        physical_features = {}
        emotional_targets = {}
        
        for participant in self.participants:
            # Load physical features
            physical_file = self.annotations_path / "physical_features" / f"{participant.replace(' ', '_')}_physical.csv"
            if physical_file.exists():
                physical_df = pd.read_csv(physical_file, index_col=0)
                physical_features[participant] = physical_df
                
            # Load emotional targets
            emotional_file = self.annotations_path / "emotional_targets" / f"{participant.replace(' ', '_')}_emotional.csv"
            if emotional_file.exists():
                emotional_df = pd.read_csv(emotional_file, index_col=0)
                emotional_targets[participant] = emotional_df
                
        return physical_features, emotional_targets
        
    def _create_sample_index(self) -> List[Tuple[str, int]]:
        """Create index mapping (participant, frame_id) for all samples."""
        
        samples = []
        
        for participant in self.participants:
            if participant in self.physical_features and participant in self.emotional_targets:
                # Each participant has 20 frames aligned with annotations
                for frame_id in range(20):
                    samples.append((participant, frame_id))
                    
        return samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single multimodal sample."""
        
        participant, frame_id = self.samples[idx]
        
        # Load video frame
        video_frame = self._load_video_frame(participant, frame_id)
        
        # Load physical features
        physical_vector = self._load_physical_features(participant, frame_id)
        
        # Load emotional targets
        emotional_vector = self._load_emotional_targets(participant, frame_id)
        
        sample = {
            'video': video_frame,
            'physical': physical_vector,
            'emotional': emotional_vector,
            'participant': participant,
            'frame_id': frame_id
        }
        
        return sample
        
    def _load_video_frame(self, participant: str, frame_id: int) -> torch.Tensor:
        """Load and preprocess video frame."""
        
        frame_path = self.video_frames_path / participant / f"frame_{frame_id:04d}.jpg"
        
        # Load image
        image = Image.open(frame_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize to 224x224 and convert to tensor
            image = image.resize((224, 224))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        return image
        
    def _load_physical_features(self, participant: str, frame_id: int) -> torch.Tensor:
        """Load physical feature vector for frame."""
        
        if participant not in self.physical_features:
            # Return zero vector if no data
            return torch.zeros(self.physical_dim, dtype=torch.float32)
            
        physical_df = self.physical_features[participant]
        
        if frame_id >= len(physical_df):
            # Return zero vector if frame out of range
            return torch.zeros(self.physical_dim, dtype=torch.float32)
            
        # Get feature row
        feature_row = physical_df.iloc[frame_id]
        
        # Convert to tensor with consistent shape
        feature_vector = torch.from_numpy(feature_row.values.astype(np.float32))
        
        # Ensure consistent dimension
        if len(feature_vector) != self.physical_dim:
            # Pad or truncate to match expected dimension
            padded_vector = torch.zeros(self.physical_dim, dtype=torch.float32)
            min_len = min(len(feature_vector), self.physical_dim)
            padded_vector[:min_len] = feature_vector[:min_len]
            feature_vector = padded_vector
        
        return feature_vector
        
    def _load_emotional_targets(self, participant: str, frame_id: int) -> torch.Tensor:
        """Load emotional target vector for frame."""
        
        if participant not in self.emotional_targets:
            # Return zero vector if no data
            return torch.zeros(self.emotional_dim, dtype=torch.float32)
            
        emotional_df = self.emotional_targets[participant]
        
        if frame_id >= len(emotional_df):
            # Return zero vector if frame out of range
            return torch.zeros(self.emotional_dim, dtype=torch.float32)
            
        # Get target row
        target_row = emotional_df.iloc[frame_id]
        
        # Convert to tensor with consistent shape
        target_vector = torch.from_numpy(target_row.values.astype(np.float32))
        
        # Ensure consistent dimension
        if len(target_vector) != self.emotional_dim:
            # Pad or truncate to match expected dimension
            padded_vector = torch.zeros(self.emotional_dim, dtype=torch.float32)
            min_len = min(len(target_vector), self.emotional_dim)
            padded_vector[:min_len] = target_vector[:min_len]
            target_vector = padded_vector
        
        return target_vector
        
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature and target names for interpretation."""
        
        return {
            "physical_features": self.processing_summary["physical_features"] + 
                               ["fixation_density", "avg_fixation_duration", "gaze_dispersion", 
                                "gsr_peak_count", "gsr_avg_amplitude"],
            "emotional_targets": self.processing_summary["emotional_targets"]
        }
        
    def get_participant_split(self, train_ratio: float = 0.7) -> Tuple[List[int], List[int]]:
        """Create participant-based train/val split to avoid data leakage."""
        
        np.random.seed(42)  # For reproducible splits
        
        # Shuffle participants
        participants_shuffled = self.participants.copy()
        np.random.shuffle(participants_shuffled)
        
        # Split participants
        n_train = int(len(participants_shuffled) * train_ratio)
        train_participants = participants_shuffled[:n_train]
        val_participants = participants_shuffled[n_train:]
        
        # Get sample indices for each split
        train_indices = []
        val_indices = []
        
        for idx, (participant, frame_id) in enumerate(self.samples):
            if participant in train_participants:
                train_indices.append(idx)
            else:
                val_indices.append(idx)
                
        return train_indices, val_indices


class MultimodalDataModule:
    """Data module for multimodal training with proper train/val splits."""
    
    def __init__(self, batch_size: int = 16, num_workers: int = 4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        from torchvision import transforms
        
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self):
        """Setup train and validation datasets."""
        
        # Create full dataset
        full_dataset = MultimodalGENEXDataset(transform=None)
        
        # Get participant-based split
        train_indices, val_indices = full_dataset.get_participant_split(train_ratio=0.75)
        
        # Create train dataset with augmentation
        self.train_dataset = MultimodalGENEXDataset(transform=self.train_transform)
        self.train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
        
        # Create val dataset without augmentation
        self.val_dataset = MultimodalGENEXDataset(transform=self.val_transform)
        self.val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
        
        print(f"Data split - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        return self.train_dataset, self.val_dataset
        
    def get_dataloaders(self):
        """Get train and validation dataloaders."""
        
        train_dataset, val_dataset = self.setup()
        
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader