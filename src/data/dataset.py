"""
PyTorch dataset classes for multimodal video emotion recognition.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import albumentations as A

from ..utils.config import Config
from ..utils.video_utils import VideoProcessor
from .data_loader import DataLoader as MLDataLoader

class EmotionVideoDataset(Dataset):
    """Dataset for video-based emotion recognition."""
    
    def __init__(self, 
                 participant_ids: List[str],
                 data_loader: MLDataLoader,
                 label_type: str = 'binary_attention',
                 transform: Optional[A.Compose] = None,
                 max_frames_per_participant: Optional[int] = None):
        """
        Args:
            participant_ids: List of participant IDs to include
            data_loader: MLDataLoader instance
            label_type: Type of labels ('binary_attention', 'attention_regression', 'emotion_multilabel')
            transform: Albumentations transform pipeline
            max_frames_per_participant: Limit frames per participant for balancing
        """
        self.participant_ids = participant_ids
        self.data_loader = data_loader
        self.label_type = label_type
        self.transform = transform
        self.max_frames_per_participant = max_frames_per_participant
        
        # Build dataset index
        self.samples = self._build_sample_index()
        
        # Determine output dimensions
        self.num_classes = self._get_num_classes()
        
    def _build_sample_index(self) -> List[Tuple[Path, Union[float, np.ndarray], str]]:
        """Build index of all samples (frame_path, label, participant_id)."""
        samples = []
        
        for participant_id in self.participant_ids:
            # Get frame-label mapping for this participant
            frame_label_pairs = self.data_loader.create_frame_label_mapping(
                participant_id, self.label_type
            )
            
            if not frame_label_pairs:
                print(f"Warning: No data found for participant {participant_id}")
                continue
                
            # Limit frames if specified
            if self.max_frames_per_participant:
                frame_label_pairs = frame_label_pairs[:self.max_frames_per_participant]
                
            # Add participant ID to each sample
            for frame_path, label in frame_label_pairs:
                samples.append((frame_path, label, participant_id))
                
        print(f"Built dataset with {len(samples)} samples from {len(self.participant_ids)} participants")
        return samples
        
    def _get_num_classes(self) -> int:
        """Determine number of output classes/dimensions."""
        if not self.samples:
            return 1
            
        sample_label = self.samples[0][1]
        if isinstance(sample_label, np.ndarray):
            return sample_label.shape[0]
        else:
            return 1
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample."""
        frame_path, label, participant_id = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(frame_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {frame_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Apply transforms
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"Error applying transform: {e}")
                # Fallback to simple tensor conversion
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # Simple tensor conversion without normalization
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        # Convert label to tensor
        if isinstance(label, np.ndarray):
            label_tensor = torch.from_numpy(label).float()
        else:
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
        return image, label_tensor, participant_id
        
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets (binary classification only)."""
        if self.label_type != 'binary_attention':
            return torch.ones(self.num_classes)
            
        # Count positive and negative samples
        labels = [sample[1] for sample in self.samples]
        labels = np.array(labels)
        
        pos_count = np.sum(labels == 1)
        neg_count = np.sum(labels == 0)
        total = len(labels)
        
        if pos_count == 0 or neg_count == 0:
            return torch.ones(2)
            
        # Inverse frequency weighting
        pos_weight = total / (2 * pos_count)
        neg_weight = total / (2 * neg_count)
        
        return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)
        
    def get_label_statistics(self) -> Dict[str, float]:
        """Get statistics about the labels in this dataset."""
        labels = [sample[1] for sample in self.samples]
        
        if self.label_type == 'binary_attention':
            labels = np.array(labels)
            return {
                'total_samples': len(labels),
                'positive_samples': int(np.sum(labels == 1)),
                'negative_samples': int(np.sum(labels == 0)),
                'positive_ratio': float(np.mean(labels)),
                'class_balance': float(min(np.mean(labels), 1 - np.mean(labels)))
            }
        else:
            labels = np.array(labels)
            if len(labels.shape) == 1:
                return {
                    'total_samples': len(labels),
                    'mean': float(np.mean(labels)),
                    'std': float(np.std(labels)),
                    'min': float(np.min(labels)),
                    'max': float(np.max(labels))
                }
            else:
                return {
                    'total_samples': len(labels),
                    'num_classes': labels.shape[1],
                    'mean_per_class': labels.mean(axis=0).tolist(),
                    'std_per_class': labels.std(axis=0).tolist()
                }

class MultiTaskEmotionDataset(Dataset):
    """Dataset for multi-task learning (attention + emotions)."""
    
    def __init__(self,
                 participant_ids: List[str],
                 data_loader: MLDataLoader,
                 transform: Optional[A.Compose] = None,
                 include_attention: bool = True,
                 include_emotions: bool = True):
        """
        Args:
            participant_ids: List of participant IDs
            data_loader: MLDataLoader instance
            transform: Transform pipeline
            include_attention: Whether to include attention prediction
            include_emotions: Whether to include emotion prediction
        """
        self.participant_ids = participant_ids
        self.data_loader = data_loader
        self.transform = transform
        self.include_attention = include_attention
        self.include_emotions = include_emotions
        
        self.samples = self._build_sample_index()
        
    def _build_sample_index(self) -> List[Tuple[Path, Dict[str, torch.Tensor], str]]:
        """Build sample index with multiple targets."""
        samples = []
        
        for participant_id in self.participant_ids:
            participant_data = self.data_loader.get_participant_data(participant_id)
            if not participant_data:
                continue
                
            frame_paths = participant_data['frame_paths']
            timestamps = participant_data['timestamps']
            
            # Get different types of labels
            labels_dict = {}
            
            if self.include_attention:
                attention_labels = self.data_loader.annotation_parser.get_binary_attention_labels(
                    participant_id, timestamps
                )
                attention_regression = self.data_loader.annotation_parser.get_regression_labels(
                    participant_id, timestamps, target='Attention'
                )
                labels_dict['attention_binary'] = attention_labels
                labels_dict['attention_regression'] = attention_regression
                
            if self.include_emotions:
                emotion_labels = self.data_loader.annotation_parser.create_frame_level_labels(
                    participant_id, timestamps
                )
                labels_dict['emotions'] = emotion_labels
                
            # Create samples
            for i, frame_path in enumerate(frame_paths):
                sample_labels = {}
                for label_type, labels in labels_dict.items():
                    if len(labels.shape) == 1:
                        sample_labels[label_type] = torch.tensor(labels[i], dtype=torch.float32)
                    else:
                        sample_labels[label_type] = torch.tensor(labels[i], dtype=torch.float32)
                        
                samples.append((frame_path, sample_labels, participant_id))
                
        return samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        """Get sample with multiple targets."""
        frame_path, labels_dict, participant_id = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(frame_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {frame_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        return image, labels_dict, participant_id

def create_data_loaders(config: Config, 
                       label_type: str = 'binary_attention',
                       batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Initialize data loader
    ml_data_loader = MLDataLoader(config)
    
    # Get transforms
    train_transform = VideoProcessor.get_augmentation_pipeline(is_training=True)
    val_transform = VideoProcessor.get_augmentation_pipeline(is_training=False)
    
    # Create datasets
    train_dataset = EmotionVideoDataset(
        participant_ids=config.TRAIN_PARTICIPANTS,
        data_loader=ml_data_loader,
        label_type=label_type,
        transform=train_transform
    )
    
    val_dataset = EmotionVideoDataset(
        participant_ids=config.VAL_PARTICIPANTS,
        data_loader=ml_data_loader,
        label_type=label_type,
        transform=val_transform
    )
    
    test_dataset = EmotionVideoDataset(
        participant_ids=config.TEST_PARTICIPANTS,
        data_loader=ml_data_loader,
        label_type=label_type,
        transform=val_transform
    )
    
    # Use provided batch size or get from config
    if batch_size is None:
        batch_size = 32  # Default fallback
        
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Print dataset statistics
    print("=== Dataset Statistics ===")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    if hasattr(train_dataset, 'get_label_statistics'):
        train_stats = train_dataset.get_label_statistics()
        print(f"Train label stats: {train_stats}")
    
    return train_loader, val_loader, test_loader