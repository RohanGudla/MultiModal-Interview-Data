"""
Data loading and preprocessing pipeline for multimodal video data.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.config import Config
from ..utils.video_utils import VideoProcessor
from .annotation_parser import AnnotationParser

class DataLoader:
    """Main data loading and preprocessing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.video_processor = VideoProcessor(
            face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            face_bbox_expansion=config.FACE_BBOX_EXPANSION
        )
        
        # Get data paths
        self.data_paths = config.get_data_paths()
        
        # Initialize annotation parser
        self.annotation_parser = AnnotationParser(
            annotation_path=self.data_paths["annotations"],
            expression_table_path=self.data_paths["expression_table"]
        )
        
    def setup_data_directories(self):
        """Create necessary data directories."""
        for path in self.data_paths.values():
            if isinstance(path, Path) and not path.name.endswith('.csv'):
                path.mkdir(parents=True, exist_ok=True)
                
    def process_all_videos(self, force_reprocess: bool = False):
        """Process all available videos and extract frames with faces."""
        self.setup_data_directories()
        
        video_dir = self.data_paths["videos"]
        processed_dir = self.data_paths["processed_frames"]
        
        # Find all video files
        video_files = list(video_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} video files")
        
        processed_data = {}
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            participant_id = VideoProcessor.extract_participant_id(video_path.name)
            
            # Check if already processed
            participant_dir = processed_dir / participant_id
            if participant_dir.exists() and not force_reprocess:
                print(f"Skipping {participant_id} - already processed")
                continue
                
            # Process video
            try:
                result = self.video_processor.process_video_frames(
                    video_path=video_path,
                    output_dir=processed_dir,
                    fps=self.config.VIDEO_FPS,
                    target_size=self.config.FRAME_SIZE
                )
                
                processed_data[participant_id] = result
                
                # Save processing metadata
                metadata_path = participant_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    # Convert Path objects to strings for JSON serialization
                    json_result = {
                        'participant_id': result['participant_id'],
                        'video_path': str(result['video_path']),
                        'valid_frames': [str(p) for p in result['valid_frames']],
                        'timestamps': result['timestamps'],
                        'fps': result['fps'],
                        'total_frames': result['total_frames'],
                        'valid_face_frames': result['valid_face_frames']
                    }
                    json.dump(json_result, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
                
        return processed_data
        
    def create_annotation_files(self):
        """Create processed annotation files for each participant."""
        labels_dir = self.data_paths["processed_labels"]
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all annotations
        all_annotations = self.annotation_parser.get_all_participant_annotations()
        
        for participant_id, annotations in all_annotations.items():
            # Save participant annotations
            annotation_file = labels_dir / f"{participant_id}_annotations.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotations, f, indent=2)
                
        # Save annotation statistics
        stats = self.annotation_parser.get_label_statistics()
        stats_file = labels_dir / "annotation_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"Created annotation files for {len(all_annotations)} participants")
        return all_annotations
        
    def create_data_splits(self):
        """Create train/validation/test splits."""
        splits_dir = self.data_paths["splits"]
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Define splits based on config
        splits = {
            'train': self.config.TRAIN_PARTICIPANTS,
            'val': self.config.VAL_PARTICIPANTS,
            'test': self.config.TEST_PARTICIPANTS
        }
        
        # Verify all participants have data
        processed_dir = self.data_paths["processed_frames"]
        available_participants = [
            d.name for d in processed_dir.iterdir() 
            if d.is_dir() and (d / "metadata.json").exists()
        ]
        
        print(f"Available participants: {available_participants}")
        
        # Create split files
        split_data = {}
        for split_name, participants in splits.items():
            split_participants = [p for p in participants if p in available_participants]
            
            if not split_participants:
                print(f"Warning: No participants available for {split_name} split")
                continue
                
            split_info = []
            for participant in split_participants:
                metadata_path = processed_dir / participant / "metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                split_info.append({
                    'participant_id': participant,
                    'num_frames': metadata['valid_face_frames'],
                    'fps': metadata['fps'],
                    'video_duration': len(metadata['timestamps'])
                })
                
            split_data[split_name] = split_info
            
            # Save split file
            split_file = splits_dir / f"{split_name}_split.json"
            with open(split_file, 'w') as f:
                json.dump(split_info, f, indent=2)
                
        # Save complete split summary
        summary_file = splits_dir / "split_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(split_data, f, indent=2)
            
        print("Data splits created:")
        for split_name, data in split_data.items():
            total_frames = sum(item['num_frames'] for item in data)
            print(f"  {split_name}: {len(data)} participants, {total_frames} frames")
            
        return split_data
        
    def get_participant_data(self, participant_id: str) -> Optional[Dict]:
        """Get processed data for a specific participant."""
        # Load frame metadata
        metadata_path = self.data_paths["processed_frames"] / participant_id / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Load annotations
        annotation_path = self.data_paths["processed_labels"] / f"{participant_id}_annotations.json"
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
        else:
            annotations = {}
            
        return {
            'metadata': metadata,
            'annotations': annotations,
            'frame_paths': [Path(p) for p in metadata['valid_frames']],
            'timestamps': metadata['timestamps']
        }
        
    def create_frame_label_mapping(self, participant_id: str, 
                                 label_type: str = 'binary_attention') -> List[Tuple[Path, float]]:
        """Create frame-to-label mapping for training."""
        participant_data = self.get_participant_data(participant_id)
        if not participant_data:
            return []
            
        frame_paths = participant_data['frame_paths']
        timestamps = participant_data['timestamps']
        
        # Get labels based on type
        if label_type == 'binary_attention':
            labels = self.annotation_parser.get_binary_attention_labels(
                participant_id, timestamps
            )
        elif label_type == 'attention_regression':
            labels = self.annotation_parser.get_regression_labels(
                participant_id, timestamps, target='Attention'
            )
        elif label_type == 'emotion_multilabel':
            labels = self.annotation_parser.create_frame_level_labels(
                participant_id, timestamps
            )
        else:
            raise ValueError(f"Unknown label type: {label_type}")
            
        # Create frame-label pairs
        if len(labels.shape) == 1:
            # Single label per frame
            frame_label_pairs = list(zip(frame_paths, labels))
        else:
            # Multiple labels per frame
            frame_label_pairs = list(zip(frame_paths, labels))
            
        return frame_label_pairs
        
    def run_full_pipeline(self, force_reprocess: bool = False):
        """Run the complete data processing pipeline."""
        print("=== Starting Full Data Processing Pipeline ===")
        
        # Step 1: Process videos
        print("\n1. Processing videos and extracting frames...")
        processed_videos = self.process_all_videos(force_reprocess=force_reprocess)
        
        # Step 2: Create annotation files
        print("\n2. Creating annotation files...")
        annotations = self.create_annotation_files()
        
        # Step 3: Create data splits
        print("\n3. Creating data splits...")
        splits = self.create_data_splits()
        
        # Step 4: Print summary
        print("\n=== Pipeline Complete ===")
        print(f"Processed {len(processed_videos)} videos")
        print(f"Created annotations for {len(annotations)} participants")
        
        for split_name, data in splits.items():
            total_frames = sum(item['num_frames'] for item in data)
            print(f"{split_name.title()} split: {len(data)} participants, {total_frames} frames")
            
        # Print annotation summary
        print("\n=== Annotation Summary ===")
        self.annotation_parser.print_annotation_summary()
        
        return {
            'processed_videos': processed_videos,
            'annotations': annotations,
            'splits': splits
        }