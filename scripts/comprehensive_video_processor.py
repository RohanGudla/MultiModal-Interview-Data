#!/usr/bin/env python3
"""
Comprehensive Video Processor for All OneDrive Videos
Phase 1: Video health check, inventory, and batch frame extraction
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import re
import os

class ComprehensiveVideoProcessor:
    """
    Processes all videos from OneDrive archive for multi-participant analysis
    """
    
    def __init__(self, 
                 videos_dir="/home/rohan/Multimodal/multimodal_video_ml/data/all_videos",
                 output_dir="/home/rohan/Multimodal/multimodal_video_ml/data/comprehensive_frames"):
        
        self.videos_dir = Path(videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.video_inventory = []
        self.working_videos = []
        self.corrupted_videos = []
        self.participant_mapping = {}
        
    def extract_participant_id(self, video_filename):
        """Extract participant ID from various video filename formats"""
        filename = str(video_filename)
        
        # Pattern 1: "Screen recording 1 - PARTICIPANT_ID.mp4"
        match1 = re.search(r'Screen recording 1 - ([A-Z]+\s*[0-9]+)', filename)
        if match1:
            return match1.group(1).replace(' ', '_')
        
        # Pattern 2: "RespCam_PARTICIPANT_ID_..."
        match2 = re.search(r'RespCam_([A-Z]+\s*[0-9]+)_', filename)
        if match2:
            return match2.group(1).replace(' ', '_')
        
        # Pattern 3: Any participant pattern
        match3 = re.search(r'([A-Z]{2,3}\s*[0-9]{3,4})', filename)
        if match3:
            return match3.group(1).replace(' ', '_')
        
        # Fallback: use filename without extension
        return Path(filename).stem.replace(' ', '_').replace('-', '_')
    
    def test_video_accessibility(self, video_path):
        """Test if video can be opened and basic properties"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return False, "Cannot open video file", {}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "Cannot read frames", {}
            
            properties = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'file_size_mb': video_path.stat().st_size / (1024*1024)
            }
            
            return True, "Video accessible", properties
            
        except Exception as e:
            return False, f"Error: {str(e)}", {}
    
    def create_video_inventory(self):
        """Create comprehensive inventory of all videos"""
        
        print("📹 Creating comprehensive video inventory...")
        
        video_files = list(self.videos_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} video files")
        
        for video_path in video_files:
            print(f"\n🔍 Processing: {video_path.name}")
            
            # Extract participant ID
            participant_id = self.extract_participant_id(video_path.name)
            
            # Test video accessibility
            is_working, status_msg, properties = self.test_video_accessibility(video_path)
            
            # Create inventory entry
            inventory_entry = {
                'filename': video_path.name,
                'filepath': str(video_path),
                'participant_id': participant_id,
                'is_working': is_working,
                'status': status_msg,
                'properties': properties
            }
            
            self.video_inventory.append(inventory_entry)
            
            if is_working:
                self.working_videos.append(inventory_entry)
                self.participant_mapping[participant_id] = inventory_entry
                print(f"✅ WORKING: {participant_id} - {properties['duration_seconds']:.1f}s, {properties['fps']:.1f}fps")
            else:
                self.corrupted_videos.append(inventory_entry)
                print(f"❌ CORRUPTED: {participant_id} - {status_msg}")
        
        # Save inventory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inventory_file = self.output_dir / f"video_inventory_{timestamp}.json"
        
        inventory_summary = {
            'timestamp': timestamp,
            'total_videos': len(video_files),
            'working_videos': len(self.working_videos),
            'corrupted_videos': len(self.corrupted_videos),
            'participants': list(self.participant_mapping.keys()),
            'detailed_inventory': self.video_inventory
        }
        
        with open(inventory_file, 'w') as f:
            json.dump(inventory_summary, f, indent=2)
        
        print(f"\n📊 Video Inventory Summary:")
        print(f"   Total videos: {len(video_files)}")
        print(f"   Working videos: {len(self.working_videos)}")
        print(f"   Corrupted videos: {len(self.corrupted_videos)}")
        print(f"   Unique participants: {len(self.participant_mapping)}")
        print(f"   Inventory saved: {inventory_file}")
        
        return inventory_summary
    
    def extract_frames_from_video(self, video_entry, frame_rate=1.0, max_frames=None):
        """Extract frames from a single video with comprehensive metadata"""
        
        participant_id = video_entry['participant_id']
        video_path = video_entry['filepath']
        
        print(f"🎬 Extracting frames from {participant_id}...")
        
        # Create participant directory
        participant_dir = self.output_dir / participant_id
        participant_dir.mkdir(exist_ok=True)
        
        # Calculate frame interval
        fps = video_entry['properties']['fps']
        frame_interval = int(fps / frame_rate) if fps > 0 else 30
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                if max_frames and saved_count >= max_frames:
                    break
                
                # Resize frame for model compatibility
                frame_resized = cv2.resize(frame, (224, 224))
                
                # Save frame
                frame_filename = f"frame_{saved_count:04d}.jpg"
                frame_path = participant_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_resized)
                
                # Calculate timestamp
                timestamp = frame_count / fps if fps > 0 else saved_count
                
                extracted_frames.append({
                    'frame_id': saved_count,
                    'original_frame_number': frame_count,
                    'timestamp_seconds': timestamp,
                    'file_path': str(frame_path),
                    'resolution': '224x224'
                })
                
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Save extraction metadata
        metadata = {
            'participant_id': participant_id,
            'video_path': video_path,
            'video_properties': video_entry['properties'],
            'extraction_settings': {
                'target_frame_rate': frame_rate,
                'frame_interval': frame_interval,
                'max_frames': max_frames
            },
            'extracted_frames': extracted_frames,
            'total_frames_extracted': saved_count
        }
        
        metadata_path = participant_dir / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Extracted {saved_count} frames to {participant_dir}")
        
        return metadata
    
    def extract_all_frames(self, frame_rate=1.0, max_frames_per_video=None):
        """Extract frames from all working videos"""
        
        print(f"\n🚀 Starting batch frame extraction...")
        print(f"   Frame rate: {frame_rate} FPS")
        print(f"   Max frames per video: {max_frames_per_video or 'unlimited'}")
        
        extraction_results = []
        total_frames = 0
        
        for video_entry in self.working_videos:
            try:
                metadata = self.extract_frames_from_video(
                    video_entry, 
                    frame_rate=frame_rate,
                    max_frames=max_frames_per_video
                )
                extraction_results.append(metadata)
                total_frames += metadata['total_frames_extracted']
                
            except Exception as e:
                print(f"❌ Failed to extract from {video_entry['participant_id']}: {e}")
        
        # Save batch extraction summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_summary = {
            'timestamp': timestamp,
            'extraction_settings': {
                'frame_rate': frame_rate,
                'max_frames_per_video': max_frames_per_video
            },
            'results': {
                'successful_extractions': len(extraction_results),
                'failed_extractions': len(self.working_videos) - len(extraction_results),
                'total_frames_extracted': total_frames,
                'participants_processed': [r['participant_id'] for r in extraction_results]
            },
            'detailed_results': extraction_results
        }
        
        summary_file = self.output_dir / f"batch_extraction_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\n📊 Batch Extraction Summary:")
        print(f"   Participants processed: {len(extraction_results)}")
        print(f"   Total frames extracted: {total_frames}")
        print(f"   Average frames per participant: {total_frames/len(extraction_results) if extraction_results else 0:.1f}")
        print(f"   Summary saved: {summary_file}")
        
        return batch_summary
    
    def create_participant_summary(self):
        """Create comprehensive participant summary"""
        
        participants_data = []
        
        for participant_id, video_entry in self.participant_mapping.items():
            # Check if frames were extracted
            participant_dir = self.output_dir / participant_id
            metadata_file = participant_dir / "extraction_metadata.json"
            
            frames_extracted = 0
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    frames_extracted = metadata['total_frames_extracted']
            
            participant_data = {
                'participant_id': participant_id,
                'video_filename': video_entry['filename'],
                'video_duration_seconds': video_entry['properties']['duration_seconds'],
                'video_fps': video_entry['properties']['fps'],
                'video_resolution': f"{video_entry['properties']['width']}x{video_entry['properties']['height']}",
                'frames_extracted': frames_extracted,
                'has_frames': frames_extracted > 0
            }
            
            participants_data.append(participant_data)
        
        # Save as CSV for easy viewing
        df = pd.DataFrame(participants_data)
        csv_file = self.output_dir / "participants_summary.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n📋 Participant Summary:")
        print(f"   Total participants: {len(participants_data)}")
        print(f"   Participants with frames: {sum(p['has_frames'] for p in participants_data)}")
        print(f"   Summary saved: {csv_file}")
        
        return participants_data

def main():
    """Main processing function"""
    
    print("🎯 Comprehensive Video Processing Pipeline")
    print("=" * 60)
    
    processor = ComprehensiveVideoProcessor()
    
    # Phase 1: Create video inventory
    print("\nPHASE 1: Video Health Check & Inventory")
    print("-" * 40)
    inventory = processor.create_video_inventory()
    
    # Phase 2: Extract frames from all working videos
    print("\nPHASE 2: Batch Frame Extraction")
    print("-" * 40)
    extraction_summary = processor.extract_all_frames(
        frame_rate=1.0,  # 1 frame per second
        max_frames_per_video=None  # Extract all frames
    )
    
    # Phase 3: Create participant summary
    print("\nPHASE 3: Participant Summary")
    print("-" * 40)
    participant_summary = processor.create_participant_summary()
    
    print("\n🎉 Comprehensive Video Processing Complete!")
    print(f"   Working videos: {inventory['working_videos']}")
    print(f"   Total frames: {extraction_summary['results']['total_frames_extracted']}")
    print(f"   Ready for annotation processing!")

if __name__ == "__main__":
    main()