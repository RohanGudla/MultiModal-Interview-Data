#!/usr/bin/env python3
"""
Enhanced Frame Extraction for All Videos
Extracts frames from all available videos with comprehensive annotation alignment
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

class EnhancedFrameExtractor:
    def __init__(self, base_path="/home/rohan/Multimodal"):
        self.base_path = Path(base_path)
        self.output_dir = self.base_path / "multimodal_video_ml" / "data" / "enhanced_frames"
        self.output_dir.mkdir(exist_ok=True)
        self.extraction_log = []
        
    def extract_participant_id(self, video_path):
        """Extract participant ID from video filename"""
        filename = video_path.name
        if " - " in filename:
            return filename.split(" - ")[-1].replace(".mp4", "")
        return filename.replace(".mp4", "")
    
    def test_video_accessibility(self, video_path):
        """Test if video can be opened and read"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "Cannot read frames from video"
            
            return True, "Video accessible"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_video_properties(self, video_path):
        """Get comprehensive video properties"""
        cap = cv2.VideoCapture(str(video_path))
        
        properties = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        properties['duration_seconds'] = properties['frame_count'] / properties['fps'] if properties['fps'] > 0 else 0
        
        cap.release()
        return properties
    
    def extract_frames_comprehensive(self, video_path, frame_rate=1.0, max_frames=None):
        """
        Extract frames at specified rate with comprehensive metadata
        
        Args:
            video_path: Path to video file
            frame_rate: Frames per second to extract (default 1.0 = 1fps)
            max_frames: Maximum number of frames to extract (None = all)
        """
        participant_id = self.extract_participant_id(video_path)
        
        # Test video accessibility first
        accessible, message = self.test_video_accessibility(video_path)
        if not accessible:
            print(f"❌ {participant_id}: {message}")
            self.extraction_log.append({
                'participant_id': participant_id,
                'video_path': str(video_path),
                'status': 'failed',
                'error': message,
                'frames_extracted': 0
            })
            return []
        
        # Get video properties
        props = self.get_video_properties(video_path)
        print(f"📹 {participant_id}: {props['duration_seconds']:.1f}s, {props['fps']:.1f}fps, {props['frame_count']} total frames")
        
        # Calculate frame interval for desired extraction rate
        frame_interval = int(props['fps'] / frame_rate) if props['fps'] > 0 else 30
        
        # Create output directory for this participant
        participant_dir = self.output_dir / participant_id
        participant_dir.mkdir(exist_ok=True)
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
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
                timestamp = frame_count / props['fps'] if props['fps'] > 0 else saved_count
                
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
        
        # Save metadata
        metadata = {
            'participant_id': participant_id,
            'video_path': str(video_path),
            'video_properties': props,
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
        
        print(f"✅ {participant_id}: Extracted {saved_count} frames to {participant_dir}")
        
        self.extraction_log.append({
            'participant_id': participant_id,
            'video_path': str(video_path),
            'status': 'success',
            'frames_extracted': saved_count,
            'duration_seconds': props['duration_seconds']
        })
        
        return extracted_frames
    
    def extract_all_videos(self, frame_rate=1.0, max_frames_per_video=None):
        """Extract frames from all available videos"""
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(list(self.base_path.rglob(f"*{ext}")))
        
        print(f"🎬 Found {len(videos)} videos to process")
        
        all_extracted = {}
        
        for video_path in videos:
            participant_id = self.extract_participant_id(video_path)
            print(f"\n📽️ Processing {participant_id}...")
            
            frames = self.extract_frames_comprehensive(
                video_path, 
                frame_rate=frame_rate,
                max_frames=max_frames_per_video
            )
            
            if frames:
                all_extracted[participant_id] = frames
        
        # Save comprehensive extraction log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.output_dir / f"extraction_log_{timestamp}.json"
        
        extraction_summary = {
            'timestamp': timestamp,
            'extraction_settings': {
                'frame_rate': frame_rate,
                'max_frames_per_video': max_frames_per_video
            },
            'results': self.extraction_log,
            'summary': {
                'total_videos_processed': len(videos),
                'successful_extractions': len([log for log in self.extraction_log if log['status'] == 'success']),
                'failed_extractions': len([log for log in self.extraction_log if log['status'] == 'failed']),
                'total_frames_extracted': sum([log['frames_extracted'] for log in self.extraction_log])
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(extraction_summary, f, indent=2)
        
        print(f"\n📊 Extraction Summary:")
        print(f"  Total videos processed: {extraction_summary['summary']['total_videos_processed']}")
        print(f"  Successful extractions: {extraction_summary['summary']['successful_extractions']}")
        print(f"  Failed extractions: {extraction_summary['summary']['failed_extractions']}")
        print(f"  Total frames extracted: {extraction_summary['summary']['total_frames_extracted']}")
        print(f"  Log saved to: {log_path}")
        
        return all_extracted, extraction_summary

def main():
    print("🚀 Starting Enhanced Frame Extraction for All Videos...")
    
    extractor = EnhancedFrameExtractor()
    
    # Extract frames at 1 FPS (as currently used)
    # But extract more frames per video to get better temporal coverage
    extracted_data, summary = extractor.extract_all_videos(
        frame_rate=1.0,  # 1 frame per second
        max_frames_per_video=None  # Extract all available frames
    )
    
    print("\n🎉 Frame extraction complete!")
    
    return extracted_data, summary

if __name__ == "__main__":
    extracted_data, summary = main()