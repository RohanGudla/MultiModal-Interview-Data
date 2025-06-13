#!/usr/bin/env python3
"""
Extract frames from working videos only (optimized version)
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import re

def extract_participant_id(video_filename):
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

def extract_frames_from_video(video_info, output_dir, frame_rate=1.0):
    """Extract frames from a single working video"""
    
    video_path = video_info['path']
    participant_id = extract_participant_id(Path(video_path).name)
    
    print(f"🎬 Extracting from {participant_id} ({video_info['duration']:.1f}s)...")
    
    # Create participant directory
    participant_dir = Path(output_dir) / participant_id
    participant_dir.mkdir(exist_ok=True)
    
    # Calculate frame interval
    fps = video_info['fps']
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
        'video_properties': {
            'fps': fps,
            'duration_seconds': video_info['duration'],
            'original_filename': Path(video_path).name
        },
        'extraction_settings': {
            'target_frame_rate': frame_rate,
            'frame_interval': frame_interval
        },
        'extracted_frames': extracted_frames,
        'total_frames_extracted': saved_count
    }
    
    metadata_path = participant_dir / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Extracted {saved_count} frames")
    
    return metadata

def main():
    """Extract frames from all working videos"""
    
    # Load working videos list
    with open('/home/rohan/Multimodal/multimodal_video_ml/data/video_test_results.json', 'r') as f:
        results = json.load(f)
    
    working_videos = results['working_videos']
    output_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/comprehensive_frames"
    
    print(f"🚀 Extracting frames from {len(working_videos)} working videos...")
    print(f"Output directory: {output_dir}")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    extraction_results = []
    total_frames = 0
    
    for i, video_info in enumerate(working_videos, 1):
        try:
            print(f"\n[{i}/{len(working_videos)}] Processing...")
            metadata = extract_frames_from_video(video_info, output_dir, frame_rate=1.0)
            extraction_results.append(metadata)
            total_frames += metadata['total_frames_extracted']
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'timestamp': timestamp,
        'total_videos_processed': len(working_videos),
        'successful_extractions': len(extraction_results),
        'total_frames_extracted': total_frames,
        'participants': [r['participant_id'] for r in extraction_results],
        'extraction_details': extraction_results
    }
    
    summary_file = Path(output_dir) / f"extraction_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Frame extraction complete!")
    print(f"   Participants: {len(extraction_results)}")
    print(f"   Total frames: {total_frames}")
    print(f"   Average per participant: {total_frames/len(extraction_results):.1f}")
    print(f"   Summary: {summary_file}")

if __name__ == "__main__":
    main()