#!/usr/bin/env python3
"""
Extract frames from ALL 17 participants in All Screens videos
Addresses colleague requirement to process ALL available videos
"""

import cv2
import os
import json
import time
from pathlib import Path
from datetime import datetime

def extract_frames_from_video(video_path, output_dir, participant_id, fps=1.0):
    """Extract frames from a single video at specified FPS"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"📹 Processing {participant_id}:")
    print(f"   Video FPS: {video_fps:.2f}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Total frames: {total_frames}")
    
    # Calculate frame interval for desired FPS
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    
    extracted_frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:04d}.jpg"
            frame_path = output_dir / frame_filename
            
            # Save frame
            cv2.imwrite(str(frame_path), frame)
            
            # Record metadata
            extracted_frames.append({
                'frame_id': extracted_count,
                'original_frame': frame_count,
                'timestamp': frame_count / video_fps if video_fps > 0 else 0,
                'filename': frame_filename
            })
            
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    
    # Save extraction metadata
    metadata = {
        'participant_id': participant_id,
        'video_path': str(video_path),
        'extraction_date': datetime.now().isoformat(),
        'video_properties': {
            'fps': video_fps,
            'total_frames': total_frames,
            'duration_seconds': duration
        },
        'extraction_settings': {
            'target_fps': fps,
            'frame_interval': frame_interval
        },
        'extracted_frames': extracted_frames,
        'total_extracted': extracted_count
    }
    
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Extracted {extracted_count} frames")
    return True

def main():
    """Extract frames from ALL 17 participants"""
    
    print("🚀 Extracting frames from ALL 17 participants")
    print("=" * 60)
    
    # All Screens video directory
    all_screens_dir = "/home/rohan/Multimodal/extracted_onedrive/MultiModal Interview Data - Chen + Anthony Collab/GENEX Interview/Video/All Screens"
    
    # Output directory
    output_base = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_frames"
    
    # All 17 participants from All Screens
    participants = [
        "AM 1355", "AR  2298", "AR 1378", "AW 8961", "BU 6095", 
        "CP 0636", "CP 6047", "CR 0863", "EV 4492", "JG 8996",
        "JM 9684", "JM IES", "JR 4166", "KW 9939", "LE 3299", 
        "YT 6156", "ZLB 8812"
    ]
    
    successful_extractions = 0
    failed_extractions = 0
    
    for participant in participants:
        # Clean participant ID for directory name
        participant_clean = participant.replace(" ", "_").replace("  ", "_")
        
        # Video file path
        video_file = f"Screen recording 1 - {participant}.mp4"
        video_path = Path(all_screens_dir) / video_file
        
        # Output directory for this participant
        participant_output = Path(output_base) / participant_clean
        
        if video_path.exists():
            print(f"\n📹 Processing participant: {participant}")
            if extract_frames_from_video(video_path, participant_output, participant_clean):
                successful_extractions += 1
            else:
                failed_extractions += 1
        else:
            print(f"\n⚠️ Video not found: {video_path}")
            failed_extractions += 1
    
    print(f"\n🎉 Frame extraction complete!")
    print(f"   ✅ Successful: {successful_extractions}")
    print(f"   ❌ Failed: {failed_extractions}")
    print(f"   📁 Output directory: {output_base}")
    
    # Create summary
    summary = {
        'extraction_date': datetime.now().isoformat(),
        'total_participants': len(participants),
        'successful_extractions': successful_extractions,
        'failed_extractions': failed_extractions,
        'output_directory': output_base,
        'participants_processed': participants
    }
    
    summary_path = Path(output_base) / "extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()