#!/usr/bin/env python3
"""
Extract REAL frames from GENEX video files using OpenCV.
This replaces the synthetic fallback frames with actual video content.
"""
import cv2
import json
import shutil
from pathlib import Path
import numpy as np

def extract_real_frames_opencv(video_path, output_dir, max_frames=20, frame_interval=30):
    """Extract real frames using OpenCV."""
    output_dir = Path(output_dir)
    
    # Remove existing synthetic frames
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video: {video_path.name}")
    print(f"  📊 FPS: {fps:.2f}")
    print(f"  ⏱️ Duration: {duration:.2f}s")
    print(f"  🎞️ Total frames: {total_frames}")
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Extract every Nth frame based on interval
        if frame_count % frame_interval == 0:
            # Resize frame to 224x224 for consistency
            frame_resized = cv2.resize(frame, (224, 224))
            
            # Save frame
            frame_filename = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame_resized)
            
            extracted_frames.append(str(frame_filename))
            saved_count += 1
            
            print(f"  ✅ Extracted frame {saved_count}/{max_frames} at time {frame_count/fps:.2f}s")
        
        frame_count += 1
    
    cap.release()
    
    print(f"  🎯 Extracted {len(extracted_frames)} real frames")
    return extracted_frames

def process_all_genex_videos():
    """Process all GENEX video files and extract real frames."""
    
    print("=" * 70)
    print("🎬 EXTRACTING REAL FRAMES FROM GENEX VIDEOS")
    print("=" * 70)
    
    # Video paths
    video_dir = Path("/home/rohan/Multimodal/GENEX Intreview/Analysis/Gaze Replays")
    output_base = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames")
    
    # Remove all existing synthetic data
    if output_base.exists():
        print("🗑️ Removing existing synthetic frames...")
        shutil.rmtree(output_base)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob("*.mp4"))
    print(f"📁 Found {len(video_files)} video files")
    
    processed_data = {}
    
    for video_path in video_files:
        # Extract participant ID from filename
        filename = video_path.name
        if "CP 0636" in filename:
            participant_id = "CP 0636"
        elif "JM 9684" in filename:
            participant_id = "JM 9684"
        elif "LE 3299" in filename:
            participant_id = "LE 3299"
        elif "MP 5114" in filename:
            participant_id = "MP 5114"
        elif "NS 4013" in filename:
            participant_id = "NS 4013"
        else:
            print(f"⚠️ Unknown participant in {filename}")
            continue
        
        print(f"\n🎭 Processing {participant_id}...")
        
        participant_dir = output_base / participant_id
        
        # Extract real frames
        frames = extract_real_frames_opencv(
            video_path, 
            participant_dir, 
            max_frames=20, 
            frame_interval=30  # Extract every 30th frame
        )
        
        if frames:
            processed_data[participant_id] = {
                'video_path': str(video_path),
                'frames': frames,
                'num_frames': len(frames),
                'method': 'opencv_real',  # Mark as REAL frames
                'extraction_details': {
                    'frame_interval': 30,
                    'target_size': '224x224',
                    'format': 'jpg'
                }
            }
            
            # Save metadata
            metadata_path = participant_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(processed_data[participant_id], f, indent=2)
            
            print(f"  ✅ {participant_id}: {len(frames)} REAL frames extracted")
        else:
            print(f"  ❌ {participant_id}: Failed to extract frames")
    
    # Save overall summary
    summary_path = output_base / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 REAL FRAME EXTRACTION COMPLETED!")
    print(f"📊 Processed {len(processed_data)} participants")
    total_frames = sum(data['num_frames'] for data in processed_data.values())
    print(f"🎞️ Total REAL frames extracted: {total_frames}")
    print(f"📝 Summary saved to: {summary_path}")
    print("=" * 70)
    
    return processed_data

def verify_real_frames():
    """Verify that we have real frames, not synthetic."""
    frame_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames")
    summary_path = frame_dir / "processing_summary.json"
    
    if not summary_path.exists():
        print("❌ No processing summary found")
        return False
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n🔍 VERIFICATION:")
    all_real = True
    for participant, data in summary.items():
        method = data.get('method', 'unknown')
        num_frames = data.get('num_frames', 0)
        
        if method == 'opencv_real':
            print(f"  ✅ {participant}: {num_frames} REAL frames")
        else:
            print(f"  ❌ {participant}: {num_frames} frames (method: {method})")
            all_real = False
    
    if all_real:
        print(f"🎉 SUCCESS: All frames are REAL video data!")
    else:
        print(f"⚠️ WARNING: Some frames may still be synthetic")
    
    return all_real

def main():
    """Extract real frames from GENEX videos."""
    try:
        # Extract real frames
        processed_data = process_all_genex_videos()
        
        # Verify extraction
        verify_real_frames()
        
        return processed_data
        
    except Exception as e:
        print(f"\n❌ Error during real frame extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()