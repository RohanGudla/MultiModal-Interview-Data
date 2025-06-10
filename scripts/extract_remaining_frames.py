#!/usr/bin/env python3
"""
Extract frames from remaining GENEX videos with better error handling.
"""
import cv2
import json
from pathlib import Path
import numpy as np

def try_extract_frames(video_path, participant_id, output_base):
    """Try to extract frames with different approaches."""
    
    print(f"\n🎭 Processing {participant_id}...")
    print(f"  📁 File: {video_path.name}")
    print(f"  📊 Size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    
    if video_path.stat().st_size == 0:
        print(f"  ❌ Empty file - skipping")
        return None
    
    participant_dir = output_base / participant_id
    participant_dir.mkdir(parents=True, exist_ok=True)
    
    # Try OpenCV with different backends
    for backend_name, backend in [
        ("Default", cv2.CAP_ANY),
        ("FFmpeg", cv2.CAP_FFMPEG),
        ("GStreamer", cv2.CAP_GSTREAMER)
    ]:
        print(f"  🔄 Trying {backend_name} backend...")
        
        cap = cv2.VideoCapture(str(video_path), backend)
        
        if not cap.isOpened():
            print(f"    ❌ Failed with {backend_name}")
            continue
        
        # Get basic properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"    📹 FPS: {fps}, Frames: {total_frames}, Size: {width}x{height}")
        
        if fps <= 0 or total_frames <= 0:
            print(f"    ⚠️ Invalid video properties")
            cap.release()
            continue
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"    ❌ Cannot read frames")
            cap.release()
            continue
        
        print(f"    ✅ Successfully opened with {backend_name}")
        
        # Extract frames
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        max_frames = 20
        
        # Calculate frame interval based on video length
        frame_interval = max(1, total_frames // max_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                break
                
            # Extract frames at intervals
            if frame_count % frame_interval == 0:
                # Resize frame to 224x224
                frame_resized = cv2.resize(frame, (224, 224))
                
                # Save frame
                frame_filename = participant_dir / f"frame_{saved_count:04d}.jpg"
                success = cv2.imwrite(str(frame_filename), frame_resized)
                
                if success:
                    extracted_frames.append(str(frame_filename))
                    saved_count += 1
                    print(f"    ✅ Frame {saved_count}/{max_frames} at {frame_count/fps:.1f}s")
                else:
                    print(f"    ⚠️ Failed to save frame {saved_count}")
            
            frame_count += 1
            
            # Safety break for very long videos
            if frame_count > total_frames * 1.1:  # 10% buffer
                break
        
        cap.release()
        
        if extracted_frames:
            # Create metadata
            result = {
                'video_path': str(video_path),
                'frames': extracted_frames,
                'num_frames': len(extracted_frames),
                'method': 'opencv_real',
                'backend_used': backend_name,
                'extraction_details': {
                    'frame_interval': frame_interval,
                    'target_size': '224x224',
                    'format': 'jpg',
                    'original_fps': fps,
                    'original_frames': total_frames,
                    'original_size': f"{width}x{height}"
                }
            }
            
            # Save metadata
            metadata_path = participant_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  🎯 SUCCESS: {len(extracted_frames)} REAL frames extracted")
            return result
        else:
            print(f"    ❌ No frames extracted with {backend_name}")
    
    print(f"  ❌ Failed to extract frames from {participant_id}")
    return None

def process_remaining_videos():
    """Process remaining GENEX videos."""
    
    print("=" * 70)
    print("🎬 EXTRACTING FRAMES FROM REMAINING GENEX VIDEOS")
    print("=" * 70)
    
    video_dir = Path("/home/rohan/Multimodal/GENEX Intreview/Analysis/Gaze Replays")
    output_base = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames")
    
    # Load existing results
    summary_path = output_base / "processing_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            processed_data = json.load(f)
        print(f"📋 Found existing data for: {list(processed_data.keys())}")
    else:
        processed_data = {}
    
    # Video files to try
    video_files = [
        ("CP 0636", "Screen recording 1 - CP 0636.mp4"),
        ("JM 9684", "Screen recording 1 - JM 9684.mp4"), 
        ("MP 5114", "Screen recording 1 - MP 5114.mp4"),
        ("NS 4013", "Screen recording 1 - NS 4013.mp4")
    ]
    
    for participant_id, filename in video_files:
        # Skip if already processed
        if participant_id in processed_data:
            print(f"✅ {participant_id} already processed - skipping")
            continue
        
        video_path = video_dir / filename
        if not video_path.exists():
            print(f"❌ {participant_id}: Video file not found")
            continue
        
        result = try_extract_frames(video_path, participant_id, output_base)
        if result:
            processed_data[participant_id] = result
    
    # Save updated summary
    with open(summary_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 PROCESSING COMPLETED!")
    
    # Summary
    total_participants = len(processed_data)
    total_frames = sum(data['num_frames'] for data in processed_data.values())
    
    print(f"📊 Total participants with data: {total_participants}")
    print(f"🎞️ Total REAL frames: {total_frames}")
    
    for participant, data in processed_data.items():
        method = data.get('method', 'unknown')
        backend = data.get('backend_used', 'unknown')
        frames = data.get('num_frames', 0)
        print(f"  ✅ {participant}: {frames} frames ({method}, {backend})")
    
    print(f"📝 Summary saved to: {summary_path}")
    print("=" * 70)
    
    return processed_data

def main():
    """Extract frames from remaining videos."""
    try:
        return process_remaining_videos()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()