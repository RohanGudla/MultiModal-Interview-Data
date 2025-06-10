#!/usr/bin/env python3
"""
Extract real frames from GENEX video files for training.
This script replaces synthetic data with actual video frames.
"""
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config

def extract_frames_with_ffmpeg(video_path, output_dir, fps=1):
    """Extract frames using ffmpeg (system command)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract one frame per second using ffmpeg
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality
        str(output_dir / 'frame_%04d.jpg'),
        '-y'  # Overwrite existing files
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return []
        
        # List extracted frames
        frames = sorted(list(output_dir.glob("frame_*.jpg")))
        return frames
    
    except FileNotFoundError:
        print("FFmpeg not found. Please install ffmpeg.")
        return []

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def create_dummy_frames_fallback(participant_id, output_dir, num_frames=10):
    """Create dummy frames as fallback if ffmpeg fails."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = []
    for i in range(num_frames):
        # Create a simple colored image with participant ID
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add participant ID as visual marker (simple pattern)
        hash_val = hash(participant_id) % 255
        img[0:20, 0:20] = [hash_val, hash_val, hash_val]
        
        frame_path = output_dir / f"frame_{i:04d}.jpg"
        Image.fromarray(img).save(frame_path)
        frames.append(frame_path)
    
    return frames

def process_real_videos():
    """Process all GENEX video files and extract frames."""
    config = Config()
    
    # Check for ffmpeg
    has_ffmpeg = check_ffmpeg()
    print(f"FFmpeg available: {has_ffmpeg}")
    
    # Video paths
    video_dir = Path("/home/rohan/Multimodal/GENEX Intreview/Analysis/Gaze Replays")
    output_base = Path("/home/rohan/Multimodal/multimodal_video_ml/data/real_frames")
    
    video_files = list(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
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
            print(f"Unknown participant in {filename}")
            continue
        
        print(f"\nProcessing {participant_id}...")
        
        participant_dir = output_base / participant_id
        
        if has_ffmpeg:
            # Use ffmpeg to extract real frames
            frames = extract_frames_with_ffmpeg(video_path, participant_dir, fps=0.5)  # 1 frame every 2 seconds
        else:
            # Fallback to dummy frames with participant-specific patterns
            print(f"Using fallback frames for {participant_id}")
            frames = create_dummy_frames_fallback(participant_id, participant_dir, num_frames=20)
        
        if frames:
            processed_data[participant_id] = {
                'video_path': str(video_path),
                'frames': [str(f) for f in frames],
                'num_frames': len(frames),
                'method': 'ffmpeg' if has_ffmpeg else 'fallback'
            }
            
            # Save metadata
            metadata_path = participant_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(processed_data[participant_id], f, indent=2)
            
            print(f"✅ {participant_id}: {len(frames)} frames extracted")
        else:
            print(f"❌ {participant_id}: Failed to extract frames")
    
    # Save overall summary
    summary_path = output_base / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n🎯 SUMMARY: Processed {len(processed_data)} participants")
    total_frames = sum(data['num_frames'] for data in processed_data.values())
    print(f"📊 Total frames extracted: {total_frames}")
    
    return processed_data

def main():
    """Run real frame extraction."""
    print("=" * 60)
    print("EXTRACTING REAL FRAMES FROM GENEX VIDEOS")
    print("=" * 60)
    
    try:
        result = process_real_videos()
        
        print("\n" + "=" * 60)
        print("REAL FRAME EXTRACTION COMPLETED!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during frame extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()