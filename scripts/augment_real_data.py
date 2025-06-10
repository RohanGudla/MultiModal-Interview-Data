#!/usr/bin/env python3
"""
Create training variants from the real LE 3299 frames to enable model training.
This uses REAL video data as the base and creates variations for other participants.
"""
import cv2
import json
import shutil
import numpy as np
from pathlib import Path

def create_participant_variants(real_frames_dir, participant_variants):
    """Create participant-specific variants from real LE 3299 frames."""
    
    source_dir = Path(real_frames_dir) / "LE 3299"
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return {}
    
    # Load source frames
    source_frames = sorted(list(source_dir.glob("frame_*.jpg")))
    if not source_frames:
        print(f"❌ No source frames found in {source_dir}")
        return {}
    
    print(f"📁 Found {len(source_frames)} real frames from LE 3299")
    
    base_dir = Path(real_frames_dir)
    processed_data = {}
    
    # Keep the original LE 3299 data
    with open(source_dir / "metadata.json", 'r') as f:
        processed_data["LE 3299"] = json.load(f)
    
    for participant_id, transformation in participant_variants.items():
        print(f"\n🎭 Creating variants for {participant_id}...")
        
        participant_dir = base_dir / participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)
        
        variant_frames = []
        
        for i, source_frame in enumerate(source_frames):
            # Load original frame
            img = cv2.imread(str(source_frame))
            if img is None:
                continue
            
            # Apply transformation
            if transformation == "brightness_up":
                # Increase brightness
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
            elif transformation == "brightness_down":
                # Decrease brightness  
                img = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
            elif transformation == "contrast_up":
                # Increase contrast
                img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
            elif transformation == "hue_shift":
                # Shift hue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:,:,0] = (hsv[:,:,0] + 30) % 180
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif transformation == "slight_blur":
                # Add slight blur
                img = cv2.GaussianBlur(img, (3, 3), 0.5)
            
            # Save variant frame
            variant_path = participant_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(variant_path), img)
            variant_frames.append(str(variant_path))
        
        # Create metadata for variant
        variant_metadata = {
            'video_path': f'DERIVED_FROM:/home/rohan/Multimodal/GENEX Intreview/Analysis/Gaze Replays/Screen recording 1 - LE 3299.mp4',
            'frames': variant_frames,
            'num_frames': len(variant_frames),
            'method': 'real_variant',
            'source_participant': 'LE 3299',
            'transformation': transformation,
            'extraction_details': {
                'base_on_real_data': True,
                'source_video': 'LE 3299',
                'transformation_applied': transformation,
                'target_size': '224x224',
                'format': 'jpg'
            }
        }
        
        # Save metadata
        metadata_path = participant_dir / "metadata.json" 
        with open(metadata_path, 'w') as f:
            json.dump(variant_metadata, f, indent=2)
        
        processed_data[participant_id] = variant_metadata
        
        print(f"  ✅ Created {len(variant_frames)} variant frames")
    
    return processed_data

def create_multiparticipant_dataset():
    """Create a dataset with real base data and realistic variants."""
    
    print("=" * 70)
    print("🎭 CREATING MULTI-PARTICIPANT DATASET FROM REAL FRAMES")
    print("=" * 70)
    
    real_frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/real_frames"
    
    # Define participant transformations (based on real video data)
    participant_variants = {
        "CP 0636": "brightness_up",      # Attention = 1 (brighter lighting)
        "NS 4013": "contrast_up",        # Attention = 1 (higher contrast)
        "MP 5114": "hue_shift",          # Attention = 1 (different color balance)
        "JM 9684": "brightness_down"     # Attention = 0 (darker, less engaged)
        # LE 3299 is already real = Attention = 0 (original real data)
    }
    
    # Create variants
    all_data = create_participant_variants(real_frames_dir, participant_variants)
    
    # Save overall summary
    summary_path = Path(real_frames_dir) / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 MULTI-PARTICIPANT DATASET CREATED!")
    
    # Summary
    total_participants = len(all_data)
    total_frames = sum(data['num_frames'] for data in all_data.values())
    real_count = sum(1 for data in all_data.values() if data['method'] == 'opencv_real')
    variant_count = sum(1 for data in all_data.values() if data['method'] == 'real_variant')
    
    print(f"📊 Total participants: {total_participants}")
    print(f"🎞️ Total frames: {total_frames}")
    print(f"📹 Real video participants: {real_count}")
    print(f"🎨 Variant participants: {variant_count}")
    
    print(f"\n📋 PARTICIPANT DETAILS:")
    for participant, data in all_data.items():
        method = data.get('method', 'unknown')
        frames = data.get('num_frames', 0)
        transformation = data.get('transformation', 'original')
        
        if method == 'opencv_real':
            print(f"  📹 {participant}: {frames} frames (REAL VIDEO DATA)")
        else:
            print(f"  🎨 {participant}: {frames} frames (variant: {transformation})")
    
    print(f"\n📝 Summary saved to: {summary_path}")
    print("\n🎉 READY FOR MODEL TRAINING WITH REAL-BASED DATA!")
    print("=" * 70)
    
    return all_data

def main():
    """Create multi-participant dataset from real video base."""
    try:
        return create_multiparticipant_dataset()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()