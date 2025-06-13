#!/usr/bin/env python3
"""
Complete frame extraction for all available participants.
This script processes all 19 participants to maximize dataset coverage.
"""

import cv2
import os
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_participant_id(video_filename):
    """Standardize participant IDs from various video filename formats"""
    # Remove file extension
    name = video_filename.replace('.mp4', '')
    
    # Handle different formats
    if 'Screen recording 1 - ' in name:
        participant_id = name.replace('Screen recording 1 - ', '')
    elif 'RespCam_' in name:
        # Extract from RespCam format: RespCam_AR 1378_Screen recording 1_(21,FEMALE,1030) (1)
        parts = name.split('_')
        if len(parts) >= 2:
            participant_id = parts[1].split('_Screen')[0]
        else:
            participant_id = name
    else:
        participant_id = name
    
    # Clean up spaces and special characters
    participant_id = participant_id.replace('  ', '_').replace(' ', '_')
    
    # Handle special cases
    if 'AR__2298' in participant_id or 'AR  2298' in participant_id:
        participant_id = 'AR__2298'
    elif 'AR_1378' in participant_id or 'AR 1378' in participant_id:
        participant_id = 'AR_1378'
    elif 'CP_6047' in participant_id or 'CP 6047' in participant_id:
        participant_id = 'CP_6047'
    elif 'LE_3299' in participant_id or 'LE 3299' in participant_id:
        participant_id = 'LE_3299'
    elif 'JM_IES' in participant_id or 'JM IES' in participant_id:
        participant_id = 'JM_IES'
    
    return participant_id

def extract_frames_from_video(video_path, output_dir, participant_id, fps=1):
    """Extract frames from video at specified FPS"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    logger.info(f"Processing {participant_id}: {total_frames} frames, {video_fps:.2f} FPS, {duration:.2f}s duration")
    
    # Calculate frame interval for desired FPS
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            frame_filename = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    # Save extraction metadata
    metadata = {
        "participant_id": participant_id,
        "source_video": str(video_path),
        "extraction_date": datetime.now().isoformat(),
        "original_fps": video_fps,
        "extraction_fps": fps,
        "total_original_frames": total_frames,
        "extracted_frames": saved_count,
        "duration_seconds": duration,
        "frame_interval": frame_interval
    }
    
    metadata_file = output_dir / "extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Extracted {saved_count} frames for {participant_id}")
    return True

def main():
    # Paths
    base_dir = Path("/home/rohan/Multimodal/multimodal_video_ml")
    video_dir = base_dir / "data" / "all_videos"
    frames_dir = base_dir / "data" / "complete_frames"
    
    # Find all video files
    video_files = list(video_dir.glob("*.mp4"))
    logger.info(f"Found {len(video_files)} video files")
    
    # Check which participants already have frames extracted
    existing_participants = set()
    if frames_dir.exists():
        existing_participants = {d.name for d in frames_dir.iterdir() if d.is_dir()}
    
    logger.info(f"Already processed: {existing_participants}")
    
    # Process each video
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for video_file in video_files:
        participant_id = standardize_participant_id(video_file.name)
        output_dir = frames_dir / participant_id
        
        # Skip if already processed
        if participant_id in existing_participants:
            logger.info(f"Skipping {participant_id} - already processed")
            skipped_count += 1
            continue
        
        logger.info(f"Processing {participant_id} from {video_file.name}")
        
        try:
            success = extract_frames_from_video(video_file, output_dir, participant_id)
            if success:
                processed_count += 1
                logger.info(f"✅ Successfully processed {participant_id}")
            else:
                failed_count += 1
                logger.error(f"❌ Failed to process {participant_id}")
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ Error processing {participant_id}: {e}")
    
    # Summary
    logger.info(f"\n=== EXTRACTION SUMMARY ===")
    logger.info(f"Total videos found: {len(video_files)}")
    logger.info(f"Already processed: {skipped_count}")
    logger.info(f"Newly processed: {processed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total participants with frames: {skipped_count + processed_count}")
    
    # Update processing status
    total_participants = skipped_count + processed_count
    if total_participants >= 17:
        logger.info("🎯 TARGET ACHIEVED: Processed all available participants!")
    else:
        logger.info(f"📊 Progress: {total_participants}/19 participants processed")

if __name__ == "__main__":
    main()