#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script
Analyzes all available videos, annotations, and temporal structure
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import cv2
from datetime import datetime

class DatasetAnalyzer:
    def __init__(self, base_path="/home/rohan/Multimodal"):
        self.base_path = Path(base_path)
        self.analysis_results = {}
        
    def discover_videos(self):
        """Discover all video files in the dataset"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        videos = []
        
        for ext in video_extensions:
            videos.extend(list(self.base_path.rglob(f"*{ext}")))
        
        print(f"📹 Found {len(videos)} video files:")
        for video in videos:
            print(f"  - {video}")
        
        self.analysis_results['videos'] = [str(v) for v in videos]
        return videos
    
    def analyze_video_properties(self, videos):
        """Analyze properties of each video file"""
        video_info = []
        
        for video_path in videos:
            try:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    info = {
                        'path': str(video_path),
                        'participant_id': self.extract_participant_id(video_path),
                        'fps': fps,
                        'frame_count': frame_count,
                        'duration_seconds': duration,
                        'resolution': f"{width}x{height}",
                        'file_size_mb': video_path.stat().st_size / (1024*1024)
                    }
                    video_info.append(info)
                    print(f"✅ {video_path.name}: {duration:.1f}s, {fps:.1f}fps, {frame_count} frames")
                else:
                    print(f"❌ Could not open: {video_path}")
                cap.release()
            except Exception as e:
                print(f"❌ Error analyzing {video_path}: {e}")
        
        self.analysis_results['video_properties'] = video_info
        return video_info
    
    def extract_participant_id(self, video_path):
        """Extract participant ID from video filename"""
        filename = video_path.name
        # Handle different naming patterns
        if " - " in filename:
            return filename.split(" - ")[-1].replace(".mp4", "")
        return filename.replace(".mp4", "")
    
    def analyze_annotations(self):
        """Analyze all annotation files and their structure"""
        annotation_path = self.base_path / "multimodal_video_ml" / "data" / "annotations"
        
        analysis = {
            'physical_features': {},
            'emotional_targets': {},
            'participants': set()
        }
        
        # Analyze physical features
        physical_dir = annotation_path / "physical_features"
        if physical_dir.exists():
            for file in physical_dir.glob("*.csv"):
                participant_id = file.stem.replace("_physical", "")
                analysis['participants'].add(participant_id)
                
                df = pd.read_csv(file)
                analysis['physical_features'][participant_id] = {
                    'file': str(file),
                    'num_frames': len(df),
                    'features': list(df.columns[1:]),  # Skip frame_id
                    'num_features': len(df.columns) - 1
                }
        
        # Analyze emotional targets
        emotional_dir = annotation_path / "emotional_targets"
        if emotional_dir.exists():
            for file in emotional_dir.glob("*.csv"):
                participant_id = file.stem.replace("_emotional", "")
                analysis['participants'].add(participant_id)
                
                df = pd.read_csv(file)
                analysis['emotional_targets'][participant_id] = {
                    'file': str(file),
                    'num_frames': len(df),
                    'features': list(df.columns[1:]),  # Skip frame_id
                    'num_features': len(df.columns) - 1
                }
        
        analysis['participants'] = list(analysis['participants'])
        
        print(f"\n📊 Annotation Analysis:")
        print(f"  Participants with annotations: {len(analysis['participants'])}")
        print(f"  Participants: {analysis['participants']}")
        
        if analysis['physical_features']:
            sample_p = list(analysis['physical_features'].keys())[0]
            print(f"  Physical features per participant: {analysis['physical_features'][sample_p]['num_features']}")
            
        if analysis['emotional_targets']:
            sample_e = list(analysis['emotional_targets'].keys())[0]
            print(f"  Emotional features per participant: {analysis['emotional_targets'][sample_e]['num_features']}")
        
        self.analysis_results['annotations'] = analysis
        return analysis
    
    def get_all_annotation_features(self):
        """Get comprehensive list of all annotation features"""
        annotations = self.analysis_results.get('annotations', {})
        
        all_features = {
            'physical': [],
            'emotional': [],
            'combined': []
        }
        
        # Get physical features
        if annotations.get('physical_features'):
            sample_participant = list(annotations['physical_features'].keys())[0]
            all_features['physical'] = annotations['physical_features'][sample_participant]['features']
        
        # Get emotional features  
        if annotations.get('emotional_targets'):
            sample_participant = list(annotations['emotional_targets'].keys())[0]
            all_features['emotional'] = annotations['emotional_targets'][sample_participant]['features']
        
        # Combine all features
        all_features['combined'] = all_features['physical'] + all_features['emotional']
        
        print(f"\n🎯 All Available Features ({len(all_features['combined'])} total):")
        print("Physical Features:")
        for i, feat in enumerate(all_features['physical'][:10]):  # Show first 10
            print(f"  {i+1:2d}. {feat}")
        if len(all_features['physical']) > 10:
            print(f"  ... and {len(all_features['physical'])-10} more")
            
        print("Emotional Features:")
        for i, feat in enumerate(all_features['emotional']):
            print(f"  {i+1:2d}. {feat}")
        
        self.analysis_results['all_features'] = all_features
        return all_features
    
    def check_temporal_annotations(self):
        """Check if annotations have temporal information (start/stop times)"""
        annotation_path = self.base_path / "multimodal_video_ml" / "data" / "annotations"
        
        # Sample one annotation file to check structure
        physical_files = list((annotation_path / "physical_features").glob("*.csv"))
        if physical_files:
            df = pd.read_csv(physical_files[0])
            
            print(f"\n⏰ Temporal Structure Analysis:")
            print(f"  Frame-based annotations: {len(df)} frames")
            print(f"  Frame IDs range: {df['frame_id'].min()} to {df['frame_id'].max()}")
            
            # Check if this represents sequential time
            frame_rate = 1.0  # Current extraction is 1 FPS
            duration_seconds = len(df) * frame_rate
            print(f"  Estimated video duration: {duration_seconds:.1f} seconds")
            
            # Check for any time-based columns
            time_columns = [col for col in df.columns if any(word in col.lower() 
                          for word in ['time', 'start', 'stop', 'duration', 'timestamp'])]
            print(f"  Time-based columns found: {time_columns}")
            
            self.analysis_results['temporal_structure'] = {
                'frame_based': True,
                'num_frames': len(df),
                'frame_rate': frame_rate,
                'duration_seconds': duration_seconds,
                'time_columns': time_columns
            }
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'analysis_timestamp': timestamp,
            'dataset_summary': {
                'total_videos': len(self.analysis_results.get('videos', [])),
                'total_participants': len(self.analysis_results.get('annotations', {}).get('participants', [])),
                'total_features': len(self.analysis_results.get('all_features', {}).get('combined', [])),
                'physical_features': len(self.analysis_results.get('all_features', {}).get('physical', [])),
                'emotional_features': len(self.analysis_results.get('all_features', {}).get('emotional', []))
            },
            'detailed_results': self.analysis_results
        }
        
        # Save to file
        output_file = self.base_path / "multimodal_video_ml" / f"dataset_analysis_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Dataset Summary:")
        print(f"  Total Videos: {summary['dataset_summary']['total_videos']}")
        print(f"  Participants with Annotations: {summary['dataset_summary']['total_participants']}")
        print(f"  Total Annotation Features: {summary['dataset_summary']['total_features']}")
        print(f"    - Physical: {summary['dataset_summary']['physical_features']}")
        print(f"    - Emotional: {summary['dataset_summary']['emotional_features']}")
        print(f"\n📁 Full analysis saved to: {output_file}")
        
        return summary

def main():
    print("🔍 Starting Comprehensive Dataset Analysis...")
    
    analyzer = DatasetAnalyzer()
    
    # Step 1: Discover all videos
    print("\n" + "="*60)
    print("STEP 1: Video Discovery")
    print("="*60)
    videos = analyzer.discover_videos()
    
    # Step 2: Analyze video properties
    print("\n" + "="*60)
    print("STEP 2: Video Properties Analysis")
    print("="*60)
    video_info = analyzer.analyze_video_properties(videos)
    
    # Step 3: Analyze annotations
    print("\n" + "="*60)
    print("STEP 3: Annotation Analysis")
    print("="*60)
    annotations = analyzer.analyze_annotations()
    
    # Step 4: Get all features
    print("\n" + "="*60)
    print("STEP 4: Feature Enumeration")
    print("="*60)
    features = analyzer.get_all_annotation_features()
    
    # Step 5: Check temporal structure
    print("\n" + "="*60)
    print("STEP 5: Temporal Structure Analysis")
    print("="*60)
    analyzer.check_temporal_annotations()
    
    # Step 6: Generate summary
    print("\n" + "="*60)
    print("STEP 6: Summary Report Generation")
    print("="*60)
    summary = analyzer.generate_summary_report()
    
    print("\n🎉 Analysis Complete!")
    
    return analyzer, summary

if __name__ == "__main__":
    analyzer, summary = main()