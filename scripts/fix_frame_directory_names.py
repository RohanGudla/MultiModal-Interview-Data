#!/usr/bin/env python3
"""
Fix Frame Directory Names to Match Annotation Naming Convention
Rename directories from "AM 1355" to "AM_1355" format
"""

import os
import shutil
from pathlib import Path

def fix_frame_directory_names():
    """Rename frame directories to match annotation naming convention"""
    
    frames_dir = Path("/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames")
    
    print("🔧 Fixing Frame Directory Names")
    print("=" * 50)
    
    # Get all directories in frames_dir
    directories = [d for d in frames_dir.iterdir() if d.is_dir()]
    
    renamed_count = 0
    
    for directory in directories:
        current_name = directory.name
        
        # Skip files (like extraction_log_*.json)
        if '.' in current_name:
            continue
            
        # Replace spaces with underscores
        new_name = current_name.replace(' ', '_')
        
        if new_name != current_name:
            old_path = directory
            new_path = frames_dir / new_name
            
            print(f"📁 Renaming: '{current_name}' → '{new_name}'")
            
            try:
                # Rename the directory
                old_path.rename(new_path)
                renamed_count += 1
                print(f"   ✅ Success")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        else:
            print(f"📁 Already correct: '{current_name}'")
    
    print(f"\n🎉 Directory renaming complete!")
    print(f"   Directories renamed: {renamed_count}")
    
    # List final directory structure
    print(f"\nFinal directory structure:")
    final_dirs = [d.name for d in frames_dir.iterdir() if d.is_dir()]
    for i, dir_name in enumerate(sorted(final_dirs), 1):
        print(f"   {i:2d}. {dir_name}")

if __name__ == "__main__":
    fix_frame_directory_names()