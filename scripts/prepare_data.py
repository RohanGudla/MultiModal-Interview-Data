#!/usr/bin/env python3
"""
Data preparation script for multimodal emotion recognition.
Run this first to process videos and create train/val/test splits.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from data.data_loader import DataLoader

def main():
    """Run the complete data preparation pipeline."""
    print("=" * 60)
    print("MULTIMODAL EMOTION RECOGNITION - DATA PREPARATION")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Run full pipeline
    try:
        results = data_loader.run_full_pipeline(force_reprocess=False)
        
        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Processed {len(results['processed_videos'])} videos")
        print(f"✅ Created annotations for {len(results['annotations'])} participants")
        
        # Print split summary
        for split_name, data in results['splits'].items():
            total_frames = sum(item['num_frames'] for item in data)
            print(f"✅ {split_name.title()} split: {len(data)} participants, {total_frames:,} frames")
            
        print("\nNext steps:")
        print("1. Run: python scripts/train_model.py --model cnn_simple")
        print("2. Run: python scripts/train_model.py --model vit_scratch")
        print("3. Run: python scripts/train_model.py --model resnet_pretrained")
        print("4. Run: python scripts/train_model.py --model vit_pretrained")
        
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()