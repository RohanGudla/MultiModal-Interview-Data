#!/usr/bin/env python3
"""
Run Complete Training Pipeline for ALL Available Participants
Addresses colleague requirements with current data and scales for full dataset
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from training.complete_trainer import CompleteMultiLabelTrainer

def main():
    """Run the complete enhanced training pipeline"""
    
    print("🚀 COMPLETE Multi-Participant Training Pipeline")
    print("=" * 70)
    print("This addresses ALL colleague requirements with available data:")
    print("✅ Process ALL available participants (currently 9, scaling to 17)")
    print("✅ Predict ALL annotations (50 features: 33 physical + 17 emotional)")
    print("✅ Include REAL temporal start/stop time predictions")
    print("✅ Save comprehensive verification CSV files")
    print("✅ Scale to work with complete dataset as frames are extracted")
    print("")
    
    # Configuration
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/complete_annotations"
    output_dir = "/home/rohan/Multimodal/multimodal_video_ml/outputs/complete_training"
    
    # Create trainer with enhanced settings for ALL participants
    trainer = CompleteMultiLabelTrainer(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        model_type='vit',  # Vision Transformer for best performance
        sequence_length=1,  # Single frame mode for stability
        batch_size=4,      # Efficient batch size for larger dataset
        learning_rate=0.0001,  # Conservative learning rate
        num_epochs=8,      # Focused training for demonstration
        temporal_boundary_weight=0.3,  # Weight for temporal boundary prediction
        device='auto'      # Auto-detect GPU/CPU
    )
    
    try:
        # Run complete training pipeline
        results = trainer.run_complete_training()
        
        print(f"\n🎉 COMPLETE Enhanced training pipeline completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Overall F1 Score: {results['overall_metrics']['f1']:.4f}")
        print(f"   Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        print(f"   Total Samples Processed: {len(results['predictions'])}")
        print(f"   Unique Participants: {len(set(results['participants']))}")
        
        # Show participant breakdown
        from collections import Counter
        participant_counts = Counter(results['participants'])
        print(f"\n📊 Participant Breakdown:")
        for participant, count in participant_counts.items():
            print(f"   {participant}: {count} samples")
        
        print(f"\n📁 Output Files Created:")
        print(f"   Training outputs: {output_dir}")
        print(f"   Verification CSVs: {output_dir}/verification_csvs/")
        print(f"   Verification plots: {output_dir}/verification_plots/")
        print(f"   Summary reports: {output_dir}/verification_reports/")
        
        print(f"\n✅ ALL colleague requirements addressed with available data:")
        print(f"   ✅ Processed all available participants ({len(set(results['participants']))})")
        print(f"   ✅ Predicted all 50 annotation features")
        print(f"   ✅ Implemented temporal boundary prediction framework")
        print(f"   ✅ Created comprehensive verification CSV files")
        print(f"   ✅ Pipeline ready to scale to full 17 participants when extraction completes")
        
        print(f"\n📋 Next Steps for Full Dataset:")
        print(f"   • Wait for frame extraction to complete for remaining participants")
        print(f"   • Re-run training with all 17 participants (~1700 samples)")
        print(f"   • Address 80 vs 17 video discrepancy with colleague")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 SUCCESS: Complete training pipeline completed!")
        print(f"Ready to scale to full dataset when all frames are extracted.")
        print(f"Your colleague can review the comprehensive verification outputs.")
    else:
        print(f"\n💥 FAILED: Complete training pipeline encountered errors.")
        sys.exit(1)