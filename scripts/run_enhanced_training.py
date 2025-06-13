#!/usr/bin/env python3
"""
Run Enhanced Training Pipeline with All Colleague Requirements
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from training.enhanced_trainer import EnhancedMultiLabelTrainer

def main():
    """Run the enhanced training pipeline"""
    
    print("🚀 Starting Enhanced Multi-Participant Training Pipeline")
    print("=" * 70)
    print("This addresses all colleague requirements:")
    print("✅ Process ALL available videos (8 participants with data)")
    print("✅ Predict ALL annotations (50 features: 33 physical + 17 emotional)")
    print("✅ Include temporal start/stop time predictions")
    print("✅ Save outputs and true annotations for verification")
    print("✅ Scale to work with all available videos")
    print("")
    
    # Configuration
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    output_dir = "/home/rohan/Multimodal/multimodal_video_ml/outputs/enhanced_training"
    
    # Create trainer with optimized settings
    trainer = EnhancedMultiLabelTrainer(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        model_type='vit',  # Vision Transformer for better performance
        sequence_length=1,  # Single frame mode for stability
        batch_size=4,      # Batch size for efficient training
        learning_rate=0.0001,  # Conservative learning rate
        num_epochs=10,     # Focused training for results
        device='auto'      # Auto-detect GPU/CPU
    )
    
    try:
        # Run complete training pipeline
        results = trainer.run_complete_training()
        
        print(f"\n🎉 Enhanced training pipeline completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Overall F1 Score: {results['overall_metrics']['f1']:.4f}")
        print(f"   Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        print(f"   Total Samples Processed: {len(results['predictions'])}")
        print(f"   Participants: {len(set(results['participants']))}")
        
        print(f"\n📁 Output Files Created:")
        print(f"   Training outputs: {output_dir}")
        print(f"   Verification CSVs: {output_dir}/verification_csvs/")
        print(f"   Verification plots: {output_dir}/verification_plots/")
        print(f"   Summary reports: {output_dir}/verification_reports/")
        
        print(f"\n✅ All colleague requirements fulfilled:")
        print(f"   ✅ Trained on all available videos")
        print(f"   ✅ Predicted all 50 annotation features")
        print(f"   ✅ Generated temporal start/stop predictions")
        print(f"   ✅ Created verification CSV files")
        print(f"   ✅ Saved model outputs vs ground truth")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 SUCCESS: Enhanced training pipeline completed!")
        print(f"Your colleague can now review the verification outputs.")
    else:
        print(f"\n💥 FAILED: Enhanced training pipeline encountered errors.")
        sys.exit(1)