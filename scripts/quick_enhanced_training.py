#!/usr/bin/env python3
"""
Quick Enhanced Training Test - 2 epochs only
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from training.enhanced_trainer import EnhancedMultiLabelTrainer

def main():
    """Run a quick enhanced training test"""
    
    print("🚀 Quick Enhanced Training Test")
    print("=" * 50)
    
    # Configuration
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    output_dir = "/home/rohan/Multimodal/multimodal_video_ml/outputs/quick_test"
    
    # Create trainer with minimal settings for testing
    trainer = EnhancedMultiLabelTrainer(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        model_type='vit',
        sequence_length=1,     # Single frame mode to avoid tensor issues
        batch_size=2,          # Small batch
        learning_rate=0.001,
        num_epochs=2,          # Just 2 epochs for testing
        device='auto'
    )
    
    try:
        # Test data setup
        print("\n1. Testing data setup...")
        trainer.setup_data()
        print(f"✅ Data setup successful")
        
        # Test model setup
        print("\n2. Testing model setup...")
        trainer.setup_model()
        print(f"✅ Model setup successful")
        
        # Test one training epoch
        print("\n3. Testing training epoch...")
        trainer.model.train()
        train_loss, train_acc = trainer.train_epoch()
        print(f"✅ Training epoch successful: loss={train_loss:.4f}, acc={train_acc:.4f}")
        
        # Test one validation epoch
        print("\n4. Testing validation epoch...")
        val_loss, val_acc = trainer.validate_epoch()
        print(f"✅ Validation epoch successful: loss={val_loss:.4f}, acc={val_acc:.4f}")
        
        # Test evaluation
        print("\n5. Testing evaluation...")
        results = trainer.evaluate()
        print(f"✅ Evaluation successful: F1={results['overall_metrics']['f1']:.4f}")
        
        # Test verification outputs
        print("\n6. Testing verification outputs...")
        trainer.verifier.create_verification_outputs(
            results['predictions'],
            results['labels'],
            results['participants'],
            results['frame_ids'],
            trainer.feature_info['all']
        )
        print(f"✅ Verification outputs created")
        
        print(f"\n🎉 All components working! Ready for full training.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Quick test completed successfully!")
    else:
        print(f"\n❌ Quick test failed!")
        sys.exit(1)