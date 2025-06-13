#!/usr/bin/env python3
"""
Quick System Demonstration
Shows that the entire pipeline works without external dependencies
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

import torch
import numpy as np
from pathlib import Path

def demo_dataset():
    """Demo the multi-label dataset"""
    print("📚 Testing Multi-Label Dataset...")
    
    try:
        from data.multilabel_dataset import MultiLabelAnnotationDataset
        
        frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
        annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
        
        # Test single frame
        dataset = MultiLabelAnnotationDataset(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            participants=["LE_3299"],
            sequence_length=1
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Dataset works! {len(dataset)} samples")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   All labels shape: {sample['all_labels'].shape}")
            print(f"   Physical features: {sample['physical_labels'].shape[0]}")
            print(f"   Emotional features: {sample['emotional_labels'].shape[0]}")
            
            # Test sequence mode
            seq_dataset = MultiLabelAnnotationDataset(
                frames_dir=frames_dir,
                annotations_dir=annotations_dir,
                participants=["LE_3299"],
                sequence_length=5
            )
            
            if len(seq_dataset) > 0:
                seq_sample = seq_dataset[0]
                print(f"✅ Sequence dataset works! {len(seq_dataset)} samples")
                print(f"   Sequence shape: {seq_sample['images'].shape}")
                return True
        
        print("❌ Dataset empty")
        return False
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def demo_models():
    """Demo the temporal models"""
    print("\n🧠 Testing Temporal Models...")
    
    try:
        from models.temporal_multilabel import create_temporal_model
        
        # Test ViT model
        model = create_temporal_model(
            model_type='vit',
            num_physical_features=33,
            num_emotional_features=17,
            sequence_length=5
        )
        
        # Test forward pass
        dummy_input = torch.randn(2, 5, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input, return_temporal_boundaries=True)
        
        print(f"✅ ViT model works!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Physical output: {output['physical_probs'].shape}")
        print(f"   Emotional output: {output['emotional_probs'].shape}")
        print(f"   Boundary output: {output['boundary_probs'].shape}")
        
        # Test ResNet model
        resnet_model = create_temporal_model(
            model_type='resnet',
            num_physical_features=33,
            num_emotional_features=17,
            sequence_length=5
        )
        
        with torch.no_grad():
            resnet_output = resnet_model(dummy_input)
        
        print(f"✅ ResNet model works!")
        print(f"   ResNet physical output: {resnet_output['physical_probs'].shape}")
        print(f"   ResNet emotional output: {resnet_output['emotional_probs'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def demo_verification():
    """Demo the output verification system"""
    print("\n📋 Testing Output Verification...")
    
    try:
        from utils.output_verification import OutputVerificationSystem
        
        # Create dummy feature names
        feature_names = {
            'physical': [f'Physical_Feature_{i}' for i in range(33)],
            'emotional': [f'Emotional_Feature_{i}' for i in range(17)],
            'all': [f'Feature_{i}' for i in range(50)]
        }
        
        # Initialize verification system
        verifier = OutputVerificationSystem(
            output_dir="/tmp/demo_verification",
            feature_names=feature_names,
            save_individual_predictions=True,
            save_aggregated_results=True,
            create_visualizations=False  # Skip matplotlib for demo
        )
        
        # Add dummy predictions
        batch_size = 5
        predictions = {
            'physical_probs': torch.rand(batch_size, 33),
            'emotional_probs': torch.rand(batch_size, 17),
            'combined_probs': torch.rand(batch_size, 50)
        }
        
        ground_truth = {
            'physical_labels': torch.randint(0, 2, (batch_size, 33), dtype=torch.float32),
            'emotional_labels': torch.randint(0, 2, (batch_size, 17), dtype=torch.float32),
            'all_labels': torch.randint(0, 2, (batch_size, 50), dtype=torch.float32)
        }
        
        metadata = {
            'participant_id': [f'P{i}' for i in range(batch_size)],
            'frame_id': torch.arange(batch_size),
            'temporal_info': [{'timestamp_seconds': i * 1.0} for i in range(batch_size)]
        }
        
        verifier.add_batch_predictions(predictions, ground_truth, metadata)
        
        # Generate report (without visualizations)
        metrics = verifier.calculate_performance_metrics()
        pred_file = verifier.save_individual_predictions("demo")
        
        print(f"✅ Verification system works!")
        print(f"   Processed {len(verifier.all_predictions)} samples")
        print(f"   Overall accuracy: {metrics['overall']['accuracy']:.3f}")
        print(f"   Physical mean accuracy: {metrics['physical']['mean_accuracy']:.3f}")
        print(f"   Emotional mean accuracy: {metrics['emotional']['mean_accuracy']:.3f}")
        print(f"   Predictions saved: {pred_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification test failed: {e}")
        return False

def demo_full_pipeline():
    """Demo the complete pipeline with real data"""
    print("\n🚀 Testing Complete Pipeline...")
    
    try:
        from data.multilabel_dataset import create_dataloaders
        from models.temporal_multilabel import create_temporal_model, MultilabelTemporalLoss
        
        frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
        annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
        
        # Create dataloaders
        train_loader, val_loader, test_loader, dataset = create_dataloaders(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            batch_size=4,
            sequence_length=3,
            num_workers=0  # No multiprocessing for demo
        )
        
        print(f"✅ Data pipeline works!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Create model
        feature_info = dataset.get_feature_names()
        model = create_temporal_model(
            model_type='vit',
            num_physical_features=len(feature_info['physical']),
            num_emotional_features=len(feature_info['emotional']),
            sequence_length=3
        )
        
        # Test one training step
        criterion = MultilabelTemporalLoss()
        
        for batch in train_loader:
            images = batch['image'].unsqueeze(1)  # Add sequence dimension
            physical_labels = batch['physical_labels']
            emotional_labels = batch['emotional_labels']
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            targets = {
                'physical_labels': physical_labels,
                'emotional_labels': emotional_labels
            }
            
            losses = criterion(predictions, targets)
            
            print(f"✅ Complete pipeline works!")
            print(f"   Batch size: {images.shape[0]}")
            print(f"   Total loss: {losses['total'].item():.4f}")
            print(f"   Physical loss: {losses['physical'].item():.4f}")
            print(f"   Emotional loss: {losses['emotional'].item():.4f}")
            
            break  # Just test one batch
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run complete system demonstration"""
    
    print("🎯 Comprehensive Multi-Label Video System Demo")
    print("=" * 60)
    
    results = []
    
    # Test components
    results.append(("Dataset", demo_dataset()))
    results.append(("Models", demo_models()))
    results.append(("Verification", demo_verification()))
    results.append(("Full Pipeline", demo_full_pipeline()))
    
    # Summary
    print("\n📊 Demo Results:")
    print("=" * 30)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:15s}: {status}")
        if result:
            passed += 1
    
    print("=" * 30)
    print(f"Overall: {passed}/{len(results)} components working")
    
    if passed == len(results):
        print("\n🎉 Complete system is working!")
        print("\n📋 Your colleague can now:")
        print("   1. ✅ Process all available videos")
        print("   2. ✅ Predict all 50 annotation features")
        print("   3. ✅ Get temporal boundary predictions")
        print("   4. ✅ Save all outputs for verification")
        print("   5. ✅ Generate comprehensive reports")
        
        print("\n🚀 Next steps:")
        print("   1. Run full training with: python3 scripts/train_comprehensive_multilabel.py")
        print("   2. Check verification outputs in training_outputs/verification/")
        print("   3. Add more videos when they become available")
    else:
        print("\n⚠️ Some components need attention")

if __name__ == "__main__":
    main()