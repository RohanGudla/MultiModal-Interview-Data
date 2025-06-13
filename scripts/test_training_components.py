#!/usr/bin/env python3
"""
Test Training Components Before Full Training
"""

import sys
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

import torch
from data.multilabel_dataset import create_dataloaders
from models.temporal_multilabel import TemporalMultiLabelViT, TemporalMultiLabelResNet

def test_training_components():
    """Test the training components work correctly"""
    
    print("🧪 Testing Training Components")
    print("=" * 50)
    
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    
    # Test 1: Data loading
    print("\n1. Testing data loading...")
    try:
        train_loader, val_loader, test_loader, dataset = create_dataloaders(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            batch_size=4,
            sequence_length=3,
            split_by_participant=True,
            num_workers=0
        )
        
        feature_info = dataset.get_feature_names()
        num_physical = len(feature_info['physical'])
        num_emotional = len(feature_info['emotional'])
        
        print(f"   ✅ Data loading successful")
        print(f"   Physical features: {num_physical}")
        print(f"   Emotional features: {num_emotional}")
        print(f"   Train batches: {len(train_loader)}")
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        # Test ViT model
        vit_model = TemporalMultiLabelViT(
            num_physical_features=num_physical,
            num_emotional_features=num_emotional,
            sequence_length=3
        ).to(device)
        
        print(f"   ✅ ViT model created")
        
        # Test ResNet model
        resnet_model = TemporalMultiLabelResNet(
            num_physical_features=num_physical,
            num_emotional_features=num_emotional,
            sequence_length=3
        ).to(device)
        
        print(f"   ✅ ResNet model created")
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    try:
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            
            if 'images' in batch:
                images = batch['images'].to(device)
            else:
                images = batch['image'].to(device).unsqueeze(1)  # Add sequence dimension
            
            print(f"   Input shape: {images.shape}")
            
            # Test ViT forward pass
            with torch.no_grad():
                results = vit_model(images, return_temporal_boundaries=True)
                
            print(f"   ✅ ViT forward pass successful")
            print(f"   Physical output: {results['physical_probs'].shape}")
            print(f"   Emotional output: {results['emotional_probs'].shape}")
            print(f"   Boundary output: {results['boundary_probs'].shape}")
            
            # Test ResNet forward pass
            with torch.no_grad():
                results = resnet_model(images, return_temporal_boundaries=True)
                
            print(f"   ✅ ResNet forward pass successful")
            
        else:
            print("   ⚠️ No data available for forward pass test")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test 4: Loss computation
    print("\n4. Testing loss computation...")
    try:
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create dummy targets
        physical_targets = torch.randint(0, 2, (batch['physical_labels'].shape), dtype=torch.float32).to(device)
        emotional_targets = torch.randint(0, 2, (batch['emotional_labels'].shape), dtype=torch.float32).to(device)
        
        # Compute losses
        physical_loss = criterion(results['physical_logits'], physical_targets)
        emotional_loss = criterion(results['emotional_logits'], emotional_targets)
        total_loss = physical_loss + emotional_loss
        
        print(f"   ✅ Loss computation successful")
        print(f"   Physical loss: {physical_loss.item():.4f}")
        print(f"   Emotional loss: {emotional_loss.item():.4f}")
        print(f"   Total loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        return False
    
    print(f"\n✅ All training components test passed!")
    print(f"   Ready for full training pipeline")
    
    return True

if __name__ == "__main__":
    success = test_training_components()
    if not success:
        print("\n❌ Component tests failed - fix issues before proceeding")
        sys.exit(1)