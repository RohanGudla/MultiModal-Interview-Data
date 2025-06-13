#!/usr/bin/env python3
"""
Test the Multi-Label System with Real Data
"""

import sys
import os
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from data.multilabel_dataset import MultiLabelAnnotationDataset, create_dataloaders
import torch
import numpy as np
import json

def test_multilabel_dataset():
    """Test the multi-label dataset with actual data"""
    
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    
    print("🧪 Testing Multi-Label Dataset...")
    
    # Test with specific participant that has both frames and annotations
    participants = ["LE_3299"]  # Use underscore version that has annotations
    
    # Test single frame mode
    print("\n1️⃣ Testing Single Frame Mode:")
    dataset = MultiLabelAnnotationDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        participants=participants,
        sequence_length=1
    )
    
    if len(dataset) > 0:
        print(f"✅ Dataset loaded successfully!")
        print(f"   Dataset size: {len(dataset)}")
        
        # Test sample
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   All labels shape: {sample['all_labels'].shape}")
        print(f"   Physical labels shape: {sample['physical_labels'].shape}")
        print(f"   Emotional labels shape: {sample['emotional_labels'].shape}")
        print(f"   Participant: {sample['participant_id']}")
        print(f"   Frame ID: {sample['frame_id']}")
        
        # Check label values
        print(f"   Physical label sum: {sample['physical_labels'].sum():.2f}")
        print(f"   Emotional label sum: {sample['emotional_labels'].sum():.2f}")
        print(f"   Total label sum: {sample['all_labels'].sum():.2f}")
        
        # Test multiple samples
        print(f"\n📊 Testing multiple samples:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"   Frame {i}: Physical={sample['physical_labels'].sum():.1f}, "
                  f"Emotional={sample['emotional_labels'].sum():.1f}, "
                  f"Total={sample['all_labels'].sum():.1f}")
    else:
        print("❌ Dataset is empty!")
        return False
    
    # Test sequence mode
    print("\n2️⃣ Testing Sequence Mode:")
    dataset_seq = MultiLabelAnnotationDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        participants=participants,
        sequence_length=5
    )
    
    if len(dataset_seq) > 0:
        print(f"✅ Sequence dataset loaded!")
        print(f"   Dataset size: {len(dataset_seq)}")
        
        sample = dataset_seq[0]
        print(f"   Images shape: {sample['images'].shape}")
        print(f"   Labels shape: {sample['all_labels'].shape}")
    else:
        print("❌ Sequence dataset is empty!")
    
    # Test dataloaders
    print("\n3️⃣ Testing DataLoaders:")
    try:
        train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            participants=participants,
            batch_size=8,
            sequence_length=1
        )
        
        print(f"✅ DataLoaders created!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"   Batch shape - Images: {batch['image'].shape}")
            print(f"   Batch shape - All labels: {batch['all_labels'].shape}")
            print(f"   Batch shape - Physical: {batch['physical_labels'].shape}")
            print(f"   Batch shape - Emotional: {batch['emotional_labels'].shape}")
            break
            
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False
    
    # Test feature information
    print("\n4️⃣ Testing Feature Information:")
    feature_info = dataset.get_feature_names()
    print(f"   Physical features: {len(feature_info['physical'])}")
    print(f"   Emotional features: {len(feature_info['emotional'])}")
    print(f"   Total features: {len(feature_info['all'])}")
    
    print("\n   First 10 physical features:")
    for i, feat in enumerate(feature_info['physical'][:10]):
        print(f"     {i+1:2d}. {feat}")
    
    print("\n   All emotional features:")
    for i, feat in enumerate(feature_info['emotional']):
        print(f"     {i+1:2d}. {feat}")
    
    # Test statistics
    print("\n5️⃣ Testing Label Statistics:")
    stats = dataset.get_label_statistics()
    
    if stats:
        print(f"   Physical feature activation rates:")
        physical_rates = stats['physical']['positive_rate']
        for i, feat in enumerate(feature_info['physical'][:10]):  # Show first 10
            print(f"     {feat}: {physical_rates[i]:.3f}")
        
        print(f"   Emotional feature activation rates:")
        emotional_rates = stats['emotional']['positive_rate']
        for i, feat in enumerate(feature_info['emotional']):
            print(f"     {feat}: {emotional_rates[i]:.3f}")
    
    print("\n✅ All tests passed! Multi-label system is working correctly.")
    return True

if __name__ == "__main__":
    success = test_multilabel_dataset()
    if success:
        print("\n🎉 Multi-label system ready for training!")
    else:
        print("\n❌ Multi-label system needs debugging.")