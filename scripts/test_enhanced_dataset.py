#!/usr/bin/env python3
"""
Test Enhanced Multi-Participant Dataset System
"""

import sys
sys.path.append('/home/rohan/Multimodal/multimodal_video_ml/src')

from data.multilabel_dataset import MultiLabelAnnotationDataset, create_dataloaders

def test_enhanced_dataset():
    """Test the enhanced multi-participant dataset system"""
    
    frames_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/enhanced_frames"
    annotations_dir = "/home/rohan/Multimodal/multimodal_video_ml/data/annotations"
    
    print("🧪 Testing Enhanced Multi-Participant Dataset System")
    print("=" * 60)
    
    # Test basic dataset loading
    print("\n1. Testing basic dataset loading...")
    dataset = MultiLabelAnnotationDataset(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        sequence_length=1
    )
    
    # Get participant summary
    participant_summary = dataset.get_participant_summary()
    print(f"\n📊 Participant Summary:")
    for participant, count in sorted(participant_summary.items()):
        print(f"  {participant}: {count} samples")
    
    # Test participant-based data splitting
    print("\n2. Testing participant-based data splitting...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        batch_size=16,
        sequence_length=1,
        split_by_participant=True,
        num_workers=0  # Avoid multiprocessing issues in testing
    )
    
    print(f"\n📈 DataLoader Summary:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading a batch from each split
    print("\n3. Testing batch loading...")
    
    if len(train_loader) > 0:
        train_batch = next(iter(train_loader))
        print(f"  Train batch shape: {train_batch['image'].shape}")
        print(f"  Train batch participants: {set(train_batch['participant_id'])}")
    
    if len(val_loader) > 0:
        val_batch = next(iter(val_loader))
        print(f"  Val batch shape: {val_batch['image'].shape}")
        print(f"  Val batch participants: {set(val_batch['participant_id'])}")
    
    if len(test_loader) > 0:
        test_batch = next(iter(test_loader))
        print(f"  Test batch shape: {test_batch['image'].shape}")
        print(f"  Test batch participants: {set(test_batch['participant_id'])}")
    
    # Test sequence mode
    print("\n4. Testing sequence mode...")
    train_loader_seq, val_loader_seq, test_loader_seq, dataset_seq = create_dataloaders(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        batch_size=8,
        sequence_length=5,
        split_by_participant=True,
        num_workers=0
    )
    
    if len(train_loader_seq) > 0:
        seq_batch = next(iter(train_loader_seq))
        print(f"  Sequence batch shape: {seq_batch['images'].shape}")
        print(f"  Sequence labels shape: {seq_batch['all_labels'].shape}")
    
    # Get label statistics
    print("\n5. Label statistics...")
    stats = dataset.get_label_statistics()
    
    print(f"  Physical features positive rate: {stats['physical']['positive_rate'].mean():.3f}")
    print(f"  Emotional features positive rate: {stats['emotional']['positive_rate'].mean():.3f}")
    print(f"  Combined features positive rate: {stats['combined']['positive_rate'].mean():.3f}")
    
    print(f"\n✅ Enhanced dataset system test completed successfully!")
    print(f"   Total participants with data: {len(participant_summary)}")
    print(f"   Total samples available: {len(dataset)}")
    print(f"   Features available: {len(dataset.feature_info['all'])}")

if __name__ == "__main__":
    test_enhanced_dataset()