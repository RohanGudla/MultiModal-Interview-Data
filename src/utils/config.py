"""
Configuration settings for the multimodal video ML project.
Optimized for RTX 4080 16GB.
"""
import torch
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    RAW_DATA_ROOT = PROJECT_ROOT.parent / "GENEX Intreview"
    
    # Hardware configuration (RTX 4080 16GB optimized)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8
    PIN_MEMORY = True
    MIXED_PRECISION = True
    
    # Data configuration
    VIDEO_FPS = 30
    FRAME_SIZE = (224, 224)
    FACE_DETECTION_CONFIDENCE = 0.7
    FACE_BBOX_EXPANSION = 1.2
    
    # Model configurations (batch sizes optimized for 16GB VRAM)
    MODEL_CONFIGS = {
        "cnn_simple": {
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4
        },
        "vit_scratch": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "patch_size": 16,
            "embed_dim": 768,
            "num_layers": 6,
            "num_heads": 8
        },
        "resnet50_pretrained": {
            "batch_size": 48,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "freeze_layers": 45  # Freeze early layers
        },
        "vit_pretrained": {
            "batch_size": 24,
            "learning_rate": 1e-5,
            "weight_decay": 1e-2,
            "freeze_layers": 10,  # Freeze first 10 transformer blocks
            "model_name": "vit_base_patch16_224"
        }
    }
    
    # Training configuration
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    GRADIENT_ACCUMULATION_STEPS = 2
    GRADIENT_CLIP_NORM = 1.0
    
    # Data splits (using actual GENEX participants)
    TRAIN_PARTICIPANTS = ["CP 0636", "JM 9684", "MP 5114"]
    VAL_PARTICIPANTS = ["NS 4013"]
    TEST_PARTICIPANTS = ["LE 3299"]
    
    # Emotion labels (7 core emotions)
    EMOTION_LABELS = [
        "Joy", "Anger", "Fear", "Disgust", 
        "Sadness", "Surprise", "Contempt"
    ]
    
    # Additional targets
    ADDITIONAL_TARGETS = [
        "Attention", "Engagement", "PositiveValence", "NegativeValence"
    ]
    
    # Augmentation parameters
    AUGMENTATION = {
        "horizontal_flip": 0.5,
        "rotation_degrees": 10,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }
    
    # Logging
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 5
    
    @classmethod
    def get_model_config(cls, model_name):
        """Get configuration for specific model."""
        return cls.MODEL_CONFIGS.get(model_name, {})
    
    @classmethod
    def get_data_paths(cls):
        """Get all relevant data paths."""
        return {
            "videos": cls.RAW_DATA_ROOT / "Analysis" / "Gaze Replays",
            "annotations": cls.RAW_DATA_ROOT / "Analysis" / "Multimodal summary metrics" / "Multimodal summary metrics.csv",
            "expression_table": cls.RAW_DATA_ROOT / "Analysis" / "Facial Coding" / "FEAExpressionTable.csv",
            "processed_frames": cls.DATA_ROOT / "processed" / "frames",
            "processed_labels": cls.DATA_ROOT / "processed" / "labels",
            "splits": cls.DATA_ROOT / "splits"
        }