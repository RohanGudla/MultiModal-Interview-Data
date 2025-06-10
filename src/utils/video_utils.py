"""
Video processing utilities for frame extraction and preprocessing.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN

class VideoProcessor:
    """Handles video frame extraction and preprocessing."""
    
    def __init__(self, face_detection_confidence: float = 0.7, 
                 face_bbox_expansion: float = 1.2):
        self.face_detection_confidence = face_detection_confidence
        self.face_bbox_expansion = face_bbox_expansion
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            keep_all=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            thresholds=[0.6, 0.7, face_detection_confidence]
        )
        
    def extract_frames(self, video_path: Path, output_dir: Path, 
                      fps: Optional[int] = None) -> List[Path]:
        """Extract frames from video at specified FPS."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps is None:
            fps = int(video_fps)
        
        frame_interval = max(1, int(video_fps / fps))
        frame_count = 0
        extracted_count = 0
        extracted_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame
                frame_filename = f"frame_{extracted_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                
                Image.fromarray(frame_rgb).save(frame_path, quality=95)
                extracted_frames.append(frame_path)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {extracted_count} frames from {video_path.name}")
        return extracted_frames
        
    def detect_and_crop_face(self, image: np.ndarray, 
                           target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """Detect face and crop with bounding box expansion."""
        # Convert to PIL Image for MTCNN
        pil_image = Image.fromarray(image)
        
        # Detect face
        boxes, _ = self.mtcnn.detect(pil_image)
        
        if boxes is None or len(boxes) == 0:
            return None
            
        # Get the first (most confident) face
        box = boxes[0]
        x1, y1, x2, y2 = box
        
        # Expand bounding box
        w, h = x2 - x1, y2 - y1
        expansion = self.face_bbox_expansion
        
        x1 = max(0, x1 - w * (expansion - 1) / 2)
        y1 = max(0, y1 - h * (expansion - 1) / 2)
        x2 = min(image.shape[1], x2 + w * (expansion - 1) / 2)
        y2 = min(image.shape[0], y2 + h * (expansion - 1) / 2)
        
        # Crop face
        face = image[int(y1):int(y2), int(x1):int(x2)]
        
        if face.size == 0:
            return None
            
        # Resize to target size
        face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return face_resized
        
    def process_video_frames(self, video_path: Path, output_dir: Path,
                           fps: int = 30, target_size: Tuple[int, int] = (224, 224)) -> Dict:
        """Extract and process all frames from a video."""
        participant_id = self.extract_participant_id(video_path.name)
        participant_dir = output_dir / participant_id
        
        # Extract raw frames
        print(f"Processing video: {video_path.name}")
        raw_frames = self.extract_frames(video_path, participant_dir / "raw", fps)
        
        # Process faces
        face_dir = participant_dir / "faces"
        face_dir.mkdir(parents=True, exist_ok=True)
        
        valid_frames = []
        timestamps = []
        
        for i, frame_path in enumerate(raw_frames):
            # Load frame
            frame = np.array(Image.open(frame_path))
            
            # Detect and crop face
            face = self.detect_and_crop_face(frame, target_size)
            
            if face is not None:
                # Save processed face
                face_filename = f"face_{i:06d}.jpg"
                face_path = face_dir / face_filename
                Image.fromarray(face).save(face_path, quality=95)
                
                valid_frames.append(face_path)
                timestamps.append(i / fps)  # Convert frame index to seconds
                
        print(f"Processed {len(valid_frames)}/{len(raw_frames)} frames with valid faces")
        
        return {
            'participant_id': participant_id,
            'video_path': video_path,
            'valid_frames': valid_frames,
            'timestamps': timestamps,
            'fps': fps,
            'total_frames': len(raw_frames),
            'valid_face_frames': len(valid_frames)
        }
        
    @staticmethod
    def extract_participant_id(filename: str) -> str:
        """Extract participant ID from filename."""
        # Examples: "Screen recording 1 - CP 0636.mp4" -> "CP 0636"
        parts = filename.replace('.mp4', '').split(' - ')
        if len(parts) > 1:
            return parts[-1].strip()
        return filename.replace('.mp4', '')
        
    @staticmethod
    def get_augmentation_pipeline(is_training: bool = True) -> A.Compose:
        """Get data augmentation pipeline."""
        if is_training:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
        return transform
        
    @staticmethod
    def check_frame_quality(image: np.ndarray, blur_threshold: float = 100.0) -> Dict[str, float]:
        """Check frame quality metrics."""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness analysis
        brightness = np.mean(gray)
        
        # Contrast analysis
        contrast = np.std(gray)
        
        return {
            'blur_score': blur_score,
            'is_blurry': blur_score < blur_threshold,
            'brightness': brightness,
            'contrast': contrast,
            'quality_score': blur_score * contrast / 10000  # Combined quality metric
        }