#!/usr/bin/env python3
"""
YOLO + EdgeFace Visual Profile Creator
======================================

Create face profiles using YOLO head detection + EdgeFace embeddings.
Consistent with the tracking system's detection pipeline.

Features:
- YOLO head detection (supports YOLOv5 and YOLOv11)
- EdgeFace embeddings for identification
- Quality validation
- Multiple images per person for robustness

Author: Integrated YOLO + EdgeFace System
Created: November 12, 2025
Updated: November 13, 2025 - Added YOLOv11 support
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

from yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VisualProfileMetadata:
    """Metadata for visual profiles"""
    person_name: str
    created_at: str
    num_images: int
    model_name: str
    embedding_dimension: int
    image_quality_score: float
    capture_environment: str
    detection_model: str = "yolov5_or_yolov11"
    profile_version: str = "2.1"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VisualProfile:
    """Visual profile with face embeddings"""
    name: str
    embeddings: List[np.ndarray]
    average_embedding: np.ndarray
    metadata: VisualProfileMetadata
    
    def save_to_file(self, filepath: Path) -> None:
        """Save profile to pickle file"""
        data = {
            'person_name': self.name,
            'embeddings': self.embeddings,
            'average_embedding': self.average_embedding,
            'metadata': self.metadata.to_dict(),
            'created_at': self.metadata.created_at,
            'model_name': self.metadata.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Visual profile saved: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'VisualProfile':
        """Load profile from pickle file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if 'metadata' not in data:
            metadata = VisualProfileMetadata(
                person_name=data['person_name'],
                created_at=data.get('created_at', 'unknown'),
                num_images=len(data['embeddings']),
                model_name=data.get('model_name', 'edgeface_xxs'),
                embedding_dimension=data['average_embedding'].shape[-1],
                image_quality_score=0.0,
                capture_environment='unknown'
            )
        else:
            metadata = VisualProfileMetadata(**data['metadata'])
        
        return cls(
            name=data['person_name'],
            embeddings=data['embeddings'],
            average_embedding=data['average_embedding'],
            metadata=metadata
        )


class EdgeFaceEmbedder:
    """EdgeFace embedding extractor"""
    
    def __init__(self, variant='edgeface_xxs', device='mps'):
        self.device = torch.device(device)
        self.input_size = (112, 112)
        
        logger.info(f"Loading EdgeFace model '{variant}'")
        
        try:
            self.model = torch.hub.load(
                'otroshi/edgeface', 
                variant, 
                source='github', 
                pretrained=True,
                trust_repo=True
            )
            logger.info(f"Successfully loaded {variant}")
        except Exception as e:
            logger.error(f"Failed to load EdgeFace: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        if str(self.device) == 'mps':
            self._register_contiguous_hooks()
    
    def _register_contiguous_hooks(self):
        def make_contiguous_hook(module, input, output):
            if isinstance(output, torch.Tensor) and not output.is_contiguous():
                return output.contiguous()
            return output
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(make_contiguous_hook)
    
    def preprocess(self, face_crop):
        resized = cv2.resize(face_crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        chw = np.transpose(normalized, (2, 0, 1))
        chw = np.ascontiguousarray(chw)
        batch = np.expand_dims(chw, axis=0)
        batch = np.ascontiguousarray(batch)
        tensor = torch.from_numpy(batch).to(self.device)
        return tensor.contiguous()
    
    def extract_embedding(self, face_crop):
        batch = self.preprocess(face_crop)
        with torch.no_grad():
            embedding = self.model(batch)
            embedding = embedding.contiguous()
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()


# Legacy YOLOHeadDetector removed - now using unified YOLODetector from yolo_detector.py


class FaceQualityAnalyzer:
    """Analyze face image quality"""
    
    @staticmethod
    def analyze_face_quality(face_crop: np.ndarray) -> Dict:
        brightness = np.mean(face_crop)
        contrast = np.std(face_crop)
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        min_dim = min(face_crop.shape[:2])
        size_score = min(1.0, min_dim / 112.0)
        
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        contrast_score = min(1.0, contrast / 50.0)
        blur_quality = min(1.0, blur_score / 100.0)
        
        quality_score = (brightness_score * 0.3 + 
                        contrast_score * 0.3 + 
                        blur_quality * 0.3 + 
                        size_score * 0.1)
        
        if quality_score > 0.7:
            quality_level = "Excellent"
        elif quality_score > 0.5:
            quality_level = "Good"
        elif quality_score > 0.3:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'blur_score': float(blur_score)
        }


class YOLOVisualProfileCreator:
    """Create visual profiles using YOLO + EdgeFace"""
    
    def __init__(self,
                 yolo_model_path: str = "./od_model/yolov5.onnx",  # Also supports yolov11.onnx
                 embeddings_dir: str = "data/visual_embeddings",
                 images_dir: str = "data/visual_images",
                 num_images: int = 5):
        
        self.embeddings_dir = Path(embeddings_dir)
        self.images_dir = Path(images_dir)
        self.num_images = num_images
        
        self.embeddings_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        
        logger.info("Initializing YOLO head detector...")
        self.detector = YOLODetector(
            yolo_model_path,
            confidence_threshold=0.5,
            nms_threshold=0.45,
            use_gpu=False
        )
        
        logger.info("Initializing EdgeFace embedder...")
        try:
            self.edgeface = EdgeFaceEmbedder(variant='edgeface_xxs', device='mps')
            logger.info("Using MPS (Apple GPU)")
        except:
            self.edgeface = EdgeFaceEmbedder(variant='edgeface_xxs', device='cpu')
            logger.info("Using CPU")
        
        self.quality_analyzer = FaceQualityAnalyzer()
        
        logger.info("Profile creator initialized successfully")
    
    def capture_face_images(self, person_name: str) -> List[Tuple[np.ndarray, Dict]]:
        """Capture multiple face images using YOLO detection"""
        
        print(f"\n{'='*60}")
        print("YOLO HEAD DETECTION - FACE CAPTURE")
        print(f"{'='*60}")
        print(f"Number of images to capture: {self.num_images}")
        print("\nFor best results:")
        print("• Look directly at the camera")
        print("• Ensure good lighting on your face")
        print("• Keep your face centered in the frame")
        print("• Vary your expression slightly between captures")
        print(f"\n{'='*60}\n")
        
        input("Press Enter to open camera...")
        
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            raise ValueError("Cannot open camera")
        
        captured_images = []
        capture_count = 0
        
        print(f"\n📸 Camera opened. Press SPACE to capture ({self.num_images} needed)")
        print("Press 'q' to cancel\n")
        
        while capture_count < self.num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect heads with YOLO
            head_bboxes = self.detector.detect_heads(frame)
            
            # Draw rectangles around detected heads
            display_frame = frame.copy()
            for bbox in head_bboxes:
                x1, y1, x2, y2, conf = bbox  # Now includes confidence score
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Head {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show capture progress
            cv2.putText(display_frame, 
                       f"Captured: {capture_count}/{self.num_images} | Press SPACE to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('YOLO Head Capture - Press SPACE', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar
                if len(head_bboxes) > 0:
                    # Take the first head
                    x1, y1, x2, y2, conf = head_bboxes[0]  # Now includes confidence
                    
                    # Add margin
                    w, h = x2 - x1, y2 - y1
                    margin = int(0.2 * max(w, h))
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Analyze quality
                    quality_metrics = self.quality_analyzer.analyze_face_quality(face_crop)
                    
                    print(f"✅ Image {capture_count + 1} captured - Quality: {quality_metrics['quality_level']} ({quality_metrics['quality_score']:.2f})")
                    
                    captured_images.append((face_crop, quality_metrics))
                    capture_count += 1
                else:
                    print("❌ No head detected. Please position your face in the frame.")
            
            elif key == ord('q'):
                print("\n❌ Capture cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return []
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Captured {len(captured_images)} images successfully!")
        return captured_images
    
    def create_visual_profile(self, 
                            person_name: str,
                            captured_images: List[Tuple[np.ndarray, Dict]],
                            environment: str = "standard") -> VisualProfile:
        """Create visual profile from captured images"""
        
        print("\n🧠 Generating face embeddings...")
        
        face_crops = [img for img, _ in captured_images]
        quality_metrics = [metrics for _, metrics in captured_images]
        
        # Generate embeddings
        embeddings_list = []
        for crop in face_crops:
            embedding = self.edgeface.extract_embedding(crop)
            embeddings_list.append(embedding)
        
        # Compute average embedding
        average_embedding = np.mean(embeddings_list, axis=0)
        
        # Average quality score
        avg_quality = np.mean([m['quality_score'] for m in quality_metrics])
        
        # Create metadata
        metadata = VisualProfileMetadata(
            person_name=person_name,
            created_at=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            num_images=len(face_crops),
            model_name="edgeface_xxs",
            embedding_dimension=average_embedding.shape[-1],
            image_quality_score=avg_quality,
            capture_environment=environment,
            detection_model="yolov5"
        )
        
        print(f"✅ Generated {len(embeddings_list)} embeddings")
        print(f"📊 Average quality: {avg_quality:.2f}/1.0")
        print(f"🎯 Embedding dimension: {metadata.embedding_dimension}")
        
        return VisualProfile(
            name=person_name,
            embeddings=embeddings_list,
            average_embedding=average_embedding,
            metadata=metadata
        )
    
    def save_visual_profile(self, profile: VisualProfile) -> Path:
        filename = self.embeddings_dir / f"{profile.name}_{profile.metadata.created_at}.pkl"
        profile.save_to_file(filename)
        return filename
    
    def save_face_images(self, person_name: str, captured_images: List[Tuple[np.ndarray, Dict]], timestamp: str) -> List[Path]:
        saved_paths = []
        person_dir = self.images_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        for idx, (face_crop, _) in enumerate(captured_images):
            filename = person_dir / f"{person_name}_{timestamp}_{idx+1}.jpg"
            cv2.imwrite(str(filename), face_crop)
            saved_paths.append(filename)
        
        logger.info(f"Saved {len(saved_paths)} face images")
        return saved_paths
    
    def create_profile_interactive(self) -> Optional[VisualProfile]:
        """Interactive visual profile creation"""
        
        print("\n🎯 YOLO + EdgeFace Profile Creator")
        print("=" * 50)
        
        person_name = input("Enter person name: ").strip()
        if not person_name:
            print("❌ Person name is required")
            return None
        
        existing_profiles = list(self.embeddings_dir.glob(f"{person_name}_*.pkl"))
        if existing_profiles:
            print(f"\n📋 Found {len(existing_profiles)} existing profile(s) for '{person_name}':")
            for profile_path in existing_profiles:
                print(f"  • {profile_path.name}")
            
            create_new = input("\nCreate additional profile? (y/n): ").lower().strip()
            if create_new != 'y':
                print("Profile creation cancelled.")
                return None
        
        environment = input("Capture environment (e.g., 'office', 'home'): ").strip() or "standard"
        
        try:
            captured_images = self.capture_face_images(person_name)
            
            if not captured_images:
                print("❌ No images captured")
                return None
            
            print("\n📊 Quality Summary:")
            for idx, (_, metrics) in enumerate(captured_images):
                print(f"  Image {idx+1}: {metrics['quality_level']} ({metrics['quality_score']:.2f})")
            
            profile = self.create_visual_profile(person_name, captured_images, environment)
            
            profile_file = self.save_visual_profile(profile)
            image_files = self.save_face_images(person_name, captured_images, profile.metadata.created_at)
            
            print(f"\n✅ Profile creation successful!")
            print(f"🧠 Profile file: {profile_file}")
            print(f"📁 Images saved: {len(image_files)} files in {self.images_dir / person_name}")
            print(f"📊 Average quality: {profile.metadata.image_quality_score:.2f}/1.0")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            print(f"❌ Error creating profile: {e}")
            return None
    
    def list_existing_profiles(self) -> List[VisualProfile]:
        """List all existing visual profiles"""
        
        profile_files = list(self.embeddings_dir.glob("*.pkl"))
        profiles = []
        
        print(f"\n📋 Existing Visual Profiles ({len(profile_files)} found):")
        print("-" * 60)
        
        for profile_file in sorted(profile_files):
            try:
                profile = VisualProfile.load_from_file(profile_file)
                profiles.append(profile)
                
                print(f"👤 {profile.name}")
                print(f"   📅 Created: {profile.metadata.created_at}")
                print(f"   🖼️  Images: {profile.metadata.num_images}")
                print(f"   🎯 Quality: {profile.metadata.image_quality_score:.2f}/1.0")
                print(f"   📁 File: {profile_file.name}")
                print()
                
            except Exception as e:
                logger.warning(f"Could not load profile {profile_file}: {e}")
                print(f"⚠️  {profile_file.name} (load error)")
        
        return profiles


def main():
    """Main function for interactive profile creation"""
    
    try:
        creator = YOLOVisualProfileCreator(num_images=5)
        
        while True:
            print("\n" + "="*60)
            print("📸 YOLO + EDGEFACE PROFILE CREATOR")
            print("="*60)
            print("1. Create new visual profile")
            print("2. List existing profiles")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                profile = creator.create_profile_interactive()
                if profile:
                    print("\n🎉 Ready for face recognition!")
                    
            elif choice == '2':
                creator.list_existing_profiles()
                
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid option. Please choose 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()