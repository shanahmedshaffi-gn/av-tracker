#!/usr/bin/env python3
"""
YOLO + EdgeFace Real-Time Verification System
=============================================

Verify detected faces against a database of known visual profiles.
Uses YOLO for detection (YOLOv5/YOLOv11 support) and EdgeFace for identification.

Features:
- YOLO head detection (supports both YOLOv5 and YOLOv11)
- Load multiple visual profiles
- Real-time face matching
- Display person names on video

"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cosine

from yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.6


@dataclass
class KnownPerson:
    """Represents a known person from the database"""
    name: str
    embedding: np.ndarray
    profile_path: Path
    quality_score: float


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


class VisualProfileDatabase:
    """Database of known visual profiles"""
    
    def __init__(self, profiles_dir: str = "data/visual_embeddings"):
        self.profiles_dir = Path(profiles_dir)
        self.known_persons: Dict[str, KnownPerson] = {}
        
        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")
            return
        
        self._load_all_profiles()
    
    def _load_all_profiles(self):
        """Load all visual profiles from directory"""
        profile_files = list(self.profiles_dir.glob("*.pkl"))
        
        logger.info(f"Loading profiles from {self.profiles_dir}")
        
        for profile_file in profile_files:
            try:
                with open(profile_file, 'rb') as f:
                    data = pickle.load(f)
                
                person_name = data['person_name']
                
                if 'average_embedding' in data:
                    embedding = data['average_embedding']
                elif 'embeddings' in data and len(data['embeddings']) > 0:
                    embedding = np.mean(data['embeddings'], axis=0)
                else:
                    logger.warning(f"No embeddings found in {profile_file}")
                    continue
                
                if 'metadata' in data:
                    quality_score = data['metadata'].get('image_quality_score', 0.0)
                else:
                    quality_score = 0.0
                
                self.known_persons[person_name] = KnownPerson(
                    name=person_name,
                    embedding=embedding,
                    profile_path=profile_file,
                    quality_score=quality_score
                )
                
                logger.info(f"Loaded profile: {person_name} (quality: {quality_score:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")
        
        logger.info(f"Loaded {len(self.known_persons)} profiles")
    
    def identify_person(self, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Tuple[Optional[str], float]:
        """Identify person from embedding"""
        if len(self.known_persons) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for person_name, known_person in self.known_persons.items():
            similarity = 1 - cosine(embedding, known_person.embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def get_all_names(self) -> List[str]:
        return list(self.known_persons.keys())


class YOLOVisualVerificationSystem:
    """Real-time visual verification with YOLO + EdgeFace"""
    
    def __init__(self, 
                 yolo_model_path: str = "./od_model/yolov11.onnx",  
                 profiles_dir: str = "data/visual_embeddings",
                 threshold: float = DEFAULT_THRESHOLD,
                 video_device: int = 0):
        
        logger.info("Initializing YOLO Visual Verification System")
        
        self.threshold = threshold
        self.video_device = video_device
        
        # Initialize YOLO detector
        logger.info("Loading YOLO head detector...")
        self.detector = YOLODetector(
            yolo_model_path,
            confidence_threshold=0.5,
            nms_threshold=0.45,
            use_gpu=False
        )
        
        # Initialize EdgeFace embedder
        logger.info("Loading EdgeFace model...")
        try:
            self.embedder = EdgeFaceEmbedder(variant='edgeface_xxs', device='mps')
        except:
            logger.warning("MPS failed, using CPU")
            self.embedder = EdgeFaceEmbedder(variant='edgeface_xxs', device='cpu')
        
        # Load profile database
        logger.info("Loading visual profiles database...")
        self.database = VisualProfileDatabase(profiles_dir)
        
        self.is_running = False
        
        logger.info(f"System initialized with {len(self.database.known_persons)} known persons")
        logger.info(f"Threshold: {self.threshold}")
    
    def _extract_face_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int, float]) -> np.ndarray:
        """Extract face crop with margin"""
        x1, y1, x2, y2, _ = bbox
        
        # Add margin
        w, h = x2 - x1, y2 - y1
        margin = int(0.1 * max(w, h))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        return frame[y1:y2, x1:x2]
    
    def _draw_identification(self, frame: np.ndarray, bbox: Tuple[int, int, int, int, float], 
                            person_name: Optional[str], similarity: float):
        """Draw identification results on frame"""
        x1, y1, x2, y2, det_conf = bbox
        
        if person_name:
            color = (0, 255, 0)  # Green for known
            label = f"{person_name} ({similarity:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = f"Unknown ({similarity:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw status indicator circle
        if person_name:
            cv2.circle(frame, (x2 - 15, y1 + 15), 8, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (x2 - 15, y1 + 15), 8, (0, 0, 255), -1)
    
    def start_verification(self):
        """Start real-time verification"""
        logger.info(f"Starting visual verification on camera {self.video_device}")
        
        if len(self.database.known_persons) == 0:
            logger.warning("No profiles loaded!")
            print("\n⚠️  No visual profiles found!")
            print("Create profiles first using: python yolo_profile_creator.py")
            return
        
        print("\n🎯 YOLO + EdgeFace Visual Verification")
        print("=" * 60)
        print(f"Loaded {len(self.database.known_persons)} known persons:")
        for name in self.database.get_all_names():
            print(f"  • {name}")
        print(f"\nThreshold: {self.threshold}")
        print("\nPress 'q' to quit")
        print("=" * 60 + "\n")
        
        cap = cv2.VideoCapture(self.video_device, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag for real-time
        
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        self.is_running = True
        
        fps_counter = 0
        fps_timer = time.time()
        fps = 0
        
        verification_log = []
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                start_time = time.time()
                
                # Detect heads with YOLO
                head_bboxes = self.detector.detect_heads(frame)
                
                # Process each detected head
                for bbox in head_bboxes:
                    try:
                        # Extract face crop
                        face_crop = self._extract_face_crop(frame, bbox)
                        
                        if face_crop.size == 0:
                            continue
                        
                        # Extract embedding
                        embedding = self.embedder.extract_embedding(face_crop)
                        
                        # Identify person
                        person_name, similarity = self.database.identify_person(
                            embedding, self.threshold
                        )
                        
                        # Draw results
                        self._draw_identification(frame, bbox, person_name, similarity)
                        
                        # Log verification
                        if person_name:
                            timestamp = time.strftime("%H:%M:%S")
                            log_entry = f"[{timestamp}] ✓ {person_name} (similarity: {similarity:.3f})"
                            if not verification_log or verification_log[-1] != log_entry:
                                logger.info(log_entry)
                                verification_log.append(log_entry)
                        
                    except Exception as e:
                        logger.error(f"Error processing face: {e}")
                        continue
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_timer > 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()
                
                processing_time = (time.time() - start_time) * 1000
                
                # Draw info overlay
                cv2.putText(frame, f"FPS: {fps:.1f} | Processing: {processing_time:.0f}ms", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Known: {len(self.database.known_persons)} | Detected: {len(head_bboxes)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "YOLO + EdgeFace Verification", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show frame
                cv2.imshow('YOLO Visual Verification - Press Q to quit', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Visual verification stopped")
            
            print(f"\n📊 Session Summary:")
            print(f"Total verifications: {len(verification_log)}")


def main():
    """Main function"""
    
    try:
        print("\n🎯 YOLO + EdgeFace Real-Time Verification")
        print("=" * 60)
        
        profiles_dir = Path("data/visual_embeddings")
        if not profiles_dir.exists():
            print("❌ No 'data/visual_embeddings' directory found.")
            print("   Create profiles first using: python yolo_profile_creator.py")
            return
        
        profile_files = list(profiles_dir.glob("*.pkl"))
        if len(profile_files) == 0:
            print("❌ No visual profiles found.")
            print("   Create profiles first using: python yolo_profile_creator.py")
            return
        
        print(f"Found {len(profile_files)} visual profile(s)")
        
        try:
            threshold_input = input(f"\nSimilarity threshold (default {DEFAULT_THRESHOLD}): ").strip()
            threshold = float(threshold_input) if threshold_input else DEFAULT_THRESHOLD
            threshold = max(0.0, min(1.0, threshold))
        except ValueError:
            threshold = DEFAULT_THRESHOLD
        
        verifier = YOLOVisualVerificationSystem(threshold=threshold)
        
        input("\nPress Enter to start camera...\n")
        
        verifier.start_verification()
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()