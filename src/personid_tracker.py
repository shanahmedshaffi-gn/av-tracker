#!/usr/bin/env python3
"""
Complete Head Tracker with Name Display
=======================================

Integrated system combining:
- YOLO head detection (YOLOv5/YOLOv11 support)
- ByteTrack motion tracking
- EdgeFace ReID (temporal matching)
- Visual profile database (identity matching)
- Name display for known persons

Features:
- Stable tracking IDs across occlusions
- Person identification from database
- Display actual names for known people
- Frame buffer storage per person
- ID reuse system

Created: November 12, 2025
Updated: November 13, 2025 - Added YOLOv11 support
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import queue
import threading
import logging
import signal
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cosine

from boxmot import ByteTrack
from yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tracker_debug.log")
    ]
)
logger = logging.getLogger(__name__)


class EdgeFaceEmbedder:
    """EdgeFace embedding extractor"""
    
    def __init__(self, variant_or_path='edgeface_xxs', device='mps', use_torchhub=True):
        self.device = torch.device(device)
        self.input_size = (112, 112)
        
        if use_torchhub and isinstance(variant_or_path, str) and not variant_or_path.endswith('.pt'):
            logging.info(f"Loading EdgeFace '{variant_or_path}' from torch.hub")
            self.model = torch.hub.load(
                'otroshi/edgeface', 
                variant_or_path, 
                source='github', 
                pretrained=True,
                trust_repo=True
            )
        else:
            raise NotImplementedError("Local file loading - use torch.hub instead")
        
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
    
    def preprocess(self, face_crops):
        if len(face_crops) == 0:
            return torch.empty((0, 3, 112, 112), device=self.device)
        
        processed = []
        for crop in face_crops:
            resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = (rgb.astype(np.float32) - 127.5) / 127.5
            chw = np.transpose(normalized, (2, 0, 1))
            chw = np.ascontiguousarray(chw)
            processed.append(chw)
        
        batch = np.stack(processed, axis=0)
        batch = np.ascontiguousarray(batch)
        tensor = torch.from_numpy(batch).to(self.device)
        tensor = tensor.contiguous()
        
        return tensor
    
    def extract_embeddings(self, face_crops):
        if len(face_crops) == 0:
            return np.array([])
        
        batch = self.preprocess(face_crops)
        
        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = embeddings.contiguous()
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()


class VisualProfileDatabase:
    """Database of known persons from visual profiles"""
    
    def __init__(self, profiles_dir: str = "data/visual_embeddings"):
        self.profiles_dir = Path(profiles_dir)
        self.known_persons: Dict[str, np.ndarray] = {}  # {name: average_embedding}
        
        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")
            return
        
        self._load_all_profiles()
    
    def _load_all_profiles(self):
        """Load all visual profiles"""
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
                    continue
                
                self.known_persons[person_name] = embedding
                logger.info(f"Loaded profile: {person_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {profile_file}: {e}")
        
        logger.info(f"Loaded {len(self.known_persons)} profiles")
    
    def identify_person(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """Identify person from embedding"""
        if len(self.known_persons) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for name, known_embedding in self.known_persons.items():
            similarity = 1 - cosine(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity


class TrackedPerson:
    """Represents a tracked person"""
    
    def __init__(self, track_id, bbox, embedding=None, frame_buffer_size=25):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.embeddings_history = deque(maxlen=30)
        if embedding is not None:
            self.embeddings_history.append(embedding)
        
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        
        self.last_seen = time.time()
        self.frames_alive = 1
        self.matched_this_frame = False
        self.total_frames_tracked = 1
        
        # Identity information
        self.identified_name: Optional[str] = None
        self.identification_confidence: float = 0.0
    
    def update(self, bbox, embedding=None, face_crop=None):
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_alive += 1
        self.total_frames_tracked += 1
        self.matched_this_frame = True
        
        if embedding is not None:
            self.embeddings_history.append(embedding)
            self.embedding = np.mean(list(self.embeddings_history), axis=0)
        
        if face_crop is not None:
            self.frame_buffer.append({
                'crop': face_crop.copy(),
                'bbox': bbox,
                'timestamp': time.time()
            })
    
    def set_identity(self, name: Optional[str], confidence: float):
        """Set identified name and confidence"""
        self.identified_name = name
        self.identification_confidence = confidence
    
    def get_display_name(self) -> str:
        """Get name to display on screen"""
        if self.identified_name:
            return f"{self.identified_name} ({self.identification_confidence:.2f})"
        else:
            return f"Person {self.track_id}"
    
    def get_average_embedding(self):
        if len(self.embeddings_history) == 0:
            return None
        return np.mean(list(self.embeddings_history), axis=0)


class ROI:
    def __init__(self, x1, y1, x2, y2, confidence=0, class_id=0, track_id=None):
        self.x1 = int(max(0, x1))
        self.y1 = int(max(0, y1))
        self.x2 = int(max(0, x2))
        self.y2 = int(max(0, y2))
        self.confidence = confidence
        self.class_id = class_id
        self.track_id = track_id
    
    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1
    
    @property
    def center(self):
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)
    
    def to_boxmot_format(self):
        return [self.x1, self.y1, self.x2, self.y2, self.confidence, self.class_id]
    
    @classmethod
    def from_boxmot_format(cls, track, class_id=0):
        return cls(
            x1=track[0], y1=track[1], x2=track[2], y2=track[3],
            confidence=track[5] if len(track) > 5 else 1.0,
            class_id=class_id,
            track_id=int(track[4]) if len(track) > 4 else None
        )


class Frame:
    def __init__(self, data):
        self.data = np.ascontiguousarray(data)
        self.height, self.width = data.shape[:2]
    
    def resize(self, width, height):
        ratio = min(width / self.width, height / self.height)
        new_w, new_h = int(self.width * ratio), int(self.height * ratio)
        resized = cv2.resize(self.data, (new_w, new_h))
        
        delta_w, delta_h = width - new_w, height - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        result = Frame(padded)
        result.padding = (top, bottom, left, right)
        result.scale_ratio = ratio
        return result
    
    def to_model_input(self):
        normalized = self.data.astype(np.float16) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)


class HeadTrackerWithNameDisplay:
    """Complete head tracker with name display"""
    
    def __init__(self, detection_model_path, 
                 confidence_threshold=0.5, nms_threshold=0.45,
                 frame_buffer_size=25, 
                 edgeface_model='edgeface_xxs',
                 use_torchhub=True,
                 profiles_dir="data/visual_embeddings",
                 identification_threshold=0.6):
        
        logging.info("Initializing Head Tracker with Name Display")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.frame_buffer_size = frame_buffer_size
        self.identification_threshold = identification_threshold
        
        # Load YOLO detection model
        logging.info(f"Loading detection model: {detection_model_path}")
        self.detector = YOLODetector(
            detection_model_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            use_gpu=False
        )
        
        # Initialize EdgeFace
        try:
            self.edgeface = EdgeFaceEmbedder(
                variant_or_path=edgeface_model, 
                device='mps',
                use_torchhub=use_torchhub
            )
            logging.info("Using MPS (Apple GPU) for EdgeFace")
        except Exception as e:
            logging.warning(f"MPS failed ({e}), falling back to CPU")
            self.edgeface = EdgeFaceEmbedder(
                variant_or_path=edgeface_model, 
                device='cpu',
                use_torchhub=use_torchhub
            )
        
        # Load visual profile database
        logging.info("Loading visual profile database...")
        self.database = VisualProfileDatabase(profiles_dir)
        
        # Initialize ByteTrack
        logging.info("Initializing ByteTrack for motion tracking")
        self.motion_tracker = ByteTrack(
            track_thresh=confidence_threshold,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        
        # Person management
        self.tracked_persons = {}
        self.next_person_id = 1
        self.available_ids = set()
        self.appearance_threshold = 0.5
        
        self.display_queue = queue.Queue(maxsize=5)
        self.running = True
        
        logging.info("Tracker initialized successfully")
        logging.info(f"Known persons in database: {len(self.database.known_persons)}")
    
    def _get_next_person_id(self):
        if self.available_ids:
            person_id = min(self.available_ids)
            self.available_ids.remove(person_id)
            return person_id
        else:
            person_id = self.next_person_id
            self.next_person_id += 1
            return person_id
    
    def _free_person_id(self, person_id):
        self.available_ids.add(person_id)
    
    def detect_heads(self, frame):
        """Detect heads using unified YOLO detector"""
        head_bboxes = self.detector.detect_heads(frame)
        
        # Convert to ROI format
        rois = []
        for bbox in head_bboxes:
            x1, y1, x2, y2, conf = bbox
            rois.append(ROI(x1, y1, x2, y2, conf, class_id=0))
        
        return rois
    
    def extract_face_crops(self, frame, rois):
        crops = []
        valid_rois = []
        
        for roi in rois:
            x1, y1, x2, y2 = int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1 and (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = frame[y1:y2, x1:x2].copy()
                crops.append(crop)
                valid_rois.append(roi)
        
        return crops, valid_rois
    
    def compute_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        return np.dot(emb1, emb2)
    
    def match_with_appearance(self, motion_tracks, embeddings, face_crops, frame):
        for person in self.tracked_persons.values():
            person.matched_this_frame = False
        
        matched_persons = []
        
        for track, embedding, face_crop in zip(motion_tracks, embeddings, face_crops):
            motion_id = track.track_id
            bbox = (track.x1, track.y1, track.x2, track.y2)
            
            # Try to match with existing persons
            best_match_id = None
            best_similarity = self.appearance_threshold
            
            for person_id, person in self.tracked_persons.items():
                if person.matched_this_frame:
                    continue
                
                person_emb = person.get_average_embedding()
                if person_emb is not None and embedding is not None:
                    similarity = self.compute_similarity(embedding, person_emb)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = person_id
            
            if best_match_id is not None:
                # Update existing person
                self.tracked_persons[best_match_id].update(bbox, embedding, face_crop)
                matched_persons.append(self.tracked_persons[best_match_id])
            else:
                # New person
                person_id = self._get_next_person_id()
                person = TrackedPerson(
                    person_id, bbox, embedding, 
                    frame_buffer_size=self.frame_buffer_size
                )
                person.update(bbox, embedding, face_crop)
                self.tracked_persons[person_id] = person
                matched_persons.append(person)
                logging.info(f"New person: ID {person_id}")
        
        # Identify persons against database
        for person in matched_persons:
            if person.embedding is not None:
                identified_name, confidence = self.database.identify_person(
                    person.embedding, 
                    self.identification_threshold
                )
                person.set_identity(identified_name, confidence)
                
                if identified_name and person.frames_alive == 1:
                    logging.info(f"Identified Person {person.track_id} as {identified_name} (confidence: {confidence:.2f})")
        
        # Remove persons not seen for too long
        current_time = time.time()
        to_remove = []
        for person_id, person in self.tracked_persons.items():
            if current_time - person.last_seen > 3.0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            logging.info(f"Removed person {person_id} (tracked {self.tracked_persons[person_id].total_frames_tracked} frames)")
            del self.tracked_persons[person_id]
            self._free_person_id(person_id)
        
        return matched_persons
    
    def process_frame(self, frame):
        # 1. Detect heads
        head_rois = self.detect_heads(frame)
        
        # 2. Motion tracking
        dets = np.array([roi.to_boxmot_format() for roi in head_rois]) if head_rois else np.empty((0, 6))
        
        motion_tracks = []
        if len(dets) > 0:
            tracks = self.motion_tracker.update(dets, frame)
            if tracks is not None and len(tracks) > 0:
                motion_tracks = [ROI.from_boxmot_format(t, class_id=0) for t in tracks]
        else:
            self.motion_tracker.update(dets, frame)
        
        # 3. Extract face crops
        face_crops, valid_tracks = self.extract_face_crops(frame, motion_tracks)
        
        # 4. Extract embeddings
        embeddings_list = []
        if len(face_crops) > 0:
            try:
                embeddings_list = self.edgeface.extract_embeddings(face_crops)
            except Exception as e:
                logging.error(f"Embedding extraction failed: {e}")
                embeddings_list = [None] * len(face_crops)
        
        # 5. Match with appearance + identify
        persons = self.match_with_appearance(valid_tracks, embeddings_list, face_crops, frame)
        
        # 6. Draw annotations
        annotated_frame = frame.copy()
        
        for person in persons:
            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            
            # Color: Green if identified, Blue if unknown
            color = (0, 255, 0) if person.identified_name else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw name/ID
            label = person.get_display_name()
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            cv2.circle(annotated_frame, center, 3, (0, 0, 255), -1)
        
        # Info overlay
        cv2.putText(annotated_frame, f"Tracked: {len(persons)} | Known DB: {len(self.database.known_persons)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_frame, persons
    
    def start_processing(self, video_device=0, video_resolution=(1280, 720), framerate=30):
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(video_device, video_resolution, framerate)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        try:
            cv2.namedWindow("Head Tracking with Names", cv2.WINDOW_NORMAL)
            while self.running:
                try:
                    frame = self.display_queue.get(timeout=1.0)
                    cv2.imshow("Head Tracking with Names", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                            
                except queue.Empty:
                    continue
        finally:
            self.running = False
            if self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            cv2.destroyAllWindows()
    
    def _capture_loop(self, video_device, video_resolution, framerate):
        try:
            cap = cv2.VideoCapture(video_device, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])
            cap.set(cv2.CAP_PROP_FPS, framerate)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag for real-time
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open camera {video_device}")
            
            fps_counter, fps_timer, fps = 0, time.time(), 0
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start = time.time()
                annotated, persons = self.process_frame(frame)
                processing_time = time.time() - start
                
                fps_counter += 1
                if time.time() - fps_timer > 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    fps_counter, fps_timer = 0, time.time()
                
                cv2.putText(annotated, f"FPS: {fps:.1f} | {processing_time*1000:.0f}ms",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                try:
                    self.display_queue.put(annotated, block=False)
                except queue.Full:
                    pass
        
        except Exception as e:
            logging.error(f"Capture error: {e}", exc_info=True)
        finally:
            if 'cap' in locals():
                cap.release()


def signal_handler(sig, frame):
    detector.running = False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    detector = HeadTrackerWithNameDisplay(
        detection_model_path="./od_model/yolov11.onnx",  # Also supports yolov11.onnx
        confidence_threshold=0.5,
        nms_threshold=0.6,
        frame_buffer_size=25,
        edgeface_model='edgeface_xxs',
        use_torchhub=True,
        profiles_dir="data/visual_embeddings",
        identification_threshold=0.2
    )
    
    detector.start_processing(video_device=0, video_resolution=(1280, 720), framerate=30)