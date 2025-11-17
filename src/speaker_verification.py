#!/usr/bin/env python3
"""
Single Speaker Real-Time Verification System
===========================================

A streamlined verification system for validating a single known speaker 
in real-time audio streams. Part of the PyAnnote multi-modal speaker 
recognition ecosystem.

Features:
- Real-time microphone verification
- Configurable similarity thresholds
- Continuous logging and monitoring
- Professional error handling
- Quality audio processing

Author: DevAgent Collaborative Development
Created: December 18, 2024 (Refactored: September 8, 2025)
Environment: pyannote-speaker-recognition
Dependencies: PyTorch, PyAnnote, SpeechBrain, SoundDevice
"""

import numpy as np
import sounddevice as sd
import torch
import pickle
import queue
import threading
import datetime
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from scipy.spatial.distance import cosine
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_THRESHOLD = 0.7
DEFAULT_CHUNK_DURATION = 3.0  # seconds
DEFAULT_SAMPLE_RATE = 16000
MIN_AUDIO_LEVEL = 0.001
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"


class RealtimeVerifier:
    """
    Real-time speaker verification system for single speaker validation
    
    This class provides real-time verification against a pre-recorded speaker profile,
    using SpeechBrain embeddings and cosine similarity matching.
    """
    
    def __init__(self, 
                 embedding_file: str, 
                 threshold: float = DEFAULT_THRESHOLD,
                 chunk_duration: float = DEFAULT_CHUNK_DURATION,
                 sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        """
        Initialize the real-time verifier
        
        Args:
            embedding_file: Path to the pre-computed embedding file (.pkl)
            threshold: Similarity threshold for verification (0.0-1.0)
            chunk_duration: Duration of audio chunks for processing (seconds)
            sample_rate: Audio sample rate (Hz)
            
        Raises:
            FileNotFoundError: If embedding file doesn't exist
            ValueError: If threshold is out of valid range
            RuntimeError: If embedding model fails to load
        """
        logger.info(f"Initializing RealtimeVerifier with {embedding_file}")
        
        # Validate inputs
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Store configuration
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # Load speaker profile
        self._load_speaker_profile(embedding_file)
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize processing components
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.session_start_time = datetime.datetime.now()
        
        # Initialize results logging
        self._initialize_logging()
        
        logger.info("RealtimeVerifier initialized successfully")
    
    def _load_speaker_profile(self, embedding_file: str) -> None:
        """Load and validate speaker profile from file"""
        try:
            with open(embedding_file, 'rb') as f:
                data = pickle.load(f)
            
            # Validate required fields
            required_fields = ['embedding', 'speaker_name']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields in profile: {missing_fields}")
            
            self.reference_embedding = data['embedding']
            self.speaker_name = data['speaker_name']
            self.profile_sample_rate = data.get('sample_rate', self.sample_rate)
            
            logger.info(f"Loaded profile for speaker: {self.speaker_name}")
            logger.info(f"Embedding shape: {self.reference_embedding.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load speaker profile: {e}")
            raise RuntimeError(f"Could not load speaker profile: {e}")
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the SpeechBrain embedding model"""
        try:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            self.embedding_model = PretrainedSpeakerEmbedding(
                MODEL_NAME,
                device=device
            )
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def _initialize_logging(self) -> None:
        """Initialize results logging"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        self.results_file = logs_dir / f"verification_results_{timestamp}.txt"
        
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Speaker Verification Session ===\n")
                f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Speaker: {self.speaker_name}\n")
                f.write(f"Threshold: {self.threshold}\n")
                f.write(f"Sample Rate: {self.sample_rate} Hz\n")
                f.write(f"Chunk Duration: {self.chunk_duration}s\n")
                f.write("=" * 50 + "\n\n")
            
            logger.info(f"Results will be logged to: {self.results_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize logging: {e}")
            raise RuntimeError(f"Could not initialize logging: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Callback for audio input stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Only queue audio if it's above minimum level
        if np.abs(indata).mean() > MIN_AUDIO_LEVEL:
            self.audio_queue.put(indata.copy())
    
    def _compute_similarity(self, audio_chunk: np.ndarray) -> float:
        """
        Compute similarity between audio chunk and reference embedding
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Similarity score (0.0-1.0)
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Prepare audio for embedding model
            audio_chunk = audio_chunk.flatten().astype(np.float32)
            
            # Normalize audio
            audio_chunk = audio_chunk - np.mean(audio_chunk)
            std = np.std(audio_chunk)
            if std > 0:
                audio_chunk = audio_chunk / std
            
            # Convert to torch tensor and add batch dimension
            waveform = torch.from_numpy(audio_chunk).unsqueeze(0).unsqueeze(0)
            waveform = waveform.to(self.embedding_model.device)
            
            # Generate embedding
            with torch.no_grad():
                current_embedding = self.embedding_model(waveform)
            
            # Compute cosine similarity
            # Ensure reference embedding is numpy array
            if hasattr(self.reference_embedding, 'cpu'):
                ref_embedding = self.reference_embedding.cpu().numpy().flatten()
            else:
                ref_embedding = self.reference_embedding.flatten()
            
            # Ensure current embedding is numpy array
            if hasattr(current_embedding, 'cpu'):
                curr_embedding = current_embedding.cpu().numpy().flatten()
            else:
                curr_embedding = current_embedding.flatten()
            
            similarity = 1 - cosine(ref_embedding, curr_embedding)
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise RuntimeError(f"Failed to compute similarity: {e}")
    
    def _log_result(self, timestamp: str, similarity: float, verified: bool) -> None:
        """Log verification results to file"""
        try:
            # Format the timestamp for better readability
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            
            # Determine confidence level and status
            if verified:
                status = "✅ VERIFIED"
                confidence = "HIGH" if similarity > 0.8 else "MEDIUM" if similarity > 0.6 else "LOW"
            else:
                status = "❌ REJECTED"
                confidence = "N/A"
            
            # Create readable log entry
            log_entry = f"[{current_time}] {status} | Similarity: {similarity:.3f} | Confidence: {confidence}\n"
            
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Error logging result: {e}")
    
    def start_verification(self) -> None:
        """Start real-time verification process"""
        if self.is_running:
            logger.warning("Verification already running")
            return
        
        logger.info("Starting real-time verification")
        self.is_running = True
        
        def processing_thread():
            """Audio processing thread"""
            try:
                with sd.InputStream(
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_samples,
                    callback=self._audio_callback
                ):
                    logger.info(f"Started verification for {self.speaker_name}")
                    logger.info(f"Sample rate: {self.sample_rate} Hz")
                    logger.info(f"Chunk duration: {self.chunk_duration}s")
                    logger.info(f"Threshold: {self.threshold}")
                    logger.info(f"Results logging to: {self.results_file}")
                    logger.info("Listening... (Call stop_verification() to stop)")
                    
                    while self.is_running:
                        try:
                            # Get audio chunk with timeout
                            audio_chunk = self.audio_queue.get(timeout=1.0)
                            
                            # Compute similarity
                            similarity = self._compute_similarity(audio_chunk)
                            verified = similarity > self.threshold
                            
                            # Create timestamp
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            
                            # Log result
                            self._log_result(timestamp, similarity, verified)
                            
                            # Print result
                            if verified:
                                logger.info(f"[{timestamp}] ✓ Verified {self.speaker_name} (similarity: {similarity:.3f})")
                            else:
                                logger.info(f"[{timestamp}] ✗ Different speaker (similarity: {similarity:.3f})")
                                
                        except queue.Empty:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error in audio stream: {e}")
                self.is_running = False
            finally:
                logger.info("Audio processing thread stopped")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=processing_thread, daemon=True)
        self.processing_thread.start()
    
    def _write_session_summary(self) -> None:
        """Write session summary to results file"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n" + "=" * 50 + "\n")
                f.write(f"Session ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total duration: {(datetime.datetime.now() - self.session_start_time).total_seconds():.1f} seconds\n")
                f.write("=" * 50 + "\n")
        except Exception as e:
            logger.error(f"Error writing session summary: {e}")

    def stop_verification(self) -> None:
        """Stop the verification process"""
        if not self.is_running:
            logger.warning("Verification not running")
            return
        
        logger.info("Stopping verification")
        self.is_running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Write session summary
        self._write_session_summary()
        
        logger.info(f"Verification stopped. Results saved to: {self.results_file}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current verification status"""
        return {
            'running': self.is_running,
            'speaker_name': self.speaker_name,
            'threshold': self.threshold,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'results_file': self.results_file,
            'queue_size': self.audio_queue.qsize()
        }


def main() -> None:
    """Main function for interactive speaker verification"""
    print("\n🎤 Single Speaker Verification System")
    print("=" * 50)
    print("DevAgent PyAnnote Speaker Recognition")
    print()
    
    try:
        # Check for embeddings directory
        embeddings_dir = Path("data/embeddings")
        if not embeddings_dir.exists():
            print("❌ No 'data/embeddings' directory found.")
            print("   Please create speaker profiles first using:")
            print("   python src/audio_profile_creator.py")
            return
        
        # Get available embedding files
        embedding_files = list(embeddings_dir.glob("*.pkl"))
        
        if not embedding_files:
            print("❌ No embedding files found in 'data/embeddings' directory.")
            print("   Please create speaker profiles first.")
            return
        
        # Display available speakers
        print("📋 Available speaker embeddings:")
        for i, file in enumerate(embedding_files, 1):
            print(f"   {i}. {file.name}")
        
        # Get user selection
        while True:
            try:
                selection = input(f"\nSelect embedding file (1-{len(embedding_files)}): ").strip()
                selection_idx = int(selection) - 1
                
                if 0 <= selection_idx < len(embedding_files):
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(embedding_files)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
        
        embedding_file = str(embedding_files[selection_idx])
        
        # Get threshold (optional)
        try:
            threshold_input = input(f"Enter similarity threshold (default {DEFAULT_THRESHOLD}): ").strip()
            threshold = float(threshold_input) if threshold_input else DEFAULT_THRESHOLD
            threshold = max(0.0, min(1.0, threshold))  # Clamp to valid range
        except ValueError:
            threshold = DEFAULT_THRESHOLD
            print(f"Using default threshold: {threshold}")
        
        # Create and start verifier
        print(f"\n🔧 Initializing verification for: {Path(embedding_file).stem}")
        verifier = RealtimeVerifier(embedding_file, threshold=threshold)
        
        print(f"\n🎯 Configuration:")
        status = verifier.get_status()
        for key, value in status.items():
            if key != 'queue_size':  # Skip dynamic values
                print(f"   {key}: {value}")
        
        print(f"\n🎤 Starting verification...")
        verifier.start_verification()
        
        try:
            input("\nPress Enter to stop verification...\n")
        except KeyboardInterrupt:
            print("\nStopping verification...")
        finally:
            verifier.stop_verification()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
