#!/usr/bin/env python3
"""
Multi-Speaker Real-Time Verification System
==========================================

Advanced verification system for identifying multiple known speakers in 
real-time audio streams. Part of the PyAnnote multi-modal speaker recognition 
ecosystem with comprehensive speaker database support.

Features:
- Multi-speaker simultaneous verification
- Real-time microphone and file processing
- High-quality audio resampling
- Comprehensive logging and monitoring
- Configurable similarity thresholds
- Professional error handling and recovery

Author: DevAgent Collaborative Development
Created: December 18, 2024 (Refactored: September 8, 2025)
Environment: pyannote-speaker-recognition
Dependencies: PyTorch, PyAnnote, SpeechBrain, SoundDevice, SciPy
"""

import numpy as np
import sounddevice as sd
import wave
import torch
import pickle
import queue
import threading
import datetime
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.spatial.distance import cosine
from scipy import signal
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHUNK_DURATION = 3.0  # seconds
DEFAULT_TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_LEVEL = 0.001
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
UNKNOWN_SPEAKER = "Unknown"
RESAMPLING_WINDOW = ('kaiser', 5.0)


class SpeakerProfile:
    """Data class to hold speaker profile information"""
    
    def __init__(self, name: str, embedding: np.ndarray, filename: str, 
                 quality_score: float = 0.0, metadata: Dict[str, Any] = None):
        self.name = name
        self.embedding = embedding
        self.filename = filename
        self.quality_score = quality_score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"SpeakerProfile(name='{self.name}', quality={self.quality_score:.3f})"


class MultiSpeakerVerifier:
    """
    Multi-speaker verification system for real-time speaker identification
    
    This class provides comprehensive speaker verification against multiple known
    speaker profiles, with support for real-time microphone input and audio file
    processing.
    """
    
    def __init__(self, 
                 embedding_directory: str,
                 threshold: float = DEFAULT_THRESHOLD,
                 target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
                 chunk_duration: float = DEFAULT_CHUNK_DURATION) -> None:
        """
        Initialize the multi-speaker verifier
        
        Args:
            embedding_directory: Directory containing speaker embedding files (.pkl)
            threshold: Similarity threshold for verification (0.0-1.0)
            target_sample_rate: Required sample rate for the model (Hz)
            chunk_duration: Duration of audio chunks for processing (seconds)
            
        Raises:
            FileNotFoundError: If embedding directory doesn't exist
            ValueError: If threshold is out of valid range
            RuntimeError: If no valid speaker profiles found or model fails to load
        """
        logger.info(f"Initializing MultiSpeakerVerifier from {embedding_directory}")
        
        # Validate inputs
        if not os.path.exists(embedding_directory):
            raise FileNotFoundError(f"Embedding directory not found: {embedding_directory}")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Store configuration
        self.embedding_directory = embedding_directory
        self.threshold = threshold
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(target_sample_rate * chunk_duration)
        
        # Load speaker profiles
        self.speakers = self._load_speaker_embeddings(embedding_directory)
        if not self.speakers:
            raise RuntimeError(f"No valid speaker profiles found in {embedding_directory}")
        
        logger.info(f"Loaded {len(self.speakers)} speaker profiles")
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize processing components
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.session_start_time = datetime.datetime.now()
        
        # Initialize results logging
        self._initialize_logging()
        
        logger.info("MultiSpeakerVerifier initialized successfully")
    
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
        
        self.results_file = logs_dir / f"multi_speaker_results_{timestamp}.txt"
        
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Multi-Speaker Verification Session ===\n")
                f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Speakers loaded: {len(self.speakers)}\n")
                f.write(f"Speaker names: {', '.join([speaker.name for speaker in self.speakers])}\n")
                f.write(f"Threshold: {self.threshold}\n")
                f.write(f"Sample Rate: {self.target_sample_rate} Hz\n")
                f.write("=" * 50 + "\n\n")
            
            logger.info(f"Results will be logged to: {self.results_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize logging: {e}")
            raise RuntimeError(f"Could not initialize logging: {e}")
    
    def resample_audio(self, audio_data: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate using high-quality resampling
        
        Args:
            audio_data: Audio samples as numpy array
            original_sample_rate: Original sample rate of the audio (Hz)
            
        Returns:
            Resampled audio data
            
        Raises:
            ValueError: If sample rates are invalid
        """
        if original_sample_rate <= 0 or self.target_sample_rate <= 0:
            raise ValueError("Sample rates must be positive")
        
        if original_sample_rate == self.target_sample_rate:
            return audio_data
        
        try:
            logger.debug(f"Resampling from {original_sample_rate}Hz to {self.target_sample_rate}Hz")
            
            # Use high-quality polyphase filtering
            resampled = signal.resample_poly(
                audio_data,
                up=self.target_sample_rate,
                down=original_sample_rate,
                window=RESAMPLING_WINDOW
            )
            
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            raise RuntimeError(f"Failed to resample audio: {e}")
    
    def _load_speaker_embeddings(self, directory: str) -> List[SpeakerProfile]:
        """
        Load all speaker embeddings from the specified directory
        
        Args:
            directory: Path to directory containing .pkl embedding files
            
        Returns:
            List of SpeakerProfile objects
        """
        speakers = []
        directory_path = Path(directory)
        
        for filepath in directory_path.glob("*.pkl"):
            try:
                logger.debug(f"Loading profile: {filepath.name}")
                
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                # Validate required fields
                if 'speaker_name' not in data or 'embedding' not in data:
                    logger.warning(f"Missing required fields in {filepath.name}")
                    continue
                
                # Extract metadata
                quality_score = 0.0
                if 'metadata' in data and isinstance(data['metadata'], dict):
                    quality_score = data['metadata'].get('audio_quality_score', 0.0)
                
                # Create speaker profile
                profile = SpeakerProfile(
                    name=data['speaker_name'],
                    embedding=data['embedding'],
                    filename=filepath.name,
                    quality_score=float(quality_score),
                    metadata=data.get('metadata', {})
                )
                
                speakers.append(profile)
                logger.debug(f"Loaded profile for {profile.name} (quality: {profile.quality_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error loading {filepath.name}: {e}")
                continue
        
        # Sort by quality score (highest first)
        speakers.sort(key=lambda x: x.quality_score, reverse=True)
        
        return speakers
    
    def _compute_similarities(self, audio_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Compute similarities between input audio and all stored speaker embeddings
        
        Args:
            audio_embedding: Input audio embedding
            
        Returns:
            Tuple of (best_speaker_name, similarity_score)
        """
        similarities = []
        
        for speaker in self.speakers:
            try:
                # Compute cosine similarity
                similarity = 1 - cosine(
                    speaker.embedding.flatten(),
                    audio_embedding.flatten()
                )
                
                # Clamp similarity to valid range
                similarity = max(0.0, min(1.0, similarity))
                similarities.append((speaker.name, similarity))
                
            except Exception as e:
                logger.error(f"Error computing similarity for {speaker.name}: {e}")
                similarities.append((speaker.name, 0.0))
        
        # Find the best match
        if similarities:
            best_match = max(similarities, key=lambda x: x[1])
            return best_match if best_match[1] > self.threshold else (UNKNOWN_SPEAKER, best_match[1])
        else:
            return (UNKNOWN_SPEAKER, 0.0)
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: Optional[int] = None) -> Tuple[str, float]:
        """
        Process a chunk of audio and return speaker verification result
        
        Args:
            audio_chunk: Audio data as numpy array
            sample_rate: Sample rate of the audio (if different from target)
            
        Returns:
            Tuple of (speaker_name, similarity_score)
        """
        try:
            # Resample if necessary
            if sample_rate and sample_rate != self.target_sample_rate:
                audio_chunk = self.resample_audio(audio_chunk, sample_rate)
            
            # Convert to mono if stereo
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.mean(axis=1)
            
            # Convert to float32 and normalize
            audio_chunk = audio_chunk.astype(np.float32)
            audio_chunk = audio_chunk - np.mean(audio_chunk)
            std = np.std(audio_chunk)
            if std > 0:
                audio_chunk = audio_chunk / std
            
            # Prepare for model input (batch, channel, samples)
            waveform = torch.from_numpy(audio_chunk).unsqueeze(0).unsqueeze(0)
            waveform = waveform.to(self.embedding_model.device)
            
            # Generate embedding
            with torch.no_grad():
                current_embedding = self.embedding_model(waveform)
            
            # Convert to numpy if it's a torch tensor
            if hasattr(current_embedding, 'cpu'):
                embedding_array = current_embedding.cpu().numpy()
            else:
                embedding_array = current_embedding
            
            # Compare with all speakers
            return self._compute_similarities(embedding_array)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return (UNKNOWN_SPEAKER, 0.0)
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Callback for microphone input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Only queue audio if it's above minimum level
        if np.abs(indata).mean() > MIN_AUDIO_LEVEL:
            self.audio_queue.put(indata.copy())
    
    def _log_result(self, timestamp: str, speaker: str, similarity: float) -> None:
        """Log verification results to file"""
        try:
            # Format the timestamp for better readability
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            
            # Determine confidence level and status
            if speaker == UNKNOWN_SPEAKER:
                status = "❓ UNKNOWN"
                confidence = "N/A"
            else:
                status = f"🎯 IDENTIFIED: {speaker}"
                confidence = "HIGH" if similarity > 0.8 else "MEDIUM" if similarity > 0.6 else "LOW"
            
            # Create readable log entry
            log_entry = f"[{current_time}] {status} | Similarity: {similarity:.3f} | Confidence: {confidence}\n"
            
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Error logging result: {e}")
    
    def process_audio_file(self, audio_file: str) -> List[Tuple[str, str, float]]:
        """
        Process a pre-recorded audio file for speaker verification
        
        Args:
            audio_file: Path to WAV audio file
            
        Returns:
            List of (timestamp, speaker_name, similarity_score) tuples
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If file processing fails
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        logger.info(f"Processing audio file: {audio_file}")
        results = []
        
        try:
            with wave.open(audio_file, 'rb') as wf:
                # Get file properties
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames_total = wf.getnframes()
                duration = frames_total / sample_rate
                
                logger.info(f"File properties: {sample_rate}Hz, {channels}ch, {duration:.1f}s")
                
                # Calculate chunk size in original sample rate
                original_chunk_samples = int(self.chunk_duration * sample_rate)
                chunk_count = 0
                
                while True:
                    frames = wf.readframes(original_chunk_samples)
                    if not frames:
                        break
                    
                    chunk_count += 1
                    
                    # Convert bytes to numpy array
                    if wf.getsampwidth() == 2:  # 16-bit
                        audio_chunk = np.frombuffer(frames, dtype=np.int16)
                        audio_chunk = audio_chunk.astype(np.float32) / 32767.0
                    elif wf.getsampwidth() == 4:  # 32-bit
                        audio_chunk = np.frombuffer(frames, dtype=np.int32)
                        audio_chunk = audio_chunk.astype(np.float32) / 2147483647.0
                    else:
                        logger.warning(f"Unsupported sample width: {wf.getsampwidth()}")
                        continue
                    
                    # Convert to mono if stereo
                    if channels > 1:
                        audio_chunk = audio_chunk.reshape(-1, channels).mean(axis=1)
                    
                    # Skip processing if audio chunk is too quiet
                    if np.abs(audio_chunk).mean() < MIN_AUDIO_LEVEL:
                        continue
                    
                    # Process chunk
                    speaker, similarity = self._process_audio_chunk(audio_chunk, sample_rate)
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    
                    # Log and store result
                    self._log_result(timestamp, speaker, similarity)
                    results.append((timestamp, speaker, similarity))
                    
                    # Print progress
                    status = "✓" if speaker != UNKNOWN_SPEAKER else "?"
                    logger.info(f"[{timestamp}] {status} Chunk {chunk_count}: {speaker} (similarity: {similarity:.3f})")
                
                logger.info(f"Processed {chunk_count} chunks from {audio_file}")
                return results
                
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise RuntimeError(f"Failed to process audio file: {e}")
    
    def start_microphone_verification(self, device_sample_rate: Optional[int] = None) -> None:
        """
        Start real-time verification from microphone input
        
        Args:
            device_sample_rate: Optional sample rate for the input device
        """
        if self.is_running:
            logger.warning("Verification already running")
            return
        
        logger.info("Starting microphone verification")
        self.is_running = True
        
        # Use device sample rate if provided, otherwise use target rate
        input_sample_rate = device_sample_rate or self.target_sample_rate
        input_chunk_samples = int(self.chunk_duration * input_sample_rate)
        
        def processing_thread():
            """Audio processing thread"""
            try:
                with sd.InputStream(
                    channels=1,
                    samplerate=input_sample_rate,
                    blocksize=input_chunk_samples,
                    callback=self._audio_callback
                ):
                    logger.info("Started real-time verification")
                    logger.info(f"Input sample rate: {input_sample_rate} Hz")
                    logger.info(f"Target sample rate: {self.target_sample_rate} Hz")
                    logger.info(f"Chunk duration: {self.chunk_duration}s")
                    logger.info(f"Threshold: {self.threshold}")
                    logger.info(f"Loaded speakers: {[s.name for s in self.speakers]}")
                    logger.info(f"Results logging to: {self.results_file}")
                    logger.info("Listening... (Call stop_verification() to stop)")
                    
                    while self.is_running:
                        try:
                            audio_chunk = self.audio_queue.get(timeout=1.0)
                            
                            # Process chunk with input sample rate for resampling
                            speaker, similarity = self._process_audio_chunk(
                                audio_chunk, input_sample_rate)
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            
                            # Log and print result
                            self._log_result(timestamp, speaker, similarity)
                            
                            status = "✓" if speaker != UNKNOWN_SPEAKER else "?"
                            logger.info(f"[{timestamp}] {status} Detected: {speaker} (similarity: {similarity:.3f})")
                            
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
                f.write(f"Speakers available: {', '.join(self.speaker_profiles.keys())}\n")
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
    
    def get_speaker_profiles(self) -> List[SpeakerProfile]:
        """Get list of loaded speaker profiles"""
        return self.speakers.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current verification status"""
        return {
            'running': self.is_running,
            'num_speakers': len(self.speakers),
            'speaker_names': [s.name for s in self.speakers],
            'threshold': self.threshold,
            'target_sample_rate': self.target_sample_rate,
            'chunk_duration': self.chunk_duration,
            'results_file': self.results_file,
            'queue_size': self.audio_queue.qsize() if hasattr(self, 'audio_queue') else 0
        }


def main() -> None:
    """Main function for interactive multi-speaker verification"""
    print("\n👥 Multi-Speaker Verification System")
    print("=" * 50)
    print("DevAgent PyAnnote Speaker Recognition")
    print()
    
    try:
        # Check for embeddings directory
        embeddings_dir = "data/embeddings"
        if not os.path.exists(embeddings_dir):
            print("❌ No 'data/embeddings' directory found.")
            print("   Please create speaker profiles first using:")
            print("   python src/audio_profile_creator.py")
            return
        
        # Initialize verifier
        print("🔧 Initializing multi-speaker verifier...")
        verifier = MultiSpeakerVerifier(embeddings_dir)
        
        # Display status
        status = verifier.get_status()
        print(f"\n📊 System Status:")
        print(f"   Speakers loaded: {status['num_speakers']}")
        print(f"   Speaker names: {', '.join(status['speaker_names'])}")
        print(f"   Threshold: {status['threshold']}")
        print(f"   Sample rate: {status['target_sample_rate']} Hz")
        
        # Main menu loop
        while True:
            print(f"\n🎯 Select verification mode:")
            print("   1. Real-time microphone input")
            print("   2. Process audio file")
            print("   3. Show speaker profiles")
            print("   4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                # Real-time microphone verification
                try:
                    device_rate_input = input("Enter microphone sample rate (or press Enter for default): ").strip()
                    device_rate = int(device_rate_input) if device_rate_input else None
                except ValueError:
                    device_rate = None
                    print("Using default sample rate")
                
                try:
                    print(f"\n🎤 Starting microphone verification...")
                    verifier.start_microphone_verification(device_rate)
                    input("\nPress Enter to stop verification...\n")
                finally:
                    verifier.stop_verification()
                    
            elif choice == "2":
                # Audio file processing
                audio_file = input("\nEnter path to WAV file: ").strip()
                if os.path.exists(audio_file):
                    try:
                        print(f"\n🎵 Processing audio file...")
                        results = verifier.process_audio_file(audio_file)
                        
                        if results:
                            print(f"\n📊 Processing Summary:")
                            speakers_detected = {}
                            for _, speaker, similarity in results:
                                if speaker not in speakers_detected:
                                    speakers_detected[speaker] = []
                                speakers_detected[speaker].append(similarity)
                            
                            for speaker, similarities in speakers_detected.items():
                                avg_sim = np.mean(similarities)
                                count = len(similarities)
                                print(f"   {speaker}: {count} chunks, avg similarity: {avg_sim:.3f}")
                        
                    except Exception as e:
                        print(f"❌ Error processing file: {e}")
                else:
                    print("❌ File not found!")
                    
            elif choice == "3":
                # Show speaker profiles
                profiles = verifier.get_speaker_profiles()
                print(f"\n👥 Loaded Speaker Profiles ({len(profiles)}):")
                print("-" * 60)
                for i, profile in enumerate(profiles, 1):
                    print(f"   {i}. {profile.name}")
                    print(f"      Quality: {profile.quality_score:.3f}")
                    print(f"      File: {profile.filename}")
                    print(f"      Embedding shape: {profile.embedding.shape}")
                    if profile.metadata:
                        created = profile.metadata.get('created_at', 'Unknown')
                        print(f"      Created: {created}")
                    print()
                    
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice! Please select 1-4.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
