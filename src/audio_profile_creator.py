#!/usr/bin/env python3
"""
Enhanced Audio Profile Creator
=============================

A clean, modular system for creating speaker audio profiles with embeddings
for multi-modal speaker recognition. Integrates with the pyannote project's
unified architecture strategy.

Features:
- Interactive and batch profile creation
- Quality validation and feedback
- Standardized embedding generation
- Professional file management
- Integration with existing speaker profile ecosystem

Author: DevAgent Collaborative Development
Created: September 8, 2025
"""

import numpy as np
import sounddevice as sd
import wave
import os
import torch
import pickle
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioProfileMetadata:
    """Metadata for audio profiles with comprehensive tracking"""
    speaker_name: str
    created_at: str
    sample_rate: int
    duration_seconds: float
    model_name: str
    embedding_dimension: int
    audio_quality_score: float
    recording_environment: str
    profile_version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass 
class SpeakerProfile:
    """Enhanced speaker profile with quality metrics and validation"""
    name: str
    embedding: np.ndarray
    metadata: AudioProfileMetadata
    
    def save_to_file(self, filepath: Path) -> None:
        """Save profile with metadata to pickle file"""
        data = {
            'speaker_name': self.name,
            'embedding': self.embedding,
            'metadata': self.metadata.to_dict(),
            # Legacy compatibility fields
            'created_at': self.metadata.created_at,
            'sample_rate': self.metadata.sample_rate,
            'model_name': self.metadata.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Speaker profile saved: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'SpeakerProfile':
        """Load profile from pickle file with backward compatibility"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle legacy format
        if 'metadata' not in data:
            metadata = AudioProfileMetadata(
                speaker_name=data['speaker_name'],
                created_at=data.get('created_at', 'unknown'),
                sample_rate=data.get('sample_rate', 16000),
                duration_seconds=0.0,
                model_name=data.get('model_name', 'speechbrain/spkrec-ecapa-voxceleb'),
                embedding_dimension=data['embedding'].shape[-1] if len(data['embedding'].shape) > 0 else 0,
                audio_quality_score=0.0,
                recording_environment='unknown'
            )
        else:
            metadata = AudioProfileMetadata(**data['metadata'])
        
        return cls(
            name=data['speaker_name'],
            embedding=data['embedding'],
            metadata=metadata
        )


class AudioQualityAnalyzer:
    """Analyze audio quality and provide feedback"""
    
    @staticmethod
    def analyze_audio_quality(audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze audio quality and return metrics"""
        
        # Basic quality metrics
        rms_level = np.sqrt(np.mean(audio_data**2))
        peak_level = np.max(np.abs(audio_data))
        zero_crossing_rate = np.mean(np.diff(np.signbit(audio_data)).astype(float))
        
        # Signal-to-noise ratio estimation (simple)
        sorted_audio = np.sort(np.abs(audio_data))
        noise_floor = np.mean(sorted_audio[:len(sorted_audio)//10])  # Bottom 10%
        signal_power = rms_level
        snr_estimate = 20 * np.log10(signal_power / (noise_floor + 1e-10))
        
        # Quality score (0-1)
        quality_score = min(1.0, max(0.0, (snr_estimate - 10) / 30))  # 10-40 dB range
        
        # Quality assessment
        if quality_score > 0.8:
            quality_level = "Excellent"
        elif quality_score > 0.6:
            quality_level = "Good"
        elif quality_score > 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'rms_level': float(rms_level),
            'peak_level': float(peak_level),
            'snr_estimate': float(snr_estimate),
            'zero_crossing_rate': float(zero_crossing_rate),
            'recommendations': AudioQualityAnalyzer._get_recommendations(quality_score, rms_level, peak_level)
        }
    
    @staticmethod
    def _get_recommendations(quality_score: float, rms_level: float, peak_level: float) -> List[str]:
        """Provide improvement recommendations based on quality metrics"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Consider recording in a quieter environment")
            
        if rms_level < 0.01:
            recommendations.append("Speak louder or move closer to the microphone")
        elif rms_level > 0.3:
            recommendations.append("Reduce microphone gain or speak more softly")
            
        if peak_level > 0.95:
            recommendations.append("Audio is clipping - reduce input volume")
            
        if quality_score > 0.8:
            recommendations.append("Excellent audio quality - profile will work well for recognition")
            
        return recommendations


class EnhancedAudioProfileCreator:
    """Enhanced audio profile creator with quality validation and user experience improvements"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 default_duration: int = 20,
                 embeddings_dir: str = "data/embeddings",
                 outputs_dir: str = "data/outputs"):
        """
        Initialize the enhanced audio profile creator
        
        Args:
            sample_rate: Audio sample rate (16000 Hz required for SpeechBrain)
            default_duration: Default recording duration in seconds
            embeddings_dir: Directory to save embedding files
            outputs_dir: Directory to save audio files
        """
        self.sample_rate = sample_rate
        self.default_duration = default_duration
        self.embeddings_dir = Path(embeddings_dir)
        self.outputs_dir = Path(outputs_dir)
        
        # Create directories
        self.embeddings_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_embedding_model()
        self.quality_analyzer = AudioQualityAnalyzer()
        
        logger.info("Enhanced Audio Profile Creator initialized successfully")
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the SpeechBrain embedding model"""
        logger.info("Loading SpeechBrain speaker embedding model...")
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use a local directory for the model to avoid symlink issues on Windows
            # and set run_opts to use the correct device
            save_dir = os.path.join(os.getcwd(), "pretrained_models", "spkrec-ecapa-voxceleb")
            os.makedirs(save_dir, exist_ok=True)
            
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=save_dir,
                run_opts={"device": device}
            )
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _get_recording_instructions(self, duration: int) -> None:
        """Provide clear recording instructions to the user"""
        print(f"\n{'='*60}")
        print("AUDIO PROFILE RECORDING INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"Duration: {duration} seconds")
        print("\nFor best results:")
        print("• Speak in a normal, conversational tone")
        print("• Include some pauses and varied speech patterns")
        print("• Avoid background noise if possible")
        print("• Stay at a consistent distance from the microphone")
        print("\nSample text (feel free to improvise):")
        print('"Hello, this is my voice profile for speaker recognition.')
        print('I am recording this sample to create a unique audio fingerprint.')
        print('The system will use this to identify me in future conversations.')
        print('I will speak naturally and clearly for the best results."')
        print(f"\n{'='*60}")
    
    def record_audio_with_countdown(self, duration: int) -> np.ndarray:
        """Record audio with user-friendly countdown and feedback"""
        
        self._get_recording_instructions(duration)
        input("\nPress Enter when ready to start recording...")
        
        print("\nPreparing to record...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            sd.sleep(1000)
        
        print("🎤 RECORDING NOW! Speak clearly...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        # Wait for recording to complete
        sd.wait()
        
        print("✅ Recording complete!")
        return audio_data.flatten()  # Ensure 1D array
    
    def analyze_and_validate_audio(self, audio_data: np.ndarray) -> Tuple[bool, Dict]:
        """Analyze audio quality and determine if it's suitable for profile creation"""
        
        print("\n🔍 Analyzing audio quality...")
        quality_metrics = self.quality_analyzer.analyze_audio_quality(audio_data, self.sample_rate)
        
        print(f"\nAudio Quality Assessment:")
        print(f"  Overall Quality: {quality_metrics['quality_level']} ({quality_metrics['quality_score']:.2f}/1.0)")
        print(f"  Signal Level: {quality_metrics['rms_level']:.3f}")
        print(f"  Peak Level: {quality_metrics['peak_level']:.3f}")
        print(f"  SNR Estimate: {quality_metrics['snr_estimate']:.1f} dB")
        
        if quality_metrics['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in quality_metrics['recommendations']:
                print(f"  • {rec}")
        
        # Determine if audio is acceptable
        is_acceptable = quality_metrics['quality_score'] >= 0.3  # Minimum threshold
        
        if not is_acceptable:
            print(f"\n⚠️  Audio quality is below recommended threshold.")
            print("Consider re-recording for better recognition performance.")
        
        return is_acceptable, quality_metrics

    def generate_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate speaker embedding from audio data"""
        
        logger.info("Generating speaker embedding...")
        
        try:
            # Prepare audio tensor
            waveform = torch.from_numpy(audio_data).float()
            
            # Ensure correct shape for the model
            # SpeechBrain expects (batch, time)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Add batch dimension
            
            # Normalize audio (SpeechBrain usually handles this, but good practice)
            # waveform = waveform - waveform.mean()
            # waveform = waveform / (waveform.std() + 1e-8)
            
            # Move to model device (handled by encode_batch if run_opts set correctly?)
            # Usually better to send to device manually if needed, but encode_batch handles it if input is on device
            device = self.embedding_model.device
            waveform = waveform.to(device)
            
            # Generate embedding
            with torch.no_grad():
                # encode_batch returns (batch, 1, emb_dim)
                embedding = self.embedding_model.encode_batch(waveform)
            
            # Convert to numpy
            # embedding is (batch, 1, 192) -> squeeze to (192,)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.squeeze().cpu().numpy()
            
            logger.info(f"Embedding generated successfully - shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def save_audio_file(self, audio_data: np.ndarray, speaker_name: str) -> Path:
        """Save audio data to WAV file"""
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.outputs_dir / f"{speaker_name}_{timestamp}.wav"
        
        try:
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                
                # Convert to 16-bit PCM
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"Audio file saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            raise
    
    def create_speaker_profile(self, 
                             speaker_name: str, 
                             audio_data: np.ndarray, 
                             quality_metrics: Dict,
                             recording_environment: str = "standard") -> SpeakerProfile:
        """Create a complete speaker profile with metadata"""
        
        # Generate embedding
        embedding = self.generate_embedding(audio_data)
        
        # Create metadata
        metadata = AudioProfileMetadata(
            speaker_name=speaker_name,
            created_at=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            sample_rate=self.sample_rate,
            duration_seconds=len(audio_data) / self.sample_rate,
            model_name="speechbrain/spkrec-ecapa-voxceleb",
            embedding_dimension=embedding.shape[-1] if len(embedding.shape) > 0 else 0,
            audio_quality_score=quality_metrics['quality_score'],
            recording_environment=recording_environment
        )
        
        return SpeakerProfile(
            name=speaker_name,
            embedding=embedding,
            metadata=metadata
        )
    
    def save_speaker_profile(self, profile: SpeakerProfile) -> Path:
        """Save speaker profile to embeddings directory"""
        
        filename = self.embeddings_dir / f"{profile.name}_{profile.metadata.created_at}.pkl"
        profile.save_to_file(filename)
        return filename
    
    def create_profile_interactive(self) -> Optional[SpeakerProfile]:
        """Interactive profile creation with quality validation and retry logic"""
        
        print("\n🎯 Enhanced Audio Profile Creator")
        print("=" * 50)
        
        # Get speaker information
        speaker_name = input("Enter speaker name: ").strip()
        if not speaker_name:
            print("❌ Speaker name is required")
            return None
        
        # Check for existing profiles
        existing_profiles = list(self.embeddings_dir.glob(f"{speaker_name}_*.pkl"))
        if existing_profiles:
            print(f"\n📋 Found {len(existing_profiles)} existing profile(s) for '{speaker_name}':")
            for profile_path in existing_profiles:
                print(f"  • {profile_path.name}")
            
            overwrite = input("\nCreate additional profile? (y/n): ").lower().strip()
            if overwrite != 'y':
                print("Profile creation cancelled.")
                return None
        
        # Get recording preferences
        try:
            duration_input = input(f"\nRecording duration in seconds (default: {self.default_duration}): ").strip()
            duration = int(duration_input) if duration_input else self.default_duration
            duration = max(5, min(60, duration))  # Limit to reasonable range
        except ValueError:
            duration = self.default_duration
        
        environment = input("Recording environment (e.g., 'office', 'home', 'studio'): ").strip() or "standard"
        
        # Recording and validation loop
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"\n📢 Recording attempt {attempt + 1}/{max_attempts}")
            
            # Record audio
            audio_data = self.record_audio_with_countdown(duration)
            
            # Analyze quality
            is_acceptable, quality_metrics = self.analyze_and_validate_audio(audio_data)
            
            if is_acceptable or attempt == max_attempts - 1:
                # Accept the recording (either good quality or final attempt)
                break
            else:
                retry = input("\nWould you like to try recording again? (y/n): ").lower().strip()
                if retry != 'y':
                    print("Using current recording despite quality concerns.")
                    break
        
        try:
            # Create profile
            print("\n🧠 Creating speaker profile...")
            profile = self.create_speaker_profile(speaker_name, audio_data, quality_metrics, environment)
            
            # Save files
            audio_file = self.save_audio_file(audio_data, speaker_name)
            profile_file = self.save_speaker_profile(profile)
            
            # Success summary
            print(f"\n✅ Profile creation successful!")
            print(f"📁 Audio file: {audio_file}")
            print(f"🧠 Profile file: {profile_file}")
            print(f"📊 Quality score: {quality_metrics['quality_score']:.2f}/1.0")
            print(f"🎯 Embedding dimension: {profile.metadata.embedding_dimension}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            print(f"❌ Error creating profile: {e}")
            return None
    
    def list_existing_profiles(self) -> List[SpeakerProfile]:
        """List all existing speaker profiles"""
        
        profile_files = list(self.embeddings_dir.glob("*.pkl"))
        profiles = []
        
        print(f"\n📋 Existing Speaker Profiles ({len(profile_files)} found):")
        print("-" * 60)
        
        for profile_file in sorted(profile_files):
            try:
                profile = SpeakerProfile.load_from_file(profile_file)
                profiles.append(profile)
                
                print(f"👤 {profile.name}")
                print(f"   📅 Created: {profile.metadata.created_at}")
                print(f"   🎯 Quality: {profile.metadata.audio_quality_score:.2f}/1.0")
                print(f"   📁 File: {profile_file.name}")
                print()
                
            except Exception as e:
                logger.warning(f"Could not load profile {profile_file}: {e}")
                print(f"⚠️  {profile_file.name} (load error)")
        
        return profiles


def main():
    """Main function for interactive audio profile creation"""
    
    try:
        # Initialize creator
        creator = EnhancedAudioProfileCreator()
        
        while True:
            print("\n" + "="*60)
            print("🎤 ENHANCED AUDIO PROFILE CREATOR")
            print("="*60)
            print("1. Create new speaker profile")
            print("2. List existing profiles")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                profile = creator.create_profile_interactive()
                if profile:
                    print("\n🎉 Ready for speaker recognition!")
                    
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
