#!/usr/bin/env python3
"""
Multi-Speaker Transcription and Separation System (Multi-View Hybrid)
===================================================================

This system overcomes separation model limits (max 3 outputs) by using a 
Multi-View approach:
1. It separates audio using a 3-speaker model to get cleaner stems.
2. It transcribes BOTH the separated stems AND the original mix.
3. It performs ID on specific time segments for every sentence found.
4. It merges results, allowing detection of 4, 5, or more speakers.

Usage:
    python src/transcribe_multispeaker.py <audio_file> [--speakers N]

Author: DevAgent
"""

import os
import sys
import argparse
import logging
import torch
import torchaudio
import numpy as np
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import timedelta

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from speechbrain.inference.separation import SepformerSeparation
    from faster_whisper import WhisperModel
    from multi_speaker_verification import MultiSpeakerVerifier
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiSpeakerTranscriber:
    def __init__(self, 
                 embeddings_dir: str = "data/embeddings", 
                 model_size: str = "medium", 
                 device: str = "auto"):
        self.embeddings_dir = embeddings_dir
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")

        # 1. Initialize Speaker Verifier
        if os.path.exists(embeddings_dir):
            try:
                self.verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.25) # Low threshold for robustness
                logger.info(f"Loaded {len(self.verifier.speakers)} speaker profiles.")
            except Exception as e:
                logger.warning(f"Could not initialize verifier: {e}")
                self.verifier = None
        else:
            self.verifier = None

        # 2. Initialize Whisper
        logger.info(f"Loading Whisper model ({model_size})...")
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.whisper = WhisperModel(model_size, device=self.device, compute_type=compute_type)

        # 3. Initialize Separation Model (Using 3-mix as the workhorse)
        logger.info("Loading Separation model...")
        try:
            self.sep_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-libri3mix",
                savedir="pretrained_models/sepformer-libri3mix",
                run_opts={"device": self.device}
            )
            logger.info("Separation model loaded.")
        except Exception as e:
            logger.error(f"Failed to load separation model: {e}")
            raise

    def _identify_segment(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Identify speaker in a specific audio segment."""
        if not self.verifier:
            return "Unknown", 0.0
        
        try:
            name, score = self.verifier._process_audio_chunk(audio_data, sample_rate=sample_rate)
            return name, score
        except Exception:
            return "Unknown", 0.0

    def _is_duplicate(self, segment: Dict, existing_transcript: List[Dict]) -> bool:
        """Check if segment is a duplicate of an existing one based on time and text."""
        for existing in existing_transcript:
            # Check time overlap
            start_overlap = max(segment['start'], existing['start'])
            end_overlap = min(segment['end'], existing['end'])
            overlap_duration = end_overlap - start_overlap
            
            if overlap_duration > 0:
                # Check text similarity
                s1 = segment['text'].lower().strip()
                s2 = existing['text'].lower().strip()
                similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
                
                # If high overlap and high similarity, it's a duplicate
                if similarity > 0.7: 
                    return True
        return False

    def process(self, audio_path: str, num_speakers: int = 2) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Processing {audio_path} (Targets: {num_speakers} speakers)...")
        
        try:
            input_path = os.path.relpath(audio_path)
        except ValueError:
            input_path = audio_path

        # --- Step 1: Separation (Get 3 clean views) ---
        logger.info("Running separation...")
        # Always use 3-speaker model to get maximum streams
        est_sources = self.sep_model.separate_file(path=input_path) 
        est_sources = est_sources.detach().cpu() # (1, time, 3)
        
        separation_sr = 8000 # SpeechBrain LibriMix models output 8k
        
        # Prepare list of audio tracks to transcribe
        # We verify on the separated tracks because they are cleaner for Whisper,
        # BUT we might also check the original for missing context.
        audio_views = []
        
        # View 0: The Original Mix (downsampled to 8k for consistency in processing loop, or keep 16k)
        # Loading original for "View 4"
        orig_wav, orig_sr = torchaudio.load(input_path)
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_sr, 16000)
            orig_wav = resampler(orig_wav)
            orig_sr = 16000
        
        # Flatten original to mono
        if orig_wav.shape[0] > 1:
            orig_wav = orig_wav.mean(dim=0)
        orig_wav_np = orig_wav.squeeze().numpy()
        audio_views.append({"data": orig_wav_np, "sr": 16000, "name": "Original_Mix"})

        # Views 1-3: Separated Sources
        for i in range(est_sources.shape[2]):
            source_data = est_sources[0, :, i].numpy()
            # Normalize
            source_data = source_data / (np.max(np.abs(source_data)) + 1e-9)
            audio_views.append({"data": source_data, "sr": separation_sr, "name": f"Source_{i+1}"})

        # --- Step 2: Transcribe & Identify All Views ---
        raw_segments = []

        for view in audio_views:
            logger.info(f"Transcribing view: {view['name']}...")
            
            # Transcribe
            # Note: We assume English output based on previous turn
            segments, _ = self.whisper.transcribe(view['data'], language="en") 
            
            for s in segments:
                start_sample = int(s.start * view['sr'])
                end_sample = int(s.end * view['sr'])
                
                # Extract slice for ID
                # Padding to ensure valid embedding
                audio_slice = view['data'][start_sample:end_sample]
                if len(audio_slice) < 500: # Skip tiny bits
                    continue
                    
                # Identify
                speaker, conf = self._identify_segment(audio_slice, view['sr'])
                
                # Store
                raw_segments.append({
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                    "speaker": speaker,
                    "confidence": conf,
                    "source_view": view['name']
                })
        
        # --- Step 3: Merge & Deduplicate ---
        logger.info(f"Collected {len(raw_segments)} raw segments. Deduplicating...")
        
        # Sort by start time
        raw_segments.sort(key=lambda x: x['start'])
        
        final_transcript = []
        
        for seg in raw_segments:
            if not seg['text']: 
                continue
                
            # If "Unknown", try to leverage "Original_Mix" context? 
            # For now, just include everything and filter duplicates
            
            if not self._is_duplicate(seg, final_transcript):
                final_transcript.append(seg)
            else:
                # If duplicate, maybe keep the one with higher ID confidence?
                pass 
                # (Simple version: first come (sorted by time) first served, 
                # but maybe we prefer "Source_X" over "Original_Mix"?)
        
        # Format output
        formatted_lines = []
        for item in final_transcript:
            # timestamp = str(timedelta(seconds=int(item['start'])))
            line = f"{item['speaker']}: {item['text']}"
            formatted_lines.append(line)
        
        # Separated tracks for saving (just the 3 sources from model)
        # We assume the user wants the clean stems
        separated_tracks_list = []
        for i in range(est_sources.shape[2]):
             separated_tracks_list.append(est_sources[0, :, i].numpy())

        return {
            "transcript": formatted_lines,
            "separated_tracks": separated_tracks_list,
            "sample_rate": separation_sr,
            "original_file": audio_path
        }

    def save_separated_audio(self, result: Dict[str, Any], output_dir: str):
        """Save just the physical separated tracks from the model."""
        os.makedirs(output_dir, exist_ok=True)
        sr = result["sample_rate"]
        tracks = result["separated_tracks"]
        
        for i, track in enumerate(tracks):
            filename = os.path.join(output_dir, f"SepModel_Track_{i+1}.wav")
            track_tensor = torch.from_numpy(track).unsqueeze(0)
            torchaudio.save(filename, track_tensor, sr)
            logger.info(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Speaker Transcriber (Multi-View)")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--speakers", type=int, default=2, help="Target number of speakers")
    
    args = parser.parse_args()
    
    transcriber = MultiSpeakerTranscriber()
    
    try:
        result = transcriber.process(args.audio_file, num_speakers=args.speakers)
        
        print("\n=== TRANSCRIPT ===")
        for line in result["transcript"]:
            print(line)
        print("==================\n")
        
        # Save results
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_dir = f"data/outputs/{base_name}_separated"
        transcriber.save_separated_audio(result, output_dir)
        
        # Save transcript txt
        # Also print to a local log file if run directly
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{base_name}_transcript.txt", "w", encoding="utf-8") as f:
            for line in result["transcript"]:
                f.write(line + "\n")
        print(f"Transcript saved to logs/{base_name}_transcript.txt")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()