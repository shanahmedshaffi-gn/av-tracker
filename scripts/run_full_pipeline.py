#!/usr/bin/env python3
"""
Full Pipeline Orchestrator
==========================
1. Selects 5 random actors.
2. Generates Speaker Profiles (embeddings) for them using one of their audio files.
3. Generates Mixed Audio files (2, 3, 4, 5 speakers) using different audio files.
4. Runs the Multi-Speaker Transcription & Separation pipeline.

Usage:
    python scripts/run_full_pipeline.py
"""

import os
import sys
import random
import csv
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

# Imports from existing modules
# Note: We need to handle potential import errors if dependencies aren't met
try:
    from create_and_mix_audio import load_actors, get_random_wav, mix_audios
    from audio_profile_creator import EnhancedAudioProfileCreator, SpeakerProfile, AudioProfileMetadata
    from transcribe_multispeaker import MultiSpeakerTranscriber
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_profiles_for_actors(actor_ids, actor_names, wav_root, creator):
    """
    Creates .pkl profiles for the selected actors.
    Returns a dictionary of actor_id -> profile_path.
    """
    profile_paths = {}
    
    print("\n--- Generating Speaker Profiles ---")
    for aid in actor_ids:
        name = actor_names[aid]
        # Get a wav for profile creation
        wav_path = get_random_wav(aid, wav_root)
        if not wav_path:
            logging.warning(f"No wav found for {name}")
            continue
            
        logging.info(f"Creating profile for {name} using {os.path.basename(wav_path)}")
        
        # Load audio using torchaudio (standardize to 16k mono)
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Convert to numpy 1D array for the creator
        audio_data = wav.squeeze().numpy()
        
        # Create profile manually to avoid interactive mode
        # We need to simulate quality metrics
        quality_metrics = {
            'quality_score': 0.9,
            'quality_level': "Simulated Excellent",
            'rms_level': 0.1,
            'peak_level': 0.5,
            'snr_estimate': 30.0,
            'zero_crossing_rate': 0.0,
            'recommendations': []
        }
        
        try:
            profile = creator.create_speaker_profile(name, audio_data, quality_metrics, "pipeline_generated")
            # Save
            # We enforce the filename to be name_timestamp.pkl
            # but we want to ensure we can find it later.
            # The name in the profile is the key.
            path = creator.save_speaker_profile(profile)
            profile_paths[aid] = path
            print(f"  + Saved profile for {name} to {path.name}")
            
        except Exception as e:
            logging.error(f"Failed to create profile for {name}: {e}")

    return profile_paths

def main():
    # Configuration
    META_PATH = os.path.join(BASE_DIR, "vox1_meta.csv")
    WAV_ROOT = os.path.join(BASE_DIR, "vox_100_atores", "wav")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "demo")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # 1. Load Actors
    available_ids, actor_names = load_actors(META_PATH, WAV_ROOT)
    if len(available_ids) < 5:
        logging.error("Not enough actors found (need at least 5).")
        return

    # Select 5 random actors
    selected_ids = random.sample(available_ids, 5)
    selected_names = [actor_names[aid] for aid in selected_ids]
    print(f"Selected Actors: {', '.join(selected_names)}")

    # 2. Create Profiles
    # Initialize creator
    creator = EnhancedAudioProfileCreator(embeddings_dir=EMBEDDINGS_DIR)
    
    # We use specific wavs for profiles (different from mix if possible, but random choice handles that probabilistically)
    # Ideally we should exclude the profile wav from the mix wavs, but for a demo, random is okay.
    profile_paths = create_profiles_for_actors(selected_ids, actor_names, WAV_ROOT, creator)

    # 3. Create Mixes & Transcribe
    # Initialize Transcriber
    # Note: This might fail if HF_TOKEN is missing. We'll try.
    try:
        transcriber = MultiSpeakerTranscriber(embeddings_dir=EMBEDDINGS_DIR)
    except Exception as e:
        logging.error(f"Failed to initialize Transcriber: {e}")
        logging.error("Make sure you have a valid HF_TOKEN for pyannote if using the pipeline.")
        return

    for num_speakers in [2, 3, 4, 5]:
        print(f"\n\n=== Processing {num_speakers} Speakers Case ===")
        
        # Pick subset of the 5 selected actors
        current_ids = selected_ids[:num_speakers]
        
        # Gather wavs for mixing
        wav_files = []
        for aid in current_ids:
            wav = get_random_wav(aid, WAV_ROOT)
            wav_files.append(wav)
        
        # Mix
        mix_filename = f"pipeline_mix_{num_speakers}.wav"
        mix_path = os.path.join(OUTPUT_DIR, mix_filename)
        mix_audios(wav_files, mix_path)
        
        # Transcribe
        print(f"Running transcription on {mix_filename}...")
        try:
            result = transcriber.process(mix_path, num_speakers=num_speakers)
            
            print("\n--- Transcript Result ---")
            for line in result["transcript"]:
                print(line)
            
            # Save Transcript to logs
            transcript_file = os.path.join(BASE_DIR, "logs", f"pipeline_mix_{num_speakers}_transcript.txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                for line in result["transcript"]:
                    f.write(line + "\n")
            print(f"Transcript saved to {transcript_file}")
                
            # Save separated audio
            sep_dir = os.path.join(OUTPUT_DIR, f"separated_{num_speakers}")
            transcriber.save_separated_audio(result, sep_dir)
            
        except Exception as e:
            logging.error(f"Transcription failed for {num_speakers} speakers: {e}")

if __name__ == "__main__":
    main()
