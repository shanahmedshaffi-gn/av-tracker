import os
import random
import csv
import torch
import torchaudio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_actors(meta_path, wav_root):
    """Load available actors from CSV and check if they exist in wav_root."""
    actors = {}
    
    # Read metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            id_ = row['VoxCeleb1 ID']
            name = row['VGGFace1 ID']
            actors[id_] = name

    # Check existence
    available_actors = []
    wav_path = Path(wav_root)
    for id_ in actors:
        actor_dir = wav_path / id_
        if actor_dir.exists() and any(actor_dir.iterdir()):
            available_actors.append(id_)
    
    logging.info(f"Found {len(available_actors)} available actors in {wav_root}")
    return available_actors, actors

def get_random_wav(actor_id, wav_root):
    """Get a random wav file for a specific actor."""
    actor_dir = Path(wav_root) / actor_id
    # Recursively find wavs or just look in subfolders
    # VoxCeleb structure: id/video_id/wav
    wavs = list(actor_dir.glob('**/*.wav'))
    if not wavs:
        return None
    return random.choice(wavs)

def mix_audios(wav_paths, output_path, target_duration=15):
    """
    Mix multiple audio files into one.
    Loops input signals to match target_duration (seconds) to ensure enough overlap.
    """
    signals = []
    sr = 16000 # Standard for these models
    target_samples = int(target_duration * sr)

    for p in wav_paths:
        wav, sample_rate = torchaudio.load(p)
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            wav = resampler(wav)
        
        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Normalize Energy (RMS) to 0.1
        rms = torch.sqrt(torch.mean(wav**2))
        if rms > 0:
            wav = wav / rms * 0.1
            
        # Loop/Repeat audio if shorter than target
        if wav.shape[1] < target_samples:
            repeats = int(target_samples / wav.shape[1]) + 1
            wav = wav.repeat(1, repeats)
        
        # Trim to exact target length
        wav = wav[:, :target_samples]
            
        signals.append(wav)

    if not signals:
        return

    # Sum signals
    mixed = torch.zeros((1, target_samples))
    for s in signals:
        mixed += s

    # Normalize Mixed Output to avoid clipping
    max_val = torch.max(torch.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, mixed, sr)
    logging.info(f"Saved mixed audio to {output_path} (Duration: {target_duration}s)")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    META_PATH = os.path.join(BASE_DIR, "vox1_meta.csv")
    WAV_ROOT = os.path.join(BASE_DIR, "vox_100_atores", "wav")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "demo")

    available_ids, actor_names = load_actors(META_PATH, WAV_ROOT)
    
    if len(available_ids) < 5:
        logging.error("Not enough actors found!")
        return

    # Create mixes for 2, 3, 4, 5 speakers
    for num_speakers in [2, 3, 4, 5]:
        selected_ids = random.sample(available_ids, num_speakers)
        wav_files = []
        selected_names = []

        print(f"\n--- Mixing {num_speakers} Speakers ---")
        for aid in selected_ids:
            wav = get_random_wav(aid, WAV_ROOT)
            if wav:
                wav_files.append(wav)
                name = actor_names[aid]
                selected_names.append(name)
                print(f"  - {name} ({aid}): {os.path.basename(wav)}")
        
        if len(wav_files) == num_speakers:
            output_file = os.path.join(OUTPUT_DIR, f"mix_{num_speakers}_speakers.wav")
            mix_audios(wav_files, output_file)
            
            # Save ground truth
            gt_file = os.path.join(OUTPUT_DIR, f"mix_{num_speakers}_ground_truth.txt")
            with open(gt_file, "w") as f:
                f.write(f"Speakers: {', '.join(selected_names)}\n")
                for i, path in enumerate(wav_files):
                    f.write(f"{selected_names[i]}: {path}\n")

if __name__ == "__main__":
    main()
