#!/usr/bin/env python3
"""
Batch-create speaker profile .pkl files from audio folders.

Usage:
  python scripts/create_embeddings_from_audio.py --roots data/voxceleb1 data/voxceleb2 --samples 3 --out data/embeddings

This will iterate speaker subfolders under each root, compute embeddings using
SpeechBrain `PretrainedSpeakerEmbedding`, average per-speaker, and save a
pickle file per speaker compatible with `MultiSpeakerVerifier`.
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import pickle
import datetime
import logging

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

try:
    import soundfile as sf
except Exception:
    sf = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_wavs(root: Path):
    exts = ('.wav', '.flac', '.m4a', '.mp3')
    return [p for p in root.rglob('*') if p.suffix.lower() in exts]


def load_audio(path: Path, target_sr=16000, max_duration=None):
    if sf is None:
        raise RuntimeError('soundfile required')
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        # lazy resample via librosa if available
        try:
            import librosa
            data = librosa.resample(data.astype('float32'), orig_sr=sr, target_sr=target_sr)
        except Exception:
            raise RuntimeError('Resampling required but librosa not available')
    if max_duration:
        max_samples = int(max_duration * target_sr)
        if data.shape[0] > max_samples:
            start = random.randint(0, data.shape[0] - max_samples)
            data = data[start:start+max_samples]
    return data.astype('float32')


def compute_embedding(model, audio_array, device):
    tensor = torch.from_numpy(audio_array).float()
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    tensor = tensor.to(device)
    with torch.no_grad():
        emb = model.encode_batch(tensor)
    emb = emb.squeeze()  # Remove batch and channel dimensions
    return emb.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', required=True, help='Root folders containing speaker subfolders')
    p.add_argument('--samples', type=int, default=3, help='Samples per speaker to average')
    p.add_argument('--out', type=str, default='data/embeddings', help='Output embeddings directory')
    p.add_argument('--segment', type=float, default=3.0, help='Segment duration (seconds) for embedding')
    p.add_argument('--speakers', nargs='+', help='Optional list of speaker IDs to process')
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Loading embedding model on {device}...')
    model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='pretrained_models/spkrec-ecapa-voxceleb',
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY
    )

    speakers_created = 0

    for root in args.roots:
        rootp = Path(root)
        if not rootp.exists():
            logger.warning(f'Skipping missing root: {rootp}')
            continue

        speaker_dirs = sorted([d for d in rootp.iterdir() if d.is_dir()])
        if args.speakers:
            speaker_dirs = [d for d in speaker_dirs if d.name in args.speakers]

        # iterate speaker directories
        for sp in speaker_dirs:
            wavs = find_wavs(sp)
            if not wavs:
                continue
            sampled = wavs if len(wavs) <= args.samples else random.sample(wavs, args.samples)
            embs = []
            for w in sampled:
                try:
                    audio = load_audio(w, target_sr=16000, max_duration=args.segment)
                    emb = compute_embedding(model, audio, device)
                    embs.append(emb)
                except Exception as e:
                    logger.warning(f'Failed to embed {w}: {e}')
            if not embs:
                continue
            avg_emb = np.mean(embs, axis=0)

            metadata = {
                'created_at': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                'num_samples': len(embs),
                'sample_rate': 16000,
                'segment_duration': args.segment,
                'model_name': 'speechbrain/spkrec-ecapa-voxceleb'
            }

            profile = {
                'speaker_name': sp.name,
                'embedding': avg_emb,
                'metadata': metadata
            }

            fname = outdir / f"{sp.name}_{metadata['created_at']}.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(profile, f)

            speakers_created += 1
            logger.info(f'Created profile: {fname} ({len(embs)} samples)')

    logger.info(f'Done. Created {speakers_created} speaker profiles in {outdir}')


if __name__ == '__main__':
    main()
