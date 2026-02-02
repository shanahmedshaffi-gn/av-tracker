
#!/usr/bin/env python3
"""
Batch-create speaker profile .pkl files from a CSV file.

Usage:
  python scripts/create_embeddings_from_csv.py --csv data/audio_files.csv --samples 10 --max_actors 10 --out data/all_embeddings.pkl
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import pickle
import datetime
import logging
import pandas as pd
from collections import defaultdict
import os

from speechbrain.pretrained import EncoderClassifier

try:
    import soundfile as sf
except Exception:
    sf = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        emb = model.encode_batch(tensor)
        emb = emb.squeeze()
    return emb.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True, help='CSV file with actor_id and audio_file columns')
    p.add_argument('--samples', type=int, default=10, help='Samples per speaker to extract')
    p.add_argument('--max_actors', type=int, default=None, help='Max number of actors to process')
    p.add_argument('--out', type=str, default='data/all_embeddings.pkl', help='Output pickle file')
    p.add_argument('--segment', type=float, default=3.0, help='Segment duration (seconds) for embedding')
    args = p.parse_args()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Loading embedding model on {device}...')
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device":device})

    all_embeddings = []
    all_labels = []

    df = pd.read_csv(args.csv)
    audio_files_by_actor = defaultdict(list)
    for index, row in df.iterrows():
        audio_files_by_actor[row['actor_id']].append(row['audio_file'])

    actor_ids = list(audio_files_by_actor.keys())
    random.shuffle(actor_ids)
    if args.max_actors is not None and args.max_actors < len(actor_ids):
        actor_ids = actor_ids[:args.max_actors]

    for actor_id in actor_ids:
        audio_files = audio_files_by_actor[actor_id]
        if not audio_files:
            continue
        
        num_samples = min(len(audio_files), args.samples)
        sampled = random.sample(audio_files, num_samples)
        
        for w in sampled:
            try:
                audio = load_audio(Path(w), target_sr=16000, max_duration=args.segment)
                emb = compute_embedding(model, audio, device)
                all_embeddings.append(emb)
                all_labels.append(actor_id)
            except Exception as e:
                logger.warning(f'Failed to embed {w}: {e}')
    
    output_data = {
        'embeddings': np.array(all_embeddings),
        'labels': all_labels
    }

    with open(out_file, 'wb') as f:
        pickle.dump(output_data, f)

    logger.info(f'Done. Saved {len(all_embeddings)} embeddings to {out_file}')


if __name__ == '__main__':
    main()
