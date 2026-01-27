#!/usr/bin/env python3
"""
Embedding Visualization (PCA / LDA / t-SNE)
=========================================

Script to extract speaker embeddings from audio files (VoxCeleb1 / VoxCeleb2)
and produce 2D visualizations using PCA, LDA and t-SNE. Designed to sample a
subset of speakers and audio files to produce quick, inspectable plots.

Usage example:
  python src/embedding_visualization.py --vox1_dir PATH_TO_VOX1 --vox2_dir PATH_TO_VOX2 \
      --max_speakers 50 --samples_per_speaker 5 --output_dir data/outputs

Requirements: numpy, torch, pyannote.audio, scikit-learn, matplotlib, seaborn,
librosa (or soundfile), tqdm

"""
from pathlib import Path
import argparse
import os
import random
import pickle
import numpy as np
import torch
from tqdm import tqdm
import logging

try:
    # speechbrain EncoderClassifier
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
except Exception as e:
    raise RuntimeError("SpeechBrain import failed. Ensure speechbrain is installed.")

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
except Exception:
    librosa = None

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_embeddings_from_pkl(embedding_dir: Path):
    """Load all .pkl files in a directory and return embeddings and labels."""
    embeddings = []
    labels = []
    for pkl_file in embedding_dir.glob('*.pkl'):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            embeddings.append(data['embedding'])
            labels.append(data['speaker_name'])
    return np.array(embeddings), labels


def find_audio_files(root: Path):
    """Recursively find .wav (and .flac/.mp3) files under root."""
    exts = ('.wav', '.flac', '.mp3', '.m4a')
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    return files


def load_audio(path: Path, target_sr: int = 16000, max_duration: float = None):
    """Load audio file, resample to target_sr. Returns 1D numpy float32 array."""
    if sf is not None:
        data, sr = sf.read(str(path))
        # soundfile can return 2D for multi-channel -> convert to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            if librosa is not None:
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            else:
                raise RuntimeError('Resampling required but librosa not installed')
    else:
        if librosa is None:
            raise RuntimeError('No audio backend available (install soundfile or librosa)')
        data, sr = librosa.load(str(path), sr=target_sr, mono=True)

    if max_duration is not None:
        max_samples = int(max_duration * target_sr)
        if data.shape[0] > max_samples:
            # sample a random segment
            start = random.randint(0, data.shape[0] - max_samples)
            data = data[start:start + max_samples]

    return data.astype(np.float32)


def collect_files_by_speaker(roots, max_speakers=None, samples_per_speaker=5, speakers_list=None):
    """Collect a sampled list of audio file paths with speaker labels.

    Assumes speaker id is the parent directory name of the audio file (VoxCeleb layout).
    Returns lists: paths, labels
    """
    speaker_to_files = {}
    for root in roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        # If a specific speaker list is given, only look into those directories
        if speakers_list:
            for speaker_id in speakers_list:
                speaker_dir = rootp / speaker_id
                if speaker_dir.is_dir():
                    files = find_audio_files(speaker_dir)
                    speaker_to_files.setdefault(speaker_id, []).extend(files)
        else:
            files = find_audio_files(rootp)
            for f in files:
                # speaker id heuristic: parent folder name
                speaker = f.parent.name
                speaker_to_files.setdefault(speaker, []).append(f)

    speakers = sorted(speaker_to_files.keys())
    if speakers_list:
        speakers = [spk for spk in speakers if spk in speakers_list]
    elif max_speakers is not None:
        speakers = speakers[:max_speakers]

    paths = []
    labels = []
    for spk in speakers:
        files = speaker_to_files.get(spk, [])
        if len(files) == 0:
            continue
        
        if samples_per_speaker is None or samples_per_speaker == -1:
            sampled = files
        else:
            sampled = files if len(files) <= samples_per_speaker else random.sample(files, samples_per_speaker)
        
        for f in sampled:
            paths.append(f)
            labels.append(spk)

    return paths, labels


def compute_embeddings(model, files, sample_rate=16000, segment_duration=3.0, device=None):
    """Compute embeddings for a list of audio file paths. Returns numpy array (N, D)."""
    model_device = model.device if hasattr(model, 'device') else device
    embeddings = []
    for path in tqdm(files, desc='Embedding files'):
        try:
            audio = load_audio(path, target_sr=sample_rate, max_duration=segment_duration)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            continue

        # Prepare tensor: (batch, samples)
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        if model_device is not None:
            try:
                tensor = tensor.to(model_device)
            except Exception:
                pass

        with torch.no_grad():
            try:
                emb = model.encode_batch(tensor)
                emb = emb.squeeze()
            except Exception as e:
                logger.error(f"Model failed on {path}: {e}")
                continue

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()

        embeddings.append(np.asarray(emb).reshape(-1))

    if len(embeddings) == 0:
        return np.empty((0, 0))
    return np.vstack(embeddings)


def reduce_and_plot(X, y, labels_map, output_dir: Path, prefix: str = '', labels_rename_map=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = X.shape[0]
    n_classes = len(labels_map)
    logger.info(f"Reducing and plotting: {n_samples} samples, {n_classes} classes")

    # save embeddings and labels for later reuse
    try:
        np.save(output_dir / f"{prefix}embeddings.npy", X)
        np.save(output_dir / f"{prefix}labels.npy", np.asarray(y))
    except Exception:
        pass
    
    numeric_to_label = {v: k for k, v in labels_map.items()}
    
    hue_labels = [numeric_to_label[i] for i in y]
    
    if labels_rename_map:
        hue_labels = [labels_rename_map.get(label, label) for label in hue_labels]

    # t-SNE
    try:
        # choose perplexity relative to n_samples but ensure 1 <= perp < n_samples
        if n_samples <= 2:
            raise ValueError('Not enough samples for t-SNE')
        # default heuristic: ~n_samples/3 but clipped
        perp = max(2, min(30, max(2, n_samples // 3)))
        if perp >= n_samples:
            perp = max(1, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perp, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=hue_labels, palette='tab20', legend='full', s=40, ax=ax)
        ax.set_title(f't-SNE (2D) perp={perp}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        tsne_path = output_dir / f"{prefix}tsne_2d.png"
        fig.savefig(tsne_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved t-SNE plot to {tsne_path}")
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")


def build_label_index(labels):
    unique = sorted(list(set(labels)))
    idx = {u: i for i, u in enumerate(unique)}
    numeric = [idx[l] for l in labels]
    return numeric, idx


def parse_args():
    p = argparse.ArgumentParser(description='Visualize speaker embeddings with PCA/LDA/t-SNE')
    p.add_argument('--vox1_dir', type=str, default=None, help='Path to VoxCeleb1 root (optional)')
    p.add_argument('--vox2_dir', type=str, default=None, help='Path to VoxCeleb2 root (optional)')
    p.add_argument('--max_speakers', type=int, default=50, help='Max number of speakers to sample')
    p.add_argument('--samples_per_speaker', type=int, default=10, help='Audio files per speaker')
    p.add_argument('--output_dir', type=str, default='data/outputs', help='Directory for saved plots')
    p.add_argument('--segment_duration', type=float, default=3.0, help='Seconds per audio segment to embed')
    p.add_argument('--random_seed', type=int, default=0, help='Random seed')
    p.add_argument('--embedding_dir', type=str, default=None, help='Path to precomputed embeddings .pkl files')
    p.add_argument('--speakers', nargs='+', help='Optional list of speaker IDs to process')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Create a mapping from speaker ID to speaker name
    speaker_id_to_name = {}
    try:
        with open('vox1_meta.csv', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    speaker_id_to_name[parts[0]] = parts[1]
    except FileNotFoundError:
        logger.warning("vox1_meta.csv not found. Speaker names will not be used in plots.")


    if args.embedding_dir:
        outdir = Path(args.output_dir)
        logger.info(f'Loading embeddings from {args.embedding_dir}')
        X, y = load_embeddings_from_pkl(Path(args.embedding_dir))
        if X.size == 0:
            raise RuntimeError('No embeddings found in the specified directory.')
        y_numeric, label_map = build_label_index(y)
        outdir.mkdir(parents=True, exist_ok=True)
        reduce_and_plot(X, y_numeric, label_map, outdir, prefix='embeddings_', labels_rename_map=speaker_id_to_name)
        return

    roots = []
    if args.vox1_dir:
        roots.append(args.vox1_dir)
    if args.vox2_dir:
        roots.append(args.vox2_dir)

    if len(roots) == 0 and not args.embedding_dir:
        raise RuntimeError('Provide at least --vox1_dir or --vox2_dir pointing to VoxCeleb folders')

    logger.info('Collecting files...')
    # Initialize embedding model
    logger.info('Loading embedding model (SpeechBrain)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='pretrained_models/spkrec-ecapa-voxceleb',
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY
    )

    outdir = Path(args.output_dir)

    # Process each dataset separately (vox1 and vox2) if provided
    if args.vox1_dir:
        files1, labels1 = collect_files_by_speaker([args.vox1_dir], max_speakers=args.max_speakers, samples_per_speaker=args.samples_per_speaker, speakers_list=args.speakers)
        if files1:
            logger.info(f'VoxCeleb1: Collected {len(files1)} samples from {len(set(labels1))} speakers')
            X1 = compute_embeddings(model, files1, sample_rate=16000, segment_duration=args.segment_duration, device=device)
            if X1.size != 0:
                y1_numeric, label_map1 = build_label_index(labels1)
                reduce_and_plot(X1, y1_numeric, label_map1, outdir, prefix='vox1_', labels_rename_map=speaker_id_to_name)
        else:
            logger.warning('No audio files found under vox1_dir')

    if args.vox2_dir:
        files2, labels2 = collect_files_by_speaker([args.vox2_dir], max_speakers=args.max_speakers, samples_per_speaker=args.samples_per_speaker, speakers_list=args.speakers)
        if files2:
            logger.info(f'VoxCeleb2: Collected {len(files2)} samples from {len(set(labels2))} speakers')
            X2 = compute_embeddings(model, files2, sample_rate=16000, segment_duration=args.segment_duration, device=device)
            if X2.size != 0:
                y2_numeric, label_map2 = build_label_index(labels2)
                reduce_and_plot(X2, y2_numeric, label_map2, outdir, prefix='vox2_', labels_rename_map=speaker_id_to_name)
        else:
            logger.warning('No audio files found under vox2_dir')

    # Combined view (both datasets together)
    files_all, labels_all = collect_files_by_speaker(roots, max_speakers=args.max_speakers, samples_per_speaker=args.samples_per_speaker, speakers_list=args.speakers)
    if not files_all:
        raise RuntimeError('No audio files found under provided roots')

    logger.info(f'Combined: Collected {len(files_all)} samples from {len(set(labels_all))} speakers')
    X = compute_embeddings(model, files_all, sample_rate=16000, segment_duration=args.segment_duration, device=device)
    if X.size == 0:
        raise RuntimeError('No embeddings computed')
    y_numeric, label_map = build_label_index(labels_all)
    reduce_and_plot(X, y_numeric, label_map, outdir, prefix='combined_', labels_rename_map=speaker_id_to_name)


if __name__ == '__main__':
    main()
