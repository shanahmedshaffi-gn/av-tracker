#!/usr/bin/env python3
"""
Embedding Visualization (PCA / LDA / t-SNE) from existing .pkl embeddings
==========================================================================

Script to load speaker embeddings from .pkl files and produce 2D visualizations
using PCA, LDA and t-SNE.

Usage example:
  python src/visualize_embeddings_from_pkl.py --input_file data/all_embeddings.pkl --output_dir data/outputs --actor_names_file vox1_meta.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pickle
import logging
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_embeddings(input_file: Path):
    """Load embeddings and labels from a single .pkl file."""
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
        labels = data['labels']
    return embeddings, labels

def load_actor_names(actor_names_file: Path):
    """Load actor ID to name mapping from a CSV file."""
    df = pd.read_csv(actor_names_file, sep='\t') # Use tab as separator
    return dict(zip(df['VoxCeleb1 ID'], df['VGGFace1 ID'])) # Use correct column names

def reduce_and_plot(X, y_str, actor_name_map, output_dir: Path, prefix: ''):
    """Reduce dimensionality and plot embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = X.shape[0]
    unique_labels = sorted(list(set(y_str)))
    n_classes = len(unique_labels)
    logger.info(f"Reducing and plotting: {n_samples} samples, {n_classes} classes")
    
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y_str])

    # Map actor IDs to names for legend
    display_labels = [actor_name_map.get(label, label) for label in y_str]

    # Save embeddings and labels for later reuse
    try:
        np.save(output_dir / f"{prefix}embeddings.npy", X)
        np.save(output_dir / f"{prefix}labels.npy", np.asarray(y_str))
        np.save(output_dir / f"{prefix}display_labels.npy", np.asarray(display_labels))
    except Exception:
        pass

    # PCA
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=display_labels, palette='tab20', legend='full', s=50, ax=ax)
    ax.set_title('PCA of Speaker Embeddings')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Actors')
    pca_path = output_dir / f"{prefix}pca_2d.png"
    fig.savefig(pca_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PCA plot to {pca_path}")

    # LDA
    if n_classes > 1 and n_samples > n_classes:
        n_components = min(2, n_classes - 1)
        try:
            lda = LDA(n_components=n_components)
            X_lda = lda.fit_transform(X, y_numeric) # Use y_numeric for LDA training
            if X_lda.shape[1] == 1:
                X_lda = np.hstack([X_lda, np.zeros((X_lda.shape[0], 1))])

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=display_labels, palette='tab20', legend='full', s=50, ax=ax)
            ax.set_title('LDA of Speaker Embeddings')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Actors')
            lda_path = output_dir / f"{prefix}lda_2d.png"
            fig.savefig(lda_path, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved LDA plot to {lda_path}")
        except Exception as e:
            logger.warning(f"LDA failed: {e}")

    # t-SNE
    try:
        if n_samples <= 2:
            raise ValueError('Not enough samples for t-SNE')
        perp = max(2, min(30, n_samples // 3))
        if perp >= n_samples:
            perp = max(1, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perp, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=display_labels, palette='tab20', legend='full', s=50, ax=ax)
        ax.set_title(f't-SNE of Speaker Embeddings (perplexity={perp})')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Actors')
        tsne_path = output_dir / f"{prefix}tsne_2d.png"
        fig.savefig(tsne_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved t-SNE plot to {tsne_path}")
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")

def main():
    p = argparse.ArgumentParser(description='Visualize speaker embeddings from .pkl files.')
    p.add_argument('--input_file', type=str, default='data/all_embeddings.pkl', help='Pickle file with embeddings and labels')
    p.add_argument('--output_dir', type=str, default='data/outputs', help='Directory for saved plots')
    p.add_argument('--actor_names_file', type=str, default='vox1_meta.csv', help='CSV file with actor ID to name mapping')
    args = p.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    X, y = load_embeddings(input_file)
    if X.shape[0] == 0:
        logger.error("No embeddings found in the input file.")
        return

    actor_name_map = {}
    if args.actor_names_file:
        actor_name_map = load_actor_names(Path(args.actor_names_file))

    reduce_and_plot(X, y, actor_name_map, output_dir, prefix='actors_')

if __name__ == '__main__':
    main()