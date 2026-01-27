#!/usr/bin/env python3
"""
Visualização de Embeddings (Versão Turbo com Torchaudio)
"""
import os
import random
import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_audio_turbo(path: str, target_sr=16000):
    """
    Carrega áudio usando Torchaudio (C++) que é muito mais rápido que Librosa.
    Já faz o resampling e converte para mono.
    """
   
    sig, sr = torchaudio.load(path)
    
    
    if sig.shape[0] > 1:
        sig = torch.mean(sig, dim=0, keepdim=True)
    
 
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        sig = resampler(sig)
        
    return sig.squeeze().numpy() # Retorna numpy array flat

def find_audio_files(root: Path):
    exts = {'.wav', '.flac', '.mp3', '.m4a'}
    # Usando list comprehension com rglob é rápido, mas vamos garantir que seja set para busca O(1)
    return [p for p in root.rglob('*') if p.suffix.lower() in exts]

def collect_files_by_speaker(roots, target_speaker_ids=None, max_speakers=None, samples_per_speaker=10):
    speaker_to_files = {}
    
    for root in roots:
        rootp = Path(root)
        if not rootp.exists(): continue
        
        # Assuming structure: INPUT_DIR/wav/idXXXXX/audio.wav
        # First, find the 'wav' directory
        wav_dir = rootp / "wav"
        if not wav_dir.is_dir():
            logger.warning(f"Diretório 'wav' não encontrado em {rootp}. Pulando.")
            continue

        # Then, iterate through speaker ID directories inside 'wav'
        for speaker_id_dir in wav_dir.iterdir():
            if speaker_id_dir.is_dir():
                s_id = speaker_id_dir.name
                # Check if this speaker ID is in our target list (if provided)
                if target_speaker_ids and s_id not in target_speaker_ids:
                    continue

                files = find_audio_files(speaker_id_dir)
                if files:
                    speaker_to_files[s_id] = files

    # Seleção e Corte (max_speakers is still applied here if needed, but target_speaker_ids is already filtered)
    speakers = sorted(speaker_to_files.keys())
    if max_speakers: # this still applies if target_speaker_ids is not set
        speakers = speakers[:max_speakers]
    
    paths = []
    labels = []
    
    for spk in speakers:
        files = speaker_to_files[spk]
        
        # Amostragem aleatória
        if samples_per_speaker > 0 and len(files) > samples_per_speaker:
            files = random.sample(files, samples_per_speaker)
            
        for f in files:
            paths.append(str(f)) # Torchaudio prefere string
            labels.append(spk)

    return paths, labels

# ==========================================
# 2. Embeddings em Batch
# ==========================================

def compute_embeddings(model, files, sample_rate=16000, segment_duration=3.0, batch_size=32, device=None):
    model_device = device
    all_embeddings = []
    target_samples = int(sample_rate * segment_duration)

    # Pré-aloca o tensor de zeros para padding (otimização)
    zeros = np.zeros(target_samples, dtype=np.float32)

    for i in tqdm(range(0, len(files), batch_size), desc='Processando Áudios'):
        batch_paths = files[i : i + batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                # Usa a função Turbo
                audio = load_audio_turbo(path, target_sr=sample_rate)
                
                # Padronização rápida
                cur_len = audio.shape[0]
                if cur_len < target_samples:
                    # Pad
                    padded = np.zeros(target_samples, dtype=np.float32)
                    padded[:cur_len] = audio
                    audio = padded
                elif cur_len > target_samples:
                    # Crop (pega o centro para evitar silêncio inicial)
                    start = (cur_len - target_samples) // 2
                    audio = audio[start : start + target_samples]
                
                batch_tensors.append(audio)
            except Exception as e:
                # Em erro, usa silêncio
                batch_tensors.append(zeros)

        if not batch_tensors: continue

        # Converte para tensor Pytorch direto
        tensor = torch.tensor(np.array(batch_tensors), dtype=torch.float32)
        if model_device:
            tensor = tensor.to(model_device)

        with torch.no_grad():
            emb = model.encode_batch(tensor)
            # Normaliza output [Batch, 1, 192] -> [Batch, 192]
            all_embeddings.append(emb.squeeze(1).cpu().numpy())

    if not all_embeddings: return np.empty((0, 0))
    return np.vstack(all_embeddings)

# ==========================================
# 3. Main
# ==========================================

def main():
    # --- AJUSTES DE PERFORMANCE ---
    SAMPLES_PER_SPEAKER = 20   # 15 é suficiente para ver clusters! 40 é muito pesado.
    BATCH_SIZE = 32            # Mantenha 32 para CPU, 64/128 para GPU
    INPUT_DIR = Path("./vox_100_atores")
    OUTPUT_DIR = Path("./resultados_tsne")
    META_FILE = Path("./vox1_meta.csv")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Definir os IDs dos atores desejados (id10001 a id10700)
    target_speaker_ids = [f"id{i}" for i in range(10001, 10701)] # 10701 para incluir id10700

    # --- Checagem de GPU ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🚀 GPU DETECTADA: {torch.cuda.get_device_name(0)}")
        print("O processamento será rápido.\n")
    else:
        device = torch.device('cpu')
        print("\n⚠️ AVISO: RODANDO EM CPU ⚠️")
        print("Isso será lento. Reduzindo expectativas...")
        print("DICA: Use o Google Colab com Runtime GPU se demorar muito.\n")

    # 1. Carregar Modelo
    print("Carregando SpeechBrain...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
        local_strategy=LocalStrategy.COPY
    )

    # 2. Coletar
    print(f"Coletando {SAMPLES_PER_SPEAKER} áudios por ator para {len(target_speaker_ids)} atores...")
    paths, labels_str = collect_files_by_speaker(
        [INPUT_DIR],
        target_speaker_ids=target_speaker_ids,
        samples_per_speaker=SAMPLES_PER_SPEAKER
    )
    
    print(f"Total para processar: {len(paths)} arquivos.")

    # 3. Processar
    X = compute_embeddings(model, paths, batch_size=BATCH_SIZE, device=device)

    # 4. Plotar
    print("Gerando t-SNE...")
    
    # Carregar nomes reais se possível
    id_to_name = {}
    if META_FILE.exists():
        try:
            df = pd.read_csv(META_FILE, sep='\t')
            # Ajuste conforme seu CSV. Geralmente: VoxCeleb1 ID | VGGFace1 ID
            # Se o CSV tiver cabeçalho:
            id_to_name = dict(zip(df.iloc[:,0], df.iloc[:,1])) 
        except: pass

    # Mapeamento numérico
    unique_labels = sorted(list(set(labels_str)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels_str])

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(paths)//10), init='pca', random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(18, 12)) # Increased figure size
    
    hue_labels = [id_to_name.get(l, l) for l in labels_str]
    
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=hue_labels, palette="tab20", alpha=0.7, s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='x-small') # Adjusted legend position and columns
    plt.title(f"t-SNE VoxCeleb (700 Atores, 20 amostras/ator)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_700_actors_20_samples.png", dpi=150)
    print(f"\n✅ Pronto! Imagem salva em: {OUTPUT_DIR / 'tsne_700_actors_20_samples.png'}")

if __name__ == "__main__":
    main()