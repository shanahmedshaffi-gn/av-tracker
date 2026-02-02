
import os
import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from huggingface_hub import login


HF_TOKEN = ""

def enroll_actors():
  
    login(token=HF_TOKEN)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Carregando modelo SpeechBrain (ECAPA-TDNN) em {device}...")
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}
    )

  
    base_audio_path = "../vox_100_atores/wav" 
    
    embeddings_dir = "../data/embeddings"   
    
    if not os.path.exists(base_audio_path):
        print(f" ERRO: Não encontrei a pasta '{base_audio_path}'.")
        print("Certifique-se de estar rodando este script de dentro da pasta 'src'.")
        return

    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    print(f" Lendo estrutura profunda de atores em: {base_audio_path}")
    print("-" * 50)

   
    try:
        atores_encontrados = [f for f in os.listdir(base_audio_path) if os.path.isdir(os.path.join(base_audio_path, f))]
    except FileNotFoundError:
        print(" Pasta raiz não encontrada.")
        return

    count = 0
    for actor_name in atores_encontrados:
        actor_path = os.path.join(base_audio_path, actor_name)
        found_wav = False
        
        
        for root, dirs, files in os.walk(actor_path):
        
            wavs = [f for f in files if f.lower().endswith(".wav")]
            
            if wavs:
              
                first_wav = wavs[0]
                full_wav_path = os.path.join(root, first_wav)
                
                try:
                   
                    signal, fs = torchaudio.load(full_wav_path)
                    
                    
                    embedding = classifier.encode_batch(signal)
                    
                  
                    emb_vector = embedding.squeeze().cpu().numpy()
                    
                    
                    save_path = os.path.join(embeddings_dir, f"{actor_name}.npy")
                    np.save(save_path, emb_vector)
                    
                    print(f" Cadastrado: {actor_name: <25} | Shape: {emb_vector.shape}")
                    count += 1
                    
                  
                    found_wav = True
                    break 
                    
                except Exception as e:
                    print(f" Erro ao ler áudio de {actor_name}: {e}")
                    
        
        if not found_wav:
            print(f" Pulei {actor_name}: Nenhum arquivo .wav válido encontrado nas subpastas.")

    print("-" * 50)
    print(f" Concluído! {count} atores cadastrados na pasta '../data/embeddings'.")

if __name__ == "__main__":
    enroll_actors()