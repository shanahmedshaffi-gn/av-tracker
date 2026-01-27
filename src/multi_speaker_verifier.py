import torch
import torchaudio
import os
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

class MultiSpeakerVerifier:
    def __init__(self, embedding_directory, threshold=0.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        
        print(f"🧠 Inicializando Verificador (Device: {self.device})")
        print(f"   📂 Embeddings: {embedding_directory}")
        
        # Carrega o modelo ECAPA-TDNN (O mesmo do enrollment)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": self.device}
        )
        
        # Carrega os bancos de dados na memória
        self.embeddings = {}
        self.load_embeddings(embedding_directory)

    def load_embeddings(self, directory):
        if not os.path.exists(directory):
            print("❌ Diretório de embeddings não encontrado!")
            return

        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                path = os.path.join(directory, filename)
                name = os.path.splitext(filename)[0]
                try:
                    # Carrega e garante formato Tensor PyTorch
                    emb_numpy = np.load(path)
                    emb_tensor = torch.from_numpy(emb_numpy).to(self.device)
                    self.embeddings[name] = emb_tensor
                    count += 1
                except Exception as e:
                    print(f"❌ Erro ao carregar {filename}: {e}")
        
        print(f"   ✅ {count} atores carregados na memória.")

    def _process_audio_chunk(self, audio_chunk, sample_rate=16000):
        # 1. Prepara o áudio (Garante Tensor [1, Time])
        if isinstance(audio_chunk, np.ndarray):
            signal = torch.from_numpy(audio_chunk).float().to(self.device)
        else:
            signal = audio_chunk.to(self.device)

        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # 2. Extrai o embedding da voz atual (Separada)
        with torch.no_grad():
            # O classifier retorna [1, 1, 192], fazemos squeeze para [192]
            output = self.classifier.encode_batch(signal)
            current_embedding = output.squeeze()

        # 3. Compara com todos os atores do banco
        best_score = -1.0
        best_speaker = "Unknown"

        # Variável para debug visual (apenas se for sobreposição/separado)
        debug_scores = []

        for speaker, stored_embedding in self.embeddings.items():
            # Similaridade de Cosseno (PyTorch)
            # A stored_embedding também precisa estar no mesmo device
            score = torch.nn.functional.cosine_similarity(current_embedding, stored_embedding, dim=0).item()
            
            debug_scores.append((speaker, score))

            if score > best_score:
                best_score = score
                best_speaker = speaker

        # 4. Lógica de decisão
        
        # Ordena para vermos os top 3 no log
        debug_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = debug_scores[:3]
        
        # Se a pontuação for muito baixa, imprimimos para entender o drama
        if best_score < 0.25:
            # Monta string de debug
            top_str = " | ".join([f"{n}: {s:.1%}" for n, s in top_3])
            # Descomente a linha abaixo se quiser ver TODOS os comparativos no terminal
            # print(f"   📊 Comparativo: {top_str}")

        if best_score >= self.threshold:
            return best_speaker, best_score
        else:
            return "Unknown", best_score