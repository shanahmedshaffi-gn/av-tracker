import os
import torch
import logging
import numpy as np
import torchaudio
import re  
import torchaudio.transforms as T
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from speechbrain.inference.separation import SepformerSeparation
from multi_speaker_verifier import MultiSpeakerVerifier
from huggingface_hub import login
from difflib import SequenceMatcher


HF_TOKEN = "" 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LargeMeetingTranscriber:
    def __init__(self, verifier, hf_token, whisper_size="large-v2", device="cuda"):
        self.verifier = verifier
        self.device = device
        
        logger.info("Autenticando...")
        login(token=hf_token)

        logger.info("Carregando PyAnnote...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(torch.device(device))

        logger.info("Carregando SepFormer (Local)...")
        local_path = "pretrained_models/sepformer"
        if not os.path.exists(os.path.join(local_path, "hyperparams.yaml")):
            logger.error("❌ ARQUIVOS NÃO ENCONTRADOS.")
            exit()

        self.separator = SepformerSeparation.from_hparams(
            source=local_path, 
            savedir=local_path,
            run_opts={"device": device}
        )

        logger.info(f"Carregando Whisper ({whisper_size})...")
        self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="int8")

    -
    def normalize_text(self, text):
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def process_audio(self, audio_path):
        if not os.path.exists(audio_path): raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        logger.info(f"Processando: {audio_path}")
        diarization = self.pipeline(audio_path)
        overlap_timeline = diarization.get_overlap()
        waveform, sample_rate = torchaudio.load(audio_path)
        
        resample_to_8k = T.Resample(sample_rate, 8000)
        resample_to_16k = T.Resample(8000, 16000)
        resample_orig_to_16k = T.Resample(sample_rate, 16000)
        
        full_transcript = []
        
        speaker_history = {} 

        logger.info("Transcrevendo segmentos...")

        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            if (turn.end - turn.start) < 0.5: continue
            
            start_frame = int(turn.start * sample_rate)
            end_frame = int(turn.end * sample_rate)
            if start_frame >= waveform.shape[1]: continue
            
            audio_segment = waveform[:, start_frame:end_frame]
            
            overlap_duration = 0.0
            for ov in overlap_timeline:
                inter = turn & ov
                if inter: overlap_duration += inter.duration
            is_overlap = (overlap_duration / (turn.end - turn.start)) > 0.30

            sources_to_transcribe = []

            if is_overlap:
                logger.info(f"Sobreposição em {turn.start:.1f}s. Separando...")
                mono = torch.mean(audio_segment, dim=0) if audio_segment.shape[0] > 1 else audio_segment.squeeze(0)
                mono_8k = resample_to_8k(mono)
                mono_8k_input = mono_8k.unsqueeze(0) 

                try:
                    est_sources = self.separator.separate_batch(mono_8k_input)
                    src1_8k = est_sources[0, :, 0].detach().cpu()
                    src2_8k = est_sources[0, :, 1].detach().cpu()

                    src1_16k = resample_to_16k(src1_8k).numpy()
                    src2_16k = resample_to_16k(src2_8k).numpy()
                    
                    sources_to_transcribe = [src1_16k, src2_16k]
                except Exception as e:
                    logger.error(f" Falha na separação: {e}. Usando áudio original.")
                    sources_to_transcribe = [resample_orig_to_16k(audio_segment).squeeze().numpy()]
            else:
                audio_16k = resample_orig_to_16k(audio_segment)
                sources_to_transcribe = [audio_16k.squeeze().numpy()]

            for idx, source_np in enumerate(sources_to_transcribe):
                source_np = source_np.squeeze()
                if source_np.ndim != 1 or source_np.size == 0: continue
                
                mx = np.abs(source_np).max()
                if np.isnan(mx) or mx == 0: continue
                source_np = source_np / mx
                
                real_name, conf = self.verifier._process_audio_chunk(source_np, sample_rate=16000)
                
                if is_overlap:
                    print(f"    [DEBUG] Voz {idx+1}: {real_name} ({conf:.1%})")

                disp = real_name if real_name != "Unknown" else speaker_id
                
                try:
                    segs, _ = self.whisper.transcribe(source_np, language="en", beam_size=1, vad_filter=True)
                    text = " ".join([s.text for s in segs]).strip()
                    
                    if text and "Amara.org" not in text:
                        
                       
                        clean_new = self.normalize_text(text) 
                        last_entry = speaker_history.get(disp)
                        is_duplicate = False
                        
                        if last_entry and clean_new: 
                            clean_old = last_entry['clean_text']
                            
                            
                            if clean_new in clean_old or clean_old in clean_new:
                                is_duplicate = True
                            
                           
                            else:
                                ratio = SequenceMatcher(None, clean_new, clean_old).ratio()
                                if ratio > 0.5: 
                                    is_duplicate = True
                        
                        if is_duplicate:
                           
                            continue
                        
                        
                        speaker_history[disp] = {'clean_text': clean_new, 'time': turn.end}
                      

                        suffix = f" (Voz {idx+1})" if is_overlap else ""
                        print(f" [{turn.start:.1f}s] {disp}{suffix}: {text}")
                        
                        full_transcript.append({
                            "speaker": disp, 
                            "text": text, 
                            "start": turn.start,
                            "suffix": suffix
                        })
                except Exception as e:
                    logger.error(f"Erro Whisper: {e}")
                
        return full_transcript

if __name__ == "__main__":
    embeddings_dir = "../data/embeddings"
    if not os.path.exists(embeddings_dir): 
        print(" Pasta de embeddings não encontrada")
        exit()

    print(" Inicializando...")
    try:
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.0)
        system = LargeMeetingTranscriber(verifier, HF_TOKEN)

        while True:
            f = input("\n Arquivo: ").strip()
            if f in ['sair', 'exit']: break
            path = f if os.path.exists(f) else f"../data/testes/{f}"
            
            if os.path.exists(path):
                transcript_result = system.process_audio(path)
                
                base_name = os.path.splitext(os.path.basename(path))[0]
                output_filename = f"{base_name}_transcricao.txt"
                
                if transcript_result:
                    print(f"\n Salvando transcrição em: {output_filename} ...")
                    with open(output_filename, "w", encoding="utf-8") as txt_file:
                        txt_file.write(f"ARQUIVO: {f}\n")
                        txt_file.write("-" * 50 + "\n\n")
                        
                        for line in transcript_result:
                            timestamp = f"[{line['start']:.1f}s]"
                            speaker = line['speaker']
                            suffix = line.get('suffix', '')
                            text = line['text']
                            txt_file.write(f"{timestamp} {speaker}{suffix}: {text}\n")
                    
                    print(f" Arquivo '{output_filename}' salvo com sucesso!")
                else:
                    print(" Nenhuma fala detectada.")
            else:
                print(" Arquivo não existe.")
    except Exception as e:
        logger.error(f"Erro Fatal: {e}")