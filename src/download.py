import os
import time
from remotezip import RemoteZip
import requests

url = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip"
QTD_ATORES = 1000  # Atenção: Isso é quase o dataset todo (o total é ~1211 atores)
PASTA_DESTINO = "./vox_100_atores"

def baixar_com_retry():
    while True:
        try:
            print("Conectando ao servidor para ler o índice...")
            # ADIÇÃO 1: Timeout de 60 segundos ou mais para evitar quedas
            with RemoteZip(url, timeout=60) as zip_remoto:
                lista_arquivos = zip_remoto.namelist()
                lista_arquivos.sort()

                ids_encontrados = set() # Set é mais rápido para verificar existência
                arquivos_para_baixar = []

                print("Filtrando lista de atores...")
                
                # Identifica os 1000 atores
                for arquivo in lista_arquivos:
                    if arquivo.endswith('/'): continue
                    
                    partes = arquivo.split('/')
                    # Procura o ID (ex: id10001)
                    ator_atual = None
                    for parte in partes:
                         if parte.startswith('id1') and len(parte) == 7:
                             ator_atual = parte
                             break
                    
                    if ator_atual:
                        # Se ainda não temos 1000 atores, adicionamos novos
                        if len(ids_encontrados) < QTD_ATORES:
                            ids_encontrados.add(ator_atual)
                        
                        # Se este arquivo é de um ator que está na nossa lista, baixa ele
                        if ator_atual in ids_encontrados:
                            arquivos_para_baixar.append(arquivo)

                total = len(arquivos_para_baixar)
                print(f"--- Resumo ---")
                print(f"Atores selecionados: {len(ids_encontrados)}")
                print(f"Total de arquivos de áudio: {total}")
                print(f"Destino: {PASTA_DESTINO}")
                print(f"--------------")

                # Loop de download
                arquivos_existentes = 0
                
                for i, arquivo in enumerate(arquivos_para_baixar):
                    caminho_local = os.path.join(PASTA_DESTINO, arquivo)
                    
                    # Lógica de Resume: Se existe e tem tamanho > 0, pula
                    if os.path.exists(caminho_local) and os.path.getsize(caminho_local) > 0:
                        arquivos_existentes += 1
                        if i % 500 == 0: # Feedback visual para não parecer travado
                            print(f"Verificando arquivos existentes... ({i}/{total})", end='\r')
                        continue
                    
                    # Mostra progresso real de download
                    print(f"Baixando [{i+1}/{total}]: {arquivo}" + " " * 20, end='\r')
                    
                    # Extrai o arquivo
                    zip_remoto.extract(arquivo, path=PASTA_DESTINO)
                
                print(f"\n\nFinalizado! {total} arquivos garantidos no disco.")
                break # Sai do while True se terminar tudo com sucesso

        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, Exception) as e:
            print(f"\n\n⚠️ Conexão instável ({e}).")
            print("⏳ Aguardando 10 segundos antes de tentar retomar de onde parou...")
            time.sleep(10)
            continue 

if __name__ == "__main__":
    if not os.path.exists(PASTA_DESTINO):
        os.makedirs(PASTA_DESTINO)
    baixar_com_retry()