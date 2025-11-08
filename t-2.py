import torch
import demucs.separate
import librosa
import numpy as np
import os
import warnings
import time
import json 
import sys

# ARQUIVO DE TESTE NUMERO 2
# Desmonta o áudio com demucs, usa o librosa para pegar o bpm pelo drums, tenta gerar bpm para cada parte da musica, ao inves de geral

# --- CONFIGURAÇÃO ---
FILE_NAME = "audio-samples/variable-bpm-song.mp3"
DEMUCS_MODEL = "htdemucs" 

# Define o caminho onde o Demucs salvará os stems
STEMS_FOLDER = f"separated/{DEMUCS_MODEL}/{os.path.splitext(os.path.basename(FILE_NAME))[0]}"

def processar_bpm_dinamico(arquivo_drums, arquivo_original):
    """
    Tenta detectar o MAPA de BPM do 'drums.wav'.
    Se falhar ou estiver silencioso, usa o arquivo original.
    """
    print("--- 3. Rodando Librosa (BPM Dinâmico V2) ---")
    arquivo_para_analise = ""
    y, sr = None, None
    
    try:
        y_drums, sr_drums = librosa.load(arquivo_drums)
        if np.sum(np.abs(y_drums)) > 1000:
            print("INFO: Bateria detectada. Usando 'drums.wav' para análise de BPM.")
            y, sr = y_drums, sr_drums
            arquivo_para_analise = "Drums"
        else:
            print("AVISO: 'drums.wav' está silencioso. Usando arquivo original para BPM.")
            y, sr = librosa.load(arquivo_original)
            arquivo_para_analise = "Original"
            
    except Exception:
        print(f"AVISO: Falha ao carregar 'drums.wav'. Usando arquivo original para BPM.")
        y, sr = librosa.load(arquivo_original)
        arquivo_para_analise = "Original"

    # --- LÓGICA DO MAPA DE BPM (A CORREÇÃO ESTÁ AQUI) ---
    
    # 1. Calcula o "envelope de força" (o "pulso" rítmico da música)
    #    Esta é a função que eu deveria ter usado!
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    
    # 2. Estima o tempo (BPM) baseado nesse *envelope* de pulso
    #    hop_length padrão é 512, o que alinha com o onset_strength
    tempo = librosa.feature.tempo(onset_envelope=onset_strength, sr=sr)
    
    # 3. Pega os carimbos de tempo (timestamps) para cada valor de BPM
    #    Isso vai criar um array [ 0.0, 0.5, 1.0, 1.5, ... ]
    times = librosa.times_like(tempo, sr=sr)
    
    # 4. Formata para o JSON (o que você quer ver)
    bpm_mapa = []
    for t, b in zip(times, tempo):
        bpm_mapa.append({"tempo_seg": round(t, 2), "bpm": round(b, 2)})

    print(f"BPM dinâmico extraído (baseado em '{arquivo_para_analise}').")
    print("-------------------------------------------\n")
    return bpm_mapa

def main():
    total_start= time.time()
    
    # VERIFICANDO SE ESTÁ USANDO GPU
    print("--- 1. Verificando Ambiente ---")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA (GPU) Disponível: {is_cuda_available}")
    print("--------------------------------\n")
    
    if not os.path.exists(FILE_NAME):
        print(f"ERRO: Arquivo '{FILE_NAME}' não encontrado.")
        return

    # SEPARAR INSTRUMENTOS (DEMUCS) ---
    print("--- 2. Rodando Demucs (Separação) ---")
    warnings.filterwarnings("ignore")
    demucs_start = time.time()
    demucs.separate.main(["-n", DEMUCS_MODEL, FILE_NAME])
    demucs_end = time.time()
    print(f"Demucs concluído! (Levou {demucs_end - demucs_start:.2f} segundos)")
    print("----------------------------------\n")

    # Define os caminhos dos arquivos que o Demucs criou
    drums_full_path = os.path.join(STEMS_FOLDER, "drums.wav")
    
    # Verifica se o Demucs realmente criou a pasta e os arquivos
    if not os.path.exists(drums_full_path):
        print("ERRO: O Demucs falhou em criar os arquivos 'stem'. Verifique a instalação.")
        return

    #  PROCESSAR O BPM DINÂMICO 
    bpm_result_map = processar_bpm_dinamico(drums_full_path, FILE_NAME)
    
    # JSON FINAL
    print("--- 5. Resultado Final (JSON) ---")
    json_response = {
        "bpm_mapa": bpm_result_map
    }
    
    print(json.dumps(json_response, indent=2))
    print("---------------------------------\n")

    total_end = time.time()
    print(f">>> Processamento completo em {total_end - total_start:.2f} segundos. <<<")


if __name__ == "__main__":
    main()