import torch
import demucs.separate
import librosa
import warnings
import os
import sys

# ARQUIVO DE TESTE NUMERO 1
# Desmonta o áudio com demucs, gera os arquivos separados e tenta pegar o bpm com librosa.

# CONFIG CONSTANTS
FILE_PATH = "./audio-samples/variable-bpm-song.mp3" 

# VERIFICANDO SE ESTÁ USANDO GPU

print("--- 1. Verificando Ambiente ---")
is_cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU) Disponível: {is_cuda_available}")

if not is_cuda_available:
    print("AVISO: CUDA não encontrado. O processamento será feito na CPU e será MUITO lento.")
print("--------------------------------\n")


# SEPARAR INSTRUMENTOS (DEMUCS) ---
# Isso é pesado, deve usar GPU
print("--- 2. Demucs (dividir instrumentos) ---")
if not os.path.exists(FILE_PATH):
    print(f"ERRO: Arquivo '{FILE_PATH}' não encontrado.")
    print("Por favor, coloque sua música na pasta e atualize o FILE_PATH no script.")
    sys.exit(1)
try:
    # Ignora avisos
    warnings.filterwarnings("ignore")
    
    # Roda o separador. 
    # '-n htdemucs' é o modelo padrão.
    # Isso vai criar uma pasta 'separated/'
    demucs.separate.main(["-n", "htdemucs", FILE_PATH])
    print("Demucs concluído! Verifique a pasta 'separated'.")
except Exception as e:
    print(f"Erro ao rodar o Demucs: {e}")
    print("Dica: O Demucs pode falhar se o 'ffmpeg' não estiver instalado e no PATH do seu sistema.")

print("----------------------------------\n")


# EXTRAIR BPM (LIBROSA) ---
# Isso é rápido, usa a CPU.
print("--- 3. Rodando Librosa (BPM) ---")
try:
    # Carrega o arquivo de áudio
    y, sr = librosa.load(FILE_PATH)
    
    # Detecta o BPM
    bpm, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    print(f"Arquivo: {FILE_PATH}")
    print(f"BPM Detectado: {float(bpm):.2f}") # .2f formata para 2 casas decimais

except Exception as e:
    print(f"Erro ao rodar o Librosa: {e}")
    
print("-------------------------------\n")

print("STATUS: OK")
