import torch
import demucs.separate
import numpy as np 
import os
import warnings
import time
import json

# --- MUDANÇAS AQUI ---
import madmom
# Precisamos de importar o 'Signal' para carregar o áudio
from madmom.audio.signal import Signal
from madmom.features.beats import BeatTrackingProcessor
# --- FIM DAS MUDANÇAS ---


# --- CONFIGURATION ---
FILE_NAME = "audio-samples/variable-bpm-song.mp3" # Your test file
DEMUCS_MODEL = "htdemucs" 
STEMS_FOLDER = f"separated/{DEMUCS_MODEL}/{os.path.splitext(os.path.basename(FILE_NAME))[0]}"


# --- BPM FUNCTION (USING MADMOM) ---
def process_bpm_madmom(drums_file_path, original_file_path):
    """
    Uses madmom's neural network to find the *actual* beat timestamps
    and calculates a precise, dynamic BPM map.
    """
    print("--- 3. Running Madmom (AI Beat Tracking) ---")
    file_to_analyze = ""
    
    try:
        # A lógica de fallback para 'drums.wav' ainda é boa
        import librosa
        y_drums, sr_drums = librosa.load(drums_file_path)
        if np.sum(np.abs(y_drums)) > 1000:
            print("INFO: Drums detected. Using 'drums.wav' for analysis.")
            file_to_analyze = drums_file_path
        else:
            print("WARNING: 'drums.wav' is silent. Using original file.")
            file_to_analyze = original_file_path
            
    except Exception:
        print(f"WARNING: Failed to load 'drums.wav'. Using original file.")
        file_to_analyze = original_file_path

    # --- MADMOM AI LOGIC (A CORREÇÃO) ---
    
    # 1. Carrega o ficheiro de áudio num objeto 'Signal' do madmom
    #    Em vez de passar a string 'file_to_analyze'
    sig = Signal(file_to_analyze)

    # 2. 'BeatTrackingProcessor' encontra os tempos das batidas
    proc = BeatTrackingProcessor(fps=100)
    
    # 3. Processa o objeto 'sig' (NÃO a string)
    beat_times = proc(sig)
    
    # --- FIM DA CORREÇÃO ---

    if len(beat_times) < 2:
        print("ERROR: Madmom couldn't find any beats.")
        return []

    # 3. Calcula o BPM a partir do tempo entre as batidas
    inter_beat_intervals = np.diff(beat_times)
    
    # 4. Converte IBIs (tempo) para BPM (batidas por minuto)
    bpms = 60 / inter_beat_intervals
    
    bpm_map = []
    
    for i in range(len(bpms)):
        bpm_map.append({
            "time_sec": round(beat_times[i], 2),
            "bpm": round(bpms[i], 2)
        })

    print(f"Dynamic BPM extracted (based on '{file_to_analyze}').")
    print("-------------------------------------------\n")
    return bpm_map

# --- MAIN (Orchestrator) ---
# (O 'main' está 100% correto e não precisa de alterações)
def main():
    total_start_time = time.time()
    
    print("--- 1. Checking Environment ---")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA (GPU) Available: {is_cuda_available}")
    print("--------------------------------\n")
    
    if not os.path.exists(FILE_NAME):
        print(f"ERROR: File '{FILE_NAME}' not found.")
        return

    print("--- 2. Running Demucs (Separation) ---")
    warnings.filterwarnings("ignore")
    demucs_start_time = time.time()
    demucs.separate.main(["-n", DEMUCS_MODEL, FILE_NAME])
    demucs_end_time = time.time()
    print(f"Demucs completed! (Took {demucs_end_time - demucs_start_time:.2f} seconds)")
    print("----------------------------------\n")

    drums_path = os.path.join(STEMS_FOLDER, "drums.wav")
    
    if not os.path.exists(drums_path):
        print("ERROR: Demucs failed to create the 'stem' files.")
        return

    # Call the new Madmom function
    bpm_map_result = process_bpm_madmom(drums_path, FILE_NAME)
    
    print("--- 5. Final Result (JSON) ---")
    final_result_json = {
        "bpm_map": bpm_map_result
    }
    print(json.dumps(final_result_json, indent=2))
    print("---------------------------------\n")

    total_end_time = time.time()
    print(f">>> Processing complete in {total_end_time - total_start_time:.2f} seconds. <<<")

if __name__ == "__main__":
    main()