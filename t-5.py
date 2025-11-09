import torch
import demucs.separate
import numpy as np 
import os
import warnings
import time
import json
import librosa

from madmom.audio.signal import Signal
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor


# --- CONFIGURATION ---
FILE_NAME = "audio-samples/variable-bpm-song.mp3"  # Arquivo de teste
DEMUCS_MODEL = "htdemucs"
STEMS_FOLDER = f"separated/{DEMUCS_MODEL}/{os.path.splitext(os.path.basename(FILE_NAME))[0]}"

def remove_outliers(bpms, z_thresh=2.5):
    mean = np.mean(bpms)
    std = np.std(bpms)
    return np.clip(bpms, mean - z_thresh * std, mean + z_thresh * std)

def bpm_by_window(beat_times, window_size=5.0):
    bpms = 60 / np.diff(beat_times)
    times = beat_times[:-1]
    averaged = []
    start = 0
    while start < times[-1]:
        mask = (times >= start) & (times < start + window_size)
        if np.any(mask):
            avg_bpm = np.mean(bpms[mask])
            averaged.append({"time_sec": round(start, 2), "bpm": round(avg_bpm, 2)})
        start += window_size
    return averaged

def smooth_bpm(bpms, window_size=8):
    smoothed = np.convolve(bpms, np.ones(window_size)/window_size, mode="valid")
    return np.concatenate(([bpms[0]]*(window_size//2), smoothed, [bpms[-1]]*(window_size//2)))


# --- BPM FUNCTION (USING MADMOM) ---
def process_bpm_madmom(drums_file_path, original_file_path):
    print("--- 3. Running Madmom (AI Beat Tracking) ---")

    try:
        y_drums, sr_drums = librosa.load(drums_file_path)
        if np.sum(np.abs(y_drums)) > 1000:
            print("INFO: Drums detected. Using 'drums.wav' for analysis.")
            file_to_analyze = drums_file_path #ORIGINAL
        else:
            print("WARNING: 'drums.wav' is silent. Using original file.")
            file_to_analyze = original_file_path
    except Exception:
        print(f"WARNING: Failed to load 'drums.wav'. Using original file.")
        file_to_analyze = original_file_path

    act = RNNBeatProcessor()(file_to_analyze)
    proc = DBNBeatTrackingProcessor(fps=100)
    beat_times = proc(act)

    if len(beat_times) < 2:
        print("ERROR: Madmom couldn't find enough beats.")
        return []

    inter_beat_intervals = np.diff(beat_times)
    bpms = 60 / inter_beat_intervals

    # --- Correções ---
    bpms = remove_outliers(bpms)
    bpms = smooth_bpm(bpms, window_size=8)
    # ------------------

    bpm_map = [
        {"time_sec": round(beat_times[i], 2), "bpm": round(bpms[i], 2)}
        for i in range(len(bpms))
    ]

    print(f"Dynamic BPM extracted successfully (using '{os.path.basename(file_to_analyze)}').")
    print("-------------------------------------------\n")
    return bpm_map


# --- MAIN ORCHESTRATOR ---
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

    # Extrair mapa dinâmico de BPM
    bpm_map_result = process_bpm_madmom(drums_path, FILE_NAME)
    
    print("--- 5. Final Result (JSON) ---")
    final_result_json = {"bpm_map": bpm_map_result}
    print(json.dumps(final_result_json, indent=2))
    print("---------------------------------\n")

    total_end_time = time.time()
    print(f">>> Processing complete in {total_end_time - total_start_time:.2f} seconds. <<<")


if __name__ == "__main__":
    main()


