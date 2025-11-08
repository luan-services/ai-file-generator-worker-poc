import torch
import demucs.separate
import librosa
import numpy as np 
import os
import warnings
import time
import json

# --- CONFIGURATION ---
FILE_NAME = "audio-samples/variable-bpm-song.mp3" # Your test file
DEMUCS_MODEL = "htdemucs" 
STEMS_FOLDER = f"separated/{DEMUCS_MODEL}/{os.path.splitext(os.path.basename(FILE_NAME))[0]}"

# --- BPM FUNCTION (SLIDING WINDOW LOGIC) ---
def process_bpm_sliding_window(drums_file_path, original_file_path):
    """
    Analyzes the audio in overlapping "slices" (windows)
    to detect the BPM over time.
    """
    print("--- 3. Running Librosa (BPM with Sliding Window) ---")
    file_to_analyze = ""
    y, sr = None, None
    
    try:
        y_drums, sr_drums = librosa.load(drums_file_path)
        if np.sum(np.abs(y_drums)) > 1000:
            print("INFO: Drums detected. Using 'drums.wav' for BPM analysis.")
            y, sr = y_drums, sr_drums
            file_to_analyze = "Drums"
        else:
            print("WARNING: 'drums.wav' is silent. Using original file for BPM.")
            y, sr = librosa.load(original_file_path)
            file_to_analyze = "Original"
            
    except Exception:
        print(f"WARNING: Failed to load 'drums.wav'. Using original file for BPM.")
        y, sr = librosa.load(original_file_path)
        file_to_analyze = "Original"

    # --- SLIDING WINDOW LOGIC ---
    
    # Window settings:
    chunk_sec = 15  # Analyze a 15-second window
    step_sec = 2    # Move the window 2 seconds forward each step
    
    # Convert seconds to samples
    samples_per_chunk = int(chunk_sec * sr)
    samples_per_step = int(step_sec * sr)
    
    bpm_map = []
    
    # Main loop: "slides" the window across the song
    for start_sample in range(0, len(y) - samples_per_chunk, samples_per_step):
        
        end_sample = start_sample + samples_per_chunk
        
        # Get the audio "chunk"
        chunk = y[start_sample:end_sample]
        
        # Run 'beat_track' ONLY on this chunk
        bpm, beats = librosa.beat.beat_track(y=chunk, sr=sr)
        
        # --- THIS IS THE FIX ---
        # If 'bpm' is an array (e.g., [110.0, 112.5]), take the average.
        if isinstance(bpm, np.ndarray):
            bpm = np.mean(bpm)
        # --- END OF FIX ---

        # Get the start time of this chunk (in seconds)
        current_time_sec = start_sample / sr
        
        # Add to our map
        bpm_map.append({
            "time_sec": round(current_time_sec, 2),
            "bpm": round(bpm, 2)
        })

    print(f"Dynamic BPM extracted (based on '{file_to_analyze}').")
    print("--------------------------------------------------\n")
    return bpm_map

# --- MAIN ---
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

    # Call the new function
    bpm_map_result = process_bpm_sliding_window(drums_path, FILE_NAME)
    
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