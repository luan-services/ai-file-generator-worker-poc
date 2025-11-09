#!/usr/bin/env python3
"""
bpm_extractor_combined.py

Opção 3 — combinação:
- RNNBeatProcessor + DBNBeatTrackingProcessor (madmom) para beat_times estáveis
- cálculo direto de BPM = 60 / diff(beat_times)
- remoção de outliers robusta (MAD)
- suavização Gaussiana preservando rampas
- limitador de aceleração (max BPM change por segundo)
- agregação em janelas (opcional) para UI

Rode: python bpm_extractor_combined.py
"""

import os
import json
import time
import warnings

import numpy as np
import librosa

import torch
import demucs.separate

from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor


# ----------------- CONFIG -----------------
FILE_NAME = "audio-samples/constant-bpm-song.mp3"
DEMUCS_MODEL = "htdemucs"
STEMS_FOLDER = f"separated/{DEMUCS_MODEL}/{os.path.splitext(os.path.basename(FILE_NAME))[0]}"

MADMOM_FPS = 100              # fps para DBN
GAUSSIAN_SIGMA = 1.2         # suavização: menor = mais responsivo, maior = mais suave
MAD_Z_THRESH = 3.0           # remoção de outliers via MAD
MAX_BPM_CHANGE_PER_SEC = 4.5 # limitar mudança de bpm (BPM por segundo). Ajuste conforme musica.
AGG_WINDOW_SEC = 2.0         # agrupar resultados para UI. 0 = sem agregação
MIN_BEATS = 3
# -----------------------------------------


def remove_outliers_mad(arr, z_thresh=MAD_Z_THRESH):
    """Remove outliers usando MAD (robusto). Retorna cópia."""
    arr = np.asarray(arr, dtype=float)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad == 0:
        return arr.copy()
    # aproximar z-score robusto
    z_scores = 0.6745 * (arr - median) / mad
    out = arr.copy()
    out[np.abs(z_scores) > z_thresh] = median
    return out


def gaussian_smooth(arr, sigma=GAUSSIAN_SIGMA):
    """Suaviza com kernel Gaussiano (1D). Retorna mesmo tamanho."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    # kernel size: cover +/- 3 sigma
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed


def limit_bpm_acceleration(bpm_arr, times, max_change_per_sec=MAX_BPM_CHANGE_PER_SEC):
    """
    Limita a variação de BPM entre batidas baseado no delta time entre batidas.
    bpm_arr: array de bpm por batida (len = n_beats-1 typically)
    times: array de times correspondentes ao bpm entries (times at which BPM value applies).
    Retorna array do mesmo tamanho.
    """
    bpm = np.asarray(bpm_arr, dtype=float).copy()
    if len(bpm) < 2:
        return bpm
    # assumimos bpm[i] corresponde a intervalo entre beat i e i+1 e time index em 'times[i]'
    for i in range(1, len(bpm)):
        dt = max(1e-6, times[i] - times[i - 1])  # segundos entre avaliações
        max_delta = max_change_per_sec * dt
        diff = bpm[i] - bpm[i - 1]
        if diff > max_delta:
            bpm[i] = bpm[i - 1] + max_delta
        elif diff < -max_delta:
            bpm[i] = bpm[i - 1] - max_delta
    return bpm


def aggregate_bpm_map(highres_map, window_sec=AGG_WINDOW_SEC):
    """Agrega o mapa de alta resolução em janelas fixas."""
    if window_sec <= 0 or not highres_map:
        return highres_map
    last_time = highres_map[-1]["time_sec"]
    aggregated = []
    t = 0.0
    while t <= last_time:
        window_vals = [e["bpm"] for e in highres_map if e["time_sec"] >= t and e["time_sec"] < t + window_sec]
        if window_vals:
            aggregated.append({"time_sec": round(t, 2), "bpm": round(float(np.mean(window_vals)), 2)})
        t += window_sec
    return aggregated


def process_bpm_combined(original_file_path):
    print(f"--- Running Madmom beat tracking on {original_file_path} ({MADMOM_FPS}fps) ---")

    # 1) Get activations (RNN) and beat_times (DBN)
    act = RNNBeatProcessor()(original_file_path)
    proc = DBNBeatTrackingProcessor(fps=MADMOM_FPS)
    beat_times = proc(act)

    if len(beat_times) < MIN_BEATS:
        print("ERROR: not enough beats found by madmom.")
        return []

    # 2) Calculate inter-beat intervals and raw BPMs
    ibis = np.diff(beat_times)  # time between beats
    ibis[ibis == 0] = 1e-6
    raw_bpms = 60.0 / ibis
    # times for each bpm value: use the time of the earlier beat (or mid-point)
    bpm_times = beat_times[:-1]  # corresponds to each IBI

    # 3) Remove outliers robustly
    bpms_no_out = remove_outliers_mad(raw_bpms, z_thresh=MAD_Z_THRESH)

    # 4) Smooth preserving ramp shapes
    bpms_smooth = gaussian_smooth(bpms_no_out, sigma=GAUSSIAN_SIGMA)

    # 5) Limit acceleration (avoid overshoot artificial)
    bpms_limited = limit_bpm_acceleration(bpms_smooth, bpm_times, max_change_per_sec=MAX_BPM_CHANGE_PER_SEC)

    # 6) Build high-res map (one entry per beat interval)
    highres_map = [{"time_sec": round(float(bpm_times[i]), 2), "bpm": round(float(bpms_limited[i]), 2)}
                   for i in range(len(bpms_limited))]

    # 7) Optionally aggregate in windows for UI
    aggregated = aggregate_bpm_map(highres_map, window_sec=AGG_WINDOW_SEC)

    # Return aggregated if used, otherwise highres
    result = aggregated if AGG_WINDOW_SEC > 0 else highres_map

    print("Dynamic BPM extracted (combined method).")
    return result


def main():
    t0 = time.time()
    print("--- 1. Environment check ---")
    print("CUDA available:", torch.cuda.is_available())
    print("---------------------------\n")

    if not os.path.exists(FILE_NAME):
        print(f"ERROR: file not found: {FILE_NAME}")
        return

    # keep Demucs step (you had it before); optional if not needed
    print("--- 2. Running Demucs (optional) ---")
    warnings.filterwarnings("ignore")
    demucs_start = time.time()
    demucs.separate.main(["-n", DEMUCS_MODEL, FILE_NAME])
    demucs_end = time.time()
    print(f"Demucs finished in {demucs_end - demucs_start:.2f}s")
    print("------------------------------------\n")

    # Run combined BPM extractor on the ORIGINAL mix
    bpm_map = process_bpm_combined(FILE_NAME)

    print("\n--- Final Result (JSON) ---")
    print(json.dumps({"bpm_map": bpm_map}, indent=2))
    print("---------------------------")
    print(f">>> total time: {time.time() - t0:.2f}s <<<")


if __name__ == "__main__":
    main()
