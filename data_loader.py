import os
import glob
import pandas as pd
import librosa
import numpy as np

def load_audio_file(file_path, target_sr=22050, target_duration=5.0):
    """
    Loads an audio file, resamples it, normalizes amplitude,
    and trims or pads it to a fixed length in seconds.

    ``file_path`` may be a path string (archive recordings) or a file-like
    object such as an uploaded audio's BytesIO buffer (custom recordings) —
    librosa.load accepts either.
    """
    if isinstance(file_path, str) and not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    signal, sr = librosa.load(file_path, sr=target_sr)
    
    
    max_samples = int(target_sr * target_duration)
    if len(signal) < max_samples:
        signal = np.pad(signal, (0, max_samples - len(signal)), mode='constant')
    else:
        signal = signal[:max_samples]
        
    # Amplitude normalization (-1 to 1)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    return signal, target_sr

def get_available_audio_files(base_path="archive"):
    
    folders = ["set_a", "set_b"]
    file_dict = {}
    
    for folder in folders:
        pattern = os.path.join(base_path, folder, "*.wav")
        files = glob.glob(pattern)
        file_dict[folder] = sorted(files)
        
    return file_dict