import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(data, lowcut=20.0, highcut=400.0, fs=22050, order=3):
    """Applies a Butterworth bandpass filter to remove background acoustic noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_envelope(signal, fs=22050, cutoff=4.0):
    """Applies rectification and lowpass filtering to extract the signal energy envelope."""
    rectified = np.abs(signal)
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype='low')
    return filtfilt(b, a, rectified)

def analyze_peaks(envelope, fs=22050):
    """
    Detects heartbeat peaks (S1/S2 peaks) and calculates BPM & HRV.
    """
   
    min_distance = int(0.4 * fs)
    min_height = np.max(envelope) * 0.20
    
    peaks, _ = find_peaks(envelope, distance=min_distance, height=min_height)
    peak_times = peaks / fs
    
    if len(peak_times) > 1:
        ibis = np.diff(peak_times) 
        bpm = 60.0 / np.mean(ibis)
        hrv = np.std(ibis) * 1000 
    else:
        bpm, hrv = 0.0, 0.0
        
    return peaks, peak_times, bpm, hrv


def label_heart_sounds(envelope, peaks, fs=22050):
    """
    Splits each detected cardiac cycle (one entry in `peaks`, from analyze_peaks)
    into its S1 ("lub") and S2 ("dub") heart sounds, for display purposes only —
    it does not affect BPM/HRV, which stay based on the cycle-level peaks from
    analyze_peaks.

    Within one cycle, S1 (mitral/tricuspid valve closure) always precedes S2
    (aortic/pulmonic valve closure), so each cycle is searched for up to two
    local sub-peaks in the envelope and the earlier one is labeled S1, the
    later S2. When only one sub-peak is resolvable (common in quieter/noisier
    recordings, where the two sounds blur together), it's labeled S1 only.
    This is a heuristic for visualization, not a validated clinical
    segmentation.

    Returns (s1_indices, s2_indices) as sample-index arrays into `envelope`.
    """
    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n = len(envelope)
    avg_gap = int(np.mean(np.diff(peaks))) if len(peaks) > 1 else int(0.8 * fs)

    boundaries = [max(0, peaks[0] - avg_gap // 2)]
    boundaries += [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    boundaries.append(min(n, peaks[-1] + avg_gap // 2))

    sub_distance = max(1, int(0.05 * fs))  # S1 and S2 are at least ~50ms apart
    s1_indices, s2_indices = [], []

    for i, peak in enumerate(peaks):
        start, end = boundaries[i], boundaries[i + 1]
        segment = envelope[start:end]
        if len(segment) == 0:
            s1_indices.append(peak)
            continue

        local_height = np.max(segment) * 0.15
        sub_peaks, _ = find_peaks(segment, distance=sub_distance, height=local_height)
        sub_peaks = sub_peaks + start

        if len(sub_peaks) == 0:
            sub_peaks = np.array([peak])
        elif len(sub_peaks) > 2:
            # Keep only the two most prominent sub-peaks, ordered by time.
            top2 = sub_peaks[np.argsort(envelope[sub_peaks])[-2:]]
            sub_peaks = np.sort(top2)

        s1_indices.append(sub_peaks[0])
        if len(sub_peaks) > 1:
            s2_indices.append(sub_peaks[1])

    return np.array(s1_indices, dtype=int), np.array(s2_indices, dtype=int)


def generate_synthetic_ecg(time_array, peak_times):
    """
    Synthesizes a clinically-proportioned P-QRS-T ECG waveform synchronized with
    detected beat peaks. Amplitudes and widths follow typical Lead II proportions
    (R ~1.0 mV reference, PR ~0.18s, QRS ~0.08s, QT ~0.40s) so the waveform reads
    as a recognizable ECG rather than a generic pulse. This is a visual simulation
    only — it is not derived from real electrical cardiac activity.
    """
    ecg_signal = np.zeros_like(time_array)

    # (offset from R-peak in seconds, amplitude, gaussian sigma in seconds)
    ecg_components = [
        (-0.20,  0.15, 0.020),   # P wave
        (-0.025, -0.10, 0.006),  # Q wave
        ( 0.000,  1.00, 0.010),  # R wave
        ( 0.025, -0.20, 0.008),  # S wave
        ( 0.300,  0.35, 0.045),  # T wave
    ]

    for r_time in peak_times:
        for offset, amp, sigma in ecg_components:
            center_time = r_time + offset
            wave = amp * np.exp(-((time_array - center_time) ** 2) / (2 * (sigma ** 2)))
            ecg_signal += wave

    baseline_noise = np.random.normal(0, 0.006, size=len(time_array))
    return ecg_signal + baseline_noise


def downsample_preserve_peaks(signal, factor):
    """
    Downsamples by keeping the largest-magnitude sample in each window, instead of
    naively striding, so sharp features (like the QRS spike) survive decimation
    for display instead of being averaged/skipped into noise.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return signal

    usable_len = (len(signal) // factor) * factor
    windows = signal[:usable_len].reshape(-1, factor)
    peak_idx = np.argmax(np.abs(windows), axis=1)
    return windows[np.arange(windows.shape[0]), peak_idx]


def get_interval_stats(peak_times):
    """Returns beat-to-beat interval statistics (ms) used in the report tables."""
    if len(peak_times) > 1:
        ibis_ms = np.diff(peak_times) * 1000.0
        return {
            "mean_ms": float(np.mean(ibis_ms)),
            "min_ms": float(np.min(ibis_ms)),
            "max_ms": float(np.max(ibis_ms)),
        }
    return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}