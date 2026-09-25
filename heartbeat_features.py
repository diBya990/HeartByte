import numpy as np
import librosa
from scipy.signal import correlate, find_peaks

from signal_processor import bandpass_filter, extract_envelope

# Order matters: training and inference must build this vector identically.
FEATURE_NAMES = [
    "low_freq_ratio", "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_rolloff_mean", "zero_crossing_rate_mean",
    "rms_mean", "rms_std", "envelope_periodicity_strength",
] + [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)] + [
    "beat_period_cv", "beat_height_cv", "beat_similarity", "noise_floor_ratio",
    "spectral_flatness_mean", "harmonic_ratio", "chroma_peak_mean",
]


def _beat_features(signal, sr):
    """
    Features that separate a real stethoscope heartbeat from a produced "beat"
    (drum/kick loops, 808s, synthesized heartbeat sound effects, music with a
    beat), which otherwise share a heartbeat's low-frequency, periodic profile:

    - beat_period_cv / beat_height_cv: a real heart's cycle length and loudness
      drift beat to beat; a sequenced beat repeats with (near) zero variation.
      Period is measured every second peak so a regular lub-dub alternation
      doesn't count as variation.
    - beat_similarity: a produced beat replays the same sample, so consecutive
      beat waveforms correlate ~1.0; real heart sounds only resemble each other.
    - noise_floor_ratio: a stethoscope always picks up body/room noise between
      beats; digitally produced beats drop to near-silence.
    - spectral_flatness_mean / harmonic_ratio / chroma_peak_mean: music and
      tonal kicks carry pitched, harmonic energy concentrated in a few pitch
      classes; heart sounds are unpitched thumps.
    """
    # Ignore the zero padding data_loader adds to short clips, otherwise it
    # would read as a silent (i.e. digitally clean) stretch.
    active = np.flatnonzero(np.abs(signal) > 1e-4)
    if len(active) > sr:
        signal = signal[:active[-1] + 1]

    filtered = bandpass_filter(signal, lowcut=20, highcut=300, fs=sr)
    envelope = extract_envelope(filtered, fs=sr, cutoff=10.0)
    peaks, _ = find_peaks(envelope, distance=int(0.25 * sr), height=0.25 * envelope.max())

    if len(peaks) >= 4:
        period2 = peaks[2:] - peaks[:-2]
        beat_period_cv = period2.std() / period2.mean()
        heights = envelope[peaks]
        beat_height_cv = heights.std() / heights.mean()

        pre, post, max_lag = int(0.05 * sr), int(0.2 * sr), int(0.03 * sr)
        similarities = []
        for a, b in zip(peaks[:-2], peaks[2:]):
            if a - pre - max_lag < 0 or b + post + max_lag > len(filtered):
                continue
            x = filtered[a - pre:a + post]
            y = filtered[b - pre - max_lag:b + post + max_lag]
            corr = correlate(y, x, mode="valid", method="fft")
            # Sliding-window energy of y for per-lag normalization.
            y_energy = np.convolve(y ** 2, np.ones(len(x)), mode="valid")
            norm = np.linalg.norm(x) * np.sqrt(np.maximum(y_energy, 1e-12))
            similarities.append((corr / norm).max())
        beat_similarity = float(np.median(similarities)) if similarities else 0.0
    else:
        beat_period_cv, beat_height_cv, beat_similarity = 1.0, 1.0, 0.0

    noise_floor_ratio = max(0.0, np.percentile(envelope, 10) / (np.percentile(envelope, 95) + 1e-12))

    power_spec = np.abs(librosa.stft(signal)) ** 2
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(power_spec))[0]
    harmonic, percussive = librosa.decompose.hpss(power_spec)
    harmonic_ratio = harmonic.sum() / (harmonic.sum() + percussive.sum() + 1e-12)
    chroma = librosa.feature.chroma_stft(S=power_spec, sr=sr, tuning=0.0)
    chroma_peak = (chroma.max(axis=0) / (chroma.sum(axis=0) + 1e-12)).mean()

    return [
        beat_period_cv, beat_height_cv, beat_similarity, noise_floor_ratio,
        flatness.mean(), harmonic_ratio, chroma_peak,
    ]


def extract_features(signal, sr):
    """
    Builds a fixed-length feature vector describing whether an audio clip
    "sounds like" a stethoscope heartbeat (PCG) recording, for the trained
    binary classifier in heartbeat_classifier.py.

    Combines: where the signal's energy sits in frequency (heart sounds are
    low-frequency and narrowband compared to speech/music), general timbre
    (MFCCs, distinguishes tonal/harmonic content from the muffled thump of a
    heartbeat), and how strongly the envelope repeats at a plausible heart-rate
    lag (periodicity a heartbeat has that random noise or a sustained tone
    doesn't), plus beat-structure/tonality features (see _beat_features) that
    reject drum beats and other produced beat sounds.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = len(signal)

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    power = np.abs(np.fft.rfft(signal * np.hanning(n))) ** 2
    total_mask = (freqs >= 20) & (freqs <= 2000)
    total_energy = power[total_mask].sum()
    band_mask = (freqs >= 20) & (freqs <= 150)
    low_freq_ratio = power[band_mask].sum() / total_energy if total_energy > 0 else 0.0

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    rms = librosa.feature.rms(y=signal)[0]
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    filtered = bandpass_filter(signal, lowcut=20, highcut=300, fs=sr)
    envelope = extract_envelope(filtered, fs=sr)
    env = envelope - envelope.mean()
    # FFT-based correlation (O(n log n)) — plain np.correlate is a direct O(n^2)
    # algorithm and takes seconds per call at this array length (~110k samples).
    autocorr = correlate(env, env, mode="full", method="fft")[len(env) - 1:]
    zero_lag = autocorr[0]
    autocorr = autocorr / zero_lag if zero_lag > 0 else autocorr

    lag_min = int(60.0 / 220 * sr)  # fastest plausible heart rate (220 bpm)
    lag_max = min(int(60.0 / 30 * sr), len(autocorr) - 1)  # slowest plausible (30 bpm)
    periodicity_strength = autocorr[lag_min:lag_max].max() if lag_max > lag_min else 0.0

    features = np.concatenate([
        [
            low_freq_ratio,
            centroid.mean(), centroid.std(),
            bandwidth.mean(),
            rolloff.mean(),
            zcr.mean(),
            rms.mean(), rms.std(),
            periodicity_strength,
        ],
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        _beat_features(signal, sr),
    ])
    return features.astype(np.float64)
