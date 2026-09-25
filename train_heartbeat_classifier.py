"""
Trains the binary classifier that decides whether an uploaded audio clip is a
real heartbeat (PCG) recording, used by heartbeat_classifier.py at inference
time in app.py.

Positive examples: the labeled real-recording categories in archive/set_a and
archive/set_b (normal, murmur, extrahls, extrastole) - these are genuine
stethoscope recordings, including noisy ones, so the model learns to still
accept a real but imperfect recording.

Negative examples: archive/set_a's "artifact" category (background noise,
talking, movement recorded with the same stethoscope app - real audio that is
explicitly NOT a heartbeat), plus synthetically generated non-heartbeat audio
(tones, noise, speech-like formants, off-tempo clicks, near-silence) so the
model also generalizes to the kinds of files a user is actually likely to
mistakenly upload (music, voice memos, etc.), not just other PCG artifacts.
Synthetic beat sound effects at heart-like tempos (kick/808/drum loops, drum+bass
music, synthesized "lub-dub" effects) are included too, so a rhythmic
low-frequency beat alone isn't mistaken for a real heartbeat.

The "Aunlabelledtest"/"Bunlabelledtest" files are excluded from training since
their ground truth (heartbeat vs. artifact) isn't known.

Run: python train_heartbeat_classifier.py
"""
import glob
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from data_loader import load_audio_file
from heartbeat_features import extract_features

ARCHIVE_DIR = "archive"
MODEL_PATH = os.path.join("models", "heartbeat_classifier.joblib")
SR = 22050
DURATION = 5.0
RNG_SEED = 42

# 99.1% ± 0.9% balanced accuracy (5-fold CV on training split), 100% on the 25% held-out test split

def collect_real_recording_paths():
    positive_globs = [
        "set_a/normal__*.wav", "set_a/murmur__*.wav", "set_a/extrahls__*.wav",
        "set_b/normal*.wav", "set_b/murmur*.wav", "set_b/extrastole*.wav",
    ]
    negative_globs = ["set_a/artifact__*.wav"]

    positives = sorted(set(sum((glob.glob(os.path.join(ARCHIVE_DIR, g)) for g in positive_globs), [])))
    negatives = sorted(set(sum((glob.glob(os.path.join(ARCHIVE_DIR, g)) for g in negative_globs), [])))
    return positives, negatives


def generate_synthetic_negatives(rng, n_per_kind=12):
    """Non-heartbeat audio a user might plausibly upload by mistake."""
    sr = SR
    t = np.arange(0, DURATION, 1.0 / sr)
    clips = []

    for _ in range(n_per_kind):
        freq = rng.uniform(100, 3000)
        clips.append(rng.uniform(0.3, 0.9) * np.sin(2 * np.pi * freq * t))

    for _ in range(n_per_kind):
        clips.append(rng.normal(0, rng.uniform(0.1, 0.5), size=len(t)))

    for _ in range(n_per_kind):
        f1, f2, f3 = rng.uniform(150, 350), rng.uniform(500, 1200), rng.uniform(1500, 3500)
        mod_rate = rng.uniform(1.5, 5.0)
        speech = (
            np.sin(2 * np.pi * f1 * t)
            + 0.6 * np.sin(2 * np.pi * f2 * t)
            + 0.4 * np.sin(2 * np.pi * f3 * t)
        ) * (0.5 + 0.5 * np.sin(2 * np.pi * mod_rate * t))
        clips.append(rng.uniform(0.3, 0.8) * speech / np.max(np.abs(speech)))

    for _ in range(n_per_kind):
        clips.append(rng.normal(0, rng.uniform(0.0005, 0.005), size=len(t)))

    for _ in range(n_per_kind):
        # rhythmic clicks well outside a plausible heart rate (e.g. a metronome/drum loop)
        click_bpm = rng.choice([260, 280, 320, 340, 20, 15])
        period = 60.0 / click_bpm
        clip = np.zeros(len(t))
        click_time = 0.0
        click_freq = rng.uniform(800, 4000)
        while click_time < DURATION:
            idx = int(click_time * sr)
            click_len = int(0.01 * sr)
            end = min(idx + click_len, len(clip))
            clip[idx:end] += np.sin(2 * np.pi * click_freq * t[idx:end]) * np.exp(-np.linspace(0, 8, end - idx))
            click_time += period
        clips.append(clip * rng.uniform(0.4, 0.9))

    for _ in range(n_per_kind):
        # chord/music-like: several harmonically related tones with slow chord changes
        base = rng.uniform(150, 400)
        chord = sum(np.sin(2 * np.pi * base * m * t) / m for m in (1, 2, 3, 4))
        clips.append(rng.uniform(0.3, 0.8) * chord / np.max(np.abs(chord)))

    return [c.astype(np.float32) for c in clips]


def _drum_hit(rng, start_hz, end_hz, decay, length_s, noise_mix=0.0):
    """One synthesized percussion hit: a pitch-swept sine (kick/tom/808) with an
    optional noise burst (snare/clap/hi-hat)."""
    sr = SR
    tt = np.arange(int(length_s * sr)) / sr
    freq = end_hz + (start_hz - end_hz) * np.exp(-tt * rng.uniform(20, 40))
    tone = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    noise = rng.normal(0, 1, len(tt))
    return ((1 - noise_mix) * tone + noise_mix * noise) * np.exp(-tt * decay)


def _sequence(rng, bpm, pattern, humanize):
    """Places `pattern` hits on a grid at `bpm`, repeating for DURATION seconds.
    With `humanize`, adds slight timing/velocity variation like a played beat."""
    sr = SR
    clip = np.zeros(int(DURATION * sr))
    step = 60.0 / bpm
    beat_time, k = rng.uniform(0, step), 0
    while beat_time < DURATION:
        for offset, hit in pattern[k % len(pattern)]:
            jitter = rng.normal(0, 0.01 * step) if humanize else 0.0
            velocity = rng.uniform(0.85, 1.0) if humanize else 1.0
            idx = int(max(0.0, beat_time + offset * step + jitter) * sr)
            end = min(idx + len(hit), len(clip))
            if idx < end:
                clip[idx:end] += velocity * hit[:end - idx]
        beat_time += step
        k += 1
    return clip


def generate_synthetic_beat_negatives(rng, n_per_kind=15):
    """
    Beat sound effects at heart-like tempos (40-200 bpm): kick loops, 808s,
    drum patterns, drum+bass and chord-backed music, and synthesized "lub-dub" heartbeat sound
    effects. These share a real heartbeat's low-frequency, rhythmic profile, so
    without them the model accepted any beat as a heartbeat.
    """
    sr = SR
    t = np.arange(int(DURATION * sr)) / sr
    clips = []

    def finish(clip):
        # Some beats are mixed/recorded with a little background noise, so the
        # model can't rely on digital silence alone.
        if rng.random() < 0.5:
            clip = clip + rng.normal(0, rng.uniform(0.002, 0.03) * np.max(np.abs(clip)), len(clip))
        return clip

    for _ in range(n_per_kind):
        # kick drum loop
        kick = _drum_hit(rng, rng.uniform(90, 220), rng.uniform(40, 65), rng.uniform(5, 18), 0.4)
        clips.append(finish(_sequence(rng, rng.uniform(40, 200), [[(0, kick)]], rng.random() < 0.5)))

    for _ in range(n_per_kind):
        # long, tonal 808 bass kick
        kick = _drum_hit(rng, rng.uniform(60, 120), rng.uniform(35, 60), rng.uniform(2, 5), 0.9)
        clips.append(finish(_sequence(rng, rng.uniform(40, 140), [[(0, kick)]], rng.random() < 0.5)))

    for _ in range(n_per_kind):
        # kick + snare backbeat, optional hi-hats
        kick = _drum_hit(rng, rng.uniform(90, 200), rng.uniform(40, 60), rng.uniform(6, 15), 0.4)
        snare = 0.7 * _drum_hit(rng, rng.uniform(180, 260), rng.uniform(150, 200), rng.uniform(15, 30), 0.25, noise_mix=0.6)
        hat = 0.25 * _drum_hit(rng, 8000, 6000, 60, 0.05, noise_mix=0.9)
        with_hats = rng.random() < 0.5
        pattern = [
            [(0, kick)] + ([(0.5, hat)] if with_hats else []),
            [(0, snare)] + ([(0.5, hat)] if with_hats else []),
        ]
        clips.append(finish(_sequence(rng, rng.uniform(60, 200), pattern, rng.random() < 0.5)))

    for _ in range(n_per_kind):
        # tom / low percussion double hit (sounds a lot like "lub-dub")
        tom = _drum_hit(rng, rng.uniform(100, 180), rng.uniform(70, 110), rng.uniform(8, 20), 0.3)
        gap = rng.uniform(0.25, 0.45)
        clips.append(finish(_sequence(rng, rng.uniform(40, 120), [[(0, tom), (gap, 0.7 * tom)]], rng.random() < 0.5)))

    for _ in range(n_per_kind):
        # synthesized "lub-dub" heartbeat sound effect
        lub = _drum_hit(rng, rng.uniform(50, 90), rng.uniform(30, 50), rng.uniform(15, 35), 0.15)
        dub = _drum_hit(rng, rng.uniform(60, 110), rng.uniform(40, 60), rng.uniform(20, 40), 0.12)
        gap = rng.uniform(0.25, 0.4)
        clips.append(finish(_sequence(rng, rng.uniform(45, 140), [[(0, lub), (gap, rng.uniform(0.5, 0.9) * dub)]], rng.random() < 0.5)))

    for _ in range(n_per_kind):
        # drum + bass line music (beat with pitched notes on top)
        bpm = rng.uniform(70, 180)
        kick = _drum_hit(rng, rng.uniform(90, 200), rng.uniform(40, 60), rng.uniform(6, 15), 0.4)
        snare = 0.6 * _drum_hit(rng, 200, 170, 20, 0.25, noise_mix=0.6)
        drums = _sequence(rng, bpm, [[(0, kick)], [(0, snare)]], rng.random() < 0.5)
        root = rng.uniform(40, 110)
        notes = root * 2 ** (rng.choice([0, 3, 5, 7, 10, 12], size=int(DURATION * bpm / 60) + 1) / 12)
        note_freq = notes[(t * bpm / 60).astype(int)]
        bass = np.sin(2 * np.pi * np.cumsum(note_freq) / sr) + 0.3 * np.sin(4 * np.pi * np.cumsum(note_freq) / sr)
        mix = drums / np.max(np.abs(drums)) + rng.uniform(0.3, 0.8) * bass
        clips.append(finish(mix))

    for _ in range(n_per_kind):
        # mellow track: soft beat under sustained chords/pads (mostly harmonic)
        bpm = rng.uniform(60, 130)
        kick = _drum_hit(rng, rng.uniform(80, 160), rng.uniform(40, 60), rng.uniform(6, 15), 0.4)
        drums = _sequence(rng, bpm, [[(0, kick)]], rng.random() < 0.5)
        chord_len = 4 * 60.0 / bpm
        roots = rng.uniform(80, 250) * 2 ** (rng.choice([0, 2, 5, 7, 9], size=int(DURATION / chord_len) + 1) / 12)
        root = roots[(t / chord_len).astype(int)]
        pad = sum(
            np.sin(2 * np.pi * np.cumsum(root * ratio) / sr) / (i + 1)
            for i, ratio in enumerate((1, 1.26, 1.5, 2))
        )
        mix = rng.uniform(0.3, 0.7) * drums / np.max(np.abs(drums)) + pad / np.max(np.abs(pad))
        clips.append(finish(mix))

    return [(c / np.max(np.abs(c))).astype(np.float32) for c in clips]


def fixed_length(signal, sr, duration=DURATION):
    max_samples = int(sr * duration)
    if len(signal) < max_samples:
        signal = np.pad(signal, (0, max_samples - len(signal)))
    else:
        signal = signal[:max_samples]
    peak = np.max(np.abs(signal))
    return signal / peak if peak > 0 else signal


def build_dataset():
    positive_paths, negative_paths = collect_real_recording_paths()
    print(f"Real recordings: {len(positive_paths)} positive (heartbeat), {len(negative_paths)} negative (artifact)")

    rng = np.random.default_rng(RNG_SEED)
    synthetic_negatives = generate_synthetic_negatives(rng) + generate_synthetic_beat_negatives(rng)
    print(f"Synthetic negatives (non-heartbeat audio): {len(synthetic_negatives)}")

    X, y, groups = [], [], []

    for path in positive_paths:
        signal, sr = load_audio_file(path, target_sr=SR, target_duration=DURATION)
        X.append(extract_features(signal, sr))
        y.append(1)
        groups.append(path)

    for path in negative_paths:
        signal, sr = load_audio_file(path, target_sr=SR, target_duration=DURATION)
        X.append(extract_features(signal, sr))
        y.append(0)
        groups.append(path)

    for i, clip in enumerate(synthetic_negatives):
        clip = fixed_length(clip, SR)
        X.append(extract_features(clip, SR))
        y.append(0)
        groups.append(f"synthetic_{i}")

    return np.array(X), np.array(y), groups


def main():
    print("Extracting features from archive/ (this reads every labeled WAV once)...")
    X, y, groups = build_dataset()
    print(f"Dataset: {len(y)} samples, {y.sum()} heartbeat / {len(y) - y.sum()} non-heartbeat")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=RNG_SEED,
        )),
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="balanced_accuracy")
    print(f"\n5-fold CV balanced accuracy on training split: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nHeld-out test set report:")
    print(classification_report(y_test, y_pred, target_names=["not_heartbeat", "heartbeat"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    # Refit on all data before shipping the model, so the deployed classifier
    # benefits from every labeled example, not just the training split.
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
