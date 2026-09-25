import numpy as np


EDUCATIONAL_DISCLAIMER = (
    "DISCLAIMER: This system is an educational signal processing demonstration tool. "
    "It is NOT a medical device and must not be used for medical evaluation or diagnosis."
)

def evaluate_heart_condition(bpm, hrv, signal, fs=22050):
    """
    Runs rule-based checks against acoustic/rhythm features extracted from the heart
    sound recording. Returns:
      status     - overall summary label
      findings   - human-readable narrative sentences (for the "Impression" section)
      checklist  - structured rows (category, value, assessment, flag) used to render
                   the tabular, physiologist-report-style breakdown. Only metrics that
                   can genuinely be derived from this audio signal are reported here —
                   no electrical-ECG intervals (PR/QRS/QT) are fabricated.
    """
    findings = []
    checklist = []
    status = "Normal Indications"

    if bpm < 50:
        findings.append("Low Heart Rate detected (Possible Bradycardia pattern: < 50 BPM).")
        checklist.append({"category": "Heart Rate", "value": f"{bpm:.1f} BPM",
                           "assessment": "Bradycardia Pattern", "flag": "warning"})
        status = "Abnormal Rate"
    elif bpm > 100:
        findings.append("Elevated Heart Rate detected (Possible Tachycardia pattern: > 100 BPM).")
        checklist.append({"category": "Heart Rate", "value": f"{bpm:.1f} BPM",
                           "assessment": "Tachycardia Pattern", "flag": "warning"})
        status = "Abnormal Rate"
    else:
        findings.append(f"Heart Rate is within nominal resting range ({bpm:.1f} BPM).")
        checklist.append({"category": "Heart Rate", "value": f"{bpm:.1f} BPM",
                           "assessment": "Within Normal Range", "flag": "ok"})

    if hrv > 120:
        findings.append(f"High Inter-Beat Variability observed (HRV: {hrv:.1f} ms — Possible Arrhythmia).")
        checklist.append({"category": "Rhythm Consistency (HRV)", "value": f"{hrv:.1f} ms",
                           "assessment": "Irregular Pattern", "flag": "warning"})
        status = "Irregular Rhythm"
    else:
        findings.append(f"Rhythm consistency is within normal bounds (HRV: {hrv:.1f} ms).")
        checklist.append({"category": "Rhythm Consistency (HRV)", "value": f"{hrv:.1f} ms",
                           "assessment": "Regular", "flag": "ok"})

    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    high_freq_energy = np.sum(fft_vals[freqs > 200]) / (np.sum(fft_vals) + 1e-6)

    if high_freq_energy > 0.40:
        findings.append("Elevated high-frequency acoustic energy detected (Possible Murmur acoustic pattern).")
        checklist.append({"category": "Acoustic Quality", "value": f"{high_freq_energy*100:.1f}% HF energy",
                           "assessment": "Possible Murmur Pattern", "flag": "warning"})
        if status == "Normal Indications":
            status = "Acoustic Anomaly"
    else:
        checklist.append({"category": "Acoustic Quality", "value": f"{high_freq_energy*100:.1f}% HF energy",
                           "assessment": "No Murmur Pattern", "flag": "ok"})

    return status, findings, checklist