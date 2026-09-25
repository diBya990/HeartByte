import os
import joblib

from heartbeat_features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "heartbeat_classifier.joblib")

_model = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Heartbeat classifier model not found at {MODEL_PATH}. "
                "Run `python train_heartbeat_classifier.py` to train and save it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_is_heartbeat(signal, sr, threshold=0.5):
    """
    Runs the trained classifier (see train_heartbeat_classifier.py) on a loaded
    audio clip and returns (is_heartbeat, diagnostics) where diagnostics carries
    the model's heartbeat-probability for display in the UI.
    """
    model = _load_model()
    features = extract_features(signal, sr).reshape(1, -1)
    probability = model.predict_proba(features)[0, 1]
    return bool(probability >= threshold), {"probability": float(probability)}
