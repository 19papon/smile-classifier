"""
Loads the trained Keras model and runs smile prediction.

IMPORTANT — preprocessing contract (must match the training notebook, Phase 8):
The model has augmentation + `preprocess_input` (Rescaling to [-1, 1]) baked IN.
So here we feed RAW 0-255 RGB pixels resized to 224x224 and do NOT normalize.
The single sigmoid output is P(smiling).
"""
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 224
# Decision threshold on P(smiling). Tuned below the naive 0.5 because gentle,
# closed-mouth smiles cluster just under 0.5 (CelebA's "Smiling" label leans
# toward broad, teeth-showing smiles), so 0.5 wrongly calls them "not smiling".
# 0.4 recovers those subtle smiles while confident neutral faces (P ~< 0.3) stay
# "not smiling". Raise it toward 0.5 for stricter smiles, lower it for looser.
THRESHOLD = 0.4   # P(smiling) >= THRESHOLD -> "smiling"

# backend/services/model_service.py  ->  backend/model/smile_classifier.keras
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "smile_classifier.keras"

_model = None


def load():
    """Load the model into memory (idempotent). Raises FileNotFoundError if missing."""
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    # Import TensorFlow lazily so the server boots fast and import problems surface clearly.
    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def is_ready() -> bool:
    return _model is not None


def predict_smiling(face_rgb: np.ndarray) -> float:
    """
    face_rgb: H x W x 3 uint8 RGB crop.
    Returns P(smiling) in [0, 1].
    """
    if _model is None:
        raise RuntimeError("Model not loaded — call load() first.")
    img = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    batch = img.astype("float32")[None, ...]          # (1, 224, 224, 3), still 0-255
    prob = float(_model.predict(batch, verbose=0).ravel()[0])
    return prob
