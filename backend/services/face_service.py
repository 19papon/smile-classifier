"""
Face detection + cropping.

Primary detector: OpenCV **YuNet** (a small deep-learning ONNX model). It is far
more robust than Haar cascades on cluttered backgrounds — Haar was flagging trees,
walls and fabric as "faces", and the smile classifier then labelled that noise
"not smiling". YuNet returns a confidence score per detection, so we simply drop
anything below `_SCORE_THRESHOLD`, which removes the background false positives.

If the YuNet model file is missing we fall back to the bundled Haar cascade so the
service still runs (just less accurately).

    model file: backend/model/face_detection_yunet_2023mar.onnx
    source:     https://github.com/opencv/opencv_zoo (face_detection_yunet)
"""
from pathlib import Path

import cv2
import numpy as np

_YUNET_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "face_detection_yunet_2023mar.onnx"
)
_SCORE_THRESHOLD = 0.7   # min detection confidence to keep a face (0-1); higher = stricter
_NMS_THRESHOLD = 0.3     # non-max suppression for overlapping boxes
_TOP_K = 5000            # keep at most this many candidates before NMS
_MAX_SIDE = 1024         # downscale large photos to this longest side for detection (speed + stability)

# Build the YuNet detector once at import. Input size is (re)set per image below.
_yunet = None
if _YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
    _yunet = cv2.FaceDetectorYN.create(
        str(_YUNET_PATH), "", (320, 320),
        _SCORE_THRESHOLD, _NMS_THRESHOLD, _TOP_K,
    )

# Haar fallback (ships with opencv, no download).
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_yunet(rgb: np.ndarray):
    """YuNet detection. Detect on a downscaled copy for big photos, then map boxes
    back to full-resolution coordinates."""
    h, w = rgb.shape[:2]
    scale = 1.0
    det = rgb
    if max(h, w) > _MAX_SIDE:
        scale = _MAX_SIDE / float(max(h, w))
        det = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    dh, dw = det.shape[:2]
    bgr = cv2.cvtColor(det, cv2.COLOR_RGB2BGR)   # YuNet expects BGR (OpenCV convention)
    _yunet.setInputSize((dw, dh))
    _, faces = _yunet.detect(bgr)

    boxes = []
    if faces is not None:
        for f in faces:
            # f = [x, y, w, h, 5 landmark xy..., score]; score already >= _SCORE_THRESHOLD
            x, y, bw, bh = (v / scale for v in f[:4])   # back to original resolution
            x, y = max(0, int(round(x))), max(0, int(round(y)))
            bw, bh = int(round(bw)), int(round(bh))
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw > 0 and bh > 0:
                boxes.append((x, y, bw, bh))
    return boxes


def _detect_haar(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
    )
    return [tuple(int(v) for v in f) for f in faces]


def detect_faces(rgb: np.ndarray):
    """
    rgb: H x W x 3 uint8 image (RGB).
    Returns a list of (x, y, w, h) integer boxes, largest area first.
    """
    boxes = _detect_yunet(rgb) if _yunet is not None else _detect_haar(rgb)
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)   # largest face first
    return boxes


def crop(rgb: np.ndarray, box, margin: float = 0.2) -> np.ndarray:
    """
    Crop a face from `rgb`, adding `margin` (fraction of box size) of context on
    each side. Faces classify better with a little forehead/chin around them.
    Coordinates are clamped to the image bounds.
    """
    x, y, w, h = box
    H, W = rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return rgb[y0:y1, x0:x1]
