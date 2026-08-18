"""
FastAPI backend for the AI Smile Classifier.

Per request:  image bytes -> RGB -> OpenCV face detection -> crop each face
-> MobileNetV2 model -> P(smiling) -> JSON (with face boxes for the UI).

Run from the backend/ folder:
    uvicorn main:app --reload --port 8000
Docs (interactive):  http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services import face_service, model_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model once at startup so the first request isn't slow.
    # Missing file doesn't crash the server — /health reports model_loaded: false.
    try:
        model_service.load()
        print("Model loaded.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}\n  -> put smile_classifier.keras in backend/model/ and restart.")
    yield


app = FastAPI(title="AI Smile Classifier API", version="1.0.0", lifespan=lifespan)

# Dev CORS: let the React dev server (and any origin) call the API.
# Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_service.is_ready()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    if not model_service.is_ready():
        raise HTTPException(503, "Model not loaded. Put smile_classifier.keras in backend/model/ and restart the server.")

    raw = await file.read()
    try:
        pil = Image.open(BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")   # honor phone-camera rotation
    except Exception:
        raise HTTPException(400, "Could not read that image.")

    rgb = np.array(pil)              # H x W x 3, uint8, 0-255
    H, W = rgb.shape[:2]

    boxes = face_service.detect_faces(rgb)   # largest first
    face_detected = len(boxes) > 0
    if not boxes:
        boxes = [(0, 0, W, H)]               # fallback: classify the whole image

    faces = []
    for (x, y, w, h) in boxes:
        crop = face_service.crop(rgb, (x, y, w, h))
        prob = model_service.predict_smiling(crop)          # P(smiling)
        smiling = prob >= model_service.THRESHOLD
        faces.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "label": "smiling" if smiling else "not_smiling",
            "smiling": bool(smiling),
            "probability": round(float(prob), 4),                       # P(smiling)
            "confidence": round(float(prob if smiling else 1 - prob), 4),
        })

    return {
        "face_detected": face_detected,
        "face_count": len(faces),
        "image": {"width": int(W), "height": int(H)},
        "primary": faces[0],
        "faces": faces,
    }


# --- Serve the built React UI (single-service deploy) --------------------------
# In the Docker / Hugging Face image the compiled frontend is copied to
# backend/static, so this same server hosts the UI and the API. The mount is at
# "/" and is registered AFTER the API routes above, so /health and /predict still
# take precedence. Locally this folder doesn't exist (Vite serves the UI on
# :5173), so the block is simply skipped and dev is unaffected.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="frontend")
