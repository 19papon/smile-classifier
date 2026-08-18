# Backend — Smile Classifier API (FastAPI)

FastAPI service that takes an image, detects faces (OpenCV YuNet), and predicts
**smiling vs not smiling** for each face using the trained MobileNetV2 model.

## 1. Get the model
Run `training/smile_classifier_training.ipynb` in Colab, download
`smile_classifier.keras`, and place it at:

```
backend/model/smile_classifier.keras
```

## 2. Create a virtual environment + install deps

> **TensorFlow + Python version (Windows):** TensorFlow may not have wheels for
> Python 3.13 yet. If install fails on `tensorflow`, use Python **3.11 or 3.12**.
> Check versions with `py -0` (Windows Python launcher).

**Git Bash:**
```bash
cd backend
py -3.12 -m venv .venv          # or: python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

**PowerShell:**
```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Run the server
```bash
uvicorn main:app --reload --port 8000
```

- Health check: http://localhost:8000/health  → `{"status":"ok","model_loaded":true}`
- Interactive docs: http://localhost:8000/docs

## API

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`  (multipart form, field name `file`)
Quick test:
```bash
curl -X POST http://localhost:8000/predict -F "file=@some_photo.jpg"
```
Response:
```json
{
  "face_detected": true,
  "face_count": 1,
  "image": { "width": 800, "height": 600 },
  "primary": {
    "box": { "x": 220, "y": 120, "w": 180, "h": 180 },
    "label": "smiling",
    "smiling": true,
    "probability": 0.973,   // P(smiling)
    "confidence": 0.973     // confidence in the shown label
  },
  "faces": [ /* one entry per detected face, largest first */ ]
}
```
If no face is found, the whole image is classified and `face_detected` is `false`.

## Layout
```
backend/
├── main.py                    # FastAPI app: /health, /predict, CORS
├── services/
│   ├── face_service.py        # OpenCV YuNet face detection + cropping
│   └── model_service.py       # loads .keras, predicts (raw 0-255 in, sigmoid out)
├── model/                     # <- put smile_classifier.keras here
└── requirements.txt
```
