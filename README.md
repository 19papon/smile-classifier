# 😊 Smile Classifier
*by Papon*

**Live demo → https://19papon.github.io/smile-classifier/**

Drop in a photo and every face is detected and scored as **smiling** or
**not smiling**, using a MobileNetV2 model transfer-learned on the CelebA
dataset. The whole thing runs **in your browser** — nothing is uploaded, and
there's no server to keep running.

```
Photo ─> YuNet face detection ─> crop each face (+margin) ─> 224×224
                                                              └─> MobileNetV2 ─> P(smiling)
                                                            <─ label + confidence + face boxes
```

Face detection (YuNet) runs via **onnxruntime-web** and the smile model
(MobileNetV2) runs via **tfjs-tflite** — both compiled to WebAssembly, so the
page works on any static host with no backend.

## Project layout
```
smile-classifier/
├── training/     # Colab notebook — download → train → export smile_classifier.keras
├── frontend/     # React + Vite app; runs all inference client-side (this is what's deployed)
├── backend/      # Optional FastAPI reference server (same models, for running locally)
└── dataset/      # empty; CelebA is downloaded inside Colab, not here (gitignored)
```

## Use it
Just open the live demo — pick or drag in a photo, hit **Analyze**. Works with
one face or many; everything happens on your device.

## Run the frontend locally
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173.

## Train the model (optional — a trained model is already bundled)
1. Open `training/smile_classifier_training.ipynb` in Google Colab.
2. `Runtime → Change runtime type → GPU`, then **Run all**.
3. It downloads CelebA, trains + fine-tunes MobileNetV2, evaluates, and exports
   **`smile_classifier.keras`**. Convert it to `smile.tflite` and drop it in
   `frontend/public/models/` to ship a retrained model to the browser app.

## Optional: the reference backend
`backend/` is the original FastAPI server — same YuNet detection and same
MobileNetV2 model, exposed over `POST /predict`. It isn't needed for the live
site (which is fully client-side) but is handy for local experiments:
```bash
cd backend
py -3.12 -m venv .venv && source .venv/Scripts/activate   # Git Bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000                     # http://localhost:8000/health
```

## Tech stack
- **Training:** TensorFlow / Keras, MobileNetV2 transfer learning, CelebA (kagglehub), Colab GPU
- **Browser app:** React 18, Vite, onnxruntime-web (YuNet face detection), tfjs-tflite (MobileNetV2)
- **Reference backend:** FastAPI, Uvicorn, OpenCV (YuNet), Pillow

## Key design note
Image preprocessing (scaling to [-1, 1]) is **baked into the model** (training
notebook, Phase 8). Both the browser app and the backend feed raw 0–255 RGB
pixels resized to 224×224 and do **not** normalize again. Keep these in sync if
you retrain.

## How it's deployed
`npm run build` in `frontend/` produces `dist/`, which is published to the
`gh-pages` branch and served by GitHub Pages at the link above. Because all
inference runs client-side with WebAssembly, the site is free to host and needs
no server.

## Credits
Dataset: **CelebA** (Liu et al., ICCV 2015), via the Kaggle mirror
`jessicali9530/celeba-dataset`.
