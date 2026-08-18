---
title: Smile Classifier
emoji: 😊
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# 😊 Smile Classifier
*by Papon*

Detects faces in a photo and classifies each as **smiling** or **not smiling**,
using a MobileNetV2 model transfer-learned on the CelebA dataset.

```
Photo ──> [React UI] ──POST /predict──> [FastAPI]
                                          ├─ Face detection (OpenCV YuNet) + crop
                                          └─ MobileNetV2 (Keras) ─> P(smiling)
                                        <── JSON: label + confidence + face boxes
```

## Project layout
```
smile-classifier/
├── training/     # Colab notebook — download → train → export smile_classifier.keras
├── backend/      # FastAPI: face detection + /predict  (runs on your laptop)
├── frontend/     # React + Vite upload UI              (runs on your laptop)
└── dataset/      # empty; CelebA is downloaded inside Colab, not here (gitignored)
```

## Run order (first → last)

### 1. Train the model (Google Colab — needs GPU)
1. Open `training/smile_classifier_training.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU`, then **Run all**.
3. It downloads CelebA, trains + fine-tunes MobileNetV2, evaluates, and downloads
   **`smile_classifier.keras`** at the end.

### 2. Install the model
Put the downloaded file at:
```
backend/model/smile_classifier.keras
```

### 3. Start the backend (see `backend/README.md`)
```bash
cd backend
py -3.12 -m venv .venv && source .venv/Scripts/activate   # Git Bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Check: http://localhost:8000/health → `{"status":"ok","model_loaded":true}`

### 4. Start the frontend (see `frontend/README.md`)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 and upload a photo. 🎉

## Tech stack
- **Training:** TensorFlow / Keras, MobileNetV2 transfer learning, CelebA (kagglehub), Colab GPU
- **Backend:** FastAPI, Uvicorn, OpenCV (YuNet deep-learning face detection), Pillow
- **Frontend:** React 18, Vite

## Key design note
Image preprocessing (scaling to [-1, 1]) is **baked into the Keras model**
(training notebook, Phase 8). So the backend feeds raw 0–255 RGB pixels resized
to 224×224 and does **not** normalize again. Keep these in sync if you retrain.

## Deploy as a live website (Hugging Face Spaces)

The included `Dockerfile` builds the React UI and runs the FastAPI server that
serves **both the UI and the API on one port** — so the whole app deploys as a
single free **Hugging Face Space** (no CORS, one link to share).

1. Create a free account at <https://huggingface.co>, then a token at
   **Settings → Access Tokens** (role **write**).
2. **New Space** → SDK **Docker** → **Blank** → name it `smile-classifier`.
3. Large files on Hugging Face use Git LFS, so track the model before pushing:
   ```bash
   git lfs track "*.keras"
   git add .gitattributes && git commit -m "Track model with LFS"
   ```
4. Push this repo to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/smile-classifier
   git push space main        # username = your HF name, password = the write token
   ```
5. The Space builds the image and goes live at
   `https://<hf-username>-smile-classifier.hf.space` — share that link with anyone.

Prefer a split host instead? Build the frontend (`npm run build`, output `dist/`)
on Vercel/Netlify and set `VITE_API_BASE` to a separately deployed backend URL.

## Push to GitHub
```bash
cd smile-classifier
git init
git add .
git commit -m "AI Smile Classifier: training notebook + FastAPI backend + React frontend"
git branch -M main
git remote add origin https://github.com/<your-username>/smile-classifier.git
git push -u origin main
```

## Credits
Dataset: **CelebA** (Liu et al., ICCV 2015), via the Kaggle mirror
`jessicali9530/celeba-dataset`.
