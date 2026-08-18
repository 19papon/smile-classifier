# Frontend — AI Smile Classifier (React + Vite)

Upload a photo, see faces boxed and labelled **Smiling / Not smiling** with a
confidence score. Talks to the FastAPI backend's `/predict` endpoint.

## Prerequisites
- **Node.js 18+** — install from https://nodejs.org (LTS). Verify: `node -v`.
- The **backend running** at http://localhost:8000 (see `../backend/README.md`).

## Run
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (default http://localhost:5173).

## Configure the API URL (optional)
The app calls `http://localhost:8000` by default. To point elsewhere:
```bash
cp .env.example .env.local      # then edit VITE_API_BASE
```

## Build for production
```bash
npm run build      # outputs to dist/
npm run preview    # preview the production build locally
```

## Files
```
frontend/
├── index.html
├── vite.config.js
├── package.json
└── src/
    ├── main.jsx      # React entry
    ├── App.jsx       # upload + call /predict + render faces/result
    ├── App.css
    └── index.css
```
