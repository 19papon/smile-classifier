# Single-service image for Hugging Face Spaces (Docker SDK).
# Stage 1 builds the React UI; stage 2 runs FastAPI, which serves BOTH the UI and
# the /predict API on one port. Hugging Face routes external traffic to port 7860.

# ---- stage 1: build the frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# empty base -> the UI calls /predict on its own origin (this same server)
ENV VITE_API_BASE=""
RUN npm run build

# ---- stage 2: python backend + compiled UI ----
FROM python:3.11-slim AS app

# native libraries OpenCV (headless) and TensorFlow load at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# run as a non-root user (Hugging Face Spaces convention)
RUN useradd -m -u 1000 user

WORKDIR /app
# install python deps first so this layer caches across code changes
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# backend code + trained model (backend/model/*.keras, *.onnx)
COPY backend/ ./
# drop the compiled UI where main.py looks for it (backend/static)
COPY --from=frontend /app/frontend/dist ./static

RUN chown -R user:user /app
USER user
ENV HOME=/home/user

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
