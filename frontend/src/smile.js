// In-browser inference engine — mirrors the original server pipeline exactly:
//   image -> YuNet face detection -> crop each face (+margin) -> 224x224 -> MobileNetV2
//   -> P(smiling) -> {smiling, confidence}. Runs 100% client-side; no network.
//
// Smile model: MobileNetV2 exported to TFLite (identical to the trained model,
// verified to <1e-7), run via WebAssembly (tfjs-tflite).
// Face detection: YuNet (face_detection_yunet_2023mar.onnx) run via onnxruntime-web
// with the same score/NMS thresholds, downscaling, padding and decode as the
// server, so the detected boxes match the server implementation to the pixel.
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-cpu'
import * as tflite from '@tensorflow/tfjs-tflite'
import * as ort from 'onnxruntime-web'

// Base path assets are served from ('/' in dev, '/smile-classifier/' on GitHub Pages).
const BASE = import.meta.env.BASE_URL

// --- smile classifier ---
const THRESHOLD = 0.4   // P(smiling) >= THRESHOLD -> "smiling"  (matches the server)
const IMG_SIZE = 224
const MARGIN = 0.2      // context added around each detected face before cropping

// --- YuNet face detector (values copied from the server's face_service.py) ---
const SCORE_THRESHOLD = 0.7   // min detection confidence to keep a face
const NMS_THRESHOLD = 0.3     // non-max suppression for overlapping boxes
const TOP_K = 5000            // keep at most this many candidates
const MAX_SIDE = 1024         // downscale big photos to this longest side for detection
const STRIDES = [8, 16, 32]   // YuNet feature-map strides
const DIV = 32                // detector input is padded to a multiple of this

let _smile = null
let _yunet = null
let _initPromise = null

// Reusable canvas for cropping/resizing faces to the smile model's input size.
const _canvas = document.createElement('canvas')
_canvas.width = IMG_SIZE
_canvas.height = IMG_SIZE
const _ctx = _canvas.getContext('2d', { willReadFrequently: true })
_ctx.imageSmoothingEnabled = true
_ctx.imageSmoothingQuality = 'high'

// Reusable canvas for feeding the (downscaled + padded) image to the detector.
const _detCanvas = document.createElement('canvas')
const _detCtx = _detCanvas.getContext('2d', { willReadFrequently: true })
_detCtx.imageSmoothingEnabled = true
_detCtx.imageSmoothingQuality = 'high'

export function initEngine() {
  if (!_initPromise) {
    _initPromise = (async () => {
      await tf.setBackend('cpu')
      await tf.ready()

      tflite.setWasmPath(`${BASE}tflite/`)
      _smile = await tflite.loadTFLiteModel(`${BASE}models/smile.tflite`)

      // onnxruntime-web: single-threaded WASM so no cross-origin isolation
      // (SharedArrayBuffer) is required — works on plain static hosts. SIMD stays on.
      ort.env.wasm.wasmPaths = `${BASE}ort/`
      ort.env.wasm.numThreads = 1
      ort.env.logLevel = 'error'   // the model's declared (fixed-size) output shapes
      _yunet = await ort.InferenceSession.create(
        `${BASE}models/face_detection_yunet_2023mar.onnx`,
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          // differ from the actual dynamic ones; silence the benign per-run warning.
          logSeverityLevel: 3,
        },
      )
    })().catch((e) => {
      _initPromise = null   // allow a retry on failure
      throw e
    })
  }
  return _initPromise
}

// Run the smile model on one (x,y,w,h) region of the image, returning P(smiling).
function scoreRegion(imgEl, x, y, w, h) {
  _ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE)
  _ctx.drawImage(imgEl, x, y, w, h, 0, 0, IMG_SIZE, IMG_SIZE)
  const { data } = _ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE) // RGBA, 0-255

  // RGBA -> RGB float32 in 0-255 (the model bakes its own normalisation)
  const rgb = new Float32Array(IMG_SIZE * IMG_SIZE * 3)
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i]
    rgb[j + 1] = data[i + 1]
    rgb[j + 2] = data[i + 2]
  }

  const input = tf.tensor4d(rgb, [1, IMG_SIZE, IMG_SIZE, 3])
  const output = _smile.predict(input)
  const prob = output.dataSync()[0]
  input.dispose()
  output.dispose()
  return prob
}

// Intersection-over-union of two {x,y,w,h} boxes.
function iou(a, b) {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

// Greedy non-max suppression, matching cv2.dnn.NMSBoxes: sort by score, keep a box,
// drop every later box that overlaps it by more than `thresh`, stop after `topK`.
function nms(boxes, thresh, topK) {
  const order = boxes.map((_, i) => i).sort((a, b) => boxes[b].score - boxes[a].score)
  const suppressed = new Array(boxes.length).fill(false)
  const kept = []
  for (const i of order) {
    if (suppressed[i]) continue
    kept.push(boxes[i])
    if (kept.length >= topK) break
    for (const j of order) {
      if (!suppressed[j] && j !== i && iou(boxes[i], boxes[j]) > thresh) suppressed[j] = true
    }
  }
  return kept
}

// Detect faces with YuNet. Returns [{x,y,w,h,score}] in original-image pixels.
// Replicates the server: downscale to MAX_SIDE, feed BGR, pad to a multiple of 32,
// decode the three feature maps, NMS, then map boxes back to full resolution.
async function detectFaces(imgEl, W, H) {
  // 1) downscale so the longest side is <= MAX_SIDE (the server uses INTER_AREA; the
  //    browser's resampler differs slightly, but YuNet is robust to it)
  const scale = Math.max(W, H) > MAX_SIDE ? MAX_SIDE / Math.max(W, H) : 1
  const dw = Math.round(W * scale)
  const dh = Math.round(H * scale)

  // 2) pad bottom/right to a multiple of 32, exactly as FaceDetectorYN does
  //    internally. Padding the far edges leaves every face coordinate unchanged.
  const padW = Math.ceil(dw / DIV) * DIV
  const padH = Math.ceil(dh / DIV) * DIV
  _detCanvas.width = padW
  _detCanvas.height = padH
  _detCtx.clearRect(0, 0, padW, padH)          // padding region stays black
  _detCtx.drawImage(imgEl, 0, 0, W, H, 0, 0, dw, dh)
  const { data } = _detCtx.getImageData(0, 0, padW, padH) // RGBA

  // 3) RGBA -> BGR, NCHW, float32 0-255 (YuNet's blob: no mean/scale, no swapRB)
  const area = padW * padH
  const chw = new Float32Array(3 * area)
  for (let p = 0, i = 0; p < area; p++, i += 4) {
    chw[p] = data[i + 2]            // B plane
    chw[area + p] = data[i + 1]     // G plane
    chw[2 * area + p] = data[i]     // R plane
  }
  const input = new ort.Tensor('float32', chw, [1, 3, padH, padW])
  const out = await _yunet.run({ [_yunet.inputNames[0]]: input })

  // 4) decode each stride's feature map exactly like OpenCV's YuNet postProcess
  const boxes = []   // in detection (downscaled) pixels
  for (const s of STRIDES) {
    const cls = out[`cls_${s}`].data
    const obj = out[`obj_${s}`].data
    const bbox = out[`bbox_${s}`].data
    const cols = padW / s
    const rows = padH / s
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c
        const cs = Math.min(Math.max(cls[idx], 0), 1)
        const os = Math.min(Math.max(obj[idx], 0), 1)
        const score = Math.sqrt(cs * os)
        if (score < SCORE_THRESHOLD) continue
        const b = idx * 4
        const cx = (c + bbox[b]) * s
        const cy = (r + bbox[b + 1]) * s
        const bw = Math.exp(bbox[b + 2]) * s
        const bh = Math.exp(bbox[b + 3]) * s
        boxes.push({ x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh, score })
      }
    }
  }

  // 5) NMS, then map boxes back to full-resolution coordinates and clamp
  const result = []
  for (const b of nms(boxes, NMS_THRESHOLD, TOP_K)) {
    const x = Math.max(0, Math.round(b.x / scale))
    const y = Math.max(0, Math.round(b.y / scale))
    const w = Math.min(Math.round(b.w / scale), W - x)
    const h = Math.min(Math.round(b.h / scale), H - y)
    if (w > 0 && h > 0) result.push({ x, y, w, h, score: b.score })
  }
  return result
}

// Analyse a loaded <img> element. Returns the same shape the server API returned.
export async function analyzePhoto(imgEl) {
  if (!_smile || !_yunet) throw new Error('Engine not ready')

  const W = imgEl.naturalWidth || imgEl.width
  const H = imgEl.naturalHeight || imgEl.height

  let boxes = await detectFaces(imgEl, W, H)

  const face_detected = boxes.length > 0
  if (!face_detected) boxes = [{ x: 0, y: 0, w: W, h: H }]     // fallback: whole image
  boxes.sort((a, b) => b.w * b.h - a.w * a.h)                  // largest face first

  const round4 = (v) => Math.round(v * 1e4) / 1e4
  const faces = boxes.map(({ x, y, w, h }) => {
    // crop with margin, clamped to the image (mirrors the server's crop())
    const mx = Math.floor(w * MARGIN)
    const my = Math.floor(h * MARGIN)
    const x0 = Math.max(0, x - mx)
    const y0 = Math.max(0, y - my)
    const x1 = Math.min(W, x + w + mx)
    const y1 = Math.min(H, y + h + my)

    const prob = scoreRegion(imgEl, x0, y0, x1 - x0, y1 - y0)
    const smiling = prob >= THRESHOLD
    return {
      box: { x, y, w, h },
      label: smiling ? 'smiling' : 'not_smiling',
      smiling,
      probability: round4(prob),
      confidence: round4(smiling ? prob : 1 - prob),
    }
  })

  return {
    face_detected,
    face_count: faces.length,
    image: { width: W, height: H },
    primary: faces[0],
    faces,
  }
}
