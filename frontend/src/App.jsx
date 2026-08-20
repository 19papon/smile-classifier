import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'
import { initEngine, analyzePhoto } from './smile'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [imgBox, setImgBox] = useState({ w: 1, h: 1, natW: 1, natH: 1 })
  const [engineReady, setEngineReady] = useState(false)
  const [engineErr, setEngineErr] = useState(null)
  const imgRef = useRef(null)
  const inputRef = useRef(null)

  // Load the models once, in the browser, on first mount.
  useEffect(() => {
    let alive = true
    initEngine()
      .then(() => alive && setEngineReady(true))
      .catch((e) => alive && setEngineErr(e?.message || 'Could not load the model in this browser.'))
    return () => { alive = false }
  }, [])

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setError('Please choose an image file.')
      return
    }
    setError(null)
    setResult(null)
    setFile(f)
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(f)
    })
  }, [])

  const analyze = useCallback(async () => {
    if (!file || !imgRef.current) return
    if (!engineReady) {
      setError('The model is still loading — please try again in a moment.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const img = imgRef.current
      if (!img.complete || !img.naturalWidth) {
        await new Promise((res, rej) => { img.onload = res; img.onerror = rej })
      }
      // let the "Analyzing…" overlay paint before inference starts
      await new Promise((r) => setTimeout(r, 20))
      setResult(await analyzePhoto(img))
    } catch (e) {
      setError(e?.message || 'Something went wrong while analyzing the photo.')
    } finally {
      setLoading(false)
    }
  }, [file, engineReady])

  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old)
      return null
    })
  }

  // measure the on-screen image so face boxes can be scaled from natural pixels
  const measure = useCallback(() => {
    const el = imgRef.current
    if (!el) return
    setImgBox({
      w: el.clientWidth,
      h: el.clientHeight,
      natW: el.naturalWidth || 1,
      natH: el.naturalHeight || 1,
    })
  }, [])

  // keep the overlay aligned when the viewport (and thus the image) resizes
  useEffect(() => {
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure])

  const scaleX = imgBox.w / imgBox.natW
  const scaleY = imgBox.h / imgBox.natH
  const overlays = result?.face_detected ? result.faces : []

  return (
    <div className="page">
      <div className="app">
        <header className="hero">
          <span className="badge"><span className="dot" /> MobileNetV2 · Deep Learning</span>
          <h1>Smile Classifier</h1>
          <div className="byline">by Papon</div>
          <p>
            Drop in a photo — every face is detected and scored as{' '}
            <strong>smiling</strong> or <strong>not</strong>, in real time.
          </p>
        </header>

        <div
          className={`stage ${dragging ? 'dragging' : ''} ${previewUrl ? 'has-image' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
          onClick={() => !previewUrl && inputRef.current?.click()}
        >
          {previewUrl ? (
            <div className="preview-wrap">
              <img ref={imgRef} src={previewUrl} alt="preview" className="preview" onLoad={measure} />
              {overlays.map((f, i) => (
                <div
                  key={i}
                  className={`facebox ${f.smiling ? 'smiling' : 'not'}`}
                  style={{
                    left: f.box.x * scaleX,
                    top: f.box.y * scaleY,
                    width: f.box.w * scaleX,
                    height: f.box.h * scaleY,
                  }}
                >
                  <span className="tag">{f.smiling ? '😊' : '😐'} {Math.round(f.confidence * 100)}%</span>
                </div>
              ))}
              {loading && (
                <div className="stage-loading">
                  <span className="spinner" />
                  <span>Analyzing…</span>
                </div>
              )}
            </div>
          ) : (
            <div className="placeholder">
              <div className="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className="lead"><strong>Click to upload</strong> or drag &amp; drop</p>
              <p className="hint">JPG, PNG or WebP · works with one or many faces</p>
            </div>
          )}
          <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => handleFile(e.target.files[0])} />
        </div>

        <div className="actions">
          <button className="primary" disabled={!file || loading || !engineReady} onClick={analyze}>
            {loading
              ? <><span className="spinner sm" /> Analyzing…</>
              : (!engineReady && !engineErr)
                ? <><span className="spinner sm" /> Loading model…</>
                : <>✨ Analyze photo</>}
          </button>
          {file && <button className="ghost" onClick={reset} disabled={loading}>Clear</button>}
        </div>

        {(error || engineErr) && <div className="error">⚠️ {error || engineErr}</div>}

        {result && (
          <div className="result">
            <div className="result-note">
              {result.face_detected
                ? <>Detected <strong>{result.face_count}</strong> face{result.face_count > 1 ? 's' : ''}</>
                : <>No face detected — scored the whole image</>}
            </div>
            <ResultCard face={result.primary} />
            {result.faces.length > 1 && (
              <details>
                <summary>Show all {result.faces.length} faces</summary>
                <div className="grid">
                  {result.faces.map((f, i) => <ResultCard key={i} face={f} small />)}
                </div>
              </details>
            )}
          </div>
        )}

        <footer>
          <span className={`live ${engineReady ? '' : 'off'}`}>
            <span className="dot" /> {engineReady ? 'On-device' : engineErr ? 'Unavailable' : 'Loading…'}
          </span>
          <code>runs in your browser · nothing is uploaded</code>
        </footer>
      </div>
    </div>
  )
}

function ResultCard({ face, small }) {
  const pct = Math.round(face.confidence * 100)
  const smiling = face.smiling
  return (
    <div className={`card ${smiling ? 'smiling' : 'not'} ${small ? 'small' : ''}`}>
      <div className="emoji">{smiling ? '😊' : '😐'}</div>
      <div className="verdict">{smiling ? 'Smiling' : 'Not smiling'}</div>
      <div className="bar"><div className="fill" style={{ width: `${pct}%` }} /></div>
      <div className="meta">
        <span className="pct">{pct}%</span>
        <span className="muted">confidence · P(smile) {Number(face.probability).toFixed(3)}</span>
      </div>
    </div>
  )
}
