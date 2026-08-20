import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

// The whole app runs in the browser — face detection (YuNet via onnxruntime-web)
// and the smile model (MobileNetV2 via tfjs-tflite) both execute client-side with
// WebAssembly, so there is no server to talk to.
//
// The tfjs-tflite package's ESM build imports "./tflite_web_api_client", but that
// file only ships under the package's wasm/ folder, so bundlers can't resolve it.
// Point the import at the real file. It's a CommonJS module exposing { tfweb }.
const tfliteClient = fileURLToPath(
  new URL('./node_modules/@tensorflow/tfjs-tflite/wasm/tflite_web_api_client.js', import.meta.url)
).replace(/\\/g, '/')

export default defineConfig(({ mode }) => ({
  // GitHub Pages serves a project site under /<repo>/, so production assets must be
  // referenced from that sub-path. Local dev and preview use the root.
  base: mode === 'production' ? '/smile-classifier/' : '/',
  plugins: [react()],
  resolve: {
    // Match the whole specifier (both "./" and "../" forms) so it is replaced
    // outright — a partial match would leave the leading "../" and break the path.
    alias: [{ find: /^\.\.?\/tflite_web_api_client$/, replacement: tfliteClient }],
  },
  optimizeDeps: {
    include: ['@tensorflow/tfjs-tflite'],
    // onnxruntime-web ships its own wasm workers and dynamic imports; let Rollup
    // bundle it as-is instead of pre-bundling with esbuild (which mangles them).
    exclude: ['onnxruntime-web'],
  },
  server: { port: 5173 },
}))
