import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Frontend dev server on :5173. The app calls the FastAPI backend directly
// (VITE_API_BASE, default http://localhost:8000) with CORS enabled on the API.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
