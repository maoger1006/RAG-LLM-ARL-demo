import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev mode the FastAPI backend runs on :8000 and Vite proxies to it,
// so the frontend always uses relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/source': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
