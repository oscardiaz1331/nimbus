import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Dev-only proxy: the SPA on :5173 talks to the FastAPI server on :8080.
// In production FastAPI serves the built dist/ itself and no proxy exists.
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/images': 'http://localhost:8080',
    },
  },
});
