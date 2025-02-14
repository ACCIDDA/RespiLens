import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/RespiLens/',  // This should match your deployment path
  plugins: [react()],
  server: {
    port: 5173,
    // Serve files from the app/public directory during development
    publicDir: 'public'
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets'
  },
  // ... rest of config
}); 