import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/',                     // root on respilens.com
  plugins: [react()],
  server: {
    watch: { usePolling: true }, // still useful on macOS/WSL
    publicDir: 'public'          // static files live here
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets'
  }
});