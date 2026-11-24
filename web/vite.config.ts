import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));

// The lab is served by the Node inference server (npm start) from web/dist.
// In dev (npm run dev:web) Vite proxies the API and the trace WebSocket to it.
export default defineConfig({
  root: here,
  base: "./",
  build: { outDir: "dist", emptyOutDir: true, target: "es2022", chunkSizeWarningLimit: 900 },
  server: {
    port: 5178,
    proxy: { "/api": { target: "http://127.0.0.1:5177", ws: true } },
  },
});
