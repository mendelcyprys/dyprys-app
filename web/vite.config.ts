import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Dev runs on :5173 — the origin `dyp serve` already allows by default — and
// proxies /api so the client never needs to know a server address. The build
// lands in dist/, which `dyp serve --web web/dist` mounts.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: true },
});
