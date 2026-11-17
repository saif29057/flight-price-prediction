import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api is proxied to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
