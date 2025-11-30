import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import sirv from "sirv";
import fs from "fs";

const reportsPath = path.resolve(__dirname, "../reports");

export default defineConfig({
  plugins: [
    react(),
    {
      name: "serve-reports",
      configureServer(server) {
        if (fs.existsSync(reportsPath)) {
          const serve = sirv(reportsPath, { dev: true, etag: true });
          server.middlewares.use("/reports", serve);
        }
      }
    }
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src")
    }
  },
  server: {
    fs: {
      allow: [".."] // allow reading ../reports
    }
  }
});
