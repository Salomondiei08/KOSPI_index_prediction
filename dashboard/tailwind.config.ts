import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Futura", "Avenir Next", "Inter", "system-ui", "sans-serif"],
        body: ["Futura", "Avenir Next", "Inter", "system-ui", "sans-serif"]
      },
      colors: {
        background: "#080C16",
        surface: "#0E1624",
        border: "#1F2A3A",
        accent: "#22C55E",
        accentSoft: "#86EFAC",
        success: "#22C55E",
        warning: "#F59E0B"
      }
    }
  },
  plugins: []
};

export default config;
