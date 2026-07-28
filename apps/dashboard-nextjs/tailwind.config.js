/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        // Neobrutalist high-contrast palette
        surface: {
          bg: "#F0EDE5",
          card: "#FFFFFF",
          accent: "#FF3366",
          highlight: "#00CCFF",
          dark: "#1A1A2E",
          muted: "#6B7280",
        },
        border: {
          DEFAULT: "#1A1A2E",
          accent: "#FF3366",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        brutal: "4px 4px 0px 0px #1A1A2E",
        "brutal-sm": "2px 2px 0px 0px #1A1A2E",
        "brutal-lg": "6px 6px 0px 0px #1A1A2E",
        "brutal-accent": "4px 4px 0px 0px #FF3366",
      },
    },
  },
  plugins: [],
};
