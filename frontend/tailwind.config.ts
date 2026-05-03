import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#090b10",
        panel: "#111722",
        line: "#263143",
        accent: "#5eead4",
        amber: "#f6c56b"
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(94,234,212,.12), 0 20px 80px rgba(0,0,0,.35)"
      }
    },
  },
  plugins: [],
};

export default config;
