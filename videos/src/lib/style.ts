// One visual language for the whole series.
import "@fontsource/rubik/hebrew-400.css";
import "@fontsource/rubik/hebrew-500.css";
import "@fontsource/rubik/hebrew-700.css";
import "@fontsource/rubik/latin-400.css";
import "@fontsource/rubik/latin-500.css";
import "@fontsource/rubik/latin-700.css";

export const FONT = "Rubik"; // bundled locally (@fontsource): renders never depend on the network
export const C = {
  bg: "#05070b",
  panel: "rgba(8,14,24,0.78)",
  line: "rgba(120,190,255,0.45)",
  text: "#e9f1ff",
  dim: "#8fa6c4",
  xray: "#7fd0ff",      // X-ray light / "the signal"
  warm: "#ffcf7f",      // "notice this"
  good: "#7dffb0",
  bad: "#ff7f8a",
  bone: "#efe6d2",
};
export const FPS = 60;
export const W = 1920, H = 1080;
