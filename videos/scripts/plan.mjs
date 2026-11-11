// Emits the GitHub Actions render matrix: one entry per (video, frame chunk).
// Usage: node scripts/plan.mjs "V00,V01" [chunkFrames]  -> prints JSON to stdout, writes serve-url bundle to ./bundle
import { bundle } from "@remotion/bundler";
import { getCompositions } from "@remotion/renderer";
import path from "node:path";
import fs from "node:fs";

const wanted = (process.argv[2] || "").split(",").map((s) => s.trim()).filter(Boolean);
const chunk = Number(process.argv[3] || 1200);
const serveUrl = await bundle({ entryPoint: path.resolve("src/index.ts"), outDir: path.resolve("bundle") });
const comps = await getCompositions(serveUrl, { chromiumOptions: { gl: "swangle" } });
const include = [];
const videos = [];
for (const c of comps) {
  if (wanted.length && !wanted.includes(c.id)) continue;
  if (c.id.startsWith("Spike")) continue;
  videos.push({ id: c.id, frames: c.durationInFrames, fps: c.fps });
  for (let a = 0, i = 0; a < c.durationInFrames; a += chunk, i++) {
    const b = Math.min(a + chunk, c.durationInFrames) - 1;
    include.push({ video: c.id, start: a, end: b, idx: String(i).padStart(3, "0") });
  }
}
fs.writeFileSync("plan.json", JSON.stringify({ include, videos }, null, 1));
console.log(JSON.stringify({ include }));
