# Brief for building one video of the series

You are building one or more videos of a Hebrew teaching series for two students (Adir and Noa) who have **zero computer-vision background** but know ML basics. The series is their main way of learning the material for their final project: reconstructing a 3D pelvis from two X-rays. Quality bar: as good as a well-made 3Blue1Brown-style explainer. Visual first, then the formula. Accurate. On *their* data.

## Read first

1. `docs/superpowers/specs/2026-11-11-video-series-design.md`: the series design, and **your video's outline** (follow it; you may refine it).
2. `videos/src/v00/Video.tsx` + `videos/src/v00/script.json`: the **reference implementation**. Copy its structure exactly.
3. `videos/src/lib/*`: the shared library (Stage, CameraPath, Bone, Detector, ScanBeam, Voxels, SlicePlane, Rays, WireBox, Captions, Title, Eq, Bullets, Panel, Label, PipelineMap, Opening, RecapCard, YourTurn, Credit, Narration, timeline helpers). **Use it; don't fork it.** If you need a new reusable piece, put it in your own folder (`videos/src/vNN/parts.tsx`). Do NOT edit files in `src/lib`, `src/videos.ts`, `src/Root.tsx` or other videos' folders.
4. The project proposal (Hebrew) for context: `C:\Users\Adir\Desktop\Adir\busi-notes\Year-3-Sem-A\Medical-CV\Project\הצעת פרויקט גמר - שחזור עצם תלת-ממדי משני צילומי רנטגן.md`

## What you produce for video NN (two-digit, e.g. 03)

- `videos/src/vNN/script.json`: Hebrew narration segments `{id, text, caption, pauseAfter?}` + `voice: "he-IL-AvriNeural"`, `rate: "-4%"`, `lead` ~3.2 (opening card), `tail` ~2.0.
  - **Length: 6–10 minutes total.** Roughly 20–35 segments of 1–3 sentences.
  - `text` is what is spoken: plain spoken Hebrew. Write English terms the way a Hebrew speaker says them (e.g. "די אר אר", "דייס", "קונבולוציה"), and numbers in words when they're spoken.
  - `caption` is what's shown: a shorter Hebrew line keeping the English technical term in Latin letters (DRR, Dice, kernel).
  - First segment(s) introduce the topic and why it matters for *the project*. End with a `recap` segment and a `turn` segment ("your turn": the notebook TODOs for that unit).
- `videos/prep/vNN_assets.py` (if you need data beyond `public/data/hero/`): writes into `videos/public/data/vNN/`. **Keep new assets under ~15 MB per video** (step_size 2 meshes, small PNGs, JSON). Reuse `public/data/hero/*` (hero pelvis s1390: meshes, DRRs, slices, ct_u8.raw + labels_u8.raw volumes, occ16/32/64.json, hu_hist.json, meta.json) whenever possible.
- `videos/src/vNN/Video.tsx`: exporting `export const tl = buildTimeline(...)` and `export const VNN = () => ...` (the registry already imports exactly these names, and a placeholder file is there for you to replace).
- Narration audio: `cd videos && .venv/Scripts/python.exe -I prep/tts.py src/vNN/script.json public/audio/vNN` (writes the mp3s + durations.json; uses network).

## Tools and paths

- Python for asset prep: `videos/.venv/Scripts/python.exe` (has numpy, nibabel, scikit-image, scipy, pillow, edge-tts). Always run with `-I`. Add `-X utf8` when reading JSON with Hebrew.
- The data: `data/raw/totalsegmentator_small/extracted/<subject>/ct.nii.gz` + `segmentations/*.nii.gz` (RAS, 1.5 mm). `data/raw/totalsegmentator_small/screen.json` lists `good` subjects (complete pelvis, no metal). Hero = **s1390**. DeepFluoro (real fluoroscopy + CT): `data/raw/deepfluoro/subject0X/` (**CC BY-NC-ND: only video 13 may use it, and its derived assets must go in `videos/public/data/df_v13/`, which is gitignored**).
- Scene convention (see `prep/hero_assets.py`): x = patient Right, y = Superior, z = Posterior, 1 mm = 0.01 units, centred on the hero pelvis centroid. Camera at −z sees the patient from the front.
- Render a still: `cd videos && npx remotion still src/index.ts VNN out/vNN_<sec>.png --frame=<sec*60> --log=error`
- Contact sheet (to review many stills at once): `.venv/Scripts/python.exe -I prep/contact.py out/sheet_vNN.png 3 out/vNN_*.png`, then look at it with the Read tool.

## Hard-won technical rules (all verified; don't relearn them)

- **Load assets in the DOM component** (`useObj/useTex/useJson/useVolume` in `VNN`), gate the `<Stage>` on `ready`, and pass the assets into the scene as props. Loading inside the R3F tree finishes after Remotion has drawn the frame, so the asset never appears.
- **Don't use `@react-three/postprocessing`** (it renders an empty canvas here). `Stage` already has bloom/vignette/grain.
- Everything is a pure function of time `t` (seconds). No `useFrame` animation state, no `Math.random()` (use deterministic pseudo-random, e.g. `Math.sin(i*12.9898)`).
- Drive visibility with `ramp/window01` over segment times from `tl.seg[id]` (or `span`). Never hard-code seconds for content.
- Shared materials: `Bone` already clones per instance. For your own materials, give each mesh its own material if opacities differ.
- Hebrew text: `direction: "rtl"` on containers (Panel/Title/Label already do). Equations are LTR (Eq handles it).
- An instanced mesh with thousands of instances is fine; ~100k-vertex meshes are fine; avoid >300k.
- 2D animated diagrams (plots, grids, kernels, arrows) are often clearer than 3D: build them with absolutely positioned divs/SVG in the DOM layer, animated by `t`.

## Content rules

- Accuracy matters more than anything. When unsure of a fact, be conservative or leave it out. Use *real computations* on the data where the video shows numbers (e.g. compute an actual Dice between two real masks in your prep script and show that number), and never invent numbers. Planned results the project hasn't produced yet (e.g. network Dice) must be presented as "what we will measure", never as results.
- Every formula appears together with a picture of what it means, and its terms get highlighted as the narration names them (`Eq` `terms`).
- Pace: one idea per segment; let visuals breathe (pauseAfter 0.6–1.0 between sections).
- Connect each concept back to the project pipeline (`PipelineMap` / `Opening active=[...]`).
- The "your turn" card must match the unit's notebook work in the design doc.

## Verification before you report done (required)

1. Render **at least 15 stills** spread over the whole timeline (one per section minimum) and review them as contact sheets. Look for: empty/black frames, objects off-screen or clipped, text overlapping other text or covering the key visual, unreadable captions, wrong orientation (the AP image must show the patient's right on the viewer's left; superior at the top), and visuals that don't match what the narration says at that moment.
2. Fix everything you find; re-render the affected stills and check again.
3. `npx remotion compositions src/index.ts` must list your video with its real duration (6–10 min).
4. Don't run full video renders (CI does that), don't commit or push (the coordinator does), and don't touch other videos.

Report back: final duration, the section list with timestamps, assets added (and sizes), anything you couldn't do or are unsure about (especially content accuracy), and the contact-sheet paths.
