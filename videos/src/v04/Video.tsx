// Video 4 · The DRR: simulating an X-ray. Parallel-beam DRR (sum of mu along an axis), perspective DRR
// (a ray per pixel from a point source; trilinear sampling vs Siddon), a real NumPy perspective DRR of the
// hero CT next to the parallel one, DiffDRR, intensity mapping and detector/pixel size.
// No DeepFluoro data here (license): the gap to real X-rays is only named, as a pointer to video 13.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import type { ReactElement, ReactNode } from "react";
import script from "./script.json";
import durations from "../../public/audio/v04/durations.json";
import { buildTimeline, Script, span } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Detector, Meta, PointSource, Rays, ScanBeam, SourcePanel } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { FanBeam, toScreen, V3 } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;

type Mark = Record<string, [number, number]>;
type D4 = {
  sdd_mm: number; sod_mm: number; M_iso: number; det_spacing_mm: number; window_L: [number, number]; gamma: number;
  img_ap: [number, number]; img_lat: [number, number]; L_hist: { counts: number[]; edges: number[] };
  ray: { s_mm: number[]; mu: number[]; L: number; rowcol: [number, number] };
  axial: { slice_k: number; cell_mm: number; grid_RA: number[][]; row_L: number[] };
  crop_rect: { ap: [number, number][]; lat: [number, number][] };
  marks: Record<string, Mark>; landmarks_scene_mm: Record<string, number[]>; ap_iso_extent: number[]; lat_iso_extent: number[];
};

// perspective rig (scene units, 1 = 100 mm): source 700 mm in front of the centroid, detector 1000 mm from it
const SRC: V3 = [0, 0, -7];
const DET_Z = 3;
type Cam = { pos: V3; look: V3 };
const CAM_P: Cam = { pos: [13.2, 2.4, -1.6], look: [0, -0.2, -2.0] };

// ---------------- 3D scene ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const box = boxOf(a.meta);
  const [cx, cy, cz] = box.c;
  const apZsrc = box.mn[2] - 1.4, apZdet = box.mx[2] + 0.3;
  const latXsrc = box.mx[0] + 1.4, latXdet = box.mn[0] - 0.3;
  const e = a.d.ap_iso_extent, M = a.d.M_iso;
  const detSize: [number, number] = [(e[1] - e[0]) * M, (e[3] - e[2]) * M];
  const detC: V3 = [((e[0] + e[1]) / 2) * M, ((e[2] + e[3]) / 2) * M, DET_Z];

  const introOn = window01(t, 2.4, S.intro2[1] + 0.4, 1.0, 0.8);
  const apOn = window01(t, S.par1[0] - 0.2, S.par2[1] + 0.4, 0.8, 0.6);
  const apScan = ramp(t, S.par1[0] + 1.5, S.par1[1] + 1.5);
  const latOn = window01(t, S.par5[0] - 0.2, S.par5[1] + 0.5, 0.8, 0.5);
  const latScan = ramp(t, S.par5[0] + 0.4, S.par5[1]);
  const pOn = Math.max(window01(t, S.persp1[0] - 0.2, S.persp1[1] + 0.4, 0.8, 0.6), window01(t, S.persp6[0] - 0.2, S.persp6[1] + 0.4, 0.8, 0.6));
  const pScan = Math.max(ramp(t, S.persp1[0] + 2.0, S.persp1[1]), ramp(t, S.persp6[0] - 0.2, S.persp6[0]) );
  const tailOn = window01(t, S.recap[0], tl.total, 1.0, 1.0);
  const boneO = Math.max(introOn, apOn * 0.85, latOn * 0.85, pOn * 0.75, tailOn * 0.8);

  const corners: V3[][] = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) =>
    [SRC, [detC[0] + sx * detSize[0] / 2, detC[1] + sy * detSize[1] / 2, DET_Z]]);

  const keys: CamKey[] = [
    { t: 0, pos: orbit(-0.9, 10.8, 1.2), look: [0, 0.5, 0] },
    { t: S.intro2[1], pos: orbit(0.55, 10.2, 1.2), look: [0, 0.5, 0] },
    { t: S.par1[0] + 1.2, pos: [8.6, 2.4, -3.6], look: [0, 0, 0.1] },
    { t: S.par2[1], pos: [8.2, 1.8, -4.4], look: [0, 0, 0.2] },
    { t: S.par5[0] - 0.3, pos: [6.8, 4.8, -7.6], look: [-0.4, -0.2, 0.3] },
    { t: S.par5[1], pos: [6.4, 5.0, -8.0], look: [-0.4, -0.2, 0.3] },
    { t: S.persp1[0] + 0.3, ...CAM_P },
    { t: S.persp6[1], ...CAM_P },
    { t: S.recap[0] + 1, pos: orbit(0.7, 8.8, 1.4), look: [cx, cy, cz] },
    { t: tl.total, pos: orbit(1.6, 9.5, 1.0) },
  ];

  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.4} opacity={Math.max(introOn, apOn, latOn, pOn, tailOn)} />
      <Bone geo={a.pelvis} material={bone} opacity={boneO} />
      <Bone geo={a.femurs} material={bone} opacity={boneO * 0.9} />
      <Bone geo={a.spine} material={bone} opacity={boneO * 0.9} />

      {/* parallel AP */}
      <group position={[cx, cy, 0]}>
        <SourcePanel size={a.meta.ap_size} position={[0, 0, apZsrc]} opacity={apOn} />
        <ScanBeam axis="ap" progress={apScan} opacity={apOn} w={a.meta.ap_size[0]} h={a.meta.ap_size[1]}
          src={apZsrc} det={apZdet} bodyIn={box.mn[2]} bodyOut={box.mx[2]} />
        <Detector tex={a.ap} progress={apScan} opacity={apOn} size={a.meta.ap_size} position={[0, 0, apZdet]} rotation={[0, Math.PI, 0]} />
      </group>
      {/* parallel LAT */}
      <group position={[0, cy, cz]}>
        <SourcePanel size={a.meta.lat_size} position={[latXsrc, 0, 0]} rotation={[0, Math.PI / 2, 0]} opacity={latOn} />
        <ScanBeam axis="lat" progress={latScan} opacity={latOn} w={a.meta.lat_size[0]} h={a.meta.lat_size[1]}
          src={latXsrc} det={latXdet} bodyIn={box.mx[0]} bodyOut={box.mn[0]} />
        <Detector tex={a.lat} progress={latScan} opacity={latOn} size={a.meta.lat_size} position={[latXdet, 0, 0]} rotation={[0, Math.PI / 2, 0]} flipX />
      </group>
      {/* perspective AP */}
      <PointSource position={SRC} opacity={pOn} r={0.1} />
      <Rays segs={corners} opacity={0.5 * pOn} />
      <FanBeam src={SRC} detZ={DET_Z} cx={detC[0]} cy={detC[1]} w={detSize[0]} h={detSize[1]} progress={pScan} opacity={pOn}
        bodyIn={box.mn[2]} bodyOut={box.mx[2]} />
      <Detector tex={a.persp} progress={pScan} opacity={pOn} size={detSize} position={detC} rotation={[0, Math.PI, 0]} />
    </>
  );
}

// ---------------- 2D pieces ----------------
// SVG text; Hebrew strings get direction rtl (anchors are swapped so `anchor` keeps meaning left/centre/right)
const T = (x: number, y: number, s: string, color: string = C.text, size = 28, anchor: "start" | "middle" | "end" = "middle", mono = false) => {
  const he = /[֐-׿]/.test(s);
  const a = he && anchor !== "middle" ? (anchor === "start" ? "end" : "start") : anchor;
  return <text x={x} y={y} fill={color} fontSize={size} fontFamily={mono ? "monospace" : FONT} textAnchor={a} direction={he ? "rtl" : "ltr"}>{s}</text>;
};
const muGray = (m: number) => { const g = Math.round(255 * Math.min(1, Math.max(0, m / 0.036)) ** 0.9); return `rgb(${g},${g},${g})`; };

/** the coarse axial mu grid: anterior on top, patient right on the left (radiology view from the feet) */
function useGridLayout(d: D4, y0 = 130) {
  const g = d.axial.grid_RA;                  // [R][A]
  const nR = g.length, nA = g[0].length, cell = 22;
  const W = nR * cell, H = nA * cell, x0 = 960 - W / 2;
  // screen col c <- R index nR-1-c ; screen row r <- A index nA-1-r (anterior = high A on top)
  const val = (c: number, r: number) => g[nR - 1 - c][nA - 1 - r];
  return { nR, nA, cell, W, H, x0, y0, val };
}

function AxialSum({ t, d }: { t: number; d: D4 }) {
  const o = window01(t, S.par3[0] - 0.2, S.par3[1] + 0.5, 0.6, 0.6);
  const G = useGridLayout(d);
  if (o <= 0) return null;
  const sweep = ramp(t, S.par3[0] + 5.0, S.par3[1] - 0.3, (x) => x);
  const cur = Math.floor(sweep * G.nR);
  const sums = Array.from({ length: G.nR }, (_, c) => { let s = 0; for (let r = 0; r < G.nA; r++) s += G.val(c, r) * d.axial.cell_mm; return s; });
  const mx = Math.max(...sums);
  const cells: ReactElement[] = [];
  for (let c = 0; c < G.nR; c++) for (let r = 0; r < G.nA; r++)
    cells.push(<rect key={`${c}_${r}`} x={G.x0 + c * G.cell} y={G.y0 + r * G.cell} width={G.cell - 1} height={G.cell - 1} fill={muGray(G.val(c, r))} />);
  const barY = G.y0 + G.H + 40, barH = 150;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      {cells}
      {sweep > 0 && sweep < 1 && <rect x={G.x0 + cur * G.cell - 1} y={G.y0 - 30} width={G.cell + 1} height={G.H + 30} fill="none" stroke={C.warm} strokeWidth={3} />}
      {sweep > 0 && sweep < 1 && <line x1={G.x0 + (cur + 0.5) * G.cell} y1={G.y0 - 60} x2={G.x0 + (cur + 0.5) * G.cell} y2={G.y0 - 8} stroke={C.xray} strokeWidth={4} />}
      {sums.map((s, c) => c <= cur || sweep >= 1 ? (
        <rect key={c} x={G.x0 + c * G.cell} y={barY + barH - (s / mx) * barH} width={G.cell - 2} height={(s / mx) * barH}
          fill={c === cur && sweep < 1 ? C.warm : C.xray} opacity={0.85} />
      ) : null)}
      <line x1={G.x0} y1={barY + barH} x2={G.x0 + G.W} y2={barY + barH} stroke={C.dim} strokeWidth={2} />
      {T(G.x0 - 20, G.y0 + 20, "קדימה (המקור)", C.dim, 24, "end")}
      {T(G.x0 - 20, G.y0 + G.H - 6, "אחורה (הגלאי)", C.dim, 24, "end")}
      {T(G.x0 - 20, barY + barH - 10, "L = Σ μ·Δs", C.xray, 28, "end")}
      {T(G.x0 + 10, G.y0 + G.H + 28, "ימין המטופל", C.dim, 22, "start")}
      {T(G.x0 + G.W - 10, G.y0 + G.H + 28, "שמאל", C.dim, 22, "end")}
    </svg>
  );
}

/** the oblique ray through the same grid: trilinear samples, then Siddon's crossings */
function RayGrid({ t, d }: { t: number; d: D4 }) {
  const [a0, a1] = span(tl, "persp2", "persp4");
  const o = window01(t, a0 - 0.2, a1 + 0.5, 0.6, 0.6);
  const G = useGridLayout(d, 250);
  if (o <= 0) return null;
  const cells: ReactElement[] = [];
  for (let c = 0; c < G.nR; c++) for (let r = 0; r < G.nA; r++)
    cells.push(<rect key={`${c}_${r}`} x={G.x0 + c * G.cell} y={G.y0 + r * G.cell} width={G.cell - 1} height={G.cell - 1} fill={muGray(G.val(c, r))} opacity={0.75} />);
  // the ray in grid units (col, row): enters the top edge, leaves the bottom edge
  const p0 = [G.nR * 0.26, 0], p1 = [G.nR * 0.52, G.nA];
  const toS = (p: number[]) => [G.x0 + p[0] * G.cell, G.y0 + p[1] * G.cell];
  const [sx0, sy0] = toS(p0), [sx1, sy1] = toS(p1);
  const draw = ramp(t, S.persp2[0] + 0.5, S.persp2[0] + 2.5);
  const centres = ramp(t, S.persp2[0] + 3.5, S.persp2[0] + 4.5);
  const samp = ramp(t, S.persp3[0] + 1.0, S.persp3[0] + 4.0);
  const zoom = window01(t, S.persp3[0] + 4.5, S.persp3[1] + 0.3, 0.6, 0.4);
  const sid = ramp(t, S.persp4[0] + 0.5, S.persp4[0] + 4.5);
  const sidOn = window01(t, S.persp4[0], a1 + 0.5, 0.5, 0.5);
  // samples every 0.6 cell along the ray
  const len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
  const ns = Math.floor(len / 0.6);
  const samples = Array.from({ length: ns }, (_, i) => (i + 0.5) / ns);
  // Siddon: parametric crossings with vertical (col) and horizontal (row) grid lines
  const cross: number[] = [0, 1];
  for (let c = Math.ceil(Math.min(p0[0], p1[0])); c <= Math.floor(Math.max(p0[0], p1[0])); c++) cross.push((c - p0[0]) / (p1[0] - p0[0]));
  for (let r = 1; r < G.nA; r++) cross.push(r / G.nA);
  cross.sort((x, y) => x - y);
  const segs = cross.slice(1).map((b, i) => ({ a: cross[i], b })).filter((s) => s.b - s.a > 1e-6);
  const hl = segs.map((s) => { const m = (s.a + s.b) / 2; return { c: Math.floor(p0[0] + (p1[0] - p0[0]) * m), r: Math.floor(p0[1] + (p1[1] - p0[1]) * m), l: (s.b - s.a) * len }; });
  const shownSegs = Math.floor(sid * hl.length);
  // the zoomed sample: one sample point between 4 cell centres (2D picture of the 8-voxel trilinear idea)
  const at = (s: number) => [p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s];
  const frac = (v: number) => v - 0.5 - Math.floor(v - 0.5);
  const zs = samples.slice(Math.floor(ns * 0.3), Math.floor(ns * 0.7)).reduce((best, s) => {
    const q = at(s), e = (frac(q[0]) - 0.3) ** 2 + (frac(q[1]) - 0.7) ** 2;
    const qb = at(best), eb = (frac(qb[0]) - 0.3) ** 2 + (frac(qb[1]) - 0.7) ** 2;
    return e < eb ? s : best;
  }, samples[Math.floor(ns * 0.45)]);
  const zp = at(zs);
  const zc = [Math.floor(zp[0] - 0.5), Math.floor(zp[1] - 0.5)];
  const fx = zp[0] - 0.5 - zc[0], fy = zp[1] - 0.5 - zc[1];
  const ZX = 1480, ZY = 250, ZS = 300;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      {cells}
      {/* voxel centres */}
      <g opacity={centres * (1 - sidOn)}>
        {Array.from({ length: G.nR * G.nA }, (_, i) => {
          const c = i % G.nR, r = Math.floor(i / G.nR);
          return <circle key={i} cx={G.x0 + (c + 0.5) * G.cell} cy={G.y0 + (r + 0.5) * G.cell} r={1.6} fill={C.xray} opacity={0.55} />;
        })}
      </g>
      {/* Siddon: crossed cells */}
      <g opacity={sidOn}>
        {hl.slice(0, shownSegs).map((h, i) => (
          <rect key={i} x={G.x0 + h.c * G.cell} y={G.y0 + h.r * G.cell} width={G.cell - 1} height={G.cell - 1} fill="none"
            stroke={C.warm} strokeWidth={2.5} opacity={0.35 + 0.65 * Math.min(1, h.l / 1.0)} />
        ))}
        {cross.slice(0, Math.floor(sid * cross.length)).map((s, i) => {
          const [x, y] = toS([p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s]);
          return <circle key={i} cx={x} cy={y} r={4} fill={C.good} />;
        })}
      </g>
      {/* the ray */}
      <line x1={sx0 - (sx1 - sx0) * 0.12} y1={sy0 - (sy1 - sy0) * 0.12} x2={lerp(sx0, sx1, draw)} y2={lerp(sy0, sy1, draw)} stroke="#cfeaff" strokeWidth={3} />
      {/* trilinear samples */}
      <g opacity={samp * (1 - sidOn)}>
        {samples.slice(0, Math.ceil(samp * ns)).map((s, i) => {
          const [x, y] = toS([p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s]);
          return <circle key={i} cx={x} cy={y} r={5} fill={C.warm} />;
        })}
      </g>
      {/* zoom inset: 4 centres in 2D (8 in 3D), weights by distance */}
      {zoom > 0 && (
        <g opacity={zoom}>
          <rect x={ZX} y={ZY} width={ZS} height={ZS} fill="rgba(8,14,24,0.9)" stroke={C.line} strokeWidth={2} />
          {[[0, 0], [1, 0], [0, 1], [1, 1]].map(([i, j], k) => {
            const w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
            const x = ZX + 50 + i * 200, y = ZY + 50 + j * 200;
            return (
              <g key={k}>
                <line x1={x} y1={y} x2={ZX + 50 + fx * 200} y2={ZY + 50 + fy * 200} stroke={C.warm} strokeWidth={1 + 7 * w} opacity={0.8} />
                <circle cx={x} cy={y} r={16} fill={muGray(G.val(zc[0] + i, zc[1] + j))} stroke={C.xray} strokeWidth={2} />
                {T(x + (i ? 26 : -26), y + (j ? 44 : -26), w.toFixed(2), C.warm, 24, i ? "start" : "end", true)}
              </g>
            );
          })}
          <circle cx={ZX + 50 + fx * 200} cy={ZY + 50 + fy * 200} r={9} fill={C.warm} />
          {T(ZX + ZS / 2, ZY + ZS + 40, "4 שכנים בדו-ממד · 8 בתלת-ממד", C.dim, 24)}
        </g>
      )}
      {sidOn > 0 && T(1660, 420, "μ × אורך הקטע", C.warm, 32, "middle")}
      {sidOn > 0 && T(1660, 470, "● חיתוך עם קו רשת", C.good, 26, "middle")}
    </svg>
  );
}

/** mu along one real ray (perspective AP, through the right femoral head); the area is the pixel value */
function RayProfile({ t, d }: { t: number; d: D4 }) {
  const o = window01(t, S.persp5[0] - 0.2, S.persp5[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const { s_mm, mu, L } = d.ray;
  const x0 = 150, y0 = 260, w = 1000, h = 480;
  const smax = s_mm[s_mm.length - 1], mmax = 0.06;
  const P = (s: number, m: number) => [x0 + (s / smax) * w, y0 + h - (m / mmax) * h];
  const k = ramp(t, S.persp5[0] + 2.5, S.persp5[0] + 8.0, (x) => x);
  const nShow = Math.max(2, Math.ceil(k * s_mm.length));
  const pts = s_mm.slice(0, nShow).map((s, i) => P(s, mu[i]));
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L${pts[pts.length - 1][0].toFixed(1)},${y0 + h} L${x0},${y0 + h} Z`;
  const areaOn = ramp(t, S.persp5[0] + 9.0, S.persp5[0] + 10.5);
  const [ih, iw] = d.img_ap, IW = 520, sc = IW / iw;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <path d={area} fill="rgba(255,207,127,0.28)" opacity={areaOn} />
        <path d={line} fill="none" stroke={C.warm} strokeWidth={3} />
        <line x1={x0} y1={y0 + h} x2={x0 + w} y2={y0 + h} stroke={C.dim} strokeWidth={2} />
        <line x1={x0} y1={y0} x2={x0} y2={y0 + h} stroke={C.dim} strokeWidth={2} />
        {T(x0 + w, y0 + h + 44, "מרחק לאורך הקרן, s (mm)", C.dim, 26, "end")}
        {T(x0 + w, y0 + h + 80, `0 … ${Math.round(smax)} mm`, C.dim, 22, "end")}
        {T(x0 + 14, y0 - 16, "μ (1/mm)", C.dim, 26, "start")}
        {[0.02, 0.04].map((m) => <g key={m}><line x1={x0 - 8} y1={P(0, m)[1]} x2={x0 + w} y2={P(0, m)[1]} stroke="rgba(143,166,196,0.2)" />{T(x0 - 14, P(0, m)[1] + 8, m.toFixed(2), C.dim, 20, "end")}</g>)}
        {T(x0 + 20, P(0, 0.02)[1] - 10, "מים ≈ 0.02", C.dim, 20, "start")}
        <g opacity={areaOn}>{T(x0 + w * 0.5, y0 + 40, `L = Σ μ·Δs ≈ ${L.toFixed(2)}`, C.warm, 40)}</g>
      </svg>
      <div style={{ position: "absolute", left: 1260, top: 250, width: IW, height: ih * sc }}>
        <Img src={staticFile("data/v04/persp_ap.png")} style={{ width: IW, height: ih * sc, borderRadius: 8, border: `1px solid ${C.line}` }} />
        <div style={{ position: "absolute", left: d.ray.rowcol[1] * sc - 14, top: d.ray.rowcol[0] * sc - 14, width: 28, height: 28,
          borderRadius: 14, border: `3px solid ${C.warm}`, boxShadow: `0 0 16px ${C.warm}`, opacity: areaOn }} />
        <Label x={IW / 2} y={-34} text="הפיקסל של הקרן הזאת" color={C.dim} size={26} />
      </div>
    </div>
  );
}

// crossfade cycle shared by the toggle and its zoom insets: hold parallel, fade, hold perspective, fade back
const toggleK = (t: number, a: number) => {
  const ph = Math.max(0, t - a - 1.5) % 3.0;
  return t - a < 1.5 ? 0 : ph < 0.9 ? 0 : ph < 1.5 ? ramp(ph, 0.9, 1.5) : ph < 2.4 ? 1 : 1 - ramp(ph, 2.4, 3.0);
};

/** magnified crop around a landmark, crossfading in sync; the cross marks the parallel position */
function Zoom({ t, a, b, sync, view, d, mark, x, y, size = 270, crop = 56, label, color }: {
  t: number; a: number; b: number; sync: number; view: "ap" | "lat"; d: D4; mark: string; x: number; y: number;
  size?: number; crop?: number; label: string; color: string;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const k = toggleK(t, sync);
  const [ih, iw] = view === "ap" ? d.img_ap : d.img_lat;
  const p = d.marks[mark][`par_${view}`], q = d.marks[mark][`persp_${view}`];
  const Z = size / crop;
  const left = -(p[1] - crop / 2) * Z, top = -(p[0] - crop / 2) * Z;
  const im = (src: string, op: number) => (
    <Img src={staticFile(src)} style={{ position: "absolute", left, top, width: iw * Z, height: ih * Z, opacity: op, imageRendering: "pixelated" }} />
  );
  const dx = (q[1] - p[1]) * Z * k, dy = (q[0] - p[0]) * Z * k;
  return (
    <div style={{ position: "absolute", left: x, top: y, opacity: o }}>
      <div style={{ position: "relative", width: size, height: size, overflow: "hidden", borderRadius: 10, border: `2px solid ${color}` }}>
        {im(`data/v04/par_${view}.png`, 1)}{im(`data/v04/persp_${view}.png`, k)}
        <div style={{ position: "absolute", left: size / 2 - 1, top: size / 2 - 16, width: 2, height: 32, background: C.xray }} />
        <div style={{ position: "absolute", left: size / 2 - 16, top: size / 2 - 1, width: 32, height: 2, background: C.xray }} />
        <div style={{ position: "absolute", left: size / 2 + dx - 7, top: size / 2 + dy - 7, width: 14, height: 14, borderRadius: 7, background: color, boxShadow: `0 0 10px ${color}` }} />
      </div>
      <Label x={size / 2} y={-28} text={label} color={color} size={26} />
      <Label x={size / 2} y={size + 30} size={24} color={C.dim}
        text={<span>הזזה: <span style={{ direction: "ltr", display: "inline-block" }}>{(Math.hypot(q[0] - p[0], q[1] - p[1]) * 1.5).toFixed(0)} mm</span></span>} />
    </div>
  );
}

/** an image with markers that move between parallel and perspective positions */
function Toggle({ t, a, b, view, d, x, y, h, marks, title }: {
  t: number; a: number; b: number; view: "ap" | "lat"; d: D4; x: number; y: number; h: number;
  marks: { key: string; label: string; color: string }[]; title?: string;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const [ih, iw] = view === "ap" ? d.img_ap : d.img_lat;
  const sc = h / ih, w = iw * sc;
  const k = toggleK(t, a);
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, opacity: o }}>
      <Img src={staticFile(`data/v04/par_${view}.png`)} style={{ position: "absolute", width: w, height: h, borderRadius: 10 }} />
      <Img src={staticFile(`data/v04/persp_${view}.png`)} style={{ position: "absolute", width: w, height: h, borderRadius: 10, opacity: k }} />
      <div style={{ position: "absolute", inset: 0, border: `2px solid ${k > 0.5 ? C.warm : C.xray}`, borderRadius: 10 }} />
      {marks.map((m) => {
        const p = d.marks[m.key][`par_${view}`], q = d.marks[m.key][`persp_${view}`];
        const r = lerp(p[0], q[0], k) * sc, c = lerp(p[1], q[1], k) * sc;
        return (
          <div key={m.key}>
            <div style={{ position: "absolute", left: c - 7, top: r - 7, width: 14, height: 14, borderRadius: 7, background: m.color, boxShadow: `0 0 12px ${m.color}` }} />
            <div style={{ position: "absolute", left: c - 22, top: r - 22, width: 44, height: 44, borderRadius: 22, border: `1.5px solid ${m.color}`, opacity: 0.7 }} />
          </div>
        );
      })}
      <Label x={w / 2} y={-36} size={34} text={<span>{title ? `${title} · ` : ""}<b style={{ color: k > 0.5 ? C.warm : C.xray }}>{k > 0.5 ? "פרספקטיבי" : "מקבילי"}</b></span>} />
    </div>
  );
}

/** intensity mapping: histogram of L with the window, and the two display curves */
function Mapping({ t, d }: { t: number; d: D4 }) {
  const o = window01(t, S.int2[0] - 0.2, S.int2[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const { counts, edges } = d.L_hist;
  const x0 = 980, y0 = 230, w = 820, h = 470;
  const Lmax = edges[edges.length - 1], cmax = Math.max(...counts);
  const X = (L: number) => x0 + (L / Lmax) * w, Y = (v: number) => y0 + h - v * h;
  const [lo, hi] = d.window_L;
  const win = (L: number) => Math.min(1, Math.max(0, (L - lo) / (hi - lo))) ** d.gamma;
  const sat = (L: number) => 1 - Math.exp(-L);
  const curve = (f: (L: number) => number) => Array.from({ length: 121 }, (_, i) => { const L = (i / 120) * Lmax; return `${i ? "L" : "M"}${X(L).toFixed(1)},${Y(f(L)).toFixed(1)}`; }).join(" ");
  const c1 = ramp(t, S.int2[0] + 1.0, S.int2[0] + 2.5), c2 = ramp(t, S.int2[0] + 5.0, S.int2[0] + 7.0);
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      {counts.map((c, i) => <rect key={i} x={X(edges[i])} y={Y(0.85 * c / cmax)} width={w / counts.length - 1} height={0.85 * (c / cmax) * h} fill="rgba(127,208,255,0.28)" />)}
      <line x1={x0} y1={y0 + h} x2={x0 + w} y2={y0 + h} stroke={C.dim} strokeWidth={2} />
      <line x1={x0} y1={y0} x2={x0} y2={y0 + h} stroke={C.dim} strokeWidth={2} />
      <path d={curve(sat)} fill="none" stroke={C.bad} strokeWidth={3} strokeDasharray="10 7" />
      <g opacity={c1}>
        <line x1={X(lo)} y1={y0} x2={X(lo)} y2={y0 + h} stroke={C.warm} strokeWidth={2} strokeDasharray="6 6" />
        <line x1={X(hi)} y1={y0} x2={X(hi)} y2={y0 + h} stroke={C.warm} strokeWidth={2} strokeDasharray="6 6" />
        {T(X(lo), y0 - 12, "2%", C.warm, 22)}{T(X(hi), y0 - 12, "99.7%", C.warm, 22)}
      </g>
      <path d={curve(win)} fill="none" stroke={C.good} strokeWidth={4} opacity={c2} />
      {T(x0 + w / 2, y0 + h + 44, "L: סכום ההנחתה", C.dim, 26)}
      {T(x0 - 14, y0 + 8, "1", C.dim, 22, "end")}{T(x0 - 14, y0 + h, "0", C.dim, 22, "end")}
      {T(x0 + w - 10, Y(0.97) + 34, "1 − e^(−L)", C.bad, 26, "end", true)}
      <g opacity={c2}>{T(x0 + w * 0.42, Y(0.35), "חלון + חזקה", C.good, 28, "end")}</g>
      {T(x0 + 10, y0 + h - 12, "היסטוגרמה של L", "rgba(127,208,255,0.8)", 22, "start")}
    </svg>
  );
}

// ---------------- the video ----------------
type Assets = {
  meta: Meta; pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry; spine: THREE.BufferGeometry;
  ap: THREE.Texture; lat: THREE.Texture; persp: THREE.Texture; d: D4;
};

export const V04 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const meta = useJson<Meta>("data/hero/meta.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const spine = useObj("data/hero/spine.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const lat = useTex("data/hero/drr_lat.png");
  const persp = useTex("data/v04/persp_ap.png");
  const d = useJson<D4>("data/v04/v04.json");
  const ready = meta && pelvis && femurs && spine && ap && lat && persp && d;

  const sc = (p: V3) => toScreen(p, CAM_P.pos, CAM_P.look);
  const [srcX, srcY] = sc(SRC);
  const [d0x, d0y] = sc([0, -2.6, SRC[2]]);
  const [d1x] = sc([0, -2.6, 0]);
  const [d2x] = sc([0, -2.6, DET_Z]);
  const p6 = window01(t, S.persp6[0] + 1, S.persp6[1] + 0.3, 0.6, 0.6);

  const imgBox = (src: string, x: number, y: number, h: number, ar: number, label: ReactNode, o: number, color = C.xray, pix = false) => o > 0 && (
    <div style={{ position: "absolute", left: x, top: y, opacity: o }}>
      <Img src={staticFile(src)} style={{ height: h, width: h * ar, borderRadius: 10, border: `1.5px solid ${C.line}`, imageRendering: pix ? "pixelated" : "auto" }} />
      <Label x={(h * ar) / 2} y={-34} text={label} color={color} size={30} />
    </div>
  );
  const arAP = d ? d.img_ap[1] / d.img_ap[0] : 1, arLAT = d ? d.img_lat[1] / d.img_lat[0] : 0.6;
  const [c1a, c4b] = span(tl, "cmp1", "cmp4");

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ meta, pelvis, femurs, spine, ap, lat, persp, d }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.62 * window01(t, S.recap[0] - 0.3, tl.total, 0.8, 0.1), pointerEvents: "none" }} />

      <Opening n={4} title="DRR: לדמות צילום רנטגן" active={[1]} t={t} end={2.9} />
      <Title text="DRR" sub="digitally reconstructed radiograph · צילום רנטגן מדומה" t={t} a={S.intro1[0] + 0.5} b={S.intro2[1] + 0.3} />

      {/* parallel */}
      <Title text="DRR מקבילי" sub="parallel beam" t={t} a={S.par1[0]} b={S.par1[1] + 0.3} size={64} />
      <Eq tex={"\\hl{L}{L} = \\sum_i \\hl{mu}{\\mu_i}\\,\\hl{ds}{\\Delta s}"} t={t} a={S.par2[0] + 0.3} b={S.par2[1] + 0.3}
        style={{ left: 40, top: 420, width: 480 }} size={54} note="סכום לאורך הקרן"
        terms={{ L: [S.par2[0] + 2, S.par2[0] + 6], mu: [S.par2[0] + 9, S.par2[1]], ds: [S.par2[0] + 11.5, S.par2[1]] }} />
      {ready && <AxialSum t={t} d={d} />}
      {ready && (() => {
        const o = window01(t, S.par4[0] - 0.2, S.par4[1] + 0.4, 0.6, 0.6);
        if (o <= 0) return null;
        const h = 640, w = h * arAP, rev = ramp(t, S.par4[0] + 0.5, S.par4[0] + 6.0, (x) => x);
        const row = (d.crop_rect.ap[1][1] - 1 - d.axial.slice_k) / d.img_ap[0];
        return (
          <div style={{ position: "absolute", left: 300, top: 180, width: w, height: h, opacity: o }}>
            <Img src={staticFile("data/v04/par_ap.png")} style={{ width: w, height: h, borderRadius: 10, clipPath: `inset(0 0 ${(1 - rev) * 100}% 0)` }} />
            <div style={{ position: "absolute", inset: 0, border: `1.5px solid ${C.line}`, borderRadius: 10 }} />
            {rev < 1 && <div style={{ position: "absolute", left: 0, right: 0, top: rev * h - 2, height: 4, background: C.xray, boxShadow: `0 0 18px ${C.xray}` }} />}
            <div style={{ position: "absolute", left: -10, right: -10, top: row * h - 2, height: 4, background: C.warm, opacity: rev > row ? window01(t, S.par4[0] + 1, S.par4[1], 0.4, 0.4) : 0 }} />
            <Label x={w + 40} y={row * h} align="left" text="השורה מהפרוסה ←" color={C.warm} size={26} o={rev > row ? 1 : 0} />
            <Panel o={ramp(t, S.par4[0] + 6.5, S.par4[0] + 7.3)} style={{ left: w + 110, top: 360, width: 560, direction: "ltr", padding: "24px 30px" }}>
              <div style={{ fontFamily: "monospace", fontSize: 34, color: C.text }}>L = mu.sum(axis=<span style={{ color: C.warm }}>?</span>) * dz</div>
              <div style={{ fontSize: 24, color: C.dim, marginTop: 12, direction: "rtl" }}>איזה ציר? זה החלק שלכם במחברת</div>
            </Panel>
          </div>
        );
      })()}

      {/* perspective */}
      <Title text="DRR פרספקטיבי" sub="קרן לכל פיקסל, ממקור נקודתי" t={t} a={S.persp1[0]} b={S.persp1[1] + 0.3} size={64} top={50} />
      {ready && <RayGrid t={t} d={d} />}
      <Title text="דגימה לאורך הקרן" sub="trilinear interpolation" t={t} a={S.persp3[0]} b={S.persp3[1] + 0.2} size={50} top={30} />
      <Title text="השיטה של Siddon" sub="חיתוכים עם קווי הרשת" t={t} a={S.persp4[0]} b={S.persp4[1] + 0.3} size={50} top={30} />
      {ready && <RayProfile t={t} d={d} />}
      <Title text="קרן אמיתית אחת" t={t} a={S.persp5[0]} b={S.persp5[1] + 0.3} size={52} top={60} />
      <Label x={srcX} y={srcY - 60} text="מקור" color={C.xray} size={32} o={p6} />
      {p6 > 0 && (
        <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: p6 }}>
          <line x1={d0x} y1={d0y} x2={d2x} y2={d0y} stroke={C.warm} strokeWidth={3} />
          <line x1={d0x} y1={d0y + 50} x2={d1x} y2={d0y + 50} stroke={C.xray} strokeWidth={3} />
          {[d0x, d1x, d2x].map((x, i) => <line key={i} x1={x} y1={d0y - 12} x2={x} y2={d0y + 62} stroke={C.dim} strokeWidth={2} />)}
          <text x={(d0x + d2x) / 2} y={d0y - 14} fill={C.warm} fontSize={28} fontFamily={FONT} textAnchor="middle">SDD = 1000 mm</text>
          <text x={(d0x + d1x) / 2} y={d0y + 90} fill={C.xray} fontSize={28} fontFamily={FONT} textAnchor="middle">SOD = 700 mm</text>
        </svg>
      )}

      {/* comparison */}
      {ready && imgBox("data/v04/par_ap.png", 250, 230, 600, arAP, "מקבילי", window01(t, S.cmp1[0], S.cmp1[1] + 0.4, 0.6, 0.6))}
      {ready && imgBox("data/v04/persp_ap.png", 1670 - 600 * arAP, 230, 600, arAP, "פרספקטיבי", window01(t, S.cmp1[0] + 0.6, S.cmp1[1] + 0.4, 0.6, 0.6), C.warm)}
      <Title text="מקבילי מול פרספקטיבי" sub="DRR אמיתי של אותו CT, מחושב ב-NumPy" t={t} a={c1a} b={S.cmp1[1] + 0.3} size={56} top={40} />
      {ready && <Toggle t={t} a={S.cmp2[0]} b={S.cmp2[1] + 0.4} view="ap" d={d} x={960 - 330 * arAP} y={180} h={660} title="AP"
        marks={[{ key: "pubis", label: "עצם החיק", color: C.warm }, { key: "asis_r", label: "עצם הכסל", color: C.warm }, { key: "sacrum_post", label: "העצה", color: C.xray }]} />}
      {ready && <Zoom t={t} a={S.cmp2[0] + 9} b={S.cmp2[1] + 0.4} sync={S.cmp2[0]} view="ap" d={d} mark="pubis" x={1560} y={560} label="עצם החיק ×5" color={C.warm} />}
      {ready && <Zoom t={t} a={S.cmp2[0] + 5} b={S.cmp2[1] + 0.4} sync={S.cmp2[0]} view="ap" d={d} mark="asis_r" x={1560} y={200} label="קצה קדמי של עצם הכסל ×5" color={C.warm} />}
      {ready && <Toggle t={t} a={S.cmp3[0]} b={S.cmp3[1] + 0.4} view="lat" d={d} x={960 - 330 * arLAT} y={180} h={660} title="LAT"
        marks={[{ key: "femhead_right", label: "ימין", color: C.warm }, { key: "femhead_left", label: "שמאל", color: C.xray }]} />}
      {ready && <Zoom t={t} a={S.cmp3[0] + 2.5} b={S.cmp3[1] + 0.4} sync={S.cmp3[0]} view="lat" d={d} mark="femhead_right" x={1300} y={200} crop={70} label={`ראש הירך הימני (קרוב): ×${(d.sod_mm / (d.sod_mm - d.landmarks_scene_mm.femhead_right[0])).toFixed(2)}`} color={C.warm} />}
      {ready && <Zoom t={t} a={S.cmp3[0] + 2.5} b={S.cmp3[1] + 0.4} sync={S.cmp3[0]} view="lat" d={d} mark="femhead_left" x={1300} y={590} crop={70} label={`ראש הירך השמאלי (רחוק): ×${(d.sod_mm / (d.sod_mm - d.landmarks_scene_mm.femhead_left[0])).toFixed(2)}`} color={C.xray} />}
      {ready && imgBox("data/v04/overlay_ap.png", 220, 220, 560, arAP, "AP: כחול = מקבילי · כתום = פרספקטיבי", window01(t, S.cmp4[0], c4b + 0.4, 0.6, 0.6), C.text)}
      {ready && imgBox("data/v04/overlay_lat.png", 1700 - 560 * arLAT, 220, 560, arLAT, "LAT", window01(t, S.cmp4[0], c4b + 0.4, 0.6, 0.6), C.text)}

      {/* DiffDRR */}
      <Title text="DiffDRR" sub="ספריית PyTorch פתוחה (MIT) ל-DRR פרספקטיבי" t={t} a={S.diff1[0]} b={S.diff3[1] + 0.3} size={64} top={60} />
      <Panel o={window01(t, S.diff1[0] + 1.0, S.diff2[1] + 0.4, 0.6, 0.6)} style={{ left: 140, top: 270, width: 980, direction: "ltr", padding: "30px 36px" }}>
        <div style={{ fontFamily: "monospace", fontSize: 30, lineHeight: 1.7, color: C.text, whiteSpace: "pre" }}>
          <span style={{ color: C.dim }}>from</span> diffdrr.drr <span style={{ color: C.dim }}>import</span> DRR{"\n"}
          drr = DRR(subject,{"\n"}
          {"          "}<span style={{ color: ramp(t, S.diff2[0] + 2.5, S.diff2[0] + 3) > 0.5 ? C.warm : C.text }}>sdd</span>=1020.0,{"\n"}
          {"          "}<span style={{ color: ramp(t, S.diff2[0] + 4.5, S.diff2[0] + 5) > 0.5 ? C.warm : C.text }}>height</span>=200,{"\n"}
          {"          "}<span style={{ color: ramp(t, S.diff2[0] + 6.0, S.diff2[0] + 6.5) > 0.5 ? C.warm : C.text }}>delx</span>=2.0){"\n"}
          img = drr(<span style={{ color: ramp(t, S.diff2[0] + 9.5, S.diff2[0] + 10) > 0.5 ? C.warm : C.text }}>rotations, translations</span>, ...)
        </div>
        <div style={{ fontSize: 22, color: C.dim, marginTop: 14, direction: "rtl" }}>שלד של השימוש, כמו בדוגמה בתיעוד; את החתימה המדויקת בודקים בגרסה שמתקינים</div>
      </Panel>
      <Panel o={window01(t, S.diff2[0] + 1.0, S.diff2[1] + 0.4, 0.6, 0.6)} style={{ left: 1170, top: 300, width: 620, padding: "28px 34px" }}>
        {[["sdd", "מרחק מקור-גלאי (מ״מ) = אורך המוקד"], ["height", "מספר הפיקסלים בגלאי"], ["delx", "גודל פיקסל (מ״מ)"], ["rot, trans", "התנוחה: [R|t]"]].map(([k, v], i) => (
          <div key={k} style={{ fontSize: 30, lineHeight: 1.6, opacity: ramp(t, S.diff2[0] + 2.5 + i * 2.2, S.diff2[0] + 3.0 + i * 2.2) }}>
            <span style={{ fontFamily: "monospace", color: C.warm, direction: "ltr", display: "inline-block" }}>{k}</span> · {v}
          </div>
        ))}
      </Panel>
      {(() => {
        const o = window01(t, S.diff3[0] + 0.3, S.diff3[1] + 0.3, 0.6, 0.6);
        if (o <= 0) return null;
        const back = ramp(t, S.diff3[0] + 4.5, S.diff3[0] + 6.0);
        const box = (x: number, label: string, c: string = C.line) => (
          <g><rect x={x} y={440} width={300} height={120} rx={18} fill="rgba(8,14,24,0.85)" stroke={c} strokeWidth={2.5} />
            <text x={x + 150} y={512} fill={C.text} fontSize={32} fontFamily={FONT} textAnchor="middle">{label}</text></g>
        );
        return (
          <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
            {box(240, "תנוחה / נפח", C.warm)}{box(810, "DRR")}{box(1380, "השוואה לצילום")}
            <line x1={540} y1={485} x2={800} y2={485} stroke={C.xray} strokeWidth={4} /><polygon points="800,475 820,485 800,495" fill={C.xray} />
            <line x1={1110} y1={485} x2={1370} y2={485} stroke={C.xray} strokeWidth={4} /><polygon points="1370,475 1390,485 1370,495" fill={C.xray} />
            <g opacity={back}>
              <line x1={1380} y1={530} x2={560} y2={530} stroke={C.warm} strokeWidth={4} strokeDasharray="14 10" /><polygon points="560,520 540,530 560,540" fill={C.warm} />
              <text x={960} y={610} fill={C.warm} fontSize={30} fontFamily={FONT} textAnchor="middle">נגזרת (gradient) חוזרת אחורה: אפשר לשפר את התנוחה או את הצורה</text>
            </g>
            <text x={960} y={720} fill={C.dim} fontSize={28} fontFamily={FONT} textAnchor="middle">נחזור לזה בסרטון 13</text>
          </svg>
        );
      })()}

      {/* intensity mapping and detector */}
      <Title text="מסכום הנחתה לגוון אפור" t={t} a={S.int1[0]} b={S.int2[1] + 0.3} size={56} top={50} />
      {ready && imgBox("data/hero/drr_ap_saturated.png", 330, 250, 560, 248 / 235, <span>כמעט הכול לבן: <span style={{ direction: "ltr", display: "inline-block" }}>1 − e^(−L)</span></span>, window01(t, S.int1[0] + 0.5, S.int1[1] + 0.4, 0.6, 0.6), C.bad)}
      {ready && imgBox("data/v04/par_ap.png", 1050, 250, 560, arAP, "L, בחלון", window01(t, S.int1[0] + 7, S.int1[1] + 0.4, 0.6, 0.6), C.good)}
      {ready && imgBox("data/v04/par_ap.png", 200, 280, 560, arAP, "L, בחלון", window01(t, S.int2[0], S.int2[1] + 0.4, 0.6, 0.6), C.good)}
      {ready && <Mapping t={t} d={d} />}
      <Title text="הגלאי: גודל וגודל פיקסל" t={t} a={S.int3[0]} b={S.int4[1] + 0.3} size={56} top={50} />
      {ready && (() => {
        const o = window01(t, S.int3[0] + 0.3, S.int3[1] + 0.4, 0.6, 0.6);
        const h = 420, w = h * arAP;
        const fov = window01(t, S.int3[0] + 1.5, S.int3[0] + 6.5, 0.5, 0.5);
        return o > 0 && (
          <div style={{ position: "absolute", inset: 0, opacity: o }}>
            {[["data/v04/par_ap.png", "1.5 mm"], ["data/v04/par_ap_6mm.png", "6 mm"], ["data/v04/par_ap_12mm.png", "12 mm"]].map(([src, lab], i) => (
              <div key={src} style={{ position: "absolute", left: 960 + (i - 1) * (w + 50) - w / 2, top: 300, opacity: ramp(t, S.int3[0] + 6 + i * 1.6, S.int3[0] + 6.6 + i * 1.6) + (i === 0 ? 1 : 0) }}>
                <Img src={staticFile(src)} style={{ width: w, height: h, borderRadius: 8, border: `1.5px solid ${C.line}`, imageRendering: "pixelated" }} />
                <Label x={w / 2} y={h + 40} text={<span style={{ direction: "ltr", display: "inline-block" }}>pixel = {lab}</span>} color={C.text} size={30} />
                {i === 0 && <div style={{ position: "absolute", left: w * 0.19, top: h * 0.2, width: w * 0.62, height: h * 0.62, border: `3px dashed ${C.warm}`, opacity: fov }} />}
                {i === 0 && <Label x={w / 2} y={-30} text="גלאי קטן יותר = שדה ראייה קטן יותר" color={C.warm} size={26} o={fov} />}
              </div>
            ))}
          </div>
        );
      })()}
      <Eq tex={"\\Delta_{\\text{body}} = \\dfrac{\\hl{det}{\\Delta_{\\text{det}}}}{\\hl{M}{M}} = \\dfrac{2}{1.43} \\approx 1.4\\ \\text{mm}"} t={t} a={S.int4[0] + 0.3} b={S.int4[1] + 0.3}
        style={{ left: 460, top: 360, width: 1000 }} size={60}
        terms={{ det: [S.int4[0] + 4, S.int4[1]], M: [S.int4[0] + 5.5, S.int4[1]] }} />

      {/* the gap to real X-rays (named only; real frames appear in video 13) */}
      <Panel o={window01(t, S.gap1[0], S.gap1[1] + 0.4, 0.6, 0.6)} style={{ left: 460, right: 460, top: 230 }}>
        <div style={{ fontSize: 46, fontWeight: 700, color: C.xray, marginBottom: 20 }}>DRR ≠ צילום אמיתי</div>
        {["רעש בגלאי", "קרינה מפוזרת (scatter)", "מקור עם טווח של אנרגיות, לא אנרגיה אחת"].map((s, i) => (
          <div key={s} style={{ fontSize: 36, lineHeight: 1.6, opacity: ramp(t, S.gap1[0] + 2.5 + i * 1.3, S.gap1[0] + 3.0 + i * 1.3) }}><span style={{ color: C.warm }}>● </span>{s}</div>
        ))}
        <div style={{ fontSize: 30, color: C.dim, marginTop: 20, opacity: ramp(t, S.gap1[0] + 8, S.gap1[0] + 9) }}>את הפער נמדוד בסרטון 13, על צילומים אמיתיים</div>
      </Panel>

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "DRR מקבילי: סכום של μ לאורך ציר אחד",
        "DRR פרספקטיבי: קרן ממקור נקודתי לכל פיקסל, ודגימה לאורכה: trilinear או Siddon",
        "פרספקטיבה: מה שקרוב למקור גדל",
        "DiffDRR: מהיר על GPU, ודיפרנציאבילי",
        "תצוגה: חלון על L · גלאי: גודל וגודל פיקסל",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/01-data-and-drr/notebook.ipynb" todos={[
        <span key="a">לכתוב את <span style={{ fontFamily: "monospace", direction: "ltr", display: "inline-block" }}>parallel_drr(volume, axis)</span>: המרה ל-μ וסכום לאורך ציר</span>,
        "להפיק ממנה צילום AP וצילום LAT",
        "ב-Colab: להתקין DiffDRR ולהפיק זוג AP+LAT ראשון",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v04" />
    </AbsoluteFill>
  );
};
