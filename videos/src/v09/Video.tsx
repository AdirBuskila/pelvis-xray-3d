// Video 9 · Convolution from scratch. Mostly 2D diagrams in the DOM layer; one 3D scene (3D convolution
// sliding through the hero occupancy grid). All numbers come from prep/v09_assets.py (real convolutions on our DRR).
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v09/durations.json";
import conv from "../../public/data/v09/conv.json";
import { buildTimeline, Script } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson } from "../lib/assets";
import { CameraPath, CamKey, Stage } from "../lib/Stage";
import { Voxels, WireBox } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Arrow, Block, CellGrid, Frame, Illustrative, Svg, T, Tile, fmt, gray, signed } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
type Occ = { n: number; cell: number; count: number; centers: number[] };

const P = conv.patch as number[][];
const K = conv.kernel as number[][];
const V = conv.valid as number[][];
const FLAT = conv.flat as number[][];
const kCol = (w: number, a = 0.85) => (w > 0 ? `rgba(255,170,80,${a})` : w < 0 ? `rgba(70,150,255,${a})` : `rgba(60,70,90,${a})`);

// ---------------- section A: the kernel sliding over the zoomed patch ----------------
const GX = 520, GY = 200, GS = 86;           // input grid
const KX = 1215, KY = 215, KS = 66;          // kernel
const OX = 1290, OY = 600, OS = 58;          // output grid
const TH = { x: 70, y: 250, w: 380 };        // DRR thumbnail
const thScale = TH.w / conv.image_hw[1];

function ConvDemo({ t }: { t: number }) {
  const [a0] = S.pix, [, b1] = S.sobel;
  const o = window01(t, a0, b1 + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const kOn = ramp(t, S.kernel[0] + 0.3, S.kernel[0] + 1.0);
  const prodOn = window01(t, S.mult[0] + 0.2, S.eq[0] + 0.2, 0.5, 0.4);
  const eqDim = window01(t, S.eq[0], S.eq[1] + 0.2, 0.4, 0.4);
  const sob = window01(t, S.sobel[0] + 0.2, b1 + 0.5, 0.5, 0.4);
  // kernel position (output index): first (0,2), then a raster sweep during "slide", then the strongest edge
  const sweep = ramp(t, S.slide[0] + 1.2, S.slide[1] - 0.3, (x) => x);
  const step = Math.min(24, Math.floor(sweep * 25));
  let pos: [number, number] = [0, 2];
  if (t >= S.slide[0] + 1.2) pos = [Math.floor(step / 5), step % 5];
  if (t >= S.sobel[0]) pos = [2, 3];
  const filled = (i: number, j: number) => (i === 0 && j === 2 && t >= S.mult[0] + 3.2) || (t >= S.slide[0] + 1.2 && i * 5 + j <= step) || t >= S.slide[1];
  const [pi, pj] = pos;
  const pv = (i: number, j: number) => P[pi + i][pj + j];
  const sum = V[pi][pj];
  const mx = TH.x + (conv.patch_center_yx[1] - 3) * thScale, my = TH.y + (conv.patch_center_yx[0] - 3) * thScale;
  const fxm = TH.x + (conv.flat_center_yx[1] - 1) * thScale, fym = TH.y + (conv.flat_center_yx[0] - 1) * thScale;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Tile src="data/hero/drr_ap.png" x={TH.x} y={TH.y} w={TH.w} label="DRR · AP" sub={`${conv.image_hw[0]} × ${conv.image_hw[1]}`} />
      <Frame x={mx - 6} y={my - 6} w={7 * thScale + 12} h={7 * thScale + 12} color={C.warm} width={3} />
      <Svg>
        <line x1={mx + 7 * thScale + 6} y1={my - 6} x2={GX} y2={GY} stroke={C.warm} strokeWidth={1.5} opacity={0.6} />
        <line x1={mx + 7 * thScale + 6} y1={my + 7 * thScale + 6} x2={GX} y2={GY + 7 * GS} stroke={C.warm} strokeWidth={1.5} opacity={0.6} />
      </Svg>
      {/* input patch */}
      <CellGrid x={GX} y={GY} rows={7} cols={7} size={GS} cell={(i, j) => ({ bg: gray(Math.round(P[i][j] * 0.72)), text: P[i][j] })} />
      {/* kernel weights on the covered cells */}
      {kOn > 0 && (
        <>
          <Frame x={GX + pj * GS - 3} y={GY + pi * GS - 3} w={3 * GS + 6} h={3 * GS + 6} o={kOn} width={5} />
          <CellGrid x={GX + pj * GS} y={GY + pi * GS} rows={3} cols={3} size={GS} o={kOn} cell={(i, j) => ({
            bg: "transparent", text: <span style={{ position: "absolute", top: 2, right: 6, fontSize: 21, color: kCol(K[i][j], 1) }}>×{K[i][j]}</span>,
          })} />
        </>
      )}
      {/* kernel itself */}
      <div style={{ opacity: kOn * (1 - eqDim) }}>
        <T x={KX + 1.5 * KS} y={KY - 34} size={28} color={C.xray}>kernel · Sobel x</T>
        <CellGrid x={KX} y={KY} rows={3} cols={3} size={KS} cell={(i, j) => ({ bg: kCol(K[i][j]), text: K[i][j], color: "white" })} font={30} />
      </div>
      {/* the nine products and their sum */}
      {prodOn * (1 - eqDim) > 0 && (
        <div style={{ position: "absolute", left: 1460, top: 200, width: 420, opacity: prodOn * (1 - eqDim), fontFamily: FONT, direction: "ltr" }}>
          {[0, 1, 2].map((i) => (
            <div key={i} style={{ display: "flex", gap: 8, fontSize: 25, color: C.text, marginBottom: 8 }}>
              {[0, 1, 2].map((j) => (
                <span key={j} style={{ width: 132, color: K[i][j] === 0 ? C.dim : K[i][j] > 0 ? "#ffb870" : "#7fb6ff" }}>
                  {K[i][j] < 0 ? `(${K[i][j]})` : K[i][j]}·{pv(i, j)}</span>
              ))}
            </div>
          ))}
          <div style={{ fontSize: 36, marginTop: 10, color: C.warm, fontWeight: 700 }}>Σ = {sum}</div>
        </div>
      )}
      {/* output 5x5 */}
      <div style={{ opacity: ramp(t, S.mult[0] + 1.5, S.mult[0] + 2.3) }}>
        <T x={OX + 2.5 * OS} y={OY - 32} size={26} color={C.dim} rtl>פלט · 5×5</T>
        <CellGrid x={OX} y={OY} rows={5} cols={5} size={OS} font={20} cell={(i, j) => filled(i, j)
          ? { bg: signed(V[i][j], 320), text: V[i][j], border: i === pi && j === pj ? `3px solid ${C.warm}` : undefined }
          : { bg: "rgba(20,30,45,0.6)", border: i === pi && j === pj && kOn ? `3px solid ${C.warm}` : undefined }} />
      </div>
      {/* sobel: compare with a flat soft-tissue patch */}
      {sob > 0 && (
        <>
          <Frame x={fxm - 6} y={fym - 6} w={3 * thScale + 12} h={3 * thScale + 12} color={C.good} width={3} o={sob} />
          <div style={{ position: "absolute", left: 1460, top: 200, opacity: sob, fontFamily: FONT }}>
            <T x={150} y={0} size={26} color={C.good} rtl>רקמה רכה (אזור אחיד)</T>
            <CellGrid x={60} y={30} rows={3} cols={3} size={60} font={22} cell={(i, j) => ({ bg: gray(Math.round(FLAT[i][j] * 0.72)), text: FLAT[i][j] })} />
            <T x={150} y={250} size={34} color={C.good} weight={700}>Σ = {conv.flat_out}</T>
            <T x={150} y={310} size={26} color={C.warm} rtl>{`ליד הקו הבהיר: Σ = ${V[2][3]}`}</T>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------- the four real feature maps ----------------
const MAPS = [
  { f: "fm_sobel_x.png", name: "Sobel x", he: "קצוות אנכיים", k: conv.kernels.sobel_x as number[][] },
  { f: "fm_sobel_y.png", name: "Sobel y", he: "קצוות אופקיים", k: conv.kernels.sobel_y as number[][] },
  { f: "fm_blur.png", name: "blur", he: "טשטוש", k: [[1, 2, 1], [2, 4, 2], [1, 2, 1]], div: "/16" },
  { f: "fm_sharpen.png", name: "sharpen", he: "חידוד", k: conv.kernels.sharpen as number[][] },
];
function MiniKernel({ x, y, k, size = 34, div }: { x: number; y: number; k: number[][]; size?: number; div?: string }) {
  return (
    <>
      <CellGrid x={x} y={y} rows={3} cols={3} size={size} font={size * 0.45} cell={(i, j) => ({ bg: kCol(k[i][j], 0.8), text: k[i][j] })} />
      {div && <T x={x + 3 * size + 34} y={y + 1.5 * size} size={24} color={C.dim}>{div}</T>}
    </>
  );
}
function FeatureMaps({ t }: { t: number }) {
  const o = window01(t, S.maps[0] - 0.2, S.learned[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const dim = 1 - 0.75 * ramp(t, S.learned[0] + 0.2, S.learned[0] + 1.0);
  const w = 400, h = w * conv.image_hw[0] / conv.image_hw[1];
  const d = S.maps[1] - S.maps[0];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <div style={{ opacity: dim }}>
        {MAPS.map((m, i) => {
          const k = ramp(t, S.maps[0] + 0.3 + i * d * 0.22, S.maps[0] + 0.9 + i * d * 0.22);
          const x = 115 + i * (w + 30);
          return (
            <div key={m.f} style={{ opacity: k }}>
              <Tile src={`data/v09/${m.f}`} x={x} y={200} w={w} label={m.he} sub={m.name} />
              <MiniKernel x={x + w / 2 - 51} y={200 + h + 100} k={m.k} div={m.div} />
            </div>
          );
        })}
        <T x={960} y={850} size={22} color={C.dim} rtl>Sobel: כתום = חיובי, כחול = שלילי (מנורמל לתצוגה)</T>
      </div>
      <LearnedKernel t={t} />
    </div>
  );
}
function LearnedKernel({ t }: { t: number }) {
  const o = window01(t, S.learned[0] + 0.3, S.learned[1] + 0.4, 0.6, 0.5);
  if (o <= 0) return null;
  const k = ramp(t, S.learned[0] + 1.5, S.learned[1] - 1.0);
  const r = (i: number) => { const s = Math.sin(i * 12.9898 + 4.1) * 43758.5453; return ((s - Math.floor(s)) - 0.5) * 4; };
  const Ks = conv.kernels.sobel_x as number[][];
  const val = (i: number, j: number) => lerp(r(i * 3 + j), Ks[i][j], k);
  return (
    <Panel o={o} style={{ left: 560, right: 560, top: 250, textAlign: "center" }}>
      <div style={{ fontSize: 36, color: C.xray, marginBottom: 20 }}>המשקלות = פרמטרים נלמדים</div>
      <div style={{ position: "relative", height: 260 }}>
        <CellGrid x={(800 - 80 - 3 * 80) / 2} y={0} rows={3} cols={3} size={80} font={28}
          cell={(i, j) => ({ bg: signed(val(i, j), 2.2), text: (Math.abs(val(i, j)) < 0.05 ? 0 : val(i, j)).toFixed(1) })} />
      </div>
      <div style={{ fontSize: 28, color: C.dim }}>{k < 0.98 ? "האימון מעדכן אותם צעד אחר צעד…" : "…עד שהם עושים משהו שימושי"}</div>
      <div style={{ position: "relative", height: 56 }}><Illustrative x={240} y={14} o={1} /></div>
    </Panel>
  );
}

// ---------------- size / padding / stride ----------------
function SizeDemo({ t }: { t: number }) {
  const o = window01(t, S.size[0] - 0.2, S.stride[1] + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const p = t >= S.pad[0] + 0.5 ? 1 : 0;
  const s = t >= S.stride[0] ? 2 : 1;
  const n = 7, N = n + 2 * p, out = Math.floor((n + 2 * p - 3) / s) + 1;
  const cs = 64, x0 = 200 + (p ? 0 : cs), y0 = 220 + (p ? 0 : cs);
  const seg = t >= S.stride[0] ? S.stride : t >= S.pad[0] ? S.pad : S.size;
  const prog = ramp(t, seg[0] + 0.8, seg[1] - 0.2, (x) => x);
  const step = Math.min(out * out - 1, Math.floor(prog * out * out));
  const [oi, oj] = [Math.floor(step / out), step % out];
  const os = 52, ox = 1060, oy = 300;
  const padArt = window01(t, S.pad[0] + 6.5, S.pad[1] + 0.4, 0.6, 0.4);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <T x={200 + 4.5 * cs} y={185} size={28} color={C.dim} rtl>{`קלט ${n}×${n}` + (p ? " + padding 1" : "")}</T>
      <CellGrid x={x0} y={y0} rows={N} cols={N} size={cs} font={22} cell={(i, j) => {
        const pad = p && (i === 0 || j === 0 || i === N - 1 || j === N - 1);
        return pad ? { bg: "rgba(40,30,10,0.7)", text: "0", color: C.warm, border: `1px dashed ${C.warm}` } : { bg: "rgba(70,95,130,0.75)" };
      }} />
      <Frame x={x0 + oj * s * cs - 3} y={y0 + oi * s * cs - 3} w={3 * cs + 6} h={3 * cs + 6} width={5} />
      <Svg>
        <Arrow x1={200 + 9 * cs + 30} y1={220 + 4.5 * cs} x2={ox - 30} y2={oy + out * os / 2} color={C.dim} />
      </Svg>
      <T x={ox + out * os / 2} y={oy - 36} size={30} color={C.warm}>{`פלט ${out}×${out}`}</T>
      <CellGrid x={ox} y={oy} rows={out} cols={out} size={os} cell={(i, j) => ({
        bg: i * out + j <= step ? "rgba(255,207,127,0.55)" : "rgba(20,30,45,0.7)",
        border: i === oi && j === oj ? `3px solid ${C.warm}` : undefined })} />
      <T x={ox + out * os / 2} y={oy + out * os + 40} size={30} color={C.text}>{`k = 3 · p = ${p} · s = ${s}`}</T>
      {padArt > 0 && (
        <div style={{ opacity: padArt }}>
          <Tile src="data/v09/fm_sobel_y.png" x={1500} y={330} w={340} label="Sobel y: קצה מזויף בשפה" labelColor={C.warm} />
          <Frame x={1492} y={322} w={356} h={26} color={C.warm} width={3} />
        </div>
      )}
    </div>
  );
}
function SizeFormula({ t }: { t: number }) {
  const [a, b] = S.formula;
  const o = window01(t, a - 0.1, b + 0.6, 0.6, 0.6);
  if (o <= 0) return null;
  const d = b - a;
  const rows: [string, number][] = [["7, 3, 0, 1", 5], ["7, 3, 1, 1", 7], ["7, 3, 1, 2", 4], ["256, 3, 1, 1", 256], ["256, 3, 1, 2", 128]];
  return (
    <>
      <Eq tex={"n_{out} = \\left\\lfloor \\frac{\\hl{n}{n} + 2\\hl{p}{p} - \\hl{k}{k}}{\\hl{s}{s}} \\right\\rfloor + 1"} t={t} a={a} b={b + 0.6}
        size={70} style={{ left: 460, width: 1000, top: 180 }}
        terms={{ n: [a + 0.2, a + d * 0.2], p: [a + d * 0.14, a + d * 0.3], k: [a + d * 0.26, a + d * 0.4], s: [a + d * 0.36, a + d * 0.5] }} />
      <div style={{ position: "absolute", left: 660, top: 560, width: 640, opacity: o, fontFamily: FONT, direction: "ltr" }}>
        <div style={{ display: "flex", fontSize: 26, color: C.dim, marginBottom: 6 }}>
          <span style={{ width: 380 }}>n, k, p, s</span><span>n_out</span></div>
        {rows.map(([r, v], i) => {
          const at = i < 3 ? a + d * 0.5 + i * 0.25 : a + d * (0.62 + 0.1 * (i - 3));
          const hi = i === 4 && t > a + d * 0.75;
          return (
            <div key={r} style={{ display: "flex", fontSize: 32, opacity: ramp(t, at, at + 0.4), color: hi ? C.warm : C.text, marginBottom: 4 }}>
              <span style={{ width: 380 }}>{r}</span><span>→ {v}</span></div>
          );
        })}
      </div>
    </>
  );
}

// ---------------- MLP vs CNN ----------------
function ParamCompare({ t }: { t: number }) {
  const o = window01(t, S.mlp[0] - 0.2, S.why[1] + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const dm = S.mlp[1] - S.mlp[0];
  const fcOn = ramp(t, S.mlp[0] + 0.4, S.mlp[0] + 2.0);
  const fcNum = ramp(t, S.mlp[0] + dm * 0.45, S.mlp[0] + dm * 0.6);
  const cnOn = ramp(t, S.cnn[0] + 0.2, S.cnn[0] + 1.4);
  const why = S.why, wd = why[1] - why[0];
  const loc = window01(t, why[0] + wd * 0.05, why[1] + 0.5, 0.5, 0.4);
  const shr = window01(t, why[0] + wd * 0.5, why[1] + 0.5, 0.5, 0.4);
  const n = 10, ys = (i: number) => 300 + i * 50;
  const P9 = conv.params;
  const wc = ["#7fb6ff", "#7dffb0", "#ffb870"];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Svg o={fcOn}>
        {Array.from({ length: n * n }, (_, k) => {
          const i = Math.floor(k / n), j = k % n;
          return <line key={k} x1={260} y1={ys(i)} x2={700} y2={ys(j)} stroke={C.xray} strokeWidth={1} opacity={0.35} />;
        })}
      </Svg>
      <Svg o={cnOn}>
        {Array.from({ length: n }, (_, j) => [-1, 0, 1].map((dd) => {
          const i = j + dd;
          if (i < 0 || i >= n) return null;
          return <line key={`${j}${dd}`} x1={1220} y1={ys(i)} x2={1660} y2={ys(j)} stroke={wc[dd + 1]} strokeWidth={3 + 2 * shr} opacity={0.85} />;
        }))}
      </Svg>
      {[260, 700].map((x) => Array.from({ length: n }, (_, i) => (
        <div key={`a${x}${i}`} style={{ position: "absolute", left: x - 12, top: ys(i) - 12, width: 24, height: 24, borderRadius: 12,
          background: C.xray, opacity: fcOn, boxShadow: `0 0 12px ${C.xray}` }} />)))}
      {[1220, 1660].map((x) => Array.from({ length: n }, (_, i) => (
        <div key={`b${x}${i}`} style={{ position: "absolute", left: x - 12, top: ys(i) - 12, width: 24, height: 24, borderRadius: 12,
          background: C.warm, opacity: cnOn, boxShadow: `0 0 12px ${C.warm}` }} />)))}
      <T x={480} y={240} size={34} color={C.xray} rtl o={fcOn}>שכבה מלאה (fully connected)</T>
      <T x={260} y={810} size={22} color={C.dim} o={fcOn}>{`${fmt(P9.n_in)} in`}</T>
      <T x={700} y={810} size={22} color={C.dim} o={fcOn}>{`${fmt(P9.n_in)} out`}</T>
      <T x={480} y={862} size={46} color={C.text} weight={700} o={fcNum}>{fmt(P9.fc_total)}</T>
      <T x={480} y={912} size={26} color={C.dim} rtl o={fcNum}>{`פרמטרים · ${P9.fc_gb_float32.toFixed(1)} GB ב-float32`}</T>
      <T x={1440} y={240} size={34} color={C.warm} rtl o={cnOn * (1 - loc)}>קונבולוציה 3×3</T>
      <T x={1440} y={862} size={46} color={C.text} weight={700} o={cnOn}>{`${P9.conv_1k}`}</T>
      <T x={1440} y={912} size={26} color={C.dim} rtl o={ramp(t, S.cnn[0] + (S.cnn[1] - S.cnn[0]) * 0.6, S.cnn[1])}>{`פרמטרים · עם 64 kernels: ${P9.conv_64k}`}</T>
      <T x={1440} y={810} size={22} color={C.dim} rtl o={cnOn}>חתך חד-ממדי, להמחשה</T>
      <Label x={1440} y={240} text="מקומיות: כל פלט רואה רק 3 שכנים" color={C.good} size={30} o={loc} />
      <Label x={1440} y={190} text="שיתוף משקלות: אותם 3 צבעים בכל מקום" color="#ffb870" size={30} o={shr} />
    </div>
  );
}

// ---------------- channels ----------------
function Channels({ t }: { t: number }) {
  const o = window01(t, S.ch1[0] - 0.2, S.ch2[1] + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const d1 = S.ch1[1] - S.ch1[0];
  const two = ramp(t, S.ch2[0] + 0.2, S.ch2[0] + 1.4);
  const w = 300, h = w * conv.image_hw[0] / conv.image_hw[1];
  const stackX = lerp(820, 300, two), stackY = 330;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <div style={{ opacity: 1 - two }}>
        <Tile src="data/hero/drr_ap.png" x={90} y={360} w={260} label="קלט: ערוץ אחד" sub="1 × 235 × 248" />
        {MAPS.map((m, i) => <MiniKernel key={i} x={480} y={230 + i * 140} k={m.k} size={30} />)}
        <Svg>
          {MAPS.map((_, i) => <Arrow key={i} x1={370} y1={480} x2={466} y2={275 + i * 140} color={C.dim} w={2} />)}
          {MAPS.map((_, i) => <Arrow key={`o${i}`} x1={590} y1={275 + i * 140} x2={800} y2={470} color={C.dim} w={2} />)}
        </Svg>
      </div>
      {MAPS.map((m, i) => {
        const k = ramp(t, S.ch1[0] + 0.4 + i * d1 * 0.15, S.ch1[0] + 1.0 + i * d1 * 0.15);
        const j = 3 - i;
        return <Tile key={m.f} src={`data/v09/${m.f}`} x={stackX + j * 48} y={stackY - j * 40} w={w} o={k}
          style={{ boxShadow: "0 0 30px rgba(0,0,0,0.9)" }} />;
      })}
      <T x={stackX + w / 2 + 72} y={stackY + h + 50} size={30} color={C.warm}>4 × 235 × 248</T>
      <T x={stackX + w / 2 + 72} y={stackY + h + 90} size={24} color={C.dim} rtl>ערוצים × גובה × רוחב</T>
      {two > 0 && (
        <div style={{ opacity: two }}>
          {[3, 2, 1, 0].map((j) => (
            <div key={j} style={{ position: "absolute", left: 900 + j * 26, top: 380 - j * 22, width: 120, height: 120,
              border: `2px solid ${C.xray}`, background: "rgba(40,90,150,0.85)", borderRadius: 6,
              backgroundImage: "linear-gradient(rgba(127,208,255,0.5) 2px, transparent 2px), linear-gradient(90deg, rgba(127,208,255,0.5) 2px, transparent 2px)",
              backgroundSize: "40px 40px" }} />
          ))}
          <T x={1010} y={560} size={28} color={C.xray}>kernel: 4 × 3 × 3</T>
          <T x={1010} y={600} size={24} color={C.dim} rtl>סוכם על כל הערוצים</T>
          <Svg>
            <Arrow x1={800} y1={440} x2={885} y2={440} color={C.dim} />
            <Arrow x1={1110} y1={440} x2={1300} y2={440} color={C.dim} />
          </Svg>
          <div style={{ position: "absolute", left: 1320, top: 330, width: 250, height: 236, border: `2px dashed ${C.warm}`, borderRadius: 8,
            display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FONT, fontSize: 28, color: C.warm, direction: "rtl" }}>ערוץ פלט אחד</div>
          <T x={1445} y={600} size={24} color={C.dim} rtl>עוד kernel ← עוד ערוץ פלט</T>
          <Eq tex={"\\#\\text{params} = C_{out}\\,(C_{in}\\cdot 9 + 1) \\;\\Rightarrow\\; 32\\,(16\\cdot 9+1) = 4{,}640"} t={t}
            a={S.ch2[0] + (S.ch2[1] - S.ch2[0]) * 0.55} b={S.ch2[1] + 0.5} size={44} style={{ left: 330, width: 1260, top: 745, padding: "10px 30px" }} />
        </div>
      )}
    </div>
  );
}

// ---------------- receptive field ----------------
function RField({ t }: { t: number }) {
  const o = window01(t, S.rf1[0] - 0.2, S.rf2[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const d = S.rf1[1] - S.rf1[0];
  const lvl = t < S.rf1[0] + d * 0.3 ? 1 : t < S.rf1[0] + d * 0.68 ? 2 : 3;
  const n = 9, cs = 34, xs = [150, 590, 1030, 1470], y0 = 330;
  const names = ["קלט", "שכבה 1", "שכבה 2", "שכבה 3"];
  const half = (g: number) => (g > lvl ? -1 : lvl - g);
  const sq = (g: number) => { const h = half(g); return { x: xs[g] + (4 - h) * cs, y: y0 + (4 - h) * cs, s: (2 * h + 1) * cs }; };
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {xs.map((x, g) => (
        <div key={g}>
          <T x={x + n * cs / 2} y={y0 - 40} size={30} color={g === lvl ? C.warm : C.dim} rtl>{names[g]}</T>
          <CellGrid x={x} y={y0} rows={n} cols={n} size={cs} gap={2} cell={(i, j) => {
            const h = half(g);
            const on = h >= 0 && Math.abs(i - 4) <= h && Math.abs(j - 4) <= h;
            return { bg: on ? (g === lvl ? "rgba(255,207,127,0.9)" : "rgba(127,208,255,0.55)") : "rgba(30,45,65,0.7)" };
          }} />
          {half(g) >= 0 && g < lvl && <T x={x + n * cs / 2} y={y0 + n * cs + 40} size={30} color={C.text}>{`${2 * half(g) + 1}×${2 * half(g) + 1}`}</T>}
        </div>
      ))}
      <Svg>
        {[0, 1, 2].filter((g) => g + 1 <= lvl).map((g) => {
          const a = sq(g), b = sq(g + 1);
          return <g key={g} opacity={0.6}>
            <line x1={a.x + a.s} y1={a.y} x2={b.x} y2={b.y} stroke={C.xray} strokeWidth={2} />
            <line x1={a.x + a.s} y1={a.y + a.s} x2={b.x} y2={b.y + b.s} stroke={C.xray} strokeWidth={2} />
          </g>;
        })}
      </Svg>
      <T x={960} y={780} size={34} color={C.warm} o={ramp(t, S.rf2[0] + 0.5, S.rf2[0] + 1.2)}>receptive field: 3 → 5 → 7 → 9 → …  (+2 per layer)</T>
    </div>
  );
}

// ---------------- pooling ----------------
function Pooling({ t }: { t: number }) {
  const [a, b] = S.pool, d = b - a;
  const o = window01(t, a - 0.2, b + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const P4 = conv.pool_patch as number[][], O4 = conv.pool_out as number[][];
  const qc = ["rgba(127,208,255,", "rgba(255,207,127,", "rgba(125,255,176,", "rgba(255,127,138,"];
  const q = (i: number, j: number) => (i >> 1) * 2 + (j >> 1);
  const outOn = ramp(t, a + d * 0.18, a + d * 0.28);
  const tiles = window01(t, a + d * 0.25, a + d * 0.62, 0.5, 0.5);
  const rf = ramp(t, a + d * 0.66, a + d * 0.76);
  const isMax = (i: number, j: number) => P4[i][j] === O4[i >> 1][j >> 1];
  const plain = [3, 5, 7, 9, 11, 13], pooled = ["3", "5", "↓2", "10", "14", "↓2", "24", "32"];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <T x={290} y={225} size={28} color={C.dim}>max pool 2×2</T>
      <CellGrid x={110} y={260} rows={4} cols={4} size={90} font={28} cell={(i, j) => ({ bg: qc[q(i, j)] + "0.32)", text: P4[i][j],
        border: isMax(i, j) && outOn > 0.5 ? `3px solid ${C.warm}` : undefined })} />
      <Svg><Arrow x1={490} y1={440} x2={570} y2={440} color={C.dim} /></Svg>
      <CellGrid x={590} y={350} rows={2} cols={2} size={90} font={30} o={outOn} cell={(i, j) => ({ bg: qc[i * 2 + j] + "0.6)", text: O4[i][j] })} />
      <div style={{ opacity: tiles }}>
        <Tile src="data/hero/drr_ap.png" x={900} y={250} w={496} sub="235 × 248" />
        <Tile src="data/v09/pool_ap.png" x={1450} y={250} w={248} sub="117 × 124" pixelated />
      </div>
      <div style={{ position: "absolute", left: 880, top: 300, opacity: rf, fontFamily: FONT, direction: "ltr" }}>
        <div style={{ fontSize: 28, color: C.dim, direction: "rtl", marginBottom: 20 }}>receptive field אחרי כל שכבת 3×3 (בפיקסלי קלט)</div>
        <div style={{ display: "flex", gap: 18, alignItems: "center", marginBottom: 30 }}>
          {plain.map((v, i) => <span key={i} style={{ width: 62, fontSize: 36, color: C.text, textAlign: "center" }}>{v}</span>)}
          <span style={{ width: 220, fontSize: 26, color: C.dim, direction: "rtl" }}>רק 3×3</span>
        </div>
        <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
          {pooled.map((v, i) => <span key={i} style={{ width: v.startsWith("↓") ? 40 : 62, fontSize: v.startsWith("↓") ? 22 : 36,
            color: v.startsWith("↓") ? C.dim : C.warm, textAlign: "center" }}>{v}</span>)}
          <span style={{ width: 220, fontSize: 26, color: C.dim, direction: "rtl" }}>עם הקטנה (↓2)</span>
        </div>
      </div>
    </div>
  );
}

/** the encoder pattern, using the same design as video 10 */
function EncoderSketch({ t }: { t: number }) {
  const [a, b] = S.pool2;
  const o = window01(t, a - 0.1, b + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const L = [
    { c: 1, s: 128, lab: "1 × 128²" }, { c: 32, s: 128, lab: "32 × 128²" }, { c: 64, s: 64, lab: "64 × 64²" },
    { c: 128, s: 32, lab: "128 × 32²" }, { c: 256, s: 16, lab: "256 × 16²" },
  ];
  const ws = L.map((l) => 22 + Math.log2(l.c) * 16), gap = 170;
  let x = (1920 - ws.reduce((p, q) => p + q, 0) - gap * (L.length - 1)) / 2;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {L.map((l, i) => {
        const h = 60 + l.s * 3.2, w = ws[i];
        const k = ramp(t, a + 0.3 + i * 0.5, a + 0.8 + i * 0.5);
        const el = (
          <div key={i}>
            <Block x={x} y={540 - h / 2} w={w} h={h} o={k} label={l.lab} color={i === 0 ? C.dim : C.xray} />
            {i > 0 && <T x={x - gap / 2} y={540} size={24} color={C.dim} o={k}>{i === 1 ? "conv ×2 →" : "↓2, conv ×2 →"}</T>}
          </div>
        );
        x += w + gap;
        return el;
      })}
      <T x={960} y={860} size={28} color={C.dim} rtl>ממדים: ערוצים × גובה × רוחב</T>
    </div>
  );
}

// ---------------- 3D convolution scene ----------------
function Conv3DScene({ t, occ }: { t: number; occ: Occ }) {
  const cell = occ.cell;
  const { c0, set, lo, hi } = useMemo(() => {
    const c0 = [occ.centers[0], occ.centers[1], occ.centers[2]];
    const n = occ.centers.length / 3;
    const set = new Set<string>();
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < n; i++) {
      const v = [0, 1, 2].map((a) => Math.round((occ.centers[3 * i + a] - c0[a]) / occ.cell));
      set.add(v.join(","));
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], v[a]); hi[a] = Math.max(hi[a], v[a]); }
    }
    return { c0, set, lo, hi };
  }, [occ]);
  const world = (v: number[]) => v.map((q, a) => c0[a] + q * cell) as [number, number, number];
  const zMid = Math.round((lo[2] + hi[2]) / 2), yMid = Math.round((lo[1] + hi[1]) / 2);
  const xs0 = lo[0] - 1, nx = hi[0] - lo[0] + 3;
  const rows = [3, 2, 1, 0, -1, -2, -3].map((dd) => yMid + dd);
  const [a1] = S.c3d1, [, b2] = S.c3d2;
  const sweep = ramp(t, a1 + 1.5, b2 - 0.5, (x) => x);
  const total = rows.length * nx;
  const step = Math.min(total - 1, Math.floor(sweep * total));
  const kpos = (s: number) => [xs0 + (s % nx), rows[Math.floor(s / nx)], zMid];
  const cur = kpos(step);
  const on = window01(t, a1 - 0.4, S.c3d3[1] + 0.5, 0.8, 0.6);
  const inside: number[][] = [];
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
    const v = [cur[0] + dx, cur[1] + dy, cur[2] + dz];
    if (set.has(v.join(","))) inside.push(v);
  }
  // output trail: fraction of occupied cells in each visited 3x3x3 window (a box-filter 3D convolution)
  const trail = useMemo(() => {
    const vals: number[] = [];
    for (let s = 0; s < total; s++) {
      const p = kpos(s); let c = 0;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++)
        if (set.has(`${p[0] + dx},${p[1] + dy},${p[2] + dz}`)) c++;
      vals.push(c / 27);
    }
    return vals;
  }, [set, total]);
  const trailMesh = useMemo(() => new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 1.2, 0.5), transparent: true, toneMapped: false }), total), [total]);
  const m4 = new THREE.Matrix4();
  let shown = 0;
  if (t > a1 + 1.5) for (let s = 0; s < step; s++) {
    if (trail[s] <= 0) continue;
    const sc = cell * 0.8 * Math.cbrt(trail[s]);
    const p = world(kpos(s));
    m4.makeScale(sc, sc, sc).setPosition(p[0], p[1], p[2]);
    trailMesh.setMatrixAt(shown++, m4);
  }
  trailMesh.count = shown;
  trailMesh.instanceMatrix.needsUpdate = true;
  (trailMesh.material as THREE.MeshBasicMaterial).opacity = on * 0.9;
  const pulled = ramp(t, S.c3d3[0], S.c3d3[0] + 2.5);
  const keys: CamKey[] = [
    { t: a1 - 0.5, pos: [2.6, 1.6, -5.6], look: [0.8, 0.1, 0] },
    { t: S.c3d2[0], pos: [3.4, 1.4, -5.0], look: [0.8, 0.1, 0] },
    { t: b2, pos: [1.6, 0.9, -5.4], look: [0.8, 0.1, 0] },
    { t: S.c3d3[1] + 1, pos: [3.4, 2.8, -7.8], look: [1.1, 0, 0] },
  ];
  const cube = occ.n * cell;
  const mid = world([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]);
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Voxels centers={occ.centers} cell={cell} opacity={on * lerp(0.2, 0.45, pulled)} gap={0.16} />
      <WireBox size={[cell * 3, cell * 3, cell * 3]} position={world(cur)} opacity={on * (1 - pulled)} color="#ffcf7f" />
      {inside.map((v, i) => (
        <mesh key={i} position={world(v)}>
          <boxGeometry args={[cell * 0.9, cell * 0.9, cell * 0.9]} />
          <meshBasicMaterial color={new THREE.Color(2.4, 1.7, 0.6)} transparent opacity={on * (1 - pulled) * 0.9} toneMapped={false} />
        </mesh>
      ))}
      <primitive object={trailMesh} />
      <WireBox size={[cube, cube, cube]} position={mid} opacity={on * 0.5} />
    </>
  );
}

export const V09 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const occ = useJson<Occ>("data/hero/occ32.json");
  const in3d = t > S.c3d1[0] - 1.5 && t < S.c3d3[1] + 1.2;
  const d = (id: string) => S[id][1] - S[id][0];
  const introK = ramp(t, S.intro2[0], S.intro2[1], (x) => x);
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {occ && in3d && (
        <Stage>
          <Conv3DScene t={t} occ={occ} />
        </Stage>
      )}

      <Opening n={9} title="קונבולוציה מאפס" active={[2]} t={t} end={2.9} />

      {/* intro: our two DRRs, a 3x3 window sliding over the AP */}
      <div style={{ opacity: window01(t, S.intro1[0], S.intro2[1] + 0.5, 0.8, 0.6) }}>
        <Tile src="data/hero/drr_ap.png" x={520} y={260} w={500} label="AP" />
        <Tile src="data/hero/drr_lat.png" x={1080} y={260} w={294} label="LAT" />
        {t > S.intro2[0] && (() => {
          const s = 500 / 248, k = Math.floor(introK * 400);
          const px = 20 + (k % 20) * 10.5, py = 30 + Math.floor(k / 20) * 9;
          return <Frame x={520 + px * s} y={260 + py * s} w={14 * s} h={14 * s} width={3} color={C.warm} radius={2} />;
        })()}
      </div>
      <Title text="kernel שזז על התמונה" t={t} a={S.pix[0]} b={S.sobel[1] + 0.4} top={60} size={54} />
      <ConvDemo t={t} />
      <Eq tex={"O[i,j] = \\sum_{u=-1}^{1}\\sum_{v=-1}^{1} \\hl{k}{K[u,v]}\\cdot \\hl{i}{I[i+u,\\,j+v]}"} t={t} a={S.eq[0]} b={S.eq[1] + 0.2}
        size={36} style={{ left: 1130, width: 760, top: 190, padding: "20px 20px" }}
        note="Conv2d ברשתות: בלי היפוך של ה-kernel"
        terms={{ k: [S.eq[0] + d("eq") * 0.22, S.eq[0] + d("eq") * 0.45], i: [S.eq[0] + d("eq") * 0.32, S.eq[0] + d("eq") * 0.5] }} />

      <Title text="ארבעה kernels על ה-DRR שלנו" t={t} a={S.maps[0]} b={S.learned[1] + 0.4} top={60} size={54} />
      <FeatureMaps t={t} />

      <Title text="גודל הפלט" t={t} a={S.size[0]} b={S.formula[1] + 0.5} top={60} size={54} />
      <SizeDemo t={t} />
      <SizeFormula t={t} />

      <Title text="שכבה מלאה מול קונבולוציה" sub="קלט: תמונה של 256 × 256" t={t} a={S.mlp[0]} b={S.why[1] + 0.5} top={20} size={50} />
      <ParamCompare t={t} />

      <Title text="ערוצים ו-feature maps" t={t} a={S.ch1[0]} b={S.ch2[1] + 0.5} top={60} size={54} />
      <Channels t={t} />

      <Title text="receptive field" t={t} a={S.rf1[0]} b={S.rf2[1] + 0.4} top={60} size={54} />
      <RField t={t} />
      <Title text="הקטנה: pooling או stride 2" t={t} a={S.pool[0]} b={S.pool[1] + 0.4} top={60} size={54} />
      <Pooling t={t} />
      <Title text="המבנה של encoder" t={t} a={S.pool2[0]} b={S.pool2[1] + 0.5} top={60} size={54} />
      <EncoderSketch t={t} />

      <Title text="קונבולוציה תלת-ממדית" sub="kernel של 3×3×3 על מפת התפוסה של האגן (32³)" t={t} a={S.c3d1[0]} b={S.c3d2[1] + 0.3} top={40} size={54} />
      <Panel o={window01(t, S.c3d2[0] + 0.5, S.c3d2[1] + 0.3)} style={{ left: 90, top: 260, width: 520, padding: "24px 30px" }}>
        <div style={{ fontSize: 30, color: C.warm, marginBottom: 10 }}>זהה ל-2D, עם אינדקס שלישי</div>
        <div style={{ fontSize: 26, color: C.text, direction: "ltr", textAlign: "left", lineHeight: 1.6 }}>
          27 weights per input channel<br />nn.Conv3d(C_in, C_out, 3, padding=1)</div>
        <div style={{ fontSize: 24, color: C.dim, marginTop: 12 }}>הקוביות הכתומות: הפלט של kernel ממוצע (חלק התאים התפוסים מתוך 27)</div>
      </Panel>
      <Panel o={window01(t, S.c3d3[0] + 0.3, S.c3d3[1] + 0.6)} style={{ left: 90, top: 260, width: 520, padding: "24px 30px" }}>
        <div style={{ fontSize: 32, color: C.text, direction: "ltr", textAlign: "left", lineHeight: 1.7 }}>
          64 × 64 = <b>{fmt(conv.params.pix64)}</b><br />64³ = <b style={{ color: C.warm }}>{fmt(conv.params.vox64)}</b></div>
        <div style={{ fontSize: 28, color: C.dim, marginTop: 8 }}>פי 64 תאים ← מעט ערוצים ברזולוציה המלאה</div>
      </Panel>

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "קונבולוציה: kernel קטן שזז, כופל ומחבר",
        "padding ו-stride קובעים את גודל הפלט: ⌊(n+2p−k)/s⌋+1",
        "מקומיות + שיתוף משקלות: 10 פרמטרים במקום 4 מיליארד",
        "שכבות והקטנות מגדילות את ה-receptive field",
        "בתלת-ממד: אותו דבר, עם kernel בצורת קובייה",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/04-cnn-unet/notebook.ipynb" todos={[
        "conv2d בעצמכם (לולאות / NumPy), והשוואה מספרית ל-nn.Conv2d",
        "תרגילי צורות: לחשב את גודל הפלט של כמה שכבות, ולבדוק בקוד",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v09" />
    </AbsoluteFill>
  );
};
