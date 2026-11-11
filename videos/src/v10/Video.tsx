// Video 10 · Encoder-decoder and U-Net: from 2D to 3D. 2D diagrams in the DOM layer; two 3D scenes
// (the 16^3 round trip of the hero occupancy, and lifting AP/LAT along their rays). Data: prep/v10_assets.py.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v10/durations.json";
import d10 from "../../public/data/v10/v10.json";
import { buildTimeline, Script } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Meta, Voxels, WireBox } from "../lib/xray";
import { Captions, Credit, Label, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Arrow, Block, CellGrid, Code, Svg, T, Tile } from "../v09/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
type Occ = { n: number; cell: number; count: number; centers: number[] };
const win = (t: number, a: string, b: string, fin = 0.6, fout = 0.6, pad = 0.4) => window01(t, S[a][0] - 0.1, S[b][1] + pad, fin, fout);
const frac = (id: string, f: number) => S[id][0] + f * (S[id][1] - S[id][0]);

/** an isometric-looking cube outline (SVG), front face at (x,y) size s */
function Cube({ x, y, s, o = 1, color = C.xray, label, sub, fill = "rgba(127,208,255,0.10)", glow = false }: {
  x: number; y: number; s: number; o?: number; color?: string; label?: string; sub?: string; fill?: string; glow?: boolean;
}) {
  if (o <= 0) return null;
  const d = s * 0.36;
  return (
    <div style={{ position: "absolute", left: 0, top: 0, opacity: o }}>
      <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, overflow: "visible",
        filter: glow ? `drop-shadow(0 0 14px ${color})` : undefined }}>
        <polygon points={`${x},${y} ${x + d},${y - d} ${x + s + d},${y - d} ${x + s},${y}`} fill={fill} stroke={color} strokeWidth={2} />
        <polygon points={`${x + s},${y} ${x + s + d},${y - d} ${x + s + d},${y + s - d} ${x + s},${y + s}`} fill={fill} stroke={color} strokeWidth={2} />
        <rect x={x} y={y} width={s} height={s} fill={fill} stroke={color} strokeWidth={2} />
      </svg>
      {label && <T x={x + s / 2 + d / 2} y={y + s + 26} size={22} color={C.text}>{label}</T>}
      {sub && <T x={x + s / 2 + d / 2} y={y + s + 54} size={20} color={C.dim} rtl>{sub}</T>}
    </div>
  );
}

// ---------------- intro ----------------
function Intro({ t }: { t: number }) {
  const o = win(t, "intro", "intro", 0.8, 0.6, 0.6);
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Tile src="data/v10/ap_128.png" x={300} y={260} w={220} label="AP" pixelated />
      <Tile src="data/v10/lat_128.png" x={300} y={580} w={220} label="LAT" pixelated />
      <Svg><Arrow x1={560} y1={530} x2={700} y2={530} /><Arrow x1={1180} y1={530} x2={1320} y2={530} /></Svg>
      <Block x={730} y={400} w={420} h={260} inner={<span style={{ fontSize: 40, direction: "rtl" }}>הרשת של היום</span>} />
      <Cube x={1360} y={430} s={200} color={C.warm} fill="rgba(255,207,127,0.12)" label="64 × 64 × 64" glow />
    </div>
  );
}

// ---------------- segmentation ----------------
function Segmentation({ t }: { t: number }) {
  const o = win(t, "seg1", "seg2");
  if (o <= 0) return null;
  const mask = ramp(t, frac("seg1", 0.45), frac("seg1", 0.7));
  const out = ramp(t, S.seg2[0] + 0.3, S.seg2[0] + 1.3);
  const w = 540;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <div style={{ position: "absolute", left: 280, top: 230, width: w }}>
        <Img src={staticFile("data/v10/seg_ct.png")} style={{ width: w, display: "block", borderRadius: 8, border: `1px solid ${C.line}` }} />
        <Img src={staticFile("data/v10/seg_mask.png")} style={{ width: w, position: "absolute", left: 0, top: 0, opacity: mask * (1 - 0.6 * out), borderRadius: 8 }} />
      </div>
      <T x={280 + w / 2} y={790} size={28} color={C.text} rtl>חתך CT קורונלי של האגן שלנו</T>
      <T x={280 + w / 2} y={830} size={24} color={C.warm} rtl o={mask}>כתום = עצם האגן (המסכה האמיתית)</T>
      <div style={{ opacity: out }}>
        <Svg><Arrow x1={850} y1={480} x2={1060} y2={480} color={C.xray} head="ahx" w={4} /></Svg>
        <T x={955} y={445} size={26} color={C.xray} rtl>רשת</T>
        <div style={{ position: "absolute", left: 1090, top: 230, width: w, background: "#000", borderRadius: 8, border: `1px solid ${C.line}` }}>
          <Img src={staticFile("data/v10/seg_mask.png")} style={{ width: w, display: "block" }} />
        </div>
        <T x={1090 + w / 2} y={790} size={28} color={C.text} rtl>פלט: הסתברות לעצם, לכל פיקסל</T>
        <T x={1090 + w / 2} y={830} size={24} color={C.dim}>{`${d10.seg_hw[0]} × ${d10.seg_hw[1]}  →  ${d10.seg_hw[0]} × ${d10.seg_hw[1]}`}</T>
      </div>
    </div>
  );
}

// ---------------- the 2D U-Net diagram ----------------
const LV = [128, 64, 32, 16], CH = [32, 64, 128, 256];
const LY = [330, 480, 610, 720];
const bh = (s: number) => Math.max(26, s * 1.45), bw = (c: number) => 14 + Math.log2(c) * 9;
function UNet2D({ t }: { t: number }) {
  const o = Math.max(win(t, "enc", "dec"), win(t, "skip2", "skip3"));
  if (o <= 0) return null;
  const encK = (i: number) => (t > S.skip2[0] - 0.5 ? 1 : ramp(t, S.enc[0] + 0.4 + i * 1.6, S.enc[0] + 1.0 + i * 1.6));
  const decK = (i: number) => (t > S.skip2[0] - 0.5 ? 1 : ramp(t, S.dec[0] + 0.5 + i * 1.4, S.dec[0] + 1.1 + i * 1.4));
  const bott = window01(t, S.bott[0], S.bott[1] + 0.3, 0.5, 0.5);
  const skip = ramp(t, S.skip2[0] + 0.8, S.skip2[0] + 2.2);
  const ex = [260, 440, 620, 820], dx = [1040, 1240, 1440];
  const enc = LV.map((s, i) => ({ x: ex[i], y: LY[i] - bh(s) / 2, w: bw(CH[i]), h: bh(s), s, c: CH[i] }));
  const dec = [2, 1, 0].map((l, i) => ({ x: dx[i], y: LY[l] - bh(LV[l]) / 2, w: bw(CH[l]), h: bh(LV[l]), s: LV[l], c: CH[l], l }));
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Tile src="data/v10/seg_ct.png" x={70} y={LY[0] - 80} w={160} sub="1 × 128²" />
      <div style={{ position: "absolute", left: 1640, top: LY[0] - 80, width: 170, opacity: decK(3) }}>
        <div style={{ background: "#000", borderRadius: 8, border: `1px solid ${C.line}` }}>
          <Img src={staticFile("data/v10/seg_mask.png")} style={{ width: 170, display: "block" }} /></div>
        <div style={{ textAlign: "center", fontSize: 24, color: C.dim, marginTop: 6, fontFamily: FONT }}>1 × 128²</div>
      </div>
      <Svg>
        {enc.slice(0, 3).map((b, i) => <Arrow key={i} x1={b.x + b.w + 6} y1={LY[i] + 10} x2={enc[i + 1].x - 8} y2={LY[i + 1] - 6} o={encK(i + 1)} />)}
        <Arrow x1={enc[3].x + enc[3].w + 8} y1={LY[3] - 6} x2={dec[0].x - 8} y2={LY[2] + 10} o={decK(0)} />
        {dec.slice(0, 2).map((b, i) => <Arrow key={`d${i}`} x1={b.x + b.w + 6} y1={LY[b.l] - 10} x2={dec[i + 1].x - 8} y2={LY[dec[i + 1].l] + 6} o={decK(i + 1)} />)}
        <Arrow x1={dec[2].x + dec[2].w + 8} y1={LY[0]} x2={1630} y2={LY[0]} o={decK(3)} />
        {skip > 0 && [0, 1, 2].map((l) => {
          const a = enc[l], b = dec[2 - l];
          return <Arrow key={`s${l}`} x1={a.x + a.w + 10} y1={LY[l] - 4} x2={b.x - 12} y2={LY[l] - 4} color={C.warm} head="ahw" w={3} dash="10 8" o={skip} />;
        })}
      </Svg>
      {enc.map((b, i) => <Block key={i} x={b.x} y={b.y} w={b.w} h={b.h} o={encK(i)} label={`${b.c} × ${b.s}²`}
        color={i === 3 && bott > 0 ? C.warm : C.xray} />)}
      {dec.map((b, i) => <Block key={`d${i}`} x={b.x} y={b.y} w={b.w} h={b.h} o={decK(i)} label={`${b.c} × ${b.s}²`} color={C.good} fill="rgba(125,255,176,0.10)" />)}
      <T x={480} y={870} size={30} color={C.xray} o={encK(0)}>encoder: ↓2 · channels ×2</T>
      <T x={1290} y={870} size={30} color={C.good} o={decK(0)}>decoder: ↑2 · fewer channels</T>
      <T x={860} y={800} size={28} color={C.warm} rtl o={bott}>צוואר הבקבוק יודע "מה", פחות "איפה"</T>
      {skip > 0 && [0, 1, 2].map((l) => <T key={l} x={(enc[l].x + dec[2 - l].x) / 2 + 40} y={LY[l] - 26} size={22} color={C.warm} o={skip}>skip · concat</T>)}
    </div>
  );
}

// ---------------- upsampling ----------------
function Upsampling({ t }: { t: number }) {
  const o = win(t, "up1", "up3");
  if (o <= 0) return null;
  const IN = [[1, 3], [2, 0]];
  const near = window01(t, S.up1[0], S.up3[0] + 0.4, 0.6, 0.5);
  const nearDim = 1 - 0.55 * ramp(t, S.up2[0], S.up2[0] + 0.8);
  const tr = window01(t, S.up2[0] + 0.2, S.up3[0] + 0.4, 0.6, 0.5);
  const ov = ramp(t, S.up3[0] + 0.3, S.up3[0] + 1.2);
  const fix = ramp(t, frac("up3", 0.62), frac("up3", 0.72));
  const cols = ["rgba(127,208,255,0.45)", "rgba(255,207,127,0.45)", "rgba(125,255,176,0.40)", "rgba(255,127,138,0.40)"];
  const stamps = [0, 1, 2, 3].map((k) => ramp(t, frac("up2", 0.35 + k * 0.12), frac("up2", 0.42 + k * 0.12)));
  const cnt = (i: number, j: number) => { let c = 0; for (let k = 0; k < 4; k++) { const si = (k >> 1) * 2, sj = (k & 1) * 2; if (stamps[k] > 0.5 && i >= si && i < si + 3 && j >= sj && j < sj + 3) c++; } return c; };
  const ov3 = d10.ov_k3 as number[][], ov2 = d10.ov_k2 as number[][];
  const shade = (v: number) => `rgba(255,207,127,${0.12 + 0.2 * v})`;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <div style={{ opacity: near * nearDim }}>
        <T x={290} y={250} size={30} color={C.xray}>nearest ↑2</T>
        <CellGrid x={200} y={330} rows={2} cols={2} size={90} font={34} cell={(i, j) => ({ bg: cols[i * 2 + j], text: IN[i][j] })} />
        <Svg><Arrow x1={410} y1={420} x2={500} y2={420} /></Svg>
        <CellGrid x={520} y={240} rows={4} cols={4} size={90} font={30} cell={(i, j) => ({ bg: cols[(i >> 1) * 2 + (j >> 1)], text: IN[i >> 1][j >> 1] })} />
        <T x={700} y={640} size={26} color={C.dim} rtl>בלי משקלות נלמדים</T>
      </div>
      <div style={{ opacity: tr }}>
        <T x={1340} y={210} size={30} color={C.warm}>transposed conv · k = 3, s = 2</T>
        <CellGrid x={1000} y={330} rows={2} cols={2} size={70} font={28} cell={(i, j) => ({ bg: cols[i * 2 + j], text: IN[i][j] })} />
        <Svg><Arrow x1={1160} y1={400} x2={1210} y2={400} /></Svg>
        <CellGrid x={1230} y={250} rows={5} cols={5} size={76} font={26} cell={(i, j) => {
          const c = cnt(i, j);
          return { bg: c === 0 ? "rgba(20,30,45,0.6)" : `rgba(255,207,127,${0.15 + 0.22 * c})`, text: c > 1 ? `×${c}` : "", color: C.bg };
        }} />
        {[0, 1, 2, 3].map((k) => stamps[k] > 0 && (
          <div key={k} style={{ position: "absolute", left: 1230 + (k & 1) * 2 * 76 + 3, top: 250 + (k >> 1) * 2 * 76 + 3, width: 3 * 76 - 6, height: 3 * 76 - 6,
            border: `3px solid ${cols[k].replace(/0\.4\d?\)/, "1)")}`, borderRadius: 6, opacity: stamps[k] }} />
        ))}
        <T x={1420} y={680} size={24} color={C.dim} rtl>כל פיקסל מטביע kernel × הערך שלו; חפיפות מתחברות</T>
      </div>
      <div style={{ opacity: ov }}>
        <T x={460} y={215} size={30} color={C.bad}>k = 3, s = 2 → checkerboard</T>
        <CellGrid x={240} y={250} rows={9} cols={9} size={50} font={22} cell={(i, j) => ({ bg: shade(ov3[i][j]), text: ov3[i][j], color: "white" })} />
        <T x={465} y={730} size={24} color={C.dim} rtl>כמה הטבעות נוחתות על כל פיקסל (קלט 4×4)</T>
        <div style={{ opacity: fix }}>
          <T x={1360} y={215} size={30} color={C.good}>k = 2, s = 2 → אחיד</T>
          <CellGrid x={1160} y={250} rows={8} cols={8} size={50} font={22} cell={(i, j) => ({ bg: shade(ov2[i][j]), text: ov2[i][j], color: "white" })} />
          <T x={1360} y={700} size={24} color={C.dim} rtl>או: nearest ↑2 ואחריו conv רגיל</T>
        </div>
      </div>
    </div>
  );
}

// ---------------- our architecture (2D diagram) ----------------
const AY = [285, 735];  // AP row, LAT row (centres)
function Arch({ t }: { t: number }) {
  const o = Math.max(win(t, "ours1", "ours2", 0.6, 0.6, 0.3), win(t, "dec3d", "head", 0.6, 0.6, 0.5));
  if (o <= 0) return null;
  const encOn = t > S.dec3d[0] - 1 ? 1 : ramp(t, S.ours2[0] + 0.4, S.ours2[0] + 2.5);
  const hiEnc = window01(t, S.ours2[0], S.ours2[1] + 0.3, 0.4, 0.4);
  const liftOn = t > S.dec3d[0] - 1 ? 1 : 0;
  const decOn = (i: number) => (t < S.dec3d[0] ? 0 : ramp(t, frac("dec3d", 0.1 + i * 0.2), frac("dec3d", 0.18 + i * 0.2)));
  const skipOn = ramp(t, frac("dec3d", 0.7), frac("dec3d", 0.82));
  const headOn = ramp(t, S.head[0] + 0.3, S.head[0] + 1.2);
  const qmark = window01(t, S.ours1[0] + 1.5, S.ours2[0] + 0.3, 0.5, 0.4);
  const ex = [250, 360, 480, 610];
  const blocks = (row: number) => LV.map((s, i) => {
    const h = Math.max(20, s * 1.1), w = bw(CH[i]);
    return <Block key={`${row}${i}`} x={ex[i]} y={AY[row] - h / 2} w={w} h={h} o={encOn} color={hiEnc > 0 ? C.warm : C.xray}
      label={i === 0 || i === 3 ? `${CH[i]}×${s}²` : undefined} />;
  });
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Tile src="data/v10/ap_128.png" x={60} y={AY[0] - 75} w={150} sub="AP · 1×128²" pixelated />
      <Tile src="data/v10/lat_128.png" x={60} y={AY[1] - 75} w={150} sub="LAT · 1×128²" pixelated />
      {blocks(0)}{blocks(1)}
      <T x={460} y={510} size={26} color={C.xray} rtl o={encOn}>encoder 2D נפרד לכל צילום</T>
      <div style={{ opacity: qmark }}>
        <T x={1000} y={510} size={90} color={C.warm}>2D → ? → 3D</T>
      </div>
      <div style={{ opacity: liftOn }}>
        <Svg>
          {[0, 1].map((r) => <Arrow key={r} x1={720} y1={AY[r]} x2={770} y2={AY[r]} />)}
          {[0, 1].map((r) => <Arrow key={`c${r}`} x1={870} y1={AY[r] + (r ? -40 : 40)} x2={930} y2={AY[r] + (r ? -150 : 150)} />)}
        </Svg>
        <Cube x={780} y={AY[0] - 40} s={70} label="64×16³" sub="lift ← AP" />
        <Cube x={780} y={AY[1] - 40} s={70} label="64×16³" sub="lift ← LAT" />
        <Cube x={940} y={470} s={84} label="128×16³" sub="concat" color={C.warm} fill="rgba(255,207,127,0.10)" />
      </div>
      {[{ x: 1110, s: 84, l: "128×16³" }, { x: 1270, s: 110, l: "64×32³" }, { x: 1460, s: 140, l: "32×64³" }].map((c, i) => (
        <Cube key={i} x={c.x} y={540 - c.s / 2} s={c.s} o={decOn(i)} label={c.l} color={C.good} fill="rgba(125,255,176,0.10)" />
      ))}
      <Svg o={decOn(0)}>
        <Arrow x1={1062} y1={512} x2={1102} y2={512} />
        <Arrow x1={1230} y1={512} x2={1262} y2={512} o={decOn(1)} />
        <Arrow x1={1420} y1={512} x2={1452} y2={512} o={decOn(2)} />
        <Arrow x1={1652} y1={512} x2={1690} y2={512} o={headOn} />
      </Svg>
      <T x={1360} y={360} size={26} color={C.good} rtl o={decOn(0)}>decoder 3D: ↑2 ↑2</T>
      <Svg o={skipOn}>
        {[0, 1].map((r) => [2, 1].map((lvl) => {
          const x1 = ex[lvl] + bw(CH[lvl]) / 2, y1 = AY[r] + (r ? -1 : 1) * (Math.max(20, LV[lvl] * 1.1) / 2 + 6);
          const x2 = lvl === 2 ? 1320 : 1520, y2 = r ? 540 + (lvl === 2 ? 55 : 70) + 40 : 540 - (lvl === 2 ? 55 : 70) - 50;
          return <path key={`${r}${lvl}`} d={`M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${y1} ${x2},${y2}`} fill="none" stroke={C.warm}
            strokeWidth={2.5} strokeDasharray="9 7" markerEnd="url(#ahw)" />;
        }))}
      </Svg>
      <T x={1180} y={210} size={24} color={C.warm} rtl o={skipOn}>skips: features מה-encoders, מורמים ל-3D באותה דרך</T>
      <Cube x={1700} y={470} s={110} o={headOn} label="1×64³" sub="logits" color={C.warm} fill="rgba(255,207,127,0.15)" glow />
      <T x={1770} y={395} size={22} color={C.dim} o={headOn}>Conv3d 1×1×1</T>
    </div>
  );
}

// ---------------- fusion options ----------------
function Fusion({ t }: { t: number }) {
  const o = win(t, "fuse1", "fuse2");
  if (o <= 0) return null;
  const two = ramp(t, S.fuse2[0] + 0.2, S.fuse2[0] + 1.2);
  const axes = ramp(t, frac("fuse1", 0.4), frac("fuse1", 0.5));
  const cond = ramp(t, frac("fuse2", 0.55), frac("fuse2", 0.65));
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Panel style={{ left: 70, top: 180, width: 800, height: 640, opacity: 1 - 0.5 * two }}>
        <div style={{ fontSize: 34, color: C.xray, fontWeight: 700 }}>דרך 1: שרשור בכניסה</div>
        <div style={{ position: "relative", height: 520, direction: "ltr" }}>
          <Tile src="data/v10/ap_128.png" x={60} y={30} w={240} pixelated />
          <Tile src="data/v10/lat_128.png" x={420} y={30} w={240} pixelated />
          <T x={180} y={300} size={24} color={C.warm} o={axes} rtl>עמודות: ימין ↔ שמאל</T>
          <T x={540} y={300} size={24} color={C.warm} o={axes} rtl>עמודות: קדימה ↔ אחורה</T>
          <T x={360} y={350} size={24} color={C.good} o={axes} rtl>שורות: אותו גובה בשני הצילומים ✓</T>
          <T x={360} y={420} size={28} color={C.text} o={axes}>cat → 2 × 128 × 128 → encoder → reshape 3D</T>
          <T x={360} y={470} size={24} color={C.dim} o={axes} rtl>את הגאומטריה הרשת צריכה ללמוד לבד</T>
        </div>
      </Panel>
      <Panel o={two} style={{ left: 930, top: 180, width: 920, height: 640, border: `1.5px solid ${C.warm}` }}>
        <div style={{ fontSize: 34, color: C.warm, fontWeight: 700 }}>דרך 2 (שלנו): הרמה לאורך ציר הקרן</div>
        <div style={{ position: "relative", height: 470 }}>
          <Code x={-10} y={20} w={860} size={22} lines={[
            "# volume axes: (B, C, D = S-I, H = A-P, W = R-L)",
            "f_ap, f_lat = enc_ap(x_ap), enc_lat(x_lat)   # 4x64x16x16",
            "v_ap  = f_ap.unsqueeze(3).expand(-1, -1, -1, 16, -1)",
            "v_lat = f_lat.unsqueeze(4).expand(-1, -1, -1, -1, 16)",
            "v = torch.cat([v_ap, v_lat], dim=1)   # 4x128x16x16x16",
          ]} hi={[2, 3]} />
          <T x={430} y={330} size={24} color={C.xray} rtl>AP נמרח לאורך A–P · LAT נמרח לאורך R–L</T>
          <T x={430} y={385} size={26} color={C.warm} o={cond} rtl>תנאי: שני הצילומים מכסים בדיוק את אותה קובייה</T>
          <T x={430} y={425} size={22} color={C.dim} o={cond} rtl>(שורה בתמונה = גובה בקובייה)</T>
        </div>
      </Panel>
    </div>
  );
}

// ---------------- the shape table ----------------
const TABLE: [string, string][] = [
  ["input (AP, LAT)", "4×1×128×128"], ["Conv2d×2", "4×32×128×128"], ["↓2 + Conv2d×2", "4×64×64×64"], ["↓2 + Conv2d×2", "4×128×32×32"],
  ["↓2 + Conv2d×2", "4×256×16×16"], ["Conv2d 1×1", "4×64×16×16"], ["lift (expand)", "4×64×16×16×16"],
];
const TABLE3: [string, string][] = [
  ["cat AP + LAT", "4×128×16×16×16"], ["Conv3d×2", "4×128×16×16×16"], ["ConvTranspose3d ↑2", "4×64×32×32×32"],
  ["cat skips (2×32)", "4×128×32×32×32"], ["Conv3d×2", "4×64×32×32×32"], ["ConvTranspose3d ↑2", "4×32×64×64×64"],
  ["cat skips (2×16)", "4×64×64×64×64"], ["Conv3d×2", "4×32×64×64×64"], ["Conv3d 1×1×1", "4×1×64×64×64"],
];
function ShapeTable({ t }: { t: number }) {
  const o = win(t, "table", "table", 0.6, 0.6, 0.6);
  if (o <= 0) return null;
  const row = (r: [string, string], i: number, last = false) => (
    <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 27, lineHeight: 1.55, direction: "ltr",
      color: last ? C.warm : C.text, borderBottom: "1px solid rgba(120,190,255,0.12)" }}>
      <span style={{ color: last ? C.warm : C.dim }}>{r[0]}</span><span style={{ fontFamily: "Consolas, monospace" }}>{r[1]}</span></div>
  );
  return (
    <Panel o={o} style={{ left: 90, right: 90, top: 150, padding: "26px 40px" }}>
      <div style={{ display: "flex", gap: 60, direction: "ltr" }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 30, color: C.xray, marginBottom: 10, direction: "rtl" }}>2D · לכל צילום בנפרד</div>
          {TABLE.map((r, i) => row(r, i))}
          <div style={{ fontSize: 22, color: C.dim, marginTop: 14, direction: "rtl" }}>skips: פלט ↓2 (64²) ו-↓4 (32²) עוברים Conv2d 1×1 ל-16 ול-32 ערוצים, ומורמים</div>
        </div>
        <div style={{ flex: 1.1 }}>
          <div style={{ fontSize: 30, color: C.good, marginBottom: 10, direction: "rtl" }}>3D · אחרי האיחוד</div>
          {TABLE3.map((r, i) => row(r, i, i === TABLE3.length - 1))}
        </div>
      </div>
      <div style={{ fontSize: 24, color: C.dim, marginTop: 12 }}>batch = 4 · צורה: B × C × (D × H × W) · Conv עם padding=1 שומר על הגודל</div>
    </Panel>
  );
}

// ---------------- 3D scenes ----------------
function RoundTrip({ t, a }: { t: number; a: Assets }) {
  const on = win(t, "skip1", "skip1", 0.8, 0.8, 0.6);
  const via = ramp(t, frac("skip1", 0.5), frac("skip1", 0.62));
  const keys: CamKey[] = [{ t: S.skip1[0] - 1, pos: [0, 0.9, -8.6], look: [0, -0.1, 0] }, { t: S.skip1[1] + 1, pos: [0.6, 1.3, -8.0], look: [0, -0.1, 0] }];
  const spin = (t - S.skip1[0]) * 0.18;
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <group position={[1.95, 0, 0]} rotation={[0, spin, 0]} scale={0.95}>
        <Voxels centers={a.occ.centers} cell={a.occ.cell} opacity={on} gap={0.08} />
      </group>
      <group position={[-1.95, 0, 0]} rotation={[0, spin, 0]} scale={0.95}>
        <Voxels centers={a.via.centers} cell={a.via.cell} opacity={on * via} gap={0.08} color="#ffcf7f" />
      </group>
    </>
  );
}

function LiftScene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const box = boxOf(a.meta);
  const [cx, cy, cz] = box.c;
  const on = win(t, "lift1", "lift2", 0.8, 0.8, 0.5);
  const apP = ramp(t, frac("lift1", 0.45), frac("lift1", 0.95), (x) => x);
  const latP = ramp(t, S.lift2[0] + 0.3, frac("lift2", 0.5), (x) => x);
  const N = 14;
  const mats = useMemo(() => Array.from({ length: 2 * N + 2 }, (_, i) => new THREE.MeshBasicMaterial({
    map: i < N + 1 ? a.ap : a.lat, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide, toneMapped: false, color: i < N + 1 ? new THREE.Color(0.55, 0.8, 1.2) : new THREE.Color(1.2, 0.85, 0.45),
  })), [a.ap, a.lat]);
  const keys: CamKey[] = [
    { t: S.lift1[0] - 1, pos: orbit(-0.55, 8.4, 2.2), look: [cx, cy - 0.1, cz] },
    { t: S.lift1[1], pos: orbit(-0.85, 8.4, 2.6), look: [cx, cy - 0.1, cz] },
    { t: S.lift2[1] + 0.5, pos: orbit(-1.25, 8.6, 3.0), look: [cx, cy - 0.1, cz] },
  ];
  const zs = Array.from({ length: N }, (_, i) => box.mx[2] - (i + 0.5) / N * box.s[2]);   // from the AP detector (posterior) back along the rays
  const xs = Array.from({ length: N }, (_, i) => box.mn[0] + (i + 0.5) / N * box.s[0]);   // from the LAT detector (patient left) back along the rays
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Bone geo={a.pelvis} material={bone} opacity={on * 0.55} />
      <WireBox size={box.s as [number, number, number]} position={[cx, cy, cz]} opacity={on * 0.5} />
      {/* the AP image in front of the body, then its copies along the A-P rays */}
      <mesh position={[cx, cy, box.mx[2] + 0.4]} rotation={[0, Math.PI, 0]} material={mats[N]}>
        <planeGeometry args={a.meta.ap_size} />
      </mesh>
      {zs.map((z, i) => {
        const k = Math.min(1, Math.max(0, apP * N - i));
        mats[i].opacity = on * k * 0.075;
        return k > 0 ? <mesh key={i} position={[cx, cy, z]} rotation={[0, Math.PI, 0]} material={mats[i]}><planeGeometry args={a.meta.ap_size} /></mesh> : null;
      })}
      <mesh position={[box.mn[0] - 0.4, cy, cz]} rotation={[0, -Math.PI / 2, 0]} material={mats[2 * N + 1]}>
        <planeGeometry args={a.meta.lat_size} />
      </mesh>
      {xs.map((x, i) => {
        const k = Math.min(1, Math.max(0, latP * N - i));
        mats[N + 1 + i].opacity = on * k * 0.04;
        return k > 0 ? <mesh key={`l${i}`} position={[x, cy, cz]} rotation={[0, -Math.PI / 2, 0]} material={mats[N + 1 + i]}><planeGeometry args={a.meta.lat_size} /></mesh> : null;
      })}
      {(() => { mats[N].opacity = on * 0.8; mats[2 * N + 1].opacity = on * ramp(t, S.lift2[0], S.lift2[0] + 0.8) * 0.95; return null; })()}
    </>
  );
}

type Assets = { meta: Meta; pelvis: THREE.BufferGeometry; ap: THREE.Texture; lat: THREE.Texture; occ: Occ; via: Occ };

export const V10 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const meta = useJson<Meta>("data/hero/meta.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const lat = useTex("data/hero/drr_lat.png");
  const occ = useJson<Occ>("data/hero/occ64.json");
  const via = useJson<Occ>("data/v10/occ64_via16.json");
  const ready = meta && pelvis && ap && lat && occ && via;
  const inRT = t > S.skip1[0] - 1.2 && t < S.skip1[1] + 1.2;
  const inLift = t > S.lift1[0] - 1.2 && t < S.lift2[1] + 1.2;
  const a = ready ? { meta, pelvis, ap, lat, occ, via } : null;
  const rt = win(t, "skip1", "skip1", 0.8, 0.8, 0.6);
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {a && (inRT || inLift) && (
        <Stage>
          {inRT && <RoundTrip t={t} a={a} />}
          {inLift && <LiftScene t={t} a={a} />}
        </Stage>
      )}

      <Opening n={10} title="Encoder-decoder ו-U-Net: מ-2D ל-3D" active={[2, 3]} t={t} end={2.9} />
      <Intro t={t} />

      <Title text="סגמנטציה = סיווג לכל פיקסל" t={t} a={S.seg1[0]} b={S.seg2[1] + 0.4} top={60} size={54} />
      <Segmentation t={t} />

      <Title text="Encoder → bottleneck → decoder" t={t} a={S.enc[0]} b={S.dec[1] + 0.4} top={60} size={54} />
      <Title text="U-Net: חיבורי דילוג" sub="Ronneberger, Fischer & Brox, 2015" t={t} a={S.skip2[0]} b={S.skip3[1] + 0.4} top={50} size={54} />
      <UNet2D t={t} />

      <Title text="Upsampling: איך מגדילים מפה" t={t} a={S.up1[0]} b={S.up3[1] + 0.4} top={60} size={54} />
      <Upsampling t={t} />

      <Title text="כמה פרטים שורדים את 16³?" t={t} a={S.skip1[0]} b={S.skip1[1] + 0.6} top={50} size={54} />
      <Label x={560} y={860} text="האמת: 64³" color={C.xray} size={32} o={rt} />
      <Label x={1360} y={860} text={"(בלי רשת) ⁦64³ → 16³ → 64³⁩"} color={C.warm} size={32} o={rt * ramp(t, frac("skip1", 0.5), frac("skip1", 0.62))} />
      <Panel o={rt * ramp(t, frac("skip1", 0.72), frac("skip1", 0.82))} style={{ left: 760, right: 760, top: 200, padding: "16px 20px", textAlign: "center" }}>
        <div style={{ fontSize: 44, color: C.warm, fontWeight: 700, direction: "ltr" }}>Dice = {d10.roundtrip_dice["16"].toFixed(2)}</div>
        <div style={{ fontSize: 22, color: C.dim, direction: "rtl" }}>{`להשוואה, דרך 32³: ${d10.roundtrip_dice["32"].toFixed(2)}`}</div>
      </Panel>

      <Title text="הרשת שלנו" t={t} a={S.ours1[0]} b={S.ours2[1] + 0.3} top={60} size={54} />
      <Title text="decoder 3D" t={t} a={S.dec3d[0]} b={S.head[1] + 0.5} top={60} size={54} />
      <Arch t={t} />

      <Title text="Lifting: מורחים לאורך הקרן" t={t} a={S.lift1[0]} b={S.lift2[1] + 0.4} top={50} size={54} />
      <Label x={960} y={190} text="להמחשה מורחים את התמונה עצמה; ברשת מורחים 64 ערוצי features" color={C.dim} size={26}
        o={win(t, "lift1", "lift2", 0.8, 0.6, 0.4)} />
      <Panel o={window01(t, frac("lift1", 0.5), S.lift2[0] + 0.2)} style={{ left: 80, top: 300, width: 470, padding: "20px 26px" }}>
        <div style={{ fontSize: 26, color: C.xray }}>AP: לאורך ציר A–P</div>
        <div style={{ fontSize: 26, direction: "ltr", textAlign: "left", marginTop: 8 }}>4×64×16×16 → 4×64×16×16×16</div>
      </Panel>
      <Panel o={window01(t, S.lift2[0] + 0.5, S.lift2[1] + 0.5)} style={{ left: 80, top: 300, width: 470, padding: "20px 26px" }}>
        <div style={{ fontSize: 26, color: C.warm }}>LAT: לאורך ציר R–L</div>
        <div style={{ fontSize: 26, direction: "ltr", textAlign: "left", marginTop: 8 }}>4×64×16×16 → 4×64×16×16×16</div>
        <div style={{ fontSize: 24, color: C.dim, marginTop: 10 }}>כל תא מקבל את שתי הקרניים שעוברות דרכו</div>
      </Panel>

      <Title text="שתי דרכים לאחד AP + LAT" t={t} a={S.fuse1[0]} b={S.fuse2[1] + 0.4} top={50} size={54} />
      <Fusion t={t} />

      <Title text="כל הצורות" t={t} a={S.table[0]} b={S.table[1] + 0.6} top={40} size={50} />
      <ShapeTable t={t} />

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "segmentation: סיווג לכל פיקסל (אצלנו: לכל תא)",
        "encoder מקטין ומעשיר; decoder מגדיל בחזרה",
        "upsampling: nearest, או transposed conv (k=2, s=2 בלי checkerboard)",
        "skip connections מחזירים את הפרטים (16³ לבד: Dice 0.67)",
        "שלנו: encoder 2D לכל צילום, lifting לאורך הקרן, decoder 3D, ופלט 64³",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/05-network-training/notebook.ipynb" todos={[
        "לבנות ב-PyTorch: encoder 2D, lifting, decoder 3D",
        "להעביר batch דמה: torch.zeros(4, 1, 128, 128) לכל צילום",
        "להדפיס את הצורה אחרי כל שלב ולהשוות לטבלה",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro[0]} />
      <Narration tl={tl} dir="v10" />
    </AbsoluteFill>
  );
};
