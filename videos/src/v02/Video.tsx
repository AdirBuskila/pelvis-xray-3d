// Video 2 · Where the CT numbers come from. One ray and Beer–Lambert (one material, many, the integral), a real ray
// through our CT, Hounsfield units and their inverse, clipping, per-tissue HU from the masks, the HU histogram of our
// CT with its real peaks, windowing (live, on a real slice), and why we display L and not 1 − e^(−L).
// Structure follows v00: narration-driven timeline, DOM-level asset loading, one <Stage>, 2D overlays on top.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { ReactNode } from "react";
import script from "./script.json";
import durations from "../../public/audio/v02/durations.json";
import info from "../../public/data/v02/info.json";
import metaJson from "../../public/data/hero/meta.json";
import { buildTimeline, Script } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Detector, Meta, PointSource, Rays, ScanBeam, SourcePanel } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, PipelineMap, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { camAt, mono, project } from "../v01/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const meta = metaJson as unknown as Meta;
const box = boxOf(meta);
const [cx, cy, cz] = box.c;
const R = info.ray;
const MUW = info.mu_w_per_mm;
const [rx, ry] = R.scene_xy;
const zSrc = box.mn[2] - 1.5, zDet = box.mx[2] + 0.5;
const zA = box.mn[2], zP = box.mx[2];                 // the ray crosses the crop from anterior (−z) to posterior (+z)

const keys: CamKey[] = [
  { t: 0, pos: orbit(-0.7, 9.0, 1.2), look: [0, 0, 0] },
  { t: S.intro2[0], pos: orbit(0.4, 8.6, 1.0), look: [0, 0, 0] },
  { t: S.ray1[0] + 0.5, pos: [10.0, ry + 1.6, -0.4], look: [0, ry + 0.25, -0.2] },
  { t: S.ray1[1] + 0.6, pos: [9.6, ry + 1.4, -0.6], look: [0, ry + 0.25, -0.2] },
  { t: S.real3[0], pos: [8.6, 2.4, -3.6], look: [0, 0, 0.1] },
  { t: S.real3[1], pos: [7.6, 2.0, -5.0], look: [0, 0, 0.2] },
  { t: S.pipe[0], pos: orbit(0.5, 9.4, 1.4), look: [0, 0, 0] },
  { t: tl.total, pos: orbit(1.3, 9.6, 1.2), look: [0, 0, 0] },
];

// ---------------- 3D ----------------
type Assets = { pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry; spine: THREE.BufferGeometry; ap: THREE.Texture };

function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const introOn = window01(t, 2.6, S.ray1[1] + 0.6, 1.0, 0.8);
  const rayOn = window01(t, S.intro2[1] - 3.0, S.ray1[1] + 0.6, 0.8, 0.6);
  const rayGrow = ramp(t, S.intro2[1] - 2.6, S.ray1[0] + 2.5);
  const ghost = ramp(t, S.ray1[0] + 0.3, S.ray1[0] + 2.0);
  const rigOn = window01(t, S.real3[0] - 0.2, S.real3[1] + 0.6, 0.8, 0.6);
  const scan = ramp(t, S.real3[0] + 1.2, S.real3[1] - 0.6);
  const tailOn = window01(t, S.pipe[0], tl.total, 1.0, 1.0);
  const boneO = introOn * (1 - 0.65 * ghost) + rigOn * 0.6 + tailOn * 0.7;

  // the ray: bright before the body, then dims with the real transmission e^(−L(s)) (shown as T^0.3 to stay visible)
  const segs: [number, number, number][][] = [];
  const cols: [number, number, number][] = [];
  const zEnd = lerp(zSrc, zDet, rayGrow);
  const n = R.n, zs = (k: number) => zA + (zP - zA) * k / n;
  const push = (z1: number, z2: number, b: number) => {
    if (z1 >= zEnd) return;
    segs.push([[rx, ry, z1], [rx, ry, Math.min(z2, zEnd)]]); cols.push([1.4 * b, 2.1 * b, 3.0 * b]);
  };
  push(zSrc, zA, 1);
  for (let k = 0; k < n; k += 3) push(zs(k), zs(Math.min(n, k + 3)), Math.pow(Math.exp(-R.L_cum[k]), 0.3));
  push(zP, zDet, Math.pow(R.T, 0.3));
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Bone geo={a.pelvis} material={bone} opacity={boneO} />
      <Bone geo={a.femurs} material={bone} opacity={boneO * 0.85} />
      <Bone geo={a.spine} material={bone} opacity={boneO * 0.85} />
      {rayOn > 0 && segs.map((s, i) => <Rays key={i} segs={[s]} opacity={rayOn} color={cols[i]} />)}
      <PointSource position={[rx, ry, zSrc]} opacity={rayOn} r={0.09} />
      {rayOn > 0 && (
        <group>
          <mesh position={[rx, ry, zDet]}>
            <planeGeometry args={[1.4, 1.4]} />
            <meshBasicMaterial color={new THREE.Color(0.05, 0.09, 0.14)} transparent opacity={0.85 * rayOn} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[rx, ry, zDet - 0.01]}>
            <planeGeometry args={[0.12, 0.12]} />
            <meshBasicMaterial color={new THREE.Color(1.2, 1.6, 2.2)} transparent opacity={rayOn * ramp(t, S.ray1[0] + 2.3, S.ray1[0] + 3.0)} side={THREE.DoubleSide} toneMapped={false} />
          </mesh>
        </group>
      )}
      {/* the AP rig: every detector pixel gets its own line integral */}
      <group position={[cx, cy, 0]}>
        <SourcePanel size={meta.ap_size} position={[0, 0, box.mn[2] - 1.4]} opacity={rigOn} />
        <ScanBeam axis="ap" progress={scan} opacity={rigOn} w={meta.ap_size[0]} h={meta.ap_size[1]}
          src={box.mn[2] - 1.4} det={box.mx[2] + 0.3} bodyIn={box.mn[2]} bodyOut={box.mx[2]} />
        <Detector tex={a.ap} progress={scan} opacity={rigOn} size={meta.ap_size} position={[0, 0, box.mx[2] + 0.3]} rotation={[0, Math.PI, 0]} />
      </group>
    </>
  );
}

// ---------------- 2D helpers ----------------
const svgFull = { position: "absolute" as const, left: 0, top: 0, width: 1920, height: 1080 };
const T = (x: number, y: number, s: ReactNode, o: { fill?: string; size?: number; anchor?: "start" | "middle" | "end"; weight?: number } = {}) => (
  <text x={x} y={y} fill={o.fill ?? C.text} fontSize={o.size ?? 28} fontFamily={FONT} textAnchor={o.anchor ?? "middle"} fontWeight={o.weight}>{s}</text>);

/** bl / many: a beam through slabs, and the intensity curve below */
function SlabSection({ t }: { t: number }) {
  const on = window01(t, S.bl1[0] - 0.2, S.many2[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const many = ramp(t, S.many1[0] + 0.4, S.many1[0] + 1.6);
  const thin = ramp(t, S.many2[0] + 0.5, S.many2[0] + 4.0);
  // geometry: x in mm along the ray, 0..200; the slab region starts at 40 mm
  const X0 = 140, X1 = 1140, mm = (X1 - X0) / 200, x = (s: number) => X0 + s * mm;
  const bandY = 300, plotY0 = 820, plotH = 330;
  const dGrow = lerp(0, 100, ramp(t, S.bl1[0] + 1.5, S.bl1[1] - 1.5));
  // layers: one water slab, then fat / muscle / bone / muscle (μ from our HU→μ rule)
  const one = [{ a: 40, b: 40 + dGrow, mu: MUW, name: "מים", c: "#4f8fd0" }];
  const layers = [
    { a: 30, b: 60, mu: MUW * (1 - 0.105), name: "שומן", c: "#d8c27a" },
    { a: 60, b: 100, mu: MUW * 1.037, name: "שריר", c: "#c86a6a" },
    { a: 100, b: 125, mu: MUW * 2.0, name: "עצם", c: "#efe6d2" },
    { a: 125, b: 165, mu: MUW * 1.037, name: "שריר", c: "#c86a6a" },
  ];
  const L = many > 0.5 ? layers : one;
  const muAt = (s: number) => { for (const l of L) if (s >= l.a && s < l.b) return l.mu; return 0; };
  const pts: string[] = [];
  let acc = 0;
  for (let s = 0; s <= 200; s += 0.5) { acc += muAt(s) * 0.5; pts.push(`${pts.length ? "L" : "M"}${x(s)},${plotY0 - Math.exp(-acc) * plotH}`); }
  const Iend = Math.exp(-acc);
  const nThin = 48;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <svg style={svgFull}>
        {/* beam: brightness follows I(s) */}
        {Array.from({ length: 100 }, (_, i) => {
          const s0 = i * 2; let a2 = 0; for (let s = 0; s < s0; s += 0.5) a2 += muAt(s) * 0.5;
          return <rect key={i} x={x(s0)} y={bandY - 16} width={2 * mm + 0.5} height={32} fill={`rgba(127,208,255,${0.15 + 0.85 * Math.exp(-a2)})`} />;
        })}
        {L.map((l, i) => (
          <g key={i} opacity={many > 0.5 ? ramp(t, S.many1[0] + 0.6 + i * 0.5, S.many1[0] + 1.1 + i * 0.5) * (1 - 0.7 * thin) : 1}>
            <rect x={x(l.a)} y={bandY - 110} width={(l.b - l.a) * mm} height={220} fill={l.c} fillOpacity={0.28} stroke={l.c} strokeWidth={2} />
            {T(x((l.a + l.b) / 2), bandY - 128, l.name, { fill: l.c, size: 28 })}
            {many > 0.5 && T(x((l.a + l.b) / 2), bandY + 150, <tspan style={{ direction: "ltr" }}>{`μ${i + 1}`}</tspan>, { fill: l.c, size: 30 })}
          </g>))}
        {many > 0.5 && thin > 0 && Array.from({ length: nThin }, (_, i) => {
          const s = 30 + (i + 0.5) * 135 / nThin, mu = muAt(s);
          return <rect key={i} x={x(30 + i * 135 / nThin)} y={bandY + 110 - mu / (MUW * 2) * 200} width={135 / nThin * mm - 1.5} height={mu / (MUW * 2) * 200}
            fill={C.warm} fillOpacity={0.5 * thin} />;
        })}
        {many < 0.5 && dGrow > 2 && (
          <g>
            <line x1={x(40)} y1={bandY + 135} x2={x(40 + dGrow)} y2={bandY + 135} stroke={C.warm} strokeWidth={3} />
            {T(x(40 + dGrow / 2), bandY + 175, <tspan>d = {Math.round(dGrow)} mm</tspan>, { fill: C.warm, size: 30 })}
          </g>)}
        {T(x(0) + 40, bandY - 40, "I₀", { fill: C.xray, size: 40, anchor: "start" })}
        {T(x(200) - 10, bandY - 40, "I", { fill: C.xray, size: 40, anchor: "end" })}
        {/* the plot */}
        <line x1={X0} y1={plotY0} x2={X1} y2={plotY0} stroke={C.dim} strokeWidth={2} />
        <line x1={X0} y1={plotY0} x2={X0} y2={plotY0 - plotH - 20} stroke={C.dim} strokeWidth={2} />
        <path d={pts.join(" ")} stroke={C.xray} strokeWidth={4} fill="none" />
        {T(X0 - 14, plotY0 - plotH + 8, "1", { fill: C.dim, size: 24, anchor: "end" })}
        {T(X0 - 14, plotY0 + 8, "0", { fill: C.dim, size: 24, anchor: "end" })}
        {T(X0 + 10, plotY0 - plotH - 30, "I / I₀", { fill: C.dim, size: 26, anchor: "start" })}
        {T(X1, plotY0 + 40, "מרחק לאורך הקרן", { fill: C.dim, size: 24, anchor: "end" })}
        {T(x(185), plotY0 - Iend * plotH - 18, <tspan style={{ direction: "ltr" }}>{(Iend * 100).toFixed(1)}%</tspan>, { fill: C.warm, size: 30 })}
      </svg>
      <Eq t={t} a={S.bl2[0]} b={S.bl3[1] + 0.3} style={{ left: 1230, top: 260, width: 600 }} size={58}
        tex={String.raw`\hl{I}{I}=\hl{I0}{I_0}\,e^{-\hl{mu}{\mu}\,\hl{d}{d}}`} note="Beer–Lambert"
        terms={{ I0: [S.bl2[0] + 1.0, S.bl2[0] + 3.4], mu: [S.bl2[0] + 3.6, S.bl2[0] + 6.4], d: [S.bl2[1] - 4.4, S.bl2[1] - 2.6], I: [S.bl2[0], S.bl2[0] + 1.0] }} />
      <Panel o={window01(t, S.bl3[0] + 0.3, S.bl3[1] + 0.3, 0.5, 0.5)} style={{ left: 1230, top: 560, width: 600, padding: "20px 30px" }}>
        <div style={{ ...mono, fontSize: 32, lineHeight: 1.7, textAlign: "left" }}>
          <div>μ<sub>water</sub> ≈ 0.02 mm⁻¹</div>
          <div>d = 100 mm</div>
          <div style={{ color: C.warm }}>e<sup>−0.02·100</sup> = e<sup>−2</sup> ≈ 13.5%</div>
        </div>
      </Panel>
      <Eq t={t} a={S.many1[0] + 4.0} b={S.many1[1] + 0.3} style={{ left: 1200, top: 280, width: 660 }} size={40}
        tex={String.raw`I=I_0\,e^{-\mu_1 d_1}e^{-\mu_2 d_2}e^{-\mu_3 d_3}e^{-\mu_4 d_4}\\[6pt]=I_0\,e^{-\hl{s}{\textstyle\sum_i \mu_i d_i}}`}
        terms={{ s: [S.many1[1] - 5.0, S.many1[1] + 0.3] }} />
      <Eq t={t} a={S.many2[0] + 0.3} b={S.many2[1] + 0.4} style={{ left: 1200, top: 280, width: 660 }} size={44}
        tex={String.raw`\sum_i \mu_i d_i \;\longrightarrow\; \hl{L}{L}=\int \mu(s)\,ds\\[10pt] I = I_0\, e^{-\hl{L}{L}}`}
        note="L: האינטגרל הקווי (line integral)" terms={{ L: [S.many2[1] - 5.5, S.many2[1] + 0.4] }} />
    </div>
  );
}

/** real1/real2: the real ray's μ profile and the running sum L */
function RealRay({ t }: { t: number }) {
  const on = window01(t, S.real1[0] - 0.2, S.real2[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const X0 = 180, X1 = 1380, Y0 = 760, HH = 420, n = R.n;
  const mx = Math.max(...R.mu) * 1.1;
  const xs = (k: number) => X0 + (X1 - X0) * k / n;
  const reveal = ramp(t, S.real1[0] + 1.0, S.real1[0] + 5.0) * n;
  const sweep = ramp(t, S.real2[0] + 1.5, S.real2[0] + 9.0);
  const kk = Math.min(n - 1, Math.floor(sweep * n));
  const Lmax = R.L_total * 1.08;
  const Lpath = R.L_cum.slice(0, kk + 1).map((v, k) => `${k ? "L" : "M"}${xs(k + 1)},${Y0 - v / Lmax * HH}`).join(" ");
  const fin = ramp(t, S.real2[0] + 9.3, S.real2[0] + 10.2);
  const thumbH = 260, z = thumbH / 235;
  const [pr, pc] = R.ap_pixel_rc;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <svg style={svgFull}>
        {R.mu.map((m, k) => k < reveal ? (
          <rect key={k} x={xs(k)} y={Y0 - m / mx * HH} width={(X1 - X0) / n - 0.8} height={m / mx * HH}
            fill={R.bone[k] ? C.bone : C.xray} fillOpacity={R.bone[k] ? 0.85 : 0.55} />) : null)}
        <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke={C.dim} strokeWidth={2} />
        <line x1={X0} y1={Y0} x2={X0} y2={Y0 - HH - 20} stroke={C.dim} strokeWidth={2} />
        {T(X0 + 8, Y0 - HH - 30, "μ (mm⁻¹)", { fill: C.xray, size: 26, anchor: "start" })}
        {T(X0 - 12, Y0 - MUW / mx * HH + 8, "0.02", { fill: C.dim, size: 22, anchor: "end" })}
        <line x1={X0} y1={Y0 - MUW / mx * HH} x2={X1} y2={Y0 - MUW / mx * HH} stroke={C.dim} strokeDasharray="6 8" strokeWidth={1.5} />
        {T(X0, Y0 + 40, "קדימה (anterior)", { fill: C.dim, size: 24, anchor: "start" })}
        {T(X1, Y0 + 40, "אחורה (posterior)", { fill: C.dim, size: 24, anchor: "end" })}
        {T(X0 + (X1 - X0) / 2, Y0 + 40, `${Math.round(n * R.ds_mm)} מ"מ, ${n} voxels`, { fill: C.dim, size: 24 })}
        {sweep > 0 && (
          <g>
            <path d={Lpath} stroke={C.warm} strokeWidth={4} fill="none" />
            <line x1={xs(kk + 1)} y1={Y0} x2={xs(kk + 1)} y2={Y0 - HH} stroke={C.warm} strokeWidth={2} opacity={0.6} />
            {T(xs(kk + 1) + 10, Y0 - R.L_cum[kk] / Lmax * HH - 16, <tspan>L = {R.L_cum[kk].toFixed(2)}</tspan>, { fill: C.warm, size: 30, anchor: "start" })}
            {T(X1 + 60, Y0 - HH - 30, "L (מצטבר)", { fill: C.warm, size: 26, anchor: "end" })}
          </g>)}
      </svg>
      <Label x={xs(63)} y={Y0 - HH + 10} text="ראש הירך" color={C.bone} size={28} o={ramp(t, S.real1[0] + 6, S.real1[0] + 7)} />
      {/* where this ray lands in the AP image */}
      <div style={{ position: "absolute", left: 1500, top: 170 }}>
        <Img src={staticFile("data/hero/drr_ap.png")} style={{ height: thumbH, display: "block", borderRadius: 6 }} />
        <div style={{ position: "absolute", left: pc * z - 9, top: pr * z - 9, width: 18, height: 18, borderRadius: 9, border: `3px solid ${C.warm}`, boxShadow: `0 0 12px ${C.warm}` }} />
        <div style={{ fontFamily: FONT, fontSize: 24, color: C.dim, textAlign: "center", marginTop: 8 }}>הפיקסל של הקרן הזאת</div>
      </div>
      <Panel o={fin} style={{ left: 1440, top: 520, width: 400, padding: "18px 26px", direction: "ltr" }}>
        <div style={{ ...mono, fontSize: 30, lineHeight: 1.7, textAlign: "left" }}>
          <div>L = Σ μ · 1.5 mm = <span style={{ color: C.warm }}>{R.L_total.toFixed(2)}</span></div>
          <div>I/I₀ = e<sup>−{R.L_total.toFixed(2)}</sup> ≈ <span style={{ color: C.warm }}>{(R.T * 100).toFixed(1)}%</span></div>
        </div>
      </Panel>
    </div>
  );
}

/** hu: the scale, formulas, the μ(HU) line and clipping, tissue medians */
function HuSection({ t }: { t: number }) {
  const on = window01(t, S.hu1[0] - 0.2, S.tis2[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  // vertical HU scale on the left
  const scaleOn = ramp(t, S.hu3[0], S.hu3[0] + 0.8);
  const Y = (hu: number) => 860 - (hu + 1100) / 2700 * 680;
  const marks = [{ hu: -1000, n: "אוויר", c: C.xray, at: S.hu3[0] + 3.5 }, { hu: 0, n: "מים", c: "#7fb2ff", at: S.hu3[0] + 0.8 }, { hu: 1000, n: "עצם צפופה", c: C.bone, at: S.hu3[1] - 4.5 }];
  const tis = info.tissue_hu as Record<string, { median: number; mu: number; p90?: number }>;
  const rows: { k: string; he: string; at: number }[] = [
    { k: "muscle (gluteus maximus)", he: "שריר הישבן", at: S.tis1[0] + 6.0 },
    { k: "bladder (urine)", he: "שתן (שלפוחית)", at: S.tis1[0] + 10.5 },
    { k: "iliac arteries (contrast)", he: "עורקים + חומר ניגוד", at: S.tis2[0] + 2.5 },
    { k: "hip bone", he: "עצם האגן", at: S.tis2[0] + 7.0 },
  ];
  const tisOn = window01(t, S.tis1[0], S.tis2[1] + 0.5, 0.6, 0.6);
  // μ(HU) plot for hu4/hu5
  const plotOn = window01(t, S.hu4[0] + 0.3, S.hu5[1] + 0.4, 0.6, 0.6);
  const clip = ramp(t, S.hu5[1] - 7.0, S.hu5[1] - 5.6);
  const PX0 = 1130, PX1 = 1820, PY0 = 760, PH = 380;
  const hx = (hu: number) => PX0 + (hu + 1200) / 2800 * (PX1 - PX0);
  const my = (mu: number) => PY0 - (mu + 0.004) / 0.036 * PH;
  const neg = ramp(t, S.hu5[0] + 3.0, S.hu5[0] + 4.2);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <Panel o={window01(t, S.hu1[0], S.hu1[1] + 0.3, 0.6, 0.5)} style={{ left: 360, right: 360, top: 260 }}>
        <div style={{ fontSize: 38, lineHeight: 1.7, textAlign: "center" }}>
          <div>CT שומר את <span style={{ color: C.warm }}>μ</span> בכל voxel</div>
          <div style={{ color: C.dim, fontSize: 32 }}>אבל μ תלוי באנרגיה של הקרן, ומכונות שונות עובדות באנרגיות שונות</div>
          <div style={{ marginTop: 12, opacity: ramp(t, S.hu1[1] - 4.5, S.hu1[1] - 3.5) }}>⇐ סקאלה יחסית למים: <b style={{ color: C.xray }}>Hounsfield units (HU)</b></div>
        </div>
      </Panel>
      {scaleOn > 0 && (
        <svg style={{ ...svgFull, opacity: scaleOn * (1 - plotOn * 0) }}>
          <defs><linearGradient id="hug" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stopColor="#000" /><stop offset="1" stopColor="#fff" /></linearGradient></defs>
          <rect x={150} y={Y(1600)} width={34} height={Y(-1100) - Y(1600)} fill="url(#hug)" stroke={C.line} />
          {[-1000, -500, 0, 500, 1000, 1500].map((h) => <g key={h}>
            <line x1={184} y1={Y(h)} x2={198} y2={Y(h)} stroke={C.dim} strokeWidth={2} />
            {T(140, Y(h) + 8, <tspan style={{ direction: "ltr" }}>{h > 0 ? `+${h}` : h}</tspan>, { fill: C.dim, size: 22, anchor: "end" })}</g>)}
          {T(167, Y(1600) - 24, "HU", { fill: C.text, size: 26 })}
          {marks.map((m) => <g key={m.hu} opacity={ramp(t, m.at, m.at + 0.6)}>
            <circle cx={167} cy={Y(m.hu)} r={9} fill={m.c} />
            {T(214, Y(m.hu) + 9, m.n, { fill: m.c, size: 28, anchor: "start" })}</g>)}
          {tisOn > 0 && rows.map((r) => <g key={r.k} opacity={tisOn * ramp(t, r.at, r.at + 0.6)}>
            <polygon points={`200,${Y(tis[r.k].median)} 214,${Y(tis[r.k].median) - 8} 214,${Y(tis[r.k].median) + 8}`} fill={C.warm} />
          </g>)}
        </svg>
      )}
      <Eq t={t} a={S.hu2[0]} b={S.hu3[1] + 0.3} style={{ left: 560, top: 300, width: 900 }} size={54}
        tex={String.raw`\mathrm{HU}=1000\cdot\frac{\hl{mu}{\mu}-\hl{w}{\mu_{w}}}{\hl{w}{\mu_{w}}}`}
        terms={{ mu: [S.hu2[0] + 2.6, S.hu2[0] + 4.0], w: [S.hu2[0] + 4.0, S.hu2[1] + 0.3] }} />
      <Panel o={window01(t, S.hu3[0] + 1.0, S.hu3[1] + 0.3, 0.5, 0.5)} style={{ left: 560, top: 640, width: 900, padding: "18px 30px", direction: "ltr" }}>
        <div style={{ ...mono, fontSize: 30, lineHeight: 1.7, textAlign: "left" }}>
          <div>μ = μ<sub>w</sub> → HU = 0</div>
          <div style={{ opacity: ramp(t, S.hu3[0] + 3.5, S.hu3[0] + 4.2) }}>μ ≈ 0 (air) → HU = −1000</div>
          <div style={{ opacity: ramp(t, S.hu3[1] - 4.5, S.hu3[1] - 3.8) }}>μ = 2μ<sub>w</sub> → HU = +1000</div>
        </div>
      </Panel>
      <Eq t={t} a={S.hu4[0] + 0.3} b={S.hu5[1] + 0.4} style={{ left: 420, top: 250, width: 620 }} size={46}
        tex={String.raw`\hl{mu}{\mu}=\hl{w}{\mu_w}\left(1+\frac{\mathrm{HU}}{1000}\right)`} note="μ_w = 0.02 mm⁻¹"
        terms={{ mu: [S.hu4[0] + 5.0, S.hu4[0] + 8.0], w: [S.hu4[1] - 4.5, S.hu4[1] + 0.3] }} />
      <Panel o={window01(t, S.hu5[1] - 7.4, S.hu5[1] + 0.4, 0.5, 0.5)} style={{ left: 380, top: 650, width: 700, padding: "18px 30px", direction: "ltr", whiteSpace: "nowrap" }}>
        <div style={{ ...mono, fontSize: 30, lineHeight: 1.7, textAlign: "left" }}>
          <div style={{ color: C.warm }}>hu = np.clip(hu, -1000, None)</div>
          <div style={{ color: C.dim, fontSize: 26 }}>our crop: {(info.hist.frac.below_m1000 * 100).toFixed(1)}% of voxels &lt; −1000 (min {info.hist.min})</div>
        </div>
      </Panel>
      {plotOn > 0 && (
        <svg style={{ ...svgFull, opacity: plotOn }}>
          <line x1={PX0} y1={my(0)} x2={PX1} y2={my(0)} stroke={C.dim} strokeWidth={2} />
          <line x1={hx(-1000)} y1={PY0} x2={hx(-1000)} y2={PY0 - PH} stroke={C.dim} strokeDasharray="5 7" />
          <line x1={hx(0)} y1={PY0} x2={hx(0)} y2={PY0 - PH} stroke={C.dim} strokeDasharray="5 7" />
          <line x1={hx(-1000)} y1={my(0)} x2={hx(1600)} y2={my(MUW * 2.6)} stroke={C.xray} strokeWidth={4} />
          <line x1={hx(-1200)} y1={my(-MUW * 0.2)} x2={hx(-1000)} y2={my(0)} stroke={C.bad} strokeWidth={4} opacity={neg * (1 - clip)} />
          <line x1={hx(-1200)} y1={my(0)} x2={hx(-1000)} y2={my(0)} stroke={C.good} strokeWidth={5} opacity={clip} />
          {T(hx(-1100), my(0) - 40, "μ < 0 ?!", { fill: C.bad, size: 26 })}
          {T(hx(-1000), PY0 + 40, "−1000", { fill: C.dim, size: 22 })}
          {T(hx(0), PY0 + 34, "0", { fill: C.dim, size: 22 })}
          {T(PX1, PY0 + 34, "HU", { fill: C.dim, size: 24, anchor: "end" })}
          {T(PX0 + 10, PY0 - PH - 14, "μ (mm⁻¹)", { fill: C.xray, size: 24, anchor: "start" })}
          {T(hx(0) + 10, my(MUW) - 12, "0.02", { fill: C.xray, size: 22, anchor: "start" })}
          <circle cx={hx(0)} cy={my(MUW)} r={7} fill={C.xray} />
        </svg>
      )}
      {/* tissue table (real medians from the masks of this CT) */}
      <Panel o={tisOn} style={{ left: 520, top: 220, width: 1100 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr 1fr 1fr", rowGap: 14, fontSize: 32, alignItems: "center" }}>
          <div style={{ color: C.dim }}>רקמה (מסכה)</div><div style={{ color: C.dim, textAlign: "center" }}>חציון HU</div>
          <div style={{ color: C.dim, textAlign: "center" }}>טווח 10%–90%</div><div style={{ color: C.dim, textAlign: "center", ...mono }}>μ (mm⁻¹)</div>
          {rows.map((r) => {
            const v = tis[r.k] as { median: number; mu: number; p10: number; p90: number }, k = ramp(t, r.at, r.at + 0.6);
            return [
              <div key={r.k + "a"} style={{ opacity: k }}>{r.he}</div>,
              <div key={r.k + "b"} style={{ opacity: k, textAlign: "center", color: C.warm, ...mono }}>{Math.round(v.median)}</div>,
              <div key={r.k + "c"} style={{ opacity: k, textAlign: "center", color: C.dim, ...mono }}>{Math.round(v.p10)} … {Math.round(v.p90)}</div>,
              <div key={r.k + "d"} style={{ opacity: k, textAlign: "center", ...mono }}>{v.mu.toFixed(4)}</div>,
            ];
          })}
        </div>
        <div style={{ fontSize: 26, color: C.dim, marginTop: 20, opacity: ramp(t, S.tis2[0] + 9.0, S.tis2[0] + 10.0) }}>
          עצם האגן: רוב הנפח עצם ספוגית; רק הקליפה החיצונית (cortex) מגיעה לאלף ומעלה
        </div>
      </Panel>
    </div>
  );
}

/** the HU histogram of our CT, with the peaks found in the prep script */
function Histogram({ t, a, b, x0 = 200, x1 = 1760, y0 = 800, h = 520, band, zoomTail = 0, labels = true, small = false }: {
  t: number; a: number; b: number; x0?: number; x1?: number; y0?: number; h?: number; band?: [number, number]; zoomTail?: number; labels?: boolean; small?: boolean;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const H = info.hist, e = H.edges, c = H.counts;
  const hmin = -1100, hmax = 1700;
  const X = (hu: number) => x0 + (hu - hmin) / (hmax - hmin) * (x1 - x0);
  const cmax = Math.max(...c) * 1.08;
  const Yc = (v: number) => y0 - Math.min(v, cmax) / cmax * h;
  const draw = ramp(t, a + 0.3, a + 2.5);
  const nShow = Math.floor(c.length * draw);
  const path = (mul: number, from: number) => {
    let d = "";
    for (let i = 0; i < nShow; i++) {
      if (e[i] < from) continue;
      d += `${d ? "L" : "M"}${X(e[i])},${Yc(c[i] * mul)} L${X(e[i + 1])},${Yc(c[i] * mul)} `;
    }
    return d;
  };
  const pk = H.peaks;
  const names = ["אוויר", "שומן", "רקמה רכה"];
  const pkT = [S.hist2[0] + 1.0, S.hist3[0] + 0.5, S.hist3[0] + 4.5];
  return (
    <svg style={{ ...svgFull, opacity: o }}>
      {band && <rect x={X(band[0])} y={y0 - h - 10} width={X(band[1]) - X(band[0])} height={h + 10} fill={C.warm} fillOpacity={0.12} stroke={C.warm} strokeWidth={2} />}
      <path d={path(1, -1e9) + `L${X(e[nShow] ?? hmax)},${y0} L${x0},${y0} Z`} fill="rgba(127,208,255,0.25)" stroke={C.xray} strokeWidth={small ? 1.5 : 2.5} />
      {zoomTail > 0 && <path d={path(30, 150)} fill="none" stroke={C.bone} strokeWidth={3} opacity={zoomTail} />}
      <line x1={x0} y1={y0} x2={x1} y2={y0} stroke={C.dim} strokeWidth={2} />
      {[-1000, -500, 0, 500, 1000, 1500].map((v) => <g key={v}>
        <line x1={X(v)} y1={y0} x2={X(v)} y2={y0 + 10} stroke={C.dim} strokeWidth={2} />
        {T(X(v), y0 + (small ? 30 : 40), <tspan style={{ direction: "ltr" }}>{v}</tspan>, { fill: C.dim, size: small ? 18 : 24 })}</g>)}
      {!small && T(x1, y0 + 80, "HU", { fill: C.dim, size: 26, anchor: "end" })}
      {!small && T(x0 + 6, y0 - h - 20, "מספר voxels", { fill: C.dim, size: 24, anchor: "start" })}
      {labels && pk.map((p, i) => (
        <g key={i} opacity={ramp(t, pkT[i], pkT[i] + 0.6)}>
          <line x1={X(p.hu)} y1={Yc(p.count) - 10} x2={X(p.hu)} y2={Yc(p.count) - 60} stroke={C.warm} strokeWidth={2} />
          {T(X(p.hu), Yc(p.count) - 104, names[i], { fill: C.warm, size: 30 })}
          {T(X(p.hu), Yc(p.count) - 70, <tspan style={{ direction: "ltr" }}>{Math.round(p.hu)}</tspan>, { fill: C.warm, size: 28 })}
        </g>))}
      {zoomTail > 0 && <g opacity={zoomTail}>
        {T(X(900), Yc(c[Math.round((600 - hmin) / 10)] * 30) - 40, "עצם (×30): זנב, בלי פסגה", { fill: C.bone, size: 30 })}
        {T(X(900), Yc(c[Math.round((600 - hmin) / 10)] * 30) + 0, <tspan style={{ direction: "ltr" }}>{`> 300 HU: ${(H.frac.bone_gt_300 * 100).toFixed(1)}%`}</tspan>, { fill: C.bone, size: 26 })}
      </g>}
    </svg>
  );
}

/** windowing: the real axial slice, re-windowed live with an SVG filter */
function WindowSection({ t }: { t: number }) {
  const on = window01(t, S.win1[0] - 0.2, S.win4[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const E = info.window_png;
  const W0: [number, number] = [E.enc_lo, E.enc_hi], W1: [number, number] = [-350, 1100], W2: [number, number] = [-160, 240], W3: [number, number] = [-500, 1300];
  const steps: [number, [number, number]][] = [[S.win1[0] + 6.0, W0], [S.win1[0] + 10.0, W1], [S.win2[0] + 1.5, W1], [S.win2[0] + 3.5, W2],
    [S.win3[0] + 0.8, W2], [S.win3[0] + 2.8, W3], [S.win4[0] + 0.5, W3], [S.win4[0] + 2.5, W1]];
  let w = W0;
  for (let i = 0; i < steps.length - 1; i++) {
    const [ta, wa] = steps[i], [tb, wb] = steps[i + 1];
    if (t >= ta) { const k = ramp(t, ta, tb); w = [lerp(wa[0], wb[0], k), lerp(wa[1], wb[1], k)]; }
  }
  const a = (w[0] - E.enc_lo) / (E.enc_hi - E.enc_lo), b = (w[1] - E.enc_lo) / (E.enc_hi - E.enc_lo);
  const slope = 1 / (b - a), icpt = -a / (b - a);
  const z = 3.5, iw = 248 * z, ih = 146 * z;
  const id = `win${Math.round(t * 60)}`;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <svg width={0} height={0} style={{ position: "absolute" }}>
        <filter id={id} colorInterpolationFilters="sRGB">
          <feComponentTransfer>
            <feFuncR type="linear" slope={slope} intercept={icpt} />
            <feFuncG type="linear" slope={slope} intercept={icpt} />
            <feFuncB type="linear" slope={slope} intercept={icpt} />
          </feComponentTransfer>
        </filter>
      </svg>
      <Img src={staticFile("data/v02/axial_hu.png")} style={{ position: "absolute", left: 90, top: 200, width: iw, height: ih, filter: `url(#${id})`, borderRadius: 6 }} />
      <Label x={90 + iw / 2} y={200 + ih + 44} size={36} color={C.warm}
        text={<span>חלון: <span style={mono}>[{Math.round(w[0])}, {Math.round(w[1])}]</span> HU</span>} />
      <Histogram t={t} a={S.win1[0] - 0.2} b={S.win4[1] + 0.5} x0={1040} x1={1820} y0={520} h={260} band={w} labels={false} small />
      <Eq t={t} a={S.win1[0] + 1.0} b={S.win4[1] + 0.5} style={{ left: 1040, top: 610, width: 780, padding: "14px 24px" }} size={38}
        tex={String.raw`g=\mathrm{clip}\!\left(\frac{\mathrm{HU}-\hl{lo}{lo}}{\hl{hi}{hi}-\hl{lo}{lo}},\,0,\,1\right)\cdot 255`}
        terms={{ lo: [S.win1[0] + 10.5, S.win1[1]], hi: [S.win1[0] + 10.5, S.win1[1]] }} />
    </div>
  );
}

/** saturation: 1 − e^(−L) vs L */
function SatSection({ t }: { t: number }) {
  const on = window01(t, S.sat1[0] - 0.2, S.sat5[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const Lp = info.L_ap;
  const satImg = window01(t, S.sat2[0], S.sat3[0] + 0.6, 0.5, 0.5);
  const plotOn = window01(t, S.sat3[0] + 0.2, S.sat3[1] + 0.4, 0.6, 0.5);
  const cmp = window01(t, S.sat4[0], S.sat5[1] + 0.5, 0.6, 0.6);
  const X0 = 260, X1 = 1500, Y0 = 800, HH = 520, Lm = 8;
  const X = (L: number) => X0 + L / Lm * (X1 - X0), Y = (v: number) => Y0 - v * HH;
  const curve = Array.from({ length: 200 }, (_, i) => { const L = i / 199 * Lm; return `${i ? "L" : "M"}${X(L)},${Y(1 - Math.exp(-L))}`; }).join(" ");
  const hc = Lp.hist.counts, he = Lp.hist.edges, hmax = Math.max(...hc);
  const band = ramp(t, S.sat3[0] + 6.0, S.sat3[0] + 7.2);
  const flat = ramp(t, S.sat3[0] + 12.0, S.sat3[0] + 13.2);
  const ih = 520, iw = ih * 248 / 235;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <Eq t={t} a={S.sat1[0] + 0.3} b={S.sat1[1] + 0.3} style={{ left: 460, top: 300, width: 1000 }} size={56}
        tex={String.raw`\frac{I}{I_0}=e^{-L}\qquad\text{display}=\hl{s}{1-e^{-L}}\;?`}
        terms={{ s: [S.sat1[1] - 5.0, S.sat1[1] + 0.3] }} />
      {satImg > 0 && (
        <div style={{ opacity: satImg }}>
          <Img src={staticFile("data/hero/drr_ap_saturated.png")} style={{ position: "absolute", left: 960 - iw / 2, top: 200, width: iw, height: ih, borderRadius: 6 }} />
          <Label x={960} y={770} text={<span style={mono}>1 − e<sup>−L</sup></span>} size={40} color={C.bad} />
        </div>
      )}
      {plotOn > 0 && (
        <svg style={{ ...svgFull, opacity: plotOn }}>
          {hc.map((v, i) => <rect key={i} x={X(he[i])} y={Y0 - v / hmax * 160} width={X(he[i + 1]) - X(he[i]) - 1} height={v / hmax * 160} fill={C.xray} fillOpacity={0.35} />)}
          <g opacity={band}>
            <rect x={X(Lp.p1)} y={Y0 - HH} width={X(Lp.p99) - X(Lp.p1)} height={HH} fill={C.warm} fillOpacity={0.08} />
            <line x1={X(Lp.p1)} y1={Y0} x2={X(Lp.p1)} y2={Y(Lp.sat_p1)} stroke={C.warm} strokeDasharray="6 6" strokeWidth={2} />
            <line x1={X(Lp.p99)} y1={Y0} x2={X(Lp.p99)} y2={Y(Lp.sat_p99)} stroke={C.warm} strokeDasharray="6 6" strokeWidth={2} />
            {T(X(Lp.p50), Y0 - 180, "הפיקסלים שלנו", { fill: C.xray, size: 26 })}
            {T(X(Lp.p1), Y0 + 72, Lp.p1.toFixed(1), { fill: C.warm, size: 24 })}
            {T(X(Lp.p99), Y0 + 72, Lp.p99.toFixed(1), { fill: C.warm, size: 24 })}
          </g>
          <g opacity={flat}>
            <rect x={X0} y={Y(Lp.sat_p99)} width={X(Lp.p99) - X0} height={Y(Lp.sat_p1) - Y(Lp.sat_p99)} fill={C.bad} fillOpacity={0.15} />
            <line x1={X0} y1={Y(Lp.sat_p1)} x2={X(Lp.p1)} y2={Y(Lp.sat_p1)} stroke={C.bad} strokeDasharray="6 6" strokeWidth={2} />
            <line x1={X0} y1={Y(Lp.sat_p99)} x2={X(Lp.p99)} y2={Y(Lp.sat_p99)} stroke={C.bad} strokeDasharray="6 6" strokeWidth={2} />
            {T(X0 - 12, Y(Lp.sat_p1) + 10, Lp.sat_p1.toFixed(2), { fill: C.bad, size: 24, anchor: "end" })}
            {T(X0 - 12, Y(Lp.sat_p99) - 2, Lp.sat_p99.toFixed(2), { fill: C.bad, size: 24, anchor: "end" })}
          </g>
          <path d={curve} stroke={C.bad} strokeWidth={4} fill="none" />
          <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke={C.dim} strokeWidth={2} />
          <line x1={X0} y1={Y0} x2={X0} y2={Y0 - HH - 20} stroke={C.dim} strokeWidth={2} />
          {T(X1, Y0 + 80, "L (האינטגרל הקווי)", { fill: C.dim, size: 26, anchor: "end" })}
          {[0, 2, 4, 6, 8].map((v) => T(X(v), Y0 + 40, String(v), { fill: C.dim, size: 22 }))}
          {T(X(6.6), Y(1) - 20, "1 − e^(−L)", { fill: C.bad, size: 30 })}
          {T(X0 - 12, Y(0) + 8, "0", { fill: C.dim, size: 22, anchor: "end" })}
        </svg>
      )}
      <Panel o={plotOn * band} style={{ left: 1540, top: 320, width: 330, padding: "18px 24px", direction: "ltr" }}>
        <div style={{ ...mono, fontSize: 26, lineHeight: 1.7, textAlign: "left" }}>
          <div>median L = {Lp.p50.toFixed(2)}</div>
          <div>1–99%: {Lp.p1.toFixed(1)} … {Lp.p99.toFixed(1)}</div>
          <div style={{ color: C.bad }}>1−e<sup>−L</sup>: {Lp.sat_p1.toFixed(2)} … {Lp.sat_p99.toFixed(2)}</div>
        </div>
      </Panel>
      {cmp > 0 && (
        <div style={{ opacity: cmp }}>
          <Img src={staticFile("data/hero/drr_ap_saturated.png")} style={{ position: "absolute", left: 250, top: 190, width: iw, height: ih, borderRadius: 6, outline: `3px solid ${C.bad}` }} />
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 1670 - iw, top: 190, width: iw, height: ih, borderRadius: 6, outline: `3px solid ${C.good}` }} />
          <Label x={250 + iw / 2} y={760} text={<span><span style={{ color: C.bad }}>✗</span> <span style={mono}>1 − e<sup>−L</sup></span></span>} size={38} />
          <Label x={1670 - iw / 2} y={760} text={<span><span style={{ color: C.good }}>✓</span> <span style={mono}>L = −ln(I/I₀)</span></span>} size={38} />
          <Label x={1670 - iw / 2} y={808} text="(עם מתיחת ניגודיות לתצוגה)" size={24} color={C.dim} />
        </div>
      )}
    </div>
  );
}

export const V02 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const spine = useObj("data/hero/spine.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const ready = pelvis && femurs && spine && ap;
  const cam = camAt(keys, t);
  const ray3d = window01(t, S.ray1[0] + 0.5, S.ray1[1] + 0.5, 0.6, 0.5);
  const pS = project(cam, [rx, ry, zSrc]), pD = project(cam, [rx, ry, zDet]);

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ pelvis, femurs, spine, ap }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.62 * window01(t, S.pipe[0] - 0.3, tl.total, 0.8, 0.1), pointerEvents: "none" }} />

      <Opening n={2} title="מאיפה מגיעים המספרים של CT" active={[0, 1]} t={t} end={2.9} />
      <Title text="למה עצם ≈ 1000, ואוויר = −1000?" t={t} a={S.intro1[0] + 4.0} b={S.intro1[1] + 0.4} top={90} size={60} />
      <Label x={pS[0]} y={pS[1] - 60} text={<span style={mono}>I₀</span>} size={46} color={C.xray} o={ray3d} />
      <Label x={pD[0]} y={pD[1] - 90} text={<span style={mono}>I &lt; I₀</span>} size={46} color={C.xray} o={ray3d * ramp(t, S.ray1[0] + 2.6, S.ray1[0] + 3.3)} />
      <Label x={pS[0]} y={pS[1] + 70} text="מקור" size={28} color={C.dim} o={ray3d} />
      <Label x={pD[0]} y={pD[1] + 110} text="גלאי" size={28} color={C.dim} o={ray3d} />

      <SlabSection t={t} />
      <RealRay t={t} />
      <Title text="צילום מדומה (DRR)" sub="לכל פיקסל: L של הקרן שלו" t={t} a={S.real3[0] + 0.3} b={S.real3[1] + 0.4} top={70} size={60} />
      <HuSection t={t} />
      <Histogram t={t} a={S.hist1[0]} b={S.hist4[1] + 0.5} zoomTail={ramp(t, S.hist4[1] - 7.0, S.hist4[1] - 5.8)} />
      <Title text="ההיסטוגרמה של הסריקה שלנו" t={t} a={S.hist1[0]} b={S.hist2[0] + 2.0} top={70} size={54} />
      <WindowSection t={t} />
      <SatSection t={t} />
      <Eq t={t} a={S.sat5[0] + 0.2} b={S.sat5[1] + 0.5} style={{ left: 660, top: 860 - 30, width: 600, padding: "6px 24px" }} size={36}
        tex={String.raw`\text{CT: } \; p = -\ln\!\left(I/I_0\right) = L`} />

      <PipelineMap active={[0, 1]} t={t} a={S.pipe[0]} b={S.pipe[1] + 0.4} top={440} />
      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        <span>Beer–Lambert: <span style={mono}>I = I₀·e^(−L)</span>, כאשר <span style={mono}>L = ∫μ ds</span> (האינטגרל הקווי)</span>,
        <span><span style={mono}>HU = 1000(μ − μ_w)/μ_w</span>: מים 0, אוויר <span style={mono}>−1000</span></span>,
        <span>בקוד: <span style={mono}>μ = μ_w(1 + HU/1000)</span>, אחרי clip ב-<span style={mono}>−1000</span></span>,
        <span>בסריקה שלנו: פסגות ב-<span style={mono}>−945, −105, +45</span>; העצם היא זנב</span>,
        <span>חלון = תצוגה בלבד; מציגים את <span style={mono}>L</span>, לא את <span style={mono}>1 − e^(−L)</span></span>,
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/01-data-and-drr/notebook.ipynb" todos={[
        "לכתוב hu_to_mu: המרה של כל voxel מ-HU ל-μ",
        <span>לחתוך (clip): ערכים מתחת ל-<span style={mono}>−1000</span> הופכים ל-<span style={mono}>−1000</span></span>,
        "לצייר היסטוגרמה של HU ולמצוא את הפסגות (אוויר, שומן, רקמה רכה)",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v02" />
    </AbsoluteFill>
  );
};
