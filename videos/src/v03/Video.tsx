// Video 3 · Cameras and projection. Pinhole model, perspective vs parallel, homogeneous coordinates,
// K and [R|t], a worked P = K[R|t] example on a real point of the hero pelvis, and the X-ray mapping.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import katex from "katex";
import type { ReactElement } from "react";
import script from "./script.json";
import durations from "../../public/audio/v03/durations.json";
import { buildTimeline, Script, span } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, Detector, PointSource, Rays } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { ArrowObject, Marker, PinholeBox, toScreen, Triad } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;

type Ex = { SDD: number; SOD: number; pix_mm: number; N: number; f_px: number; c: number[]; X: number[]; Xc: number[]; uv: number[]; mag_point: number };
type Pts = { points: number[]; landmarks_mm: Record<string, number[]>; example: Ex };
type V04 = { ap_iso_extent: number[]; M_iso: number };
type V3 = [number, number, number];

// ---- geometry of the X-ray "camera" (scene units, 1 = 100 mm) ----
const SRC: V3 = [0, 0, -7];       // SOD = 700 mm in front of the pelvis centroid
const DET_Z = 3;                  // SDD = 1000 mm from the source
const hitDet = (p: number[]): V3 => {
  const s = (DET_Z - SRC[2]) / (p[2] - SRC[2]);
  return [SRC[0] + (p[0] - SRC[0]) * s, SRC[1] + (p[1] - SRC[1]) * s, DET_Z];
};

// static cameras, reused for screen-space labels
type Cam = { pos: V3; look: V3 };
const CAM_EX: Cam = { pos: [19.0, 3.4, -2.6], look: [0, 0, 1.8] };
const CAM_XR: Cam = { pos: [13.2, 2.4, -1.6], look: [0, -0.2, -2.0] };
const CAM_RT: Cam = { pos: [7.8, 3.6, -9.8], look: [-1.6, 0.2, -3.4] };

const texLine = (s: string) => <span dangerouslySetInnerHTML={{ __html: katex.renderToString(s, { strict: false }) }} />;

// ---------------- 3D scene ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const ex = a.pts.example;
  const P = ex.X.map((v) => v * 0.01) as V3;
  const hit = hitDet(P);
  const e = a.v04.ap_iso_extent, M = a.v04.M_iso;
  const detSize: [number, number] = [(e[1] - e[0]) * M, (e[3] - e[2]) * M];
  const detPos: V3 = [((e[0] + e[1]) / 2) * M, ((e[2] + e[3]) / 2) * M, DET_Z];

  const introOn = window01(t, 2.4, S.intro2[1] + 0.5, 1.0, 0.8);
  const pinOn = window01(t, S.pin1[0] - 0.2, S.pin2[1] + 0.5, 0.8, 0.6);
  const raysK = ramp(t, S.pin2[0] + 0.3, S.pin2[0] + 4.5);
  const imgOn = ramp(t, S.pin2[0] + 4.0, S.pin2[0] + 5.5);
  const [r1, r3] = span(tl, "rt1", "rt3");
  const rtOn = window01(t, r1, r3 + 0.4, 0.8, 0.6);
  const rtMove = ramp(t, S.rt2[0] + 2.0, S.rt2[1] - 1.0);
  const [e1, e3] = span(tl, "ex1", "ex3");
  const exOn = window01(t, e1, e3 + 0.5, 0.8, 0.6);
  const exRay = ramp(t, S.ex2[0] + 0.5, S.ex2[0] + 2.5);
  const exPix = ramp(t, S.ex3[0] + 3.0, S.ex3[0] + 4.0);
  const [x1, x3] = span(tl, "xr1", "xr3");
  const xrOn = window01(t, x1, x3 + 0.6, 0.8, 0.6);
  const tailOn = window01(t, S.recap[0], tl.total, 1.0, 1.0);

  const rigOn = Math.max(introOn, exOn, xrOn, tailOn * 0.8);
  const boneO = Math.max(introOn, rtOn * 0.35, exOn * 0.5, xrOn * 0.75, tailOn * 0.8);

  // fan of rays: source -> grid of detector points
  const fan: V3[][] = [];
  for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) {
    fan.push([SRC, [detPos[0] + (i / 6 - 0.5) * detSize[0], detPos[1] + (j / 6 - 0.5) * detSize[1], DET_Z]]);
  }
  const corners: V3[][] = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) =>
    [SRC, [detPos[0] + sx * detSize[0] / 2, detPos[1] + sy * detSize[1] / 2, DET_Z]]);

  // pinhole: object at z = -3, rays from object points through the hole (0,0,0) to the back wall
  const OBJ_Z = -3, BACK = 2.2, k = BACK / -OBJ_Z;
  const objPts: V3[] = [[0, 1.1, 0], [0, 0.6, 0], [0, -0.9, 0], [0.49, 0.58, 0], [0.49, 0.32, 0]];
  const pinRays = objPts.slice(0, Math.max(0, Math.ceil(raysK * objPts.length))).map(([x, y]) =>
    [[x, y, OBJ_Z], [-x * k, -y * k, BACK - 0.01]] as V3[]);

  const keys: CamKey[] = [
    { t: 0, pos: [10.5, 3.4, -8.5], look: [0, 0, -2.0] },
    { t: S.intro2[1], pos: [12.0, 2.2, -4.0], look: [0, 0, -2.0] },
    { t: S.pin1[0] + 0.2, pos: [5.6, 2.4, -6.2], look: [0, 0, -0.5] },
    { t: S.pin2[1], pos: [6.4, 1.4, -2.6], look: [0, 0, -0.3] },
    { t: S.pin3[0], pos: [6.4, 1.4, -2.6], look: [0, 0, -0.3] },
    { t: r1 - 0.4, ...CAM_RT },
    { t: r3, ...CAM_RT },
    { t: e1, ...CAM_EX },
    { t: e3, ...CAM_EX },
    { t: x1, ...CAM_XR },
    { t: x3, ...CAM_XR },
    { t: S.recap[0] + 1, pos: [10.5, 3.4, -9.0], look: [0, 0, -2.0] },
    { t: tl.total, pos: [12.0, 2.6, -5.5], look: [0, 0, -2.0] },
  ];

  // [R|t] demo: a copy of the world triad rotates 180 deg about z (R), then moves to the source (t)
  const rot = Math.PI * ramp(rtMove, 0, 0.5, (x) => x);
  const mv = ramp(rtMove, 0.5, 1, (x) => x);

  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.4} opacity={0.6 * Math.max(rigOn, pinOn, rtOn)} />
      <Bone geo={a.pelvis} material={bone} opacity={boneO} />

      {/* X-ray rig */}
      <PointSource position={SRC} opacity={Math.max(rigOn, rtOn)} r={0.1} />
      <Detector tex={a.persp} progress={1} opacity={rigOn} size={detSize} position={detPos} rotation={[0, Math.PI, 0]} />
      <Rays segs={fan} opacity={0.18 * Math.max(introOn, xrOn, tailOn)} />
      <Rays segs={corners} opacity={0.55 * Math.max(introOn, xrOn, exOn * 0.6, tailOn)} />

      {/* worked example: one landmark, its ray, its pixel */}
      <Marker position={P} opacity={exOn + xrOn * ramp(t, S.xr3[0], S.xr3[0] + 1)} r={0.08} />
      {exOn > 0 && <Rays segs={[[SRC, [lerp(SRC[0], hit[0], exRay), lerp(SRC[1], hit[1], exRay), lerp(SRC[2], hit[2], exRay)]]]} opacity={exOn} color={[3.0, 2.2, 0.9]} />}
      <Marker position={[hit[0], hit[1], hit[2] - 0.02]} opacity={exOn * exPix} r={0.09} color={[3.4, 2.4, 0.8]} />

      {/* coordinate frames */}
      <Triad position={[0, 0, 0]} opacity={rtOn} scale={1.5} />
      <Triad position={[0, 0, lerp(0, SRC[2], mv)]} rotation={[0, 0, rot]} opacity={rtOn * ramp(t, S.rt2[0] + 1.0, S.rt2[0] + 2.0)} scale={0.9} />
      {rtOn > 0 && <Rays segs={[[[0, 0, 0], SRC]]} opacity={0.35 * rtOn * ramp(t, S.rt2[0] + 1, S.rt2[0] + 2)} color={[1.6, 1.4, 0.6]} />}

      {/* pinhole camera */}
      <PinholeBox opacity={pinOn} depth={BACK} />
      <group position={[0, 0, OBJ_Z]}><ArrowObject opacity={pinOn} /></group>
      <group position={[0, 0, BACK - 0.03]} scale={[-k, -k, 1]}>
        <ArrowObject opacity={pinOn * imgOn} flat emissive={new THREE.Color(1.6, 1.2, 0.4)} />
      </group>
      {pinOn > 0 && <Rays segs={pinRays} opacity={pinOn} color={[2.2, 1.8, 0.8]} />}
    </>
  );
}

// ---------------- 2D diagrams ----------------
const svgText = (x: number, y: number, s: string, color: string = C.text, size = 34, italic = true, anchor: "start" | "middle" | "end" = "middle") => (
  <text x={x} y={y} fill={color} fontSize={size} fontFamily={italic ? "KaTeX_Math, serif" : FONT} fontStyle={italic ? "italic" : "normal"} textAnchor={anchor}>{s}</text>
);

/** side view of the pinhole: similar triangles, then the virtual plane in front */
function SideView({ t }: { t: number }) {
  const o = window01(t, S.pin3[0] - 0.2, S.pin5[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const ox = 1020, oy = 560, f = 250, Z = 620, X = 230;
  const objX = ox - Z, objY = oy - X;
  const imgX = ox + f, imgY = oy + X * f / Z;
  const virtX = ox - f, virtY = oy - X * f / Z;
  const tri = ramp(t, S.pin4[0] + 0.5, S.pin4[0] + 1.5);
  const virt = ramp(t, S.pin5[0] + 0.5, S.pin5[0] + 2.0);
  const lab = ramp(t, S.pin3[0] + 2.0, S.pin3[0] + 3.0);
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      <line x1={250} y1={oy} x2={1460} y2={oy} stroke={C.dim} strokeWidth={2} strokeDasharray="10 10" />
      <line x1={imgX} y1={oy - 300} x2={imgX} y2={oy + 300} stroke={C.xray} strokeWidth={5} opacity={1 - 0.6 * virt} />
      <line x1={ox} y1={oy - 300} x2={ox} y2={oy - 10} stroke={C.line} strokeWidth={6} />
      <line x1={ox} y1={oy + 10} x2={ox} y2={oy + 300} stroke={C.line} strokeWidth={6} />
      <circle cx={ox} cy={oy} r={8} fill={C.warm} />
      <polygon points={`${ox},${oy} ${objX},${oy} ${objX},${objY}`} fill="rgba(255,207,127,0.18)" opacity={tri} stroke={C.warm} strokeWidth={2} />
      <polygon points={`${ox},${oy} ${imgX},${oy} ${imgX},${imgY}`} fill="rgba(127,208,255,0.22)" opacity={tri * (1 - virt)} stroke={C.xray} strokeWidth={2} />
      <line x1={objX} y1={oy} x2={objX} y2={objY + 14} stroke={C.warm} strokeWidth={8} />
      <polygon points={`${objX - 18},${objY + 22} ${objX + 18},${objY + 22} ${objX},${objY - 6}`} fill={C.warm} />
      <line x1={objX} y1={objY} x2={imgX} y2={imgY} stroke="#fff3c4" strokeWidth={3} />
      <circle cx={imgX} cy={imgY} r={9} fill={C.xray} opacity={1 - 0.6 * virt} />
      <g opacity={virt}>
        <line x1={virtX} y1={oy - 300} x2={virtX} y2={oy + 300} stroke={C.good} strokeWidth={5} />
        <polygon points={`${ox},${oy} ${virtX},${oy} ${virtX},${virtY}`} fill="rgba(125,255,176,0.2)" stroke={C.good} strokeWidth={2} />
        <circle cx={virtX} cy={virtY} r={10} fill={C.good} />
        {svgText(virtX, oy + 345, "image plane (virtual)", C.good, 26, false)}
        {svgText(virtX - 30, (oy + virtY) / 2 + 10, "x", C.good, 38, true, "end")}
      </g>
      <g opacity={lab}>
        {svgText((objX + ox) / 2, oy + 50, "Z")}
        {svgText(objX - 30, (oy + objY) / 2 + 10, "X", C.warm, 38, true, "end")}
        {svgText((ox + imgX) / 2, oy - 18, "f", C.xray)}
        {svgText(imgX + 30, (oy + imgY) / 2 + 12, "x", C.xray, 38, true, "start")}
        {svgText(ox, oy - 320, "pinhole", C.warm, 26, false)}
        {svgText(imgX + 40, oy - 320, "image plane", C.xray, 26, false)}
      </g>
    </svg>
  );
}

/** two equal arrows at two depths: perspective (rays to a point) vs parallel (rays straight) */
function DepthDemo({ t }: { t: number }) {
  const o = window01(t, S.persp1[0] - 0.2, S.persp2[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const par = ramp(t, S.persp2[0] + 0.8, S.persp2[0] + 3.0);
  const oy = 700, cx = 260, ipX = 520, h = 230;
  const objs = [{ x: 900, col: C.warm, name: "Z₁" }, { x: 1500, col: C.xray, name: "Z₂" }];
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      <line x1={180} y1={oy} x2={1720} y2={oy} stroke={C.dim} strokeWidth={2} strokeDasharray="10 10" />
      <line x1={ipX} y1={oy - 330} x2={ipX} y2={oy + 60} stroke={C.text} strokeWidth={4} />
      {svgText(ipX, oy + 100, "image plane", C.dim, 26, false)}
      <g opacity={1 - par}>
        <circle cx={cx} cy={oy} r={10} fill={C.warm} />
        {svgText(cx, oy + 50, "centre", C.dim, 26, false)}
      </g>
      {objs.map((ob, i) => {
        const hp = h * (ipX - cx) / (ob.x - cx);
        const hi = lerp(hp, h, par);
        const fromX = lerp(cx, ipX, par);
        const fromY = lerp(oy, oy - h, par);
        const barX = ipX + 10 + i * 16;
        return (
          <g key={i}>
            <line x1={ob.x} y1={oy} x2={ob.x} y2={oy - h + 16} stroke={ob.col} strokeWidth={8} />
            <polygon points={`${ob.x - 18},${oy - h + 24} ${ob.x + 18},${oy - h + 24} ${ob.x},${oy - h - 6}`} fill={ob.col} />
            <line x1={ob.x} y1={oy - h} x2={fromX} y2={fromY} stroke={ob.col} strokeWidth={2.5} opacity={0.9} />
            <line x1={barX} y1={oy} x2={barX} y2={oy - hi} stroke={ob.col} strokeWidth={10} />
            {svgText(ob.x, oy + 50, ob.name, ob.col)}
          </g>
        );
      })}
      <g opacity={par}>{svgText(1100, 330, "parallel: size does not depend on Z", C.good, 32, false)}</g>
      <g opacity={1 - par}>{svgText(1100, 330, "perspective: farther = smaller", C.warm, 32, false)}</g>
    </svg>
  );
}

/** the hero pelvis point cloud projected, from parallel to strong perspective */
function CloudProjection({ t, pts }: { t: number; pts: number[] }) {
  const o = window01(t, S.persp3[0] - 0.2, S.persp3[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const k = ramp(t, S.persp3[0] + 4.0, S.persp3[1] - 2.0);
  const D = 1 / lerp(1 / 20000, 1 / 380, k);                 // source distance, mm (interpolated in 1/D)
  const Du = D * 0.01;
  const sc = 200, cx = 960, cy = 500;
  const n = pts.length / 3;
  const ghost: ReactElement[] = [], dots: ReactElement[] = [];
  for (let i = 0; i < n; i++) {
    const x = pts[3 * i], y = pts[3 * i + 1], z = pts[3 * i + 2];
    const m = Du / (Du + z);
    const front = Math.max(0, Math.min(1, 0.5 - z * 0.55));   // 1 = anterior (warm), 0 = posterior (cyan)
    const col = `rgb(${Math.round(lerp(127, 255, front))},${Math.round(lerp(208, 190, front))},${Math.round(lerp(255, 110, front))})`;
    ghost.push(<circle key={i} cx={cx - x * sc} cy={cy - y * sc} r={1.6} fill="rgba(200,210,230,0.2)" />);
    dots.push(<circle key={i} cx={cx - x * m * sc} cy={cy - y * m * sc} r={2.1} fill={col} />);
  }
  const lo = ramp(t, S.persp3[0] + 6, S.persp3[0] + 7);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>{ghost}{dots}</svg>
      <Label x={960} y={890} size={34} color={C.text}
        text={<span>מרחק המקור ממרכז האגן: <b style={{ color: C.warm, direction: "ltr", display: "inline-block" }}>{k < 0.02 ? "∞ (מקבילי)" : `${Math.round(D)} mm`}</b></span>} />
      <Label x={1650} y={330} size={28} color={C.warm} text="● קדמי (קרוב למקור)" o={lo} />
      <Label x={1650} y={380} size={28} color={C.xray} text="● אחורי (רחוק)" o={lo} />
      <Label x={1650} y={430} size={24} color={C.dim} text="אפור: ההטלה המקבילית" o={lo} />
    </div>
  );
}

/** homogeneous coordinates: all points on a ray through the origin are the same image point */
function HomDiagram({ t }: { t: number }) {
  const o = window01(t, S.hom2[0] - 0.2, S.hom3[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const ox = 1200, oy = 840, s = 160, xv = 1.4;
  const P = (x: number, w: number) => [ox + x * s, oy - w * s];
  const ws = [0.5, 1, 2];
  const line = ramp(t, S.hom2[0] + 4.5, S.hom2[0] + 7.5);
  const [ex, ey] = P(xv * 2.9, 2.9);
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      <line x1={ox - 40} y1={oy} x2={ox + 4 * s} y2={oy} stroke={C.dim} strokeWidth={2} />
      <line x1={ox} y1={oy + 30} x2={ox} y2={oy - 3.2 * s} stroke={C.dim} strokeWidth={2} />
      {svgText(ox + 4 * s + 20, oy + 10, "x", C.dim, 32, true, "start")}
      {svgText(ox, oy - 3.2 * s - 16, "w", C.dim, 32)}
      <line x1={ox - 40} y1={oy - s} x2={ox + 4 * s} y2={oy - s} stroke={C.xray} strokeWidth={3} strokeDasharray="12 8" />
      {svgText(ox + 4 * s + 14, oy - s + 10, "w = 1", C.xray, 28, false, "start")}
      <line x1={ox} y1={oy} x2={ox + (ex - ox) * line} y2={oy + (ey - oy) * line} stroke={C.warm} strokeWidth={3} />
      {ws.map((w, i) => {
        const [x, y] = P(xv * w, w);
        const on = w === 1;
        return (
          <g key={i} opacity={ramp(t, S.hom2[0] + 4.5 + i * 0.8, S.hom2[0] + 5.2 + i * 0.8)}>
            <circle cx={x} cy={y} r={on ? 12 : 9} fill={on ? C.xray : C.warm} />
            {svgText(x - 20, y - 18, `(${(xv * w).toFixed(1)}, ${w})`, on ? C.xray : C.warm, 26, false, "end")}
          </g>
        );
      })}
      <circle cx={ox} cy={oy} r={7} fill={C.text} />
    </svg>
  );
}

/** the image grid: pixel (0,0) top-left, principal point in the middle */
function PixelGrid({ t }: { t: number }) {
  const o = window01(t, S.k3[0] - 0.2, S.k3[1] + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const x0 = 1150, y0 = 260, side = 480, n = 10;
  const c = ramp(t, S.k3[0] + 0.8, S.k3[0] + 1.8);
  const orig = ramp(t, S.k3[0] + 5.0, S.k3[0] + 6.0);
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: o }}>
      <rect x={x0} y={y0} width={side} height={side} fill="rgba(8,14,24,0.7)" stroke={C.line} strokeWidth={2} />
      {Array.from({ length: n - 1 }, (_, i) => (
        <g key={i}>
          <line x1={x0 + (i + 1) * side / n} y1={y0} x2={x0 + (i + 1) * side / n} y2={y0 + side} stroke="rgba(120,190,255,0.15)" />
          <line x1={x0} y1={y0 + (i + 1) * side / n} x2={x0 + side} y2={y0 + (i + 1) * side / n} stroke="rgba(120,190,255,0.15)" />
        </g>
      ))}
      <g opacity={c}>
        <circle cx={x0 + side / 2} cy={y0 + side / 2} r={11} fill={C.warm} />
        {svgText(x0 + side / 2, y0 + side / 2 - 24, "c = (150, 150)", C.warm, 28, false)}
      </g>
      <g opacity={orig}>
        <circle cx={x0} cy={y0} r={10} fill={C.good} />
        {svgText(x0 - 14, y0 - 16, "(0, 0)", C.good, 26, false, "end")}
        <line x1={x0} y1={y0} x2={x0 + 160} y2={y0} stroke={C.good} strokeWidth={4} />
        <line x1={x0} y1={y0} x2={x0} y2={y0 + 160} stroke={C.good} strokeWidth={4} />
        {svgText(x0 + 180, y0 + 10, "u", C.good, 32, true, "start")}
        {svgText(x0 - 24, y0 + 190, "v", C.good, 32, true)}
      </g>
      {svgText(x0 + side / 2, y0 + side + 50, "300 × 300 pixels, 2 mm each", C.dim, 26, false)}
    </svg>
  );
}

// ---------------- the video ----------------
type Assets = { pelvis: THREE.BufferGeometry; persp: THREE.Texture; pts: Pts; v04: V04 };

export const V03 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const pelvis = useObj("data/hero/pelvis.obj");
  const persp = useTex("data/v04/persp_ap.png");
  const pts = useJson<Pts>("data/v03/points.json");
  const v04 = useJson<V04>("data/v04/v04.json");
  const ready = pelvis && persp && pts && v04;
  const ex = pts?.example;
  const f1 = (v: number) => (Math.round(v * 10) / 10).toString();

  const sc = (p: V3, cam: Cam) => toScreen(p, cam.pos, cam.look);
  const P = ex ? (ex.X.map((v) => v * 0.01) as V3) : ([0, 0, 0] as V3);
  const hit = hitDet(P);
  const [srcX, srcY] = sc(SRC, CAM_XR);
  const [detX, detY] = sc([0, 2.6, DET_Z], CAM_XR);
  const [bodyX, bodyY] = sc([0, 2.1, 0], CAM_XR);
  const [d0x, d0y] = sc([0, -2.6, SRC[2]], CAM_XR);
  const [d1x] = sc([0, -2.6, 0], CAM_XR);
  const [d2x] = sc([0, -2.6, DET_Z], CAM_XR);
  const [exPx, exPy] = sc(P, CAM_EX);
  const [exHx, exHy] = sc(hit, CAM_EX);
  const [exSx, exSy] = sc(SRC, CAM_EX);
  const [rtWx, rtWy] = sc([0, 1.3, 0], CAM_RT);
  const [rtCx, rtCy] = sc(SRC, CAM_RT);

  const [x1, x3] = span(tl, "xr1", "xr3");
  const xr = (a: number, b: number) => window01(t, a, b, 0.6, 0.6);

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ pelvis, persp, pts, v04 }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.62 * window01(t, S.recap[0] - 0.3, tl.total, 0.8, 0.1), pointerEvents: "none" }} />

      <Opening n={3} title="מצלמות והטלה" active={[1]} t={t} end={2.9} />

      {/* intro */}
      <Label x={srcX} y={srcY - 60} text="מקור" color={C.xray} o={window01(t, S.intro2[1] - 4, S.intro2[1] + 0.4, 0.6, 0.6)} />

      {/* pinhole */}
      <Title text="מצלמת חור" sub="pinhole camera" t={t} a={S.pin1[0]} b={S.pin2[1] + 0.3} />
      <SideView t={t} />
      <Eq tex={"\\hl{x}{x} = \\hl{f}{f}\\,\\dfrac{\\hl{X}{X}}{\\hl{Z}{Z}}"} t={t} a={S.pin4[0] + 3} b={S.pin5[1] + 0.3}
        style={{ left: 1420, top: 140, width: 380 }} size={72}
        terms={{ x: [S.pin4[0] + 7, S.pin4[1]], f: [S.pin4[0] + 8, S.pin4[1]], X: [S.pin4[0] + 9, S.pin4[1]], Z: [S.pin4[0] + 10, S.pin4[1]] }} />

      {/* perspective vs parallel */}
      <Title text="פרספקטיבי מול מקבילי" t={t} a={S.persp1[0]} b={S.persp2[1] + 0.3} size={64} />
      <DepthDemo t={t} />
      <CloudProjection t={t} pts={pts?.points ?? []} />
      <Title text="האגן שלנו: מקבילי ← פרספקטיבי" t={t} a={S.persp3[0]} b={S.persp3[1] + 0.3} size={52} top={40} />
      {window01(t, S.persp4[0], S.persp4[1] + 0.4) > 0 && (
        <div style={{ position: "absolute", left: 690, top: 200, width: 540, opacity: window01(t, S.persp4[0], S.persp4[1] + 0.4, 0.6, 0.6) }}>
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ width: 540, borderRadius: 12, border: `1.5px solid ${C.line}` }} />
          <Label x={270} y={-50} text="DRR מקבילי: סכום לאורך ציר" color={C.xray} size={38} />
        </div>
      )}

      {/* homogeneous coordinates */}
      <Title text="קואורדינטות הומוגניות" sub="homogeneous coordinates" t={t} a={S.hom1[0]} b={S.hom3[1] + 0.3} size={64} top={60} />
      <Eq tex={"x = f\\,\\dfrac{X}{\\hl{Z}{Z}}"} note="÷Z: לא ליניארי" t={t} a={S.hom1[0] + 0.4} b={S.hom1[1] + 0.3}
        style={{ left: 660, top: 330, width: 600 }} terms={{ Z: [S.hom1[0] + 2, S.hom1[1]] }} />
      <Eq tex={"\\begin{pmatrix}x\\\\y\\\\\\hl{one}{1}\\end{pmatrix} \\sim \\begin{pmatrix}\\hl{w}{w}x\\\\\\hl{w}{w}y\\\\\\hl{w}{w}\\end{pmatrix}"}
        t={t} a={S.hom2[0] + 0.3} b={S.hom2[1] + 0.3} style={{ left: 140, top: 300, width: 700 }}
        terms={{ one: [S.hom2[0] + 2.5, S.hom2[0] + 5], w: [S.hom2[0] + 5, S.hom2[1]] }} />
      <Eq tex={"\\begin{pmatrix}u'\\\\v'\\\\\\hl{w}{w}\\end{pmatrix} = P\\begin{pmatrix}X\\\\Y\\\\Z\\\\1\\end{pmatrix},\\quad u=\\dfrac{u'}{\\hl{w}{w}},\\; v=\\dfrac{v'}{\\hl{w}{w}}"}
        t={t} a={S.hom3[0]} b={S.hom3[1] + 0.3} style={{ left: 70, top: 300, width: 960 }} size={40}
        terms={{ w: [S.hom3[0] + 4, S.hom3[1]] }} />
      <HomDiagram t={t} />

      {/* intrinsics */}
      <Title text="הפרמטרים הפנימיים K" sub="intrinsics" t={t} a={S.k1[0]} b={S.k3[1] + 0.3} size={64} top={60} />
      <Eq tex={"K = \\begin{pmatrix}\\hl{f}{f} & 0 & \\hl{c}{c_x}\\\\ 0 & \\hl{f}{f} & \\hl{c}{c_y}\\\\ 0 & 0 & 1\\end{pmatrix}"}
        t={t} a={S.k1[0] + 0.5} b={S.k3[1] + 0.3} style={{ left: 150, top: 290, width: 700 }}
        terms={{ f: [S.k2[0], S.k2[1]], c: [S.k3[0], S.k3[1]] }} />
      <Eq tex={"f_{px} = \\dfrac{f_{mm}}{\\Delta} = \\dfrac{1000}{2} = 500"} t={t} a={S.k2[0] + 1.5} b={S.k2[1] + 0.3}
        style={{ left: 1000, top: 380, width: 820 }} size={46} />
      <PixelGrid t={t} />

      {/* extrinsics */}
      <Title text="הפרמטרים החיצוניים [R|t]" sub="extrinsics" t={t} a={S.rt1[0]} b={S.rt3[1] + 0.3} size={60} top={50} />
      <Label x={rtWx} y={rtWy - 40} text="העולם: מערכת ה-CT" color={C.text} o={window01(t, S.rt1[0] + 1, S.rt3[1])} />
      <Label x={rtCx} y={rtCy - 70} text="המצלמה (המקור)" color={C.xray} o={window01(t, S.rt2[0] + 1.5, S.rt3[1])} />
      <Eq tex={"\\begin{pmatrix}X_c\\\\Y_c\\\\Z_c\\end{pmatrix} = \\hl{R}{R}\\begin{pmatrix}X\\\\Y\\\\Z\\end{pmatrix} + \\hl{t}{t}"}
        t={t} a={S.rt2[0] + 0.3} b={S.rt2[1] + 0.3} style={{ left: 1150, top: 200, width: 640 }} size={46}
        terms={{ R: [S.rt2[0] + 3.5, S.rt2[0] + 7], t: [S.rt2[0] + 7, S.rt2[1]] }} />
      <Eq tex={"P = \\hl{K}{K}\\,[\\,\\hl{R}{R}\\mid\\hl{R}{t}\\,]"} note="3×4: נקודה בעולם ← פיקסל"
        t={t} a={S.rt3[0] + 0.3} b={S.rt3[1] + 0.3} style={{ left: 1150, top: 200, width: 640 }} size={64}
        terms={{ K: [S.rt3[0] + 1.5, S.rt3[1]], R: [S.rt3[0] + 2.5, S.rt3[1]] }} />

      {/* worked example */}
      <Title text="דוגמה: P = K[R|t]" t={t} a={S.ex1[0]} b={S.ex3[1] + 0.3} size={60} top={50} />
      {ex && (
        <Panel o={window01(t, S.ex1[0] + 0.5, S.ex3[1] + 0.4, 0.6, 0.6)} style={{ left: 40, top: 190, width: 640, direction: "ltr", padding: "24px 30px" }}>
          {[
            { a: S.ex1[0] + 3, el: texLine(`X = (${ex.X.join(",\\ ")})\\ \\text{mm}`) },
            { a: S.ex2[0] + 3, el: texLine(`R = \\mathrm{diag}(-1,-1,1)`) },
            { a: S.ex2[0] + 5, el: texLine(`t = (0,\\ 0,\\ ${ex.SOD})`) },
            { a: S.ex2[0] + 9, el: texLine(`R X + t = (${ex.Xc.join(",\\ ")})`) },
            { a: S.ex3[0] + 0.5, el: texLine(`u = \\frac{${ex.f_px}\\cdot(${ex.Xc[0]})}{${ex.Xc[2]}} + ${ex.c[0]} \\approx ${f1(ex.uv[0])}`) },
            { a: S.ex3[0] + 5.5, el: texLine(`v = \\frac{${ex.f_px}\\cdot(${ex.Xc[1]})}{${ex.Xc[2]}} + ${ex.c[1]} \\approx ${f1(ex.uv[1])}`) },
          ].map((r, i) => (
            <div key={i} style={{ opacity: ramp(t, r.a, r.a + 0.6), fontSize: 30, marginBottom: 12, color: "white" }}>{r.el}</div>
          ))}
        </Panel>
      )}
      <Label x={exPx + 20} y={exPy - 46} text="הקצה הקדמי של עצם הכסל" color={C.warm} size={28} o={window01(t, S.ex1[0] + 3, S.ex3[1])} />
      <Label x={exSx} y={exSy - 50} text="מקור (מרכז ההטלה)" color={C.xray} size={28} o={window01(t, S.ex2[0] + 0.5, S.ex3[1])} />
      {ex && <Label x={exHx} y={exHy + 60} text={<span style={{ direction: "ltr", display: "inline-block" }}>{`(u, v) ≈ (${Math.round(ex.uv[0])}, ${Math.round(ex.uv[1])})`}</span>} color={C.warm} size={30} o={window01(t, S.ex3[0] + 3.5, S.ex3[1] + 0.3)} />}

      {/* the X-ray mapping */}
      <Title text="רנטגן = מצלמת חור" t={t} a={x1} b={S.xr2[1] + 0.3} size={60} top={50} />
      <Label x={srcX} y={srcY - 70} text="מקור = מרכז ההטלה" color={C.xray} size={32} o={xr(S.xr1[0] + 1.5, x3 + 0.3)} />
      <Label x={detX} y={detY - 40} text="גלאי = מישור התמונה" color={C.xray} size={32} o={xr(S.xr1[0] + 3.0, x3 + 0.3)} />
      <Label x={bodyX} y={bodyY - 40} text="הגוף ביניהם" color={C.warm} size={32} o={xr(S.xr1[0] + 7.5, x3 + 0.3)} />
      {xr(S.xr2[0] + 4, x3 + 0.3) > 0 && (
        <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: xr(S.xr2[0] + 4, x3 + 0.3) }}>
          <line x1={d0x} y1={d0y} x2={d2x} y2={d0y} stroke={C.warm} strokeWidth={3} />
          <line x1={d0x} y1={d0y + 50} x2={d1x} y2={d0y + 50} stroke={C.xray} strokeWidth={3} />
          {[d0x, d1x, d2x].map((x, i) => <line key={i} x1={x} y1={d0y - 12} x2={x} y2={d0y + 62} stroke={C.dim} strokeWidth={2} />)}
          <text x={(d0x + d2x) / 2} y={d0y - 14} fill={C.warm} fontSize={28} fontFamily={FONT} textAnchor="middle">SDD = 1000 mm</text>
          <text x={(d0x + d1x) / 2} y={d0y + 90} fill={C.xray} fontSize={28} fontFamily={FONT} textAnchor="middle">SOD = 700 mm</text>
        </svg>
      )}
      <Eq tex={"M = \\dfrac{\\mathrm{SDD}}{Z}:\\quad \\dfrac{1000}{700}\\approx 1.43,\\quad \\dfrac{1000}{621}\\approx 1.61"} t={t} a={S.xr3[0] + 0.5} b={x3 + 0.4}
        style={{ left: 880, top: 150, width: 980 }} size={36} />

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "מצלמת חור: x = f·X/Z (כל נקודה לאורך קרן דרך מרכז ההטלה)",
        "פרספקטיבי: רחוק = קטן. מקבילי: הגודל לא תלוי בעומק",
        "קואורדינטות הומוגניות: ההטלה = כפל במטריצה, ובסוף חלוקה ב-w",
        "P = K[R|t]: פנימיים (f, c) כפול חיצוניים (R, t)",
        "רנטגן: מקור = מרכז ההטלה, גלאי = מישור התמונה, הגוף ביניהם",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/01-data-and-drr/notebook.ipynb" todos={[
        "לבנות את K (f, c) ואת [R|t] של מצלמת AP",
        "להטיל נקודות תלת-ממדיות לפיקסלים עם K[R|t]",
        "להשוות הטלה מקבילית מול פרספקטיבית על אותן נקודות",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v03" />
    </AbsoluteFill>
  );
};
