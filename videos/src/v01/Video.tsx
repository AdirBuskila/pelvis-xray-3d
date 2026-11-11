// Video 1 · What is an image. Pixels as numbers, (row, col) vs (x, y), grayscale vs RGB, spacing and resolution;
// CT as a 3D voxel grid (live slices from the hero volume), voxel spacing, RAS/LPS, the affine, masks.
// Structure follows v00: narration-driven timeline, DOM-level asset loading, one <Stage>, 2D overlays on top.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { CSSProperties, ReactNode } from "react";
import script from "./script.json";
import durations from "../../public/audio/v01/durations.json";
import info from "../../public/data/v01/info.json";
import metaJson from "../../public/data/hero/meta.json";
import { buildTimeline, Script } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useObj, useVolume } from "../lib/assets";
import { CameraPath, CamKey, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Meta, WireBox } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, PipelineMap, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Arrow2D, Arrow3D, camAt, Dot, IsoBox, mono, NumGrid, PlaneOutline, project, VolBox, VolumeSlice } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const meta = metaJson as unknown as Meta & { occ_cube_mm: number };
const box = boxOf(meta);
const [cx, cy, cz] = box.c;
const vbox: VolBox = { bmin: meta.box_min as [number, number, number], bmax: meta.box_max as [number, number, number] };
const DRR = { H: info.drr.shape[0], W: info.drr.shape[1] };
const PR = info.drr.patch_rc[0], PC = info.drr.patch_rc[1], PN = info.drr.n;

// camera helper: on a circle around the box centre at angle a (0 = in front), the look point shifted along the
// camera's right vector so the subject sits left (shift > 0) or right (shift < 0) of the frame
const obl = (t: number, a: number, r: number, y: number, shift = 0, fov?: number): CamKey => ({
  t, pos: [cx + r * Math.sin(a), y, cz - r * Math.cos(a)],
  look: [cx - Math.cos(a) * shift, cy * 0.6, cz - Math.sin(a) * shift], fov,
});
const keys: CamKey[] = [
  obl(0, 0.75, 10.8, 4.2, 1.5),
  obl(S.ct1[0] - 0.4, 0.75, 10.8, 4.2, 1.5),
  obl(S.ct3[1], 0.55, 10.2, 3.8, 1.6),
  obl(S.pl1[0] + 0.8, 0.45, 9.8, 5.0, 1.7),
  obl(S.pl1[1], 0.4, 9.8, 4.8, 1.7),
  obl(S.pl2[0] + 0.9, 0.3, 10.0, 2.0, 1.7),
  obl(S.pl2[1] + 0.2, 0.3, 10.0, 2.0, 1.7),
  obl(S.pl3[0] + 1.0, -0.95, 10.0, 2.2, 1.7),
  obl(S.pl3[1], -0.62, 10.4, 3.0, 1.7),
  obl(S.or1[0], -0.75, 8.6, 2.2, 0),
  obl(S.or3[1], -0.55, 8.6, 1.8, 0),
  obl(S.af1[0], 0.0, 9.8, 0.9, -2.7),
  obl(S.af3[1], 0.25, 9.6, 1.0, -2.7),
  obl(S.m3[0], 0.0, 8.2, 1.0, -1.6),
  obl(S.m4[1], -0.45, 8.0, 1.4, -1.6),
  obl(S.pipe[0], 0.3, 9.4, 1.6, 0),
  obl(tl.total, 1.1, 9.6, 1.4, 0),
];

// ---------------- the 3D scene ----------------
type Assets = {
  vol: THREE.Data3DTexture; lab: THREE.Data3DTexture; pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry;
  spine: THREE.BufferGeometry; hipL: THREE.BufferGeometry; hipR: THREE.BufferGeometry; sacrum: THREE.BufferGeometry;
};
const STACK = 9;

function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const orange = useBoneMaterial("#ff9a50");
  const blue = useBoneMaterial("#5cb8ff");
  const green = useBoneMaterial("#74f08e");
  const [x0, y0, z0] = vbox.bmin, [x1, y1, z1] = vbox.bmax;

  const volOn = window01(t, S.ct1[0] - 0.2, S.pl3[1] + 0.5, 0.8, 0.8);
  const stackOn = window01(t, S.ct1[0], S.pl1[0] + 0.8, 0.5, 0.8);
  const dense = ramp(t, S.ct2[0] + 0.2, S.ct2[0] + 2.4);
  const voxOn = window01(t, S.ct2[0] + 2.0, S.ct3[1] + 0.2, 0.6, 0.6);
  const axOn = window01(t, S.pl1[0] + 0.3, S.pl3[1] + 0.5, 0.6, 0.6) * (1 - 0.75 * window01(t, S.pl2[0], S.pl3[0] + 5.0, 0.5, 0.6));
  const coOn = window01(t, S.pl2[0] + 0.2, S.pl3[1] + 0.5, 0.6, 0.6) * (1 - 0.75 * window01(t, S.pl3[0], S.pl3[0] + 5.0, 0.5, 0.6));
  const sgOn = window01(t, S.pl3[0] + 0.2, S.pl3[1] + 0.5, 0.6, 0.6);
  // sweeps: go across the whole box, then come back to the pelvis centre (scene origin)
  const sweep = (ta: number, tb: number, lo: number, hi: number) => {
    const k1 = ramp(t, ta, ta + (tb - ta) * 0.7), k2 = ramp(t, ta + (tb - ta) * 0.7, tb);
    return lerp(lerp(lo, hi, k1), 0, k2);
  };
  const ayy = sweep(S.pl1[0] + 0.5, S.pl1[1] - 0.3, y0 + 0.06, y1 - 0.06);
  const czz = sweep(S.pl2[0] + 0.4, S.pl2[1] + 0.2, z1 + 0.05, z0 - 0.05);
  const sxx = sweep(S.pl3[0] + 0.4, S.pl3[0] + 4.6, x1 - 0.05, x0 + 0.05);

  const orOn = window01(t, S.or1[0], S.or3[1] + 0.5, 0.8, 0.6);
  const flip = ramp(t, S.or3[0] + 0.8, S.or3[0] + 2.6);
  const afOn = window01(t, S.af1[0], S.af3[1] + 0.4, 0.8, 0.6);
  const mOn = window01(t, S.m3[0] - 0.2, S.m4[1] + 0.4, 0.8, 0.6);
  const union = ramp(t, S.m3[1] - 3.2, S.m3[1] - 1.6) * (1 - ramp(t, S.m4[0], S.m4[0] + 1.0));
  const m4 = ramp(t, S.m4[0], S.m4[0] + 1.0);
  const tailOn = window01(t, S.pipe[0], tl.total, 1.0, 1.0);
  const boneShow = orOn * 0.8 + afOn * 0.85 + tailOn * 0.6;

  const n = Math.round(STACK + dense * 31);
  const slices: number[] = [];
  for (let i = 0; i < n; i++) slices.push(y0 + 0.05 + (i + 0.5) / n * (y1 - y0 - 0.1));
  const sAx: [number, number] = [box.s[0], box.s[2]], sCo: [number, number] = [box.s[0], box.s[1]], sSg: [number, number] = [box.s[2], box.s[1]];
  const corner: [number, number, number] = [x0, y0, z0];
  const vcell = 0.2;

  return (
    <>
      <CameraPath t={t} keys={keys} />
      {/* ---- CT volume: stack of slices, then the three planes ---- */}
      <WireBox size={box.s as [number, number, number]} position={box.c as [number, number, number]} opacity={0.55 * Math.min(1, volOn + orOn * 0.7 + afOn * 0.4)} />
      {stackOn > 0 && slices.map((y, i) => {
        const appear = i < STACK ? ramp(t, S.ct1[0] + 0.6 + i * 0.55, S.ct1[0] + 1.2 + i * 0.55) : dense;
        const drop = (1 - (i < STACK ? appear : 1)) * 0.6;
        return <VolumeSlice key={i} vol={a.vol} box={vbox} axis="axial" at={y + drop} size={sAx}
          opacity={stackOn * appear * (i < STACK ? 0.92 : 0.55)} gain={0.95} />;
      })}
      {stackOn > 0 && slices.slice(0, STACK).map((y, i) => (
        <PlaneOutline key={`o${i}`} axis="axial" at={y} box={vbox} opacity={stackOn * (1 - dense) * 0.6 * ramp(t, S.ct1[0] + 0.6 + i * 0.55, S.ct1[0] + 1.2 + i * 0.55)} />))}
      {/* magnified voxels next to the box */}
      {voxOn > 0 && (
        <group position={[x1 + 0.9, y1 - 0.5, z1]}>
          {[0, 1, 2].flatMap((i) => [0, 1, 2].flatMap((j) => [0, 1, 2].map((k) => {
            const hot = i === 2 && j === 2 && k === 0;
            return (
              <mesh key={`${i}${j}${k}`} position={[(i - 1) * vcell * 1.12, (j - 1) * vcell * 1.12, (k - 1) * vcell * 1.12]}>
                <boxGeometry args={[vcell, vcell, vcell]} />
                <meshStandardMaterial color={hot ? "#ffcf7f" : "#9fd6ff"} transparent opacity={voxOn * 0.9}
                  emissive={new THREE.Color(hot ? "#ffcf7f" : "#3a6c96")} emissiveIntensity={0.5} />
              </mesh>);
          })))}
        </group>
      )}
      <VolumeSlice vol={a.vol} box={vbox} axis="axial" at={ayy} size={sAx} opacity={axOn} />
      <PlaneOutline axis="axial" at={ayy} box={vbox} opacity={axOn} />
      <VolumeSlice vol={a.vol} box={vbox} axis="coronal" at={czz} size={sCo} opacity={coOn} />
      <PlaneOutline axis="coronal" at={czz} box={vbox} opacity={coOn} color={[2.2, 1.6, 0.6]} />
      <VolumeSlice vol={a.vol} box={vbox} axis="sagittal" at={sxx} size={sSg} opacity={sgOn} />
      <PlaneOutline axis="sagittal" at={sxx} box={vbox} opacity={sgOn} color={[0.7, 2.2, 1.0]} />

      {/* ---- orientation: index axes at voxel (0,0,0), then RAS vs LPS ---- */}
      <Bone geo={a.pelvis} material={bone} opacity={boneShow * (1 - mOn)} />
      <Bone geo={a.femurs} material={bone} opacity={boneShow * (1 - mOn) * 0.8 + mOn * 0.25} />
      <Bone geo={a.spine} material={bone} opacity={boneShow * (1 - mOn) * 0.8 + mOn * 0.25} />
      <Arrow3D from={corner} dir={[1, 0, 0]} len={1.5 * (1 - 2 * flip)} color="#ff7f8a" opacity={orOn} />
      <Arrow3D from={corner} dir={[0, 0, -1]} len={1.5 * (1 - 2 * flip)} color="#7dffb0" opacity={orOn} />
      <Arrow3D from={corner} dir={[0, 1, 0]} len={1.5} color="#7fd0ff" opacity={orOn} />
      <Dot p={corner} opacity={orOn} r={0.05} />
      {/* ---- affine: the centre voxel ---- */}
      <Dot p={[0, 0, 0]} opacity={afOn * ramp(t, S.af3[0] + 0.5, S.af3[0] + 1.2)} r={0.07} />

      {/* ---- masks: one file per bone, then their union ---- */}
      <Bone geo={a.hipL} material={orange} opacity={mOn * (1 - union)} />
      <Bone geo={a.hipR} material={blue} opacity={mOn * (1 - union) * (1 - 0.8 * m4)} />
      <Bone geo={a.sacrum} material={green} opacity={mOn * (1 - union) * (1 - 0.8 * m4)} />
      <Bone geo={a.pelvis} material={bone} opacity={mOn * union} />
    </>
  );
}

// ---------------- DOM pieces ----------------
const grayBg = (v: number) => `rgb(${v},${v},${v})`;
const fgFor = (v: number) => (v > 150 ? "#05070b" : "white");

/** the AP DRR, zoomable around an image point (pixel-exact rendering) */
function PixelView({ z, ccx, ccy, sx = 960, sy = 470, o, children }: {
  z: number; ccx: number; ccy: number; sx?: number; sy?: number; o: number; children?: ReactNode;
}) {
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden", opacity: o }}>
      <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 0, top: 0, width: DRR.W, imageRendering: "pixelated",
        transformOrigin: "0 0", transform: `translate(${sx - ccx * z}px, ${sy - ccy * z}px) scale(${z})` }} />
      {children}
    </div>
  );
}

function PixelSection({ t }: { t: number }) {
  const on = window01(t, 2.7, S.px3[1] + 0.5, 0.8, 0.6);
  if (on <= 0) return null;
  const z0 = 760 / DRR.H, z1 = 86;
  const kin = ramp(t, S.px1[0] + 0.4, S.px1[1] - 0.6);
  const kout = ramp(t, S.px3[0] + 0.2, S.px3[0] + 2.4);
  const k = kin * (1 - kout);
  const z = z0 * Math.pow(z1 / z0, k);
  const tx = PC + PN / 2, ty = PR + PN / 2, ix = DRR.W / 2, iy = DRR.H / 2;
  const ccx = tx + (ix - tx) * (z0 / z) * (1 - k), ccy = ty + (iy - ty) * (z0 / z) * (1 - k);
  // intro: image on the left, the table of numbers the computer sees on the right
  const split = window01(t, S.intro1[0] + 4.5, S.px1[0] + 0.5, 1.0, 0.8);
  const sx = 960 - 340 * split;
  const grid = ramp(t, S.px1[1] - 2.0, S.px1[1] - 0.6) * (1 - kout);
  const nums = ramp(t, S.px2[0] + 0.3, S.px2[0] + 1.2) * (1 - kout);
  const shapeOn = ramp(t, S.px3[0] + 2.6, S.px3[0] + 3.4);
  const ds = info.ds8_values as number[][];
  const lines: ReactNode[] = [];
  if (grid > 0) {
    const c0 = Math.floor(ccx - sx / z) - 1, c1 = Math.ceil(ccx + (1920 - sx) / z) + 1;
    const r0 = Math.floor(ccy - 470 / z) - 1, r1 = Math.ceil(ccy + 610 / z) + 1;
    for (let c = c0; c <= c1; c++) lines.push(<div key={`c${c}`} style={{ position: "absolute", left: sx + (c - ccx) * z, top: 0, width: 1.5, height: 1080, background: "rgba(5,7,11,0.7)" }} />);
    for (let r = r0; r <= r1; r++) lines.push(<div key={`r${r}`} style={{ position: "absolute", top: 470 + (r - ccy) * z, left: 0, height: 1.5, width: 1920, background: "rgba(5,7,11,0.7)" }} />);
  }
  const px = (c: number) => sx + (c - ccx) * z, py = (r: number) => 470 + (r - ccy) * z;
  return (
    <>
      <PixelView z={z} ccx={ccx} ccy={ccy} sx={sx} o={on}>
        <div style={{ opacity: grid }}>{lines}</div>
        {nums > 0 && (
          <div style={{ position: "absolute", left: px(PC) - 4, top: py(PR) - 4, width: PN * z + 8, height: PN * z + 8, opacity: nums,
            border: `4px solid ${C.warm}`, boxShadow: `0 0 30px rgba(255,207,127,0.5)` }}>
            {info.drr.values.flatMap((row, r) => row.map((v, c) => (
              <div key={`${r}-${c}`} style={{ position: "absolute", left: c * z, top: r * z, width: z, height: z, display: "flex",
                alignItems: "center", justifyContent: "center", fontSize: z * 0.34, color: fgFor(v), ...mono }}>{v}</div>)))}
          </div>
        )}
      </PixelView>
      {/* the computer's view: the same picture as a table (8x8-pixel block means) */}
      <div style={{ position: "absolute", left: 1090, top: 112, opacity: on * split, ...mono }}>
        {ds.map((row, r) => (
          <div key={r} style={{ display: "flex", height: 23 }}>
            {row.map((v, c) => <div key={c} style={{ width: 25, fontSize: 10.5, color: `rgb(${80 + v * 0.68},${110 + v * 0.57},${140 + v * 0.45})`, textAlign: "center" }}>{v}</div>)}
          </div>
        ))}
      </div>
      <Label x={1450} y={810} text="מה שהמחשב רואה: מערך של מספרים" color={C.dim} size={28} o={on * split} />
      <Label x={sx} y={880} text="מה שאנחנו רואים" color={C.dim} size={28} o={on * split} />
      {/* gray scale legend */}
      <div style={{ position: "absolute", right: 80, top: 120, opacity: nums, padding: "14px 18px", background: C.panel, borderRadius: 14, ...mono }}>
        <div style={{ width: 300, height: 26, background: "linear-gradient(90deg, #000, #fff)", border: `1px solid ${C.line}` }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 26, marginTop: 6, color: C.text }}><span>0</span><span>255</span></div>
      </div>
      {/* shape annotation */}
      {shapeOn > 0 && (
        <div style={{ opacity: shapeOn * on }}>
          <div style={{ position: "absolute", left: 960 - DRR.W * z0 / 2, width: DRR.W * z0, top: 80, borderTop: `3px solid ${C.warm}` }} />
          <Label x={960} y={56} text={<span><span style={mono}>248</span> עמודות (cols)</span>} color={C.warm} size={30} />
          <div style={{ position: "absolute", left: 960 - DRR.W * z0 / 2 - 22, top: 90, height: 760, borderLeft: `3px solid ${C.warm}` }} />
          <Label x={960 - DRR.W * z0 / 2 - 40} y={470} text={<span><span style={mono}>235</span> שורות (rows)</span>} color={C.warm} size={30} align="right" />
          <Panel o={shapeOn} style={{ right: 50, top: 380, width: 440, padding: "22px 26px", whiteSpace: "nowrap" }}>
            <div style={{ ...mono, fontSize: 32, lineHeight: 1.7, textAlign: "left" }}>
              <div>img.shape = (235, 248)</div>
              <div style={{ color: C.dim }}>img.dtype = uint8</div>
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}

function CoordSection({ t }: { t: number }) {
  const on = window01(t, S.co1[0] - 0.2, S.co3[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const tabOn = 1 - ramp(t, S.co3[0] + 0.2, S.co3[0] + 1.0);
  const hl = ramp(t, S.co1[0] + 6.0, S.co1[0] + 6.8);
  const xy = ramp(t, S.co2[0] + 0.5, S.co2[0] + 1.3);
  const bug = ramp(t, S.co3[0] + 0.6, S.co3[0] + 1.6);
  const z = 2.2, thx = 1180, thy = 250;
  const vals = info.drr.values;
  const cell = 74, gx = 250, gy = 250;
  const bh = 520, bw = bh * DRR.W / DRR.H;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      {/* thumbnail with index axes from the top-left corner */}
      <div style={{ position: "absolute", left: thx, top: thy, opacity: tabOn }}>
        <Img src={staticFile("data/hero/drr_ap.png")} style={{ width: DRR.W * z, imageRendering: "pixelated", display: "block" }} />
        <div style={{ position: "absolute", left: PC * z, top: PR * z, width: PN * z, height: PN * z, border: `2px solid ${C.warm}` }} />
        <div style={{ position: "absolute", left: -10, top: -10, width: 20, height: 20, borderRadius: 10, background: C.warm }} />
      </div>
      <Arrow2D x1={thx} y1={thy - 34} x2={thx + 260} y2={thy - 34} o={tabOn} color={xy > 0.5 ? C.warm : C.xray} />
      <Arrow2D x1={thx - 34} y1={thy} x2={thx - 34} y2={thy + 260} o={tabOn} color={xy > 0.5 ? C.warm : C.xray} />
      <Label x={thx + 280} y={thy - 34} text={<span style={mono}>col{xy > 0.5 ? " = x" : ""}</span>} color={xy > 0.5 ? C.warm : C.xray} size={30} align="left" o={tabOn} />
      <Label x={thx - 34} y={thy + 295} text={<span style={mono}>row{xy > 0.5 ? " = y" : ""}</span>} color={xy > 0.5 ? C.warm : C.xray} size={30} o={tabOn} />
      <Label x={thx - 40} y={thy - 76} text={<span style={mono}>(0, 0)</span>} color={C.warm} size={26} o={tabOn} />
      {/* the patch as a table with its real indices */}
      <NumGrid rows={PN} cols={PN} x={gx} y={gy} size={cell} o={tabOn}
        cell={(r, c) => ({ text: vals[r][c], bg: grayBg(vals[r][c]), fg: fgFor(vals[r][c]) })}
        rowLabels={vals.map((_, r) => PR + r)} colLabels={vals[0].map((_, c) => PC + c)}
        highlight={{ r: 2, c: 5, o: hl }} />
      <div style={{ position: "absolute", left: gx, top: gy + PN * cell + 34, opacity: tabOn * hl, ...mono, fontSize: 36, color: C.text }}>
        img[row, col] = img[{PR + 2}, {PC + 5}] = <span style={{ color: C.warm }}>{vals[2][5]}</span>
      </div>
      <div style={{ position: "absolute", left: gx, top: gy + PN * cell + 86, opacity: tabOn * xy, ...mono, fontSize: 36, color: C.warm }}>
        (x, y) = ({PC + 5}, {PR + 2})  →  img[y, x]
      </div>
      {/* the bug: transposed image */}
      {bug > 0 && (
        <div style={{ opacity: bug }}>
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 300, top: 220, width: bw, height: bh, borderRadius: 8 }} />
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 1100, top: 220, width: bw, height: bh,
            transformOrigin: "center", transform: "rotate(90deg) scaleY(-1)", borderRadius: 8, outline: `3px solid ${C.bad}` }} />
          <Label x={300 + bw / 2} y={800} text={<span><span style={{ color: C.good }}>✓</span> <span style={mono}>img</span></span>} size={40} />
          <Label x={1100 + bw / 2} y={800} text={<span><span style={{ color: C.bad }}>✗</span> <span style={mono}>img.T</span>: שורות ועמודות התחלפו</span>} size={36} />
        </div>
      )}
    </div>
  );
}

function RgbSection({ t }: { t: number }) {
  const on = window01(t, S.rgb1[0] - 0.2, S.rgb3[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const rgbIn = ramp(t, S.rgb1[0] + 4.5, S.rgb1[0] + 5.5);
  const split = ramp(t, S.rgb2[0] + 0.5, S.rgb2[0] + 2.2);
  const probe = ramp(t, S.rgb2[1] - 9.5, S.rgb2[1] - 8.5);
  const dimRgb = ramp(t, S.rgb3[0] + 0.3, S.rgb3[0] + 1.3);
  const z = 2.6, w = 248 * z, h = 146 * z;
  const [pr, pc] = info.rgb.probe_rc, P = info.rgb.probe;
  const chans = [{ f: "axial_r", c: "#ff6b6b", n: "R" }, { f: "axial_g", c: "#6bff8e", n: "G" }, { f: "axial_b", c: "#6bb5ff", n: "B" }];
  const gx = 140, gy = 300, rx = 1030, ry = 250;
  const sc = 1 - 0.18 * split;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <Img src={staticFile("data/v01/axial_gray.png")} style={{ position: "absolute", left: gx, top: gy, width: w, imageRendering: "pixelated",
        borderRadius: 6, outline: dimRgb > 0.5 ? `3px solid ${C.warm}` : "none" }} />
      <Label x={gx + w / 2} y={gy - 50} text={<span>גווני אפור · <span style={mono}>(146, 248)</span></span>} size={32} />
      <Label x={gx + w / 2} y={gy + h + 44} text="מספר אחד לכל פיקסל" color={dimRgb > 0.5 ? C.warm : C.dim} size={30} />
      <div style={{ opacity: rgbIn * (1 - 0.55 * dimRgb) }}>
        {chans.map((ch, i) => {
          const d = (i + 1) * 66 * split;
          return (
            <div key={ch.n} style={{ position: "absolute", left: rx + d, top: ry + d, opacity: split, zIndex: 3 - i }}>
              <Img src={staticFile(`data/v01/${ch.f}.png`)} style={{ width: w * 0.82, imageRendering: "pixelated", display: "block",
                outline: `3px solid ${ch.c}`, borderRadius: 4 }} />
              <div style={{ position: "absolute", left: w * 0.82 + 14, bottom: 0, color: ch.c, fontSize: 36, fontWeight: 700, fontFamily: FONT }}>{ch.n}</div>
            </div>
          );
        })}
        <div style={{ position: "absolute", left: rx, top: ry, zIndex: 5 }}>
          <Img src={staticFile("data/v01/axial_rgb.png")} style={{ width: w * sc, imageRendering: "pixelated", display: "block", borderRadius: 6 }} />
          {probe > 0 && (
            <div style={{ position: "absolute", left: (pc + 0.5) * z * sc - 14, top: (pr + 0.5) * z * sc - 14, width: 28, height: 28,
              border: `4px solid white`, borderRadius: 14, opacity: probe, boxShadow: "0 0 18px white" }} />)}
        </div>
        <Label x={rx + w / 2} y={ry - 50} text={<span>צבע · RGB · <span style={mono}>(146, 248, 3)</span></span>} size={32} />
        {probe > 0 && (
          <Panel o={probe} style={{ left: rx + 20, top: ry + h + 170, padding: "14px 26px", direction: "ltr" }}>
            <div style={{ ...mono, fontSize: 34 }}>
              img[{pr}, {pc}] = (<span style={{ color: "#ff8b8b" }}>{P[0]}</span>, <span style={{ color: "#8bffa6" }}>{P[1]}</span>, <span style={{ color: "#8bc5ff" }}>{P[2]}</span>)
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}

function ResSection({ t }: { t: number }) {
  const on = window01(t, S.res1[0] - 0.2, S.res4[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const sOn = window01(t, S.res1[0] - 0.2, S.res1[1] + 0.3, 0.6, 0.5);
  const imgOn = window01(t, S.res2[0], S.res3[1] + 0.3, 0.6, 0.5);
  const bracket = ramp(t, S.res3[0] + 3.0, S.res3[0] + 4.0);
  const cmpOn = window01(t, S.res4[0], S.res4[1] + 0.5, 0.6, 0.6);
  // res1: an illustrative continuous profile, sampled by averaging over cells
  const f = (x: number) => 0.25 + 0.55 / (1 + Math.exp(-(x - 0.32) * 40)) - 0.45 / (1 + Math.exp(-(x - 0.55) * 30)) + 0.08 * Math.sin(x * 23) + 0.35 * Math.exp(-((x - 0.8) ** 2) / 0.003);
  const X0 = 260, X1 = 1660, Y0 = 760, HH = 440, N = 16;
  const path = Array.from({ length: 300 }, (_, i) => { const x = i / 299; return `${i ? "L" : "M"}${X0 + x * (X1 - X0)},${Y0 - f(x) * HH}`; }).join(" ");
  const cells = Array.from({ length: N }, (_, i) => {
    let s = 0; for (let k = 0; k < 20; k++) s += f((i + (k + 0.5) / 20) / N);
    return s / 20;
  });
  const shown = ramp(t, S.res1[0] + 4.0, S.res1[1] - 1.0) * N;
  const z = 640 / DRR.H, ix = 250, iy = 160;
  const inset = 3, iz = 120;
  const ds = [{ f: "data/hero/drr_ap.png", mm: "1.5" }, { f: "data/v01/drr_ap_ds4.png", mm: "6" }, { f: "data/v01/drr_ap_ds8.png", mm: "12" }];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      {sOn > 0 && (
        <svg style={{ position: "absolute", left: 0, top: 0, width: 1920, height: 1080, opacity: sOn }}>
          {cells.map((v, i) => {
            const k = Math.max(0, Math.min(1, shown - i));
            const x = X0 + i / N * (X1 - X0), w = (X1 - X0) / N;
            return k > 0 ? <rect key={i} opacity={k} x={x + 2} y={Y0 - v * HH} width={w - 4} height={v * HH}
              fill={`rgba(127,208,255,${0.12 + v * 0.3})`} stroke={C.xray} strokeWidth={2} /> : null;
          })}
          <path d={path} stroke={C.warm} strokeWidth={5} fill="none" />
          <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke={C.dim} strokeWidth={2} />
          <text x={X1} y={Y0 + 50} fill={C.dim} fontSize={28} fontFamily={FONT} textAnchor="end">מיקום</text>
          <text x={X0 + 30} y={Y0 - HH - 40} fill={C.warm} fontSize={32} fontFamily={FONT} textAnchor="start">העולם: רציף (המחשה)</text>
          <text x={X1 - 30} y={Y0 - HH - 40} fill={C.xray} fontSize={32} fontFamily={FONT} textAnchor="end" opacity={Math.min(1, shown)}>פיקסל: ממוצע על תא</text>
        </svg>
      )}
      {imgOn > 0 && (
        <div style={{ opacity: imgOn }}>
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: ix, top: iy, width: DRR.W * z, imageRendering: "pixelated" }} />
          <div style={{ position: "absolute", left: ix + (PC + 3) * z, top: iy + PR * z, width: inset * z, height: inset * z, border: `2px solid ${C.warm}` }} />
          <div style={{ position: "absolute", left: 1240, top: 190, width: inset * iz, height: inset * iz, overflow: "hidden", border: `2px solid ${C.warm}` }}>
            {[0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => (
              <div key={`${r}${c}`} style={{ position: "absolute", left: c * iz, top: r * iz, width: iz, height: iz, background: grayBg(info.drr.values[r][c + 3]) }} />)))}
            {[1, 2].map((i) => <div key={i} style={{ position: "absolute", left: i * iz, top: 0, width: 2, height: inset * iz, background: "rgba(5,7,11,0.8)" }} />)}
            {[1, 2].map((i) => <div key={`h${i}`} style={{ position: "absolute", top: i * iz, left: 0, height: 2, width: inset * iz, background: "rgba(5,7,11,0.8)" }} />)}
          </div>
          <Arrow2D x1={1240} y1={190 + 3 * iz + 30} x2={1240 + iz} y2={190 + 3 * iz + 30} o={1} color={C.warm} width={3} />
          <Arrow2D x1={1240 + iz} y1={190 + 3 * iz + 30} x2={1240} y2={190 + 3 * iz + 30} o={1} color={C.warm} width={3} />
          <Label x={1240 + iz / 2} y={190 + 3 * iz + 68} text={<span style={mono}>1.5 mm</span>} color={C.warm} size={30} />
          <Label x={1240 + 1.5 * iz} y={150} text={<span>פיקסל = <span style={mono}>1.5 × 1.5</span> מ"מ</span>} size={34} />
          {bracket > 0 && (
            <div style={{ opacity: bracket }}>
              <div style={{ position: "absolute", left: ix, top: iy + DRR.H * z + 16, width: DRR.W * z, height: 18, borderLeft: `3px solid ${C.warm}`,
                borderRight: `3px solid ${C.warm}`, borderBottom: `3px solid ${C.warm}` }} />
              <Label x={ix + DRR.W * z / 2} y={iy + DRR.H * z + 66} text={<span style={mono}>248 × 1.5 mm = 372 mm</span>} color={C.warm} size={34} />
            </div>
          )}
        </div>
      )}
      <Eq t={t} a={S.res3[0]} b={S.res3[1] + 0.3} style={{ left: 1100, top: 640, width: 700, padding: "14px 30px" }} size={46}
        tex={String.raw`\text{size} = \hl{n}{N}\cdot\hl{d}{\Delta}`} note="מספר פיקסלים × spacing"
        terms={{ n: [S.res3[0] + 1.2, S.res3[0] + 3.0], d: [S.res3[0] + 2.4, S.res3[0] + 4.4] }} />
      {cmpOn > 0 && (
        <div style={{ opacity: cmpOn }}>
          {ds.map((d, i) => {
            const k = ramp(t, S.res4[0] + 0.3 + i * 2.2, S.res4[0] + 1.0 + i * 2.2);
            const hh = 420, ww = hh * DRR.W / DRR.H;
            return (
              <div key={i} style={{ opacity: k }}>
                <Img src={staticFile(d.f)} style={{ position: "absolute", left: 150 + i * 570, top: 200, width: ww, height: hh, imageRendering: "pixelated", borderRadius: 6 }} />
                <Label x={150 + i * 570 + ww / 2} y={670} text={<span>פיקסל <span style={mono}>{d.mm}</span> מ"מ</span>} size={34} color={i ? C.text : C.warm} />
              </div>
            );
          })}
          <Label x={960} y={780} text={<span>פלט הרשת: <span style={mono}>64³</span> תאים על קובייה של <span style={mono}>{Math.round(meta.occ_cube_mm)}</span> מ"מ ≈ <span style={mono}>{(meta.occ_cube_mm / 64).toFixed(1)}</span> מ"מ לתא</span>}
            color={C.warm} size={32} o={ramp(t, S.res4[1] - 6.5, S.res4[1] - 5.5)} />
        </div>
      )}
    </div>
  );
}

function SliceInset({ src, w, h, x, y, o, name, he, top, left }: {
  src: string; w: number; h: number; x: number; y: number; o: number; name: string; he: string; top: string; left: string;
}) {
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, opacity: o }}>
      <Img src={staticFile(src)} style={{ width: w, height: h, display: "block", borderRadius: 6, border: `1.5px solid ${C.line}` }} />
      <div style={{ position: "absolute", top: 4, left: w / 2 - 20, width: 40, textAlign: "center", color: C.warm, fontSize: 24, fontFamily: FONT, textShadow: "0 0 6px black" }}>{top}</div>
      <div style={{ position: "absolute", top: h / 2 - 16, left: 8, color: C.warm, fontSize: 24, fontFamily: FONT, textShadow: "0 0 6px black" }}>{left}</div>
      <div style={{ textAlign: "center", fontFamily: FONT, fontSize: 28, color: C.text, marginTop: 6, direction: "rtl", whiteSpace: "nowrap" }}>{he} · <span style={mono}>{name}</span></div>
    </div>
  );
}

function VolumeOverlays({ t }: { t: number }) {
  const ct3 = window01(t, S.ct3[0] + 0.3, S.ct3[1] + 0.4, 0.6, 0.5);
  const vox = window01(t, S.ct2[0] + 2.4, S.ct3[1] + 0.2, 0.6, 0.6);
  if (t < S.ct1[0] - 1 || t > S.pl3[1] + 1) return null;
  const cam = camAt(keys, t);
  const vp = project(cam, [vbox.bmax[0] + 0.9, vbox.bmax[1] - 0.5 + 0.5, vbox.bmax[2]]);
  // insets: each plane alone, then all three together
  const all = ramp(t, S.pl3[0] + 5.0, S.pl3[0] + 5.8);
  const ax = window01(t, S.pl1[1] - 2.6, S.pl3[1] + 0.4, 0.6, 0.5) * (1 - ramp(t, S.pl2[0] - 0.2, S.pl2[0] + 0.4) * (1 - all));
  const co = window01(t, S.pl2[1] - 1.4, S.pl3[1] + 0.4, 0.6, 0.5) * (1 - ramp(t, S.pl3[0], S.pl3[0] + 0.6) * (1 - all));
  const sg = window01(t, S.pl3[0] + 4.2, S.pl3[1] + 0.4, 0.6, 0.5);
  const big = 1 - all;
  const X = 1330;
  const L = (a: number, b: number) => a * big + b * all;
  return (
    <>
      <Label x={vp[0]} y={vp[1] - 20} text="voxel" color={C.warm} size={34} o={vox} />
      <Panel o={ct3} style={{ right: 70, top: 250, width: 540, padding: "24px 30px", whiteSpace: "nowrap" }}>
        <div style={{ ...mono, fontSize: 32, lineHeight: 1.75, textAlign: "left" }}>
          <div>ct.shape = ({info.nifti.shape.join(", ")})</div>
          <div style={{ color: C.dim }}>ct.dtype = {info.nifti.dtype}</div>
          <div style={{ color: C.warm }}>≈ {(info.nifti.shape.reduce((a, b) => a * b, 1) / 1e6).toFixed(1)} M voxels</div>
        </div>
      </Panel>
      <SliceInset src="data/hero/slice_axial.png" name="axial" he="אקסיאלי" top="A" left="R" o={ax}
        x={L(X, 1440)} y={L(260, 90)} w={L(500, 330)} h={L(500, 330) * 146 / 248} />
      <SliceInset src="data/hero/slice_coronal.png" name="coronal" he="קורונלי" top="S" left="R" o={co}
        x={L(X + 40, 1490)} y={L(190, 340)} w={L(440, 220) * 248 / 235} h={L(440, 220)} />
      <SliceInset src="data/hero/slice_sagittal.png" name="sagittal" he="סגיטלי" top="S" left="A" o={sg}
        x={L(X + 130, 1540)} y={L(190, 620)} w={L(440, 220) * 146 / 235} h={L(440, 220)} />
    </>
  );
}

function SpacingSection({ t }: { t: number }) {
  const on = window01(t, S.sp1[0] - 0.2, S.sp3[1] + 0.5, 0.7, 0.6);
  if (on <= 0) return null;
  const k = ramp(t, S.sp2[0] + 3.0, S.sp2[0] + 5.0);
  const boxOn = 1 - ramp(t, S.sp3[0] - 0.2, S.sp3[0] + 0.5);
  const dist = ramp(t, S.sp3[0] + 0.3, S.sp3[0] + 1.2);
  const wpl = lerp(1.5, 0.8, k);
  const sq = lerp(1, 0.8 / 1.5, ramp(t, S.sp3[0] + 1.6, S.sp3[0] + 3.4));
  const ih = 560, iw = ih * 248 / 235;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: on }}>
      <div style={{ opacity: boxOn }}>
        <IsoBox w={wpl} d={wpl} h={1.5} x={520} y={600} scale={170} color={k > 0.5 ? C.warm : C.xray}
          labels={{ w: `${wpl.toFixed(1)} mm`, d: `${wpl.toFixed(1)} mm`, h: "1.5 mm" }} />
        <Panel o={1} style={{ right: 110, top: 300, width: 800 }}>
          <div style={{ fontSize: 34, lineHeight: 1.6 }}>
            <div style={{ color: C.xray, fontWeight: 700 }}>TotalSegmentator (המאגר שלנו)</div>
            <div style={{ ...mono, textAlign: "right" }}>spacing = (1.5, 1.5, 1.5) mm · isotropic</div>
            <div style={{ opacity: ramp(t, S.sp2[0] + 0.5, S.sp2[0] + 1.3), marginTop: 26 }}>
              <div style={{ color: C.warm, fontWeight: 700 }}>סריקה מקורית (דוגמה)</div>
              <div style={{ ...mono, textAlign: "right" }}>spacing = (0.8, 0.8, 1.5) mm · anisotropic</div>
            </div>
          </div>
        </Panel>
      </div>
      {dist > 0 && (
        <div style={{ opacity: dist }}>
          <Img src={staticFile("data/hero/slice_coronal.png")} style={{ position: "absolute", left: 300, top: 190, width: iw, height: ih, borderRadius: 6 }} />
          <Img src={staticFile("data/hero/slice_coronal.png")} style={{ position: "absolute", left: 1030, top: 190 + ih * (1 - sq) / 2, width: iw, height: ih * sq,
            borderRadius: 6, outline: `3px solid ${C.bad}` }} />
          <Label x={300 + iw / 2} y={800} text={<span><span style={{ color: C.good }}>✓</span> לפי ה-spacing</span>} size={34} />
          <Label x={1030 + iw / 2} y={800} text={<span><span style={{ color: C.bad }}>✗</span> כל voxel צויר כקובייה</span>} size={34} />
        </div>
      )}
    </div>
  );
}

function OrientLabels({ t }: { t: number }) {
  const on = window01(t, S.or1[0] + 0.5, S.or3[1] + 0.5, 0.6, 0.6);
  if (on <= 0) return null;
  const cam = camAt(keys, t);
  const [x0, y0, z0] = vbox.bmin;
  const flip = ramp(t, S.or3[0] + 0.8, S.or3[0] + 2.6);
  const ras = ramp(t, S.or2[0] + 0.3, S.or2[0] + 1.2);
  const s = 1 - 2 * flip;
  const L = 1.5 * s + 0.28 * (s >= 0 ? 1 : -1);
  const pi = project(cam, [x0 + L, y0, z0]), pj = project(cam, [x0, y0, z0 - L]), pk = project(cam, [x0, y0 + 1.8, z0]);
  const sw = flip > 0.5;
  const lab = (p: [number, number], idx: string, ana: string, col: string) => (
    <Label x={p[0]} y={p[1]} size={34} color={col} o={Math.abs(s) > 0.3 || idx === "k" || idx === "z" ? 1 : 0}
      text={<span><span style={mono}>{idx}</span>{ras > 0 ? <span style={{ opacity: ras }}> · {ana}</span> : null}</span>} />);
  return (
    <div style={{ opacity: on }}>
      {lab(pi, sw ? "x" : "i", sw ? "L שמאל" : "R ימין", "#ff9aa3")}
      {lab(pj, sw ? "y" : "j", sw ? "P אחורה" : "A קדימה", "#9dffc0")}
      {lab(pk, sw ? "z" : "k", "S למעלה", "#9fdcff")}
      <Label x={960} y={120} size={60} color={sw ? C.warm : "white"} text={<span style={mono}>{sw ? "LPS" : ras > 0.3 ? "RAS" : "(i, j, k)"}</span>} />
    </div>
  );
}

function AffineSection({ t }: { t: number }) {
  const on = window01(t, S.af1[0] + 0.3, S.af3[1] + 0.4, 0.6, 0.6);
  if (on <= 0) return null;
  const A = info.nifti.affine;
  const f = (v: number) => (Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1));
  const tex = String.raw`\begin{pmatrix}x\\y\\z\\1\end{pmatrix}=\begin{pmatrix}\hl{d}{${f(A[0][0])}}&0&0&\hl{o}{${f(A[0][3])}}\\0&\hl{d}{${f(A[1][1])}}&0&\hl{o}{${f(A[1][3])}}\\0&0&\hl{d}{${f(A[2][2])}}&\hl{o}{${f(A[2][3])}}\\0&0&0&1\end{pmatrix}\begin{pmatrix}\hl{i}{i}\\\hl{i}{j}\\\hl{i}{k}\\1\end{pmatrix}`;
  const ex = ramp(t, S.af3[0] + 0.5, S.af3[0] + 1.3);
  const [i, j, k] = info.nifti.centroid_vox_full, mm = info.nifti.centroid_mm;
  const cam = camAt(keys, t);
  const p = project(cam, [0, 0, 0]);
  return (
    <>
      <Eq t={t} a={S.af1[0] + 0.3} b={S.af3[1] + 0.4} style={{ left: 90, top: 200, width: 880 }} size={44}
        tex={tex} note="affine: voxel index (i, j, k) → mm (x, y, z)"
        terms={{ i: [S.af2[0] + 0.5, S.af2[0] + 4.0], d: [S.af2[0] + 5.0, S.af2[0] + 9.5], o: [S.af2[0] + 9.6, S.af2[1] + 0.3] }} />
      <Panel o={ex} style={{ left: 90, top: 700, width: 880, padding: "16px 30px", direction: "ltr" }}>
        <div style={{ ...mono, fontSize: 36, textAlign: "center" }}>
          ({i}, {j}, {k}) <span style={{ color: C.dim }}>→</span> <span style={{ color: C.warm }}>({mm.map((v) => v.toFixed(1)).join(", ")}) mm</span>
        </div>
      </Panel>
      <Label x={p[0]} y={p[1] - 60} text={<span style={mono}>({i}, {j}, {k})</span>} color={C.warm} size={30} o={ex} />
    </>
  );
}

function MaskSection({ t }: { t: number }) {
  const on = window01(t, S.m1[0] - 0.2, S.m2[1] + 0.5, 0.7, 0.6);
  const filesOn = window01(t, S.m3[0] + 0.3, S.m3[1] + 0.4, 0.6, 0.6);
  const cntOn = window01(t, S.m4[0] + 0.3, S.m4[1] + 0.4, 0.6, 0.6);
  const z = 2.15, w = 248 * z, h = 146 * z;
  const m = info.mask;
  const triOn = window01(t, S.m1[0] - 0.2, S.m2[0] + 0.6, 0.6, 0.6);
  const gridOn = ramp(t, S.m2[0] + 0.4, S.m2[0] + 1.2);
  const maskGrid = ramp(t, S.m2[0] + 6.5, S.m2[0] + 7.3);
  const cell = 56;
  const win = (v: number) => Math.round(Math.max(0, Math.min(1, (v + 350) / 1450)) * 255);
  const files = ["…", "femur_left.nii.gz", "femur_right.nii.gz", "gluteus_maximus_left.nii.gz", "…", "hip_left.nii.gz", "hip_right.nii.gz",
    "…", "sacrum.nii.gz", "…", "vertebrae_L5.nii.gz", "…"];
  const hotC: Record<string, string> = { "hip_left.nii.gz": "#ffa870", "hip_right.nii.gz": "#7cc6ff", "sacrum.nii.gz": "#8cf5a2" };
  const L = info.labels;
  const maskCss: CSSProperties = { WebkitMaskImage: `url(${staticFile("data/v01/axial_mask_hip_left.png")})`, WebkitMaskSize: "100% 100%",
    maskMode: "luminance", maskImage: `url(${staticFile("data/v01/axial_mask_hip_left.png")})`, maskSize: "100% 100%" };
  return (
    <>
      {on > 0 && (
        <div style={{ position: "absolute", inset: 0, opacity: on }}>
          {triOn > 0 && (
            <div style={{ opacity: triOn }}>
              {[0, 1, 2].map((i) => {
                const x = 90 + i * 600;
                return (
                  <div key={i} style={{ position: "absolute", left: x, top: 320, width: w, height: h }}>
                    {i !== 1 && <Img src={staticFile("data/v01/axial_gray.png")} style={{ position: "absolute", width: w, height: h, imageRendering: "pixelated" }} />}
                    {i === 1 && <Img src={staticFile("data/v01/axial_mask_hip_left.png")} style={{ position: "absolute", width: w, height: h, imageRendering: "pixelated", outline: `1px solid ${C.line}` }} />}
                    {i === 2 && <div style={{ position: "absolute", width: w, height: h, background: "#ff9a50", opacity: 0.8, ...maskCss }} />}
                    <div style={{ position: "absolute", top: -52, width: w, textAlign: "center", fontSize: 30, color: C.text, fontFamily: FONT, direction: "rtl" }}>
                      {["סריקה (HU)", "מסכה: hip_left", "יחד"][i]}</div>
                    <div style={{ position: "absolute", top: h + 14, width: w, textAlign: "center", fontSize: 26, color: C.dim, ...mono }}>
                      {i < 2 ? "shape = (146, 248)" : ""}</div>
                  </div>
                );
              })}
            </div>
          )}
          {gridOn > 0 && (
            <div style={{ opacity: gridOn * (1 - triOn) }}>
              <NumGrid rows={m.n} cols={m.n} x={150} y={250} size={cell} font={0.3}
                cell={(r, c) => { const v = m.hu[r][c], g = win(v); return { text: v, bg: grayBg(g), fg: fgFor(g) }; }} />
              <NumGrid rows={m.n} cols={m.n} x={1770 - m.n * cell} y={250} size={cell} font={0.42} o={maskGrid}
                cell={(r, c) => { const v = m.m[r][c]; return { text: v, bg: v ? "rgba(255,154,80,0.85)" : "rgba(20,30,45,0.9)", fg: v ? "#05070b" : C.dim }; }} />
              <Label x={150 + m.n * cell / 2} y={210} text="מספרי CT (HU)" size={32} />
              <Label x={1770 - m.n * cell / 2} y={210} text="מסכה hip_left" size={32} color={C.warm} o={maskGrid} />
              <div style={{ position: "absolute", left: 960 - 160, top: 330, width: 320 }}>
                <Img src={staticFile("data/v01/axial_gray.png")} style={{ width: 320, display: "block" }} />
                <div style={{ position: "absolute", left: m.rc[1] * 320 / 248, top: m.rc[0] * 320 / 248, width: m.n * 320 / 248, height: m.n * 320 / 248,
                  border: `2px solid ${C.warm}`, boxShadow: `0 0 10px ${C.warm}` }} />
              </div>
              <Label x={960} y={560} text="חלון 10×10 מהפרוסה" size={26} color={C.dim} />
            </div>
          )}
        </div>
      )}
      <Panel o={filesOn} style={{ left: 90, top: 190, width: 640, padding: "26px 34px" }}>
        <div style={{ fontSize: 34, fontWeight: 700, color: C.xray, marginBottom: 12 }}>segmentations/ · <span style={mono}>{info.nifti.n_seg_files}</span> קבצים</div>
        {files.map((f, i) => (
          <div key={i} style={{ ...mono, textAlign: "left", fontSize: 28, lineHeight: 1.45, color: hotC[f] ?? C.dim,
            fontWeight: hotC[f] ? 700 : 400, opacity: ramp(t, S.m3[0] + 0.5 + i * 0.12, S.m3[0] + 0.9 + i * 0.12) }}>{f}</div>))}
        <div style={{ ...mono, textAlign: "left", fontSize: 28, marginTop: 14, color: C.warm, opacity: ramp(t, S.m3[1] - 3.4, S.m3[1] - 2.6) }}>
          pelvis = hip_left | hip_right | sacrum</div>
      </Panel>
      <Panel o={cntOn} style={{ left: 90, top: 260, width: 700 }}>
        <div style={{ ...mono, fontSize: 32, lineHeight: 1.8, textAlign: "left" }}>
          <div>hip_left.sum() = <span style={{ color: C.warm }}>{L.hip_left.voxels.toLocaleString("en-US")}</span></div>
          <div style={{ color: C.dim }}>voxel = 1.5 × 1.5 × 1.5 = 3.375 mm³</div>
          <div>volume ≈ <span style={{ color: C.warm }}>{Math.round(L.hip_left.ml)} ml</span></div>
        </div>
      </Panel>
    </>
  );
}

export const V01 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const dims = meta.vol_dims_xyz;
  const vol = useVolume("data/hero/ct_u8.raw", dims);
  const lab = useVolume("data/hero/labels_u8.raw", dims);
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const spine = useObj("data/hero/spine.obj");
  const hipL = useObj("data/hero/hip_left.obj");
  const hipR = useObj("data/hero/hip_right.obj");
  const sacrum = useObj("data/hero/sacrum.obj");
  const ready = vol && lab && pelvis && femurs && spine && hipL && hipR && sacrum;
  const or4 = window01(t, S.or4[0] - 0.2, S.or4[1] + 0.5, 0.7, 0.6);
  const z4 = 600 / DRR.H;
  
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ vol, lab, pelvis, femurs, spine, hipL, hipR, sacrum }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.62 * window01(t, S.pipe[0] - 0.3, tl.total, 0.8, 0.1), pointerEvents: "none" }} />

      <Opening n={1} title="מהי תמונה" active={[0]} t={t} end={2.9} />

      <PixelSection t={t} />
      <CoordSection t={t} />
      <RgbSection t={t} />
      <ResSection t={t} />
      <VolumeOverlays t={t} />
      <SpacingSection t={t} />
      <OrientLabels t={t} />
      {or4 > 0 && (
        <div style={{ position: "absolute", inset: 0, opacity: or4 }}>
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 960 - DRR.W * z4 / 2, top: 190, width: DRR.W * z4, borderRadius: 8 }} />
          <Label x={960 - DRR.W * z4 / 2 - 90} y={470} text="R" size={84} color={C.warm} />
          <Label x={960 + DRR.W * z4 / 2 + 90} y={470} text="L" size={84} color={C.warm} />
          <Label x={960 - DRR.W * z4 / 2 - 90} y={550} text="ימין המטופל" size={26} color={C.dim} />
          <Label x={960 + DRR.W * z4 / 2 + 90} y={550} text="שמאל המטופל" size={26} color={C.dim} />
        </div>
      )}
      <AffineSection t={t} />
      <MaskSection t={t} />

      <Title text="CT = מערך תלת-ממדי" sub="voxels" t={t} a={S.ct2[0]} b={S.ct3[0] + 0.3} top={70} size={64} />
      <Title text="שלושה חתכים" sub="axial · coronal · sagittal" t={t} a={S.pl1[0]} b={S.pl1[1] - 2.6} top={70} size={64} />
      <PipelineMap active={[0]} t={t} a={S.pipe[0]} b={S.pipe[1] + 0.4} top={440} />
      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        <span>תמונה = מערך דו-ממדי של מספרים; <span style={mono}>img[row, col] = img[y, x]</span></span>,
        "לכל פיקסל spacing במ\"מ; גודל אמיתי = N × spacing",
        "CT = מערך תלת-ממדי של voxels; שלושה חתכים",
        "affine: אינדקס → מ\"מ (RAS אצלנו, LPS ב-DICOM)",
        "מסכה = אותו מערך, 0/1; האגן = hip_left ∪ hip_right ∪ sacrum",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/01-data-and-drr/notebook.ipynb" todos={[
        "לטעון ct.nii.gz עם nibabel; להדפיס shape, spacing ו-affine",
        "למצוא את קבצי האגן: hip_left, hip_right, sacrum, ולאחד אותם",
        "להציג את שלושת החתכים (axial, coronal, sagittal) דרך מרכז האגן",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v01" />
    </AbsoluteFill>
  );
};
