// Text-free 3D clips for the RESULTS deck (presentation/results.html). Same rules as Deck.tsx: assets load in
// the DOM component, Stage is gated on `ready`, everything is a pure function of the frame.
// Data: videos/public/data/results/ (written by the training workspace's export_deck_assets.py) and the hero pelvis.
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useMemo } from "react";
import { ramp, window01, lerp } from "../lib/anim";
import { C } from "../lib/style";
import { useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, Rays, WireBox } from "../lib/xray";
import { DeckDetector as Detector } from "./parts";

export const RES = { trio: 720, predict: 660, lift: 720, repose: 600, data: 660 }; // frames @ 60 fps

const useT = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return { frame, t: frame / fps };
};

const RimLights = ({ k = 1 }: { k?: number }) => (
  <>
    <directionalLight position={[-3, 3, 7]} intensity={2.6 * k} color="#8fc8ff" />
    <directionalLight position={[4, -1, 6]} intensity={1.4 * k} color="#5f8dff" />
  </>
);

/** the prediction: vertex colours = distance to the true surface (green 0 mm, warm 5 mm, red >= 15 mm) */
function usePredMaterial() {
  return useMemo(() => new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.45, clearcoat: 0.4,
    clearcoatRoughness: 0.35, transparent: true }), []);
}

/** the truth as a faint "X-ray vision" ghost: only its own front surface, drawn over the prediction */
function Ghost({ geo, o, color = C.bone }: { geo: THREE.BufferGeometry; o: number; color?: string }) {
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: new THREE.Color(color), emissive: new THREE.Color("#ffe2b0"),
    emissiveIntensity: 0.1, roughness: 0.6, transparent: true, depthWrite: false }), [color]);
  mat.opacity = o;
  if (o <= 0.001) return null;
  return <mesh geometry={geo} material={mat} renderOrder={5} />;
}

function Pred({ geo, material, opacity = 1, clipY }: {
  geo: THREE.BufferGeometry; material: THREE.MeshPhysicalMaterial; opacity?: number; clipY?: number;
}) {
  const mat = useMemo(() => material.clone(), [material]);
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, -1, 0), 0), []);
  plane.constant = clipY ?? 1e3;
  mat.clippingPlanes = clipY === undefined ? [] : [plane];
  mat.opacity = opacity;
  mat.depthWrite = opacity > 0.6;
  if (opacity <= 0.001) return null;
  return <mesh geometry={geo} material={mat} />;
}

const EnableClipping = () => {
  const { gl } = useThree();
  gl.localClippingEnabled = true;
  return null;
};

type Case = { truth: THREE.BufferGeometry; pred: THREE.BufferGeometry };
function useCase(kind: string): Case | null {
  const truth = useObj(`data/results/${kind}_truth.obj`);
  const pred = useObj(`data/results/${kind}_pred.obj`);
  return truth && pred ? { truth, pred } : null;
}

// =====================================================================================
// 1. ResTrio: worst / median / best test patient, prediction coloured by error, truth as a ghost; one revolution
// =====================================================================================
function TrioScene({ frame, cases }: { frame: number; cases: Case[] }) {
  const pm = usePredMaterial();
  const ang = (2 * Math.PI * frame) / RES.trio;          // seamless loop
  const xs = [-3.4, 0, 3.4];
  return (
    <>
      <CameraPath t={0} keys={[{ t: 0, pos: [0, 1.5, -10.4], look: [0, -0.1, 0], fov: 34 }]} />
      <RimLights k={0.8} />
      <Floor y={-1.75} opacity={0.5} />
      {cases.map((c, i) => (
        <group key={i} position={[xs[i], 0, 0]} scale={1.0} rotation={[0, ang + i * 0.35, 0]}>
          <Pred geo={c.pred} material={pm} />
          <Ghost geo={c.truth} o={0.16} />
        </group>
      ))}
    </>
  );
}

export const ResTrio = () => {
  const { frame } = useT();
  const w = useCase("worst"), m = useCase("median"), b = useCase("best");
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {w && m && b && <Stage bloom={0.6}><TrioScene frame={frame} cases={[w, m, b]} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// 2. ResPredict: the median patient's two X-rays light up, the prediction builds bottom-up, the truth shows through
// =====================================================================================
const CUBE = 3.2;                                        // 320 mm cube, 0.01 scene units per mm

function PredictScene({ t, c, ap, lat }: { t: number; c: Case; ap: THREE.Texture; lat: THREE.Texture }) {
  const pm = usePredMaterial();
  const det = 2.75;                                      // detector panels just outside the cube
  const panelIn = ramp(t, 0.1, 1.0);
  const apScan = ramp(t, 0.4, 2.0, (x) => x), latScan = ramp(t, 0.9, 2.5, (x) => x);
  const build = ramp(t, 2.6, 5.4, (x) => x);
  const clipY = lerp(-1.7, 1.7, build);
  const boxO = 0.45 * (1 - 0.7 * ramp(t, 5.6, 6.6));
  const ghost = 0.2 * ramp(t, 6.0, 7.2);
  const glow = window01(t, 2.6, 5.6, 0.2, 0.4);
  const keys: CamKey[] = [
    { t: 0, pos: orbit(0.62, 9.2, 2.6), look: [-0.4, -0.1, 0.4], fov: 36 },
    { t: 5.5, pos: orbit(0.2, 8.0, 1.8), look: [-0.2, -0.15, 0.3], fov: 36 },
    { t: 11, pos: orbit(-0.55, 7.6, 1.2), look: [0, -0.15, 0], fov: 36 },
  ];
  return (
    <>
      <EnableClipping />
      <CameraPath t={t} keys={keys} />
      <RimLights k={0.8} />
      <Floor y={-2.2} opacity={0.6} />
      <WireBox size={[CUBE, CUBE, CUBE]} opacity={boxO} />
      {/* AP detector behind the patient (+z = posterior), LAT detector on the patient's right (-x side is left) */}
      <Detector tex={ap} progress={apScan} opacity={panelIn} size={[CUBE * 1.1, CUBE * 1.1]} position={[0, 0, det]} rotation={[0, Math.PI, 0]} />
      <Detector tex={lat} progress={latScan} opacity={panelIn} size={[CUBE * 1.1, CUBE * 1.1]} position={[-det, 0, 0]} rotation={[0, Math.PI / 2, 0]} flipX />
      {/* the build line */}
      {glow > 0 && (
        <group position={[0, clipY, 0]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[CUBE, CUBE]} />
            <meshBasicMaterial color={new THREE.Color(0.3, 0.6, 1.2)} transparent opacity={0.1 * glow}
              blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} side={THREE.DoubleSide} /></mesh>
          <WireBox size={[CUBE, 0.001, CUBE]} opacity={0.9 * glow} color="#9fdcff" />
        </group>
      )}
      {build > 0 && <Pred geo={c.pred} material={pm} clipY={clipY} />}
      <Ghost geo={c.truth} o={ghost} />
    </>
  );
}

export const ResPredict = () => {
  const { t } = useT();
  const c = useCase("median");
  const ap = useTex("data/results/median_ap.png");
  const lat = useTex("data/results/median_lat.png");
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {c && ap && lat && <Stage bloom={0.8}><PredictScene t={t} c={c} ap={ap} lat={lat} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// 3. ResLift: what the network assumed (parallel lift) vs where the X-ray really went (perspective)
// Real geometry, AP view, seen from above: source 750 mm in front of the cube centre, detector 1,020 mm from it.
// Parallel lift: a pixel at detector position u feeds the voxel column x = u / 1.6 (the 512 mm detector
// stretched over the 320 mm cube). The real ray through that pixel sits at x = u (z + 7.5) / 10.2.
// The two agree only where (z + 7.5) / 10.2 = 1 / 1.6, i.e. 112 mm in front of the centre.
// =====================================================================================
const SRC_Z = -7.5, DET_Z = 2.7, SDD = 10.2, STRETCH = 1.6;
const rayX = (u: number, z: number) => (u * (z - SRC_Z)) / SDD;

function LiftScene({ t, pelvis, ap }: { t: number; pelvis: THREE.BufferGeometry; ap: THREE.Texture }) {
  const bone = useBoneMaterial("#cdbfa5");
  const us = [-2.3, -1.72, -1.15, -0.57, 0, 0.57, 1.15, 1.72, 2.3];
  const fan = ramp(t, 0.5, 2.4);
  const par = ramp(t, 2.8, 4.4);
  const focus = window01(t, 4.8, 8.6, 0.5, 0.6);
  const fix = ramp(t, 8.6, 10.4);
  const half = CUBE / 2;
  // true rays (blue): source -> pixel, drawn up to the growing tip
  const trueSegs: [number, number, number][][] = us.map((u) => {
    const zEnd = lerp(SRC_Z, DET_Z, fan);
    return [[0, 0, SRC_Z], [rayX(u, zEnd), 0, zEnd]];
  });
  // the parallel lift (warm): inside the cube at x = u / 1.6, morphing onto the true ray when fixed
  const parSegs: [number, number, number][][] = [];
  for (const u of us) {
    const n = 12;
    for (let i = 0; i < n; i++) {
      const z0 = -half + (CUBE * i) / n, z1 = -half + (CUBE * (i + 1)) / n;
      parSegs.push([[lerp(u / STRETCH, rayX(u, z0), fix), 0.004, z0], [lerp(u / STRETCH, rayX(u, z1), fix), 0.004, z1]]);
    }
  }
  // the focus pixel: a voxel at the back of the cube reads pixel uF, but its real ray lands at uTrue
  const uF = 2.3, zV = 1.25, xV = uF / STRETCH, uTrue = (xV * SDD) / (zV - SRC_Z);
  const keys: CamKey[] = [
    { t: 0, pos: [3.4, 9.6, -6.4], look: [0, 0, -1.6], fov: 40 },
    { t: 4.6, pos: [2.6, 8.2, -2.4], look: [0.4, 0, -0.6], fov: 38 },
    { t: 6.0, pos: [3.6, 4.6, 0.4], look: [1.3, 0, 1.4], fov: 34 },
    { t: 8.6, pos: [3.4, 4.8, 0.2], look: [1.3, 0, 1.4], fov: 34 },
    { t: 12, pos: [2.8, 9.4, -4.6], look: [0, 0, -1.2], fov: 40 },
  ];
  const red = new THREE.Color(3.0, 1.2, 1.3), green = new THREE.Color(1.0, 3.0, 1.6);
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <RimLights k={0.7} />
      <Floor y={-1.9} opacity={0.45} />
      <WireBox size={[CUBE, CUBE, CUBE]} opacity={0.5} />
      <Bone geo={pelvis} material={bone} opacity={0.22} />
      {/* source */}
      <mesh position={[0, 0, SRC_Z]}><sphereGeometry args={[0.12, 24, 16]} />
        <meshBasicMaterial color={new THREE.Color(2.6, 3.4, 5.0)} toneMapped={false} /></mesh>
      <Detector tex={ap} progress={1} opacity={0.85} size={[5.12, 5.12]} position={[0, 0, DET_Z]} rotation={[0, Math.PI, 0]} />
      <Rays segs={trueSegs} opacity={fan * (1 - 0.45 * focus)} color={[0.9, 1.6, 2.8]} />
      <Rays segs={parSegs} opacity={par * (1 - 0.5 * focus)}
        color={[lerp(3.0, 0.9, fix), lerp(1.7, 2.6, fix), lerp(0.6, 1.4, fix)]} />
      {/* the focus: one voxel, the pixel it read (red) and where its own ray really lands (green) */}
      {focus > 0 && (
        <>
          <mesh position={[xV, 0, zV]}><boxGeometry args={[0.16, 0.16, 0.16]} />
            <meshBasicMaterial color={new THREE.Color(2.4, 2.0, 1.0)} transparent opacity={focus} toneMapped={false} /></mesh>
          <Rays segs={[[[xV, 0.01, zV], [xV, 0.01, DET_Z]], [[xV, 0.01, DET_Z], [uF, 0.01, DET_Z - 0.01]]]} opacity={focus} color={[3.0, 1.2, 1.3]} />
          <Rays segs={[[[0, 0.02, SRC_Z], [uTrue, 0.02, DET_Z]]]} opacity={focus} color={[1.0, 3.0, 1.6]} />
          <mesh position={[uF, 0, DET_Z - 0.03]}><sphereGeometry args={[0.07, 16, 12]} />
            <meshBasicMaterial color={red} transparent opacity={focus} toneMapped={false} /></mesh>
          <mesh position={[uTrue, 0, DET_Z - 0.03]}><sphereGeometry args={[0.07, 16, 12]} />
            <meshBasicMaterial color={green} transparent opacity={focus} toneMapped={false} /></mesh>
          {/* the error, as a bar along the detector */}
          <mesh position={[(uF + uTrue) / 2, 0, DET_Z - 0.05]}><boxGeometry args={[Math.abs(uF - uTrue), 0.03, 0.03]} />
            <meshBasicMaterial color={new THREE.Color(3.0, 2.4, 1.0)} transparent opacity={focus} toneMapped={false} /></mesh>
        </>
      )}
    </>
  );
}

export const ResLift = () => {
  const { t } = useT();
  const pelvis = useObj("data/results/median_truth.obj");
  const ap = useTex("data/results/median_ap.png");
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {pelvis && ap && <Stage bloom={1.0}><LiftScene t={t} pelvis={pelvis} ap={ap} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// 4. ResRepose: the re-posed training copies (tilt ±12°, obliquity ±6°, rotation ±10°, scale 0.95-1.05)
// The pelvis visits four poses, leaving a translucent copy at each; seamless loop.
// =====================================================================================
const D2R = Math.PI / 180;
const POSES: [number, number, number, number][] = [   // tilt (x), rotation (y), obliquity (z), scale
  [12, 10, -6, 1.05], [-12, -10, 6, 0.95], [8, -10, -6, 1.0], [-10, 8, 6, 1.03],
];

function ReposeScene({ t, parts }: { t: number; parts: THREE.BufferGeometry[] }) {
  const bone = useBoneMaterial("#c2b59c");
  // one material per copy: per-object opacity on a shared material doesn't refresh reliably (see lib Bone)
  const ghostMats = useMemo(() => POSES.map(() => new THREE.MeshStandardMaterial({ color: new THREE.Color("#79bff2"),
    emissive: new THREE.Color("#3f7fd0"), emissiveIntensity: 0.35, roughness: 0.5, transparent: true, depthWrite: false })), []);
  const T = RES.repose / 60, leg = T / (POSES.length + 1);
  // pose path: base -> P0 -> P1 -> P2 -> P3 -> base
  const path = [[0, 0, 0, 1], ...POSES, [0, 0, 0, 1]];
  const i = Math.min(path.length - 2, Math.floor(t / leg));
  const k = ramp(t, i * leg + 0.35 * leg, (i + 1) * leg);
  const p = path[i].map((v, j) => lerp(v, path[i + 1][j], k));
  const fadeAll = 1 - ramp(t, T - 1.2, T - 0.2);
  return (
    <>
      <CameraPath t={0} keys={[{ t: 0, pos: orbit(0.55, 8.2, 2.0), look: [0, -0.1, 0], fov: 36 }]} />
      <RimLights k={0.8} />
      <Floor y={-2.0} opacity={0.55} />
      <WireBox size={[CUBE, CUBE, CUBE]} opacity={0.35} />
      {POSES.map((q, j) => {
        const born = (j + 1) * leg;
        const o = 0.22 * ramp(t, born - 0.1, born + 0.5) * fadeAll;
        if (o <= 0.001) return null;
        ghostMats[j].opacity = o;
        return (
          <group key={j} rotation={[q[0] * D2R, q[1] * D2R, q[2] * D2R]} scale={q[3]}>
            {parts.map((g, n) => <mesh key={n} geometry={g} material={ghostMats[j]} />)}
          </group>
        );
      })}
      <group rotation={[p[0] * D2R, p[1] * D2R, p[2] * D2R]} scale={p[3]}>
        {parts.map((g, n) => <Bone key={n} geo={g} material={bone} />)}
      </group>
    </>
  );
}

export const ResRepose = () => {
  const { t } = useT();
  const hipL = useObj("data/hero/hip_left.obj");
  const hipR = useObj("data/hero/hip_right.obj");
  const sacrum = useObj("data/hero/sacrum.obj");
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {hipL && hipR && sacrum && <Stage bloom={0.8}><ReposeScene t={t} parts={[hipL, hipR, sacrum]} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// 5. ResData: 1,228 CT scans, one cube each, sorted by what the screening decided (real counts)
// order left -> right: no complete pelvis bone 637, pelvis cut by the scan 202, metal 110, larger than the cube 15,
// usable 264. The deck's HTML labels sit under the five stacks.
// =====================================================================================
const FATES = [
  { n: 637, color: "#36506d" }, { n: 202, color: "#4d6f96" }, { n: 110, color: "#a86c7c" },
  { n: 15, color: "#8fa6c4" }, { n: 264, color: "#ffcf7f" },
];
export const DATA_STACK_X = [-4.2, -2.1, 0, 2.1, 4.2];

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function DataScene({ t }: { t: number }) {
  const S = 0.16, G = 0.2;                                // cube size, pitch
  const { fate, wall, stack, delay } = useMemo(() => {
    const fate: number[] = [];
    FATES.forEach((f, j) => { for (let i = 0; i < f.n; i++) fate.push(j); });
    const rnd = mulberry32(7);
    const order = fate.map((_, i) => i).sort(() => rnd() - 0.5);    // deterministic shuffle of wall slots
    const cols = 56, rows = Math.ceil(fate.length / cols);
    const wall = fate.map((_, i) => {
      const s = order[i];
      return [((s % cols) - (cols - 1) / 2) * G, (rows - 1 - Math.floor(s / cols)) * G - 1.2, 0] as [number, number, number];
    });
    const seen = [0, 0, 0, 0, 0];
    const stack = fate.map((f) => {
      const k = seen[f]++;
      const layer = Math.floor(k / 64), r = k % 64;              // 8 x 8 footprint, built upward
      return [DATA_STACK_X[f] + ((r % 8) - 3.5) * G, -1.9 + S / 2 + layer * G, ((Math.floor(r / 8)) - 3.5) * G] as [number, number, number];
    });
    const delay = fate.map(() => rnd());
    return { fate, wall, stack, delay };
  }, []);
  const mesh = useMemo(() => {
    const m = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05, emissive: new THREE.Color("#ffffff"), emissiveIntensity: 0.0 });
    const im = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), m, fate.length);
    return im;
  }, [fate]);
  const base = new THREE.Color("#5f9fd8");
  const cols = FATES.map((f) => new THREE.Color(f.color));
  const m4 = new THREE.Matrix4(), col = new THREE.Color();
  const wallIn = ramp(t, 0.0, 1.0);
  const tint = ramp(t, 1.4, 2.8);
  for (let i = 0; i < fate.length; i++) {
    const d = delay[i];
    const fly = ramp(t, 3.2 + 2.2 * d, 4.6 + 2.2 * d);
    const a = wall[i], b = stack[i];
    const arc = Math.sin(Math.PI * fly) * 1.2;
    const x = lerp(a[0], b[0], fly), y = lerp(a[1], b[1], fly) + arc, z = lerp(a[2], b[2], fly) - arc * 0.6;
    const s = S * wallIn * (fate[i] === 4 ? 1 + 0.15 * window01(t, 7.4, 10.6, 0.5, 0.8) : 1);
    m4.makeScale(s, s, s).setPosition(x, y, z);
    mesh.setMatrixAt(i, m4);
    col.copy(base).lerp(cols[fate[i]], tint);
    if (fate[i] === 4) col.multiplyScalar(1 + 0.9 * ramp(t, 7.2, 8.2));
    mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  const keys: CamKey[] = [
    { t: 0, pos: [0, 1.8, -12.5], look: [0, 0.1, 0], fov: 38 },
    { t: 7.2, pos: [0, 1.7, -10.4], look: [0, -1.9, 0], fov: 38 },
    { t: 11, pos: [0, 2.3, -10.0], look: [0, -1.95, 0], fov: 38 },
  ];
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <RimLights k={0.6} />
      <Floor y={-1.9} opacity={0.6} />
      <primitive object={mesh} />
    </>
  );
}

export const ResData = () => {
  const { t } = useT();
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      <Stage bloom={0.9}><DataScene t={t} /></Stage>
    </AbsoluteFill>
  );
};
