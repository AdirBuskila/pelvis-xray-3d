// Text-free 3D clips for the presentation deck (reveal.js). 1920x1080, 60 fps, series background.
// Same rules as the series: assets load in the DOM component, Stage is gated on `ready`,
// everything is a pure function of the frame.
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { useMemo } from "react";
import { ramp, window01, lerp } from "../lib/anim";
import { C } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Meta, Rays, ScanBeam, SourcePanel, WireBox } from "../lib/xray";
import { DeckDetector as Detector, GridVox, Grid, PlaneStack, useMaskTexture, ViewPlane } from "./parts";

export const DECK = { hero: 600, rig: 600, ambiguity: 480, voxels: 480, hull: 480 }; // frames @ 60 fps

const BONE = "#cdbfa5";   // a touch darker than C.bone: the deck lighting is hotter (rim lights)
type Occ = { n: number; cell: number; count: number; centers: number[] };
type Vox = Grid & { truth: number[]; extra: number[]; ap: number[]; lat: number[] };

const useT = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return { frame, t: frame / fps };
};

/** soft radial glow texture (deterministic) for a backdrop plane */
function useGlowTex(inner: string, outer: string) {
  return useMemo(() => {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 512;
    const g = cv.getContext("2d")!;
    const gr = g.createRadialGradient(256, 256, 0, 256, 256, 256);
    gr.addColorStop(0, inner);
    gr.addColorStop(0.45, outer);
    gr.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = gr;
    g.fillRect(0, 0, 512, 512);
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [inner, outer]);
}

function Glow({ position, scale, opacity = 1, inner = "rgba(70,140,255,0.55)", outer = "rgba(30,70,160,0.18)", rotation }: {
  position: [number, number, number]; scale: [number, number]; opacity?: number; inner?: string; outer?: string;
  rotation?: [number, number, number];
}) {
  const tex = useGlowTex(inner, outer);
  if (opacity <= 0.001) return null;
  return (
    <mesh position={position} rotation={rotation}>
      <planeGeometry args={scale} />
      <meshBasicMaterial map={tex} transparent opacity={opacity} blending={THREE.AdditiveBlending} depthWrite={false}
        toneMapped={false} fog={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

/** cool rim light from behind + a warm key, on top of the Stage lights */
const RimLights = ({ k = 1 }: { k?: number }) => (
  <>
    <directionalLight position={[-3, 3, 7]} intensity={2.6 * k} color="#8fc8ff" />
    <directionalLight position={[4, -1, 6]} intensity={1.4 * k} color="#5f8dff" />
  </>
);

// =====================================================================================
// 1. DeckHero: the hero pelvis turning one full revolution (seamless loop)
// =====================================================================================
type HeroA = { hipL: THREE.BufferGeometry; hipR: THREE.BufferGeometry; sacrum: THREE.BufferGeometry; femurs: THREE.BufferGeometry };

function HeroScene({ frame, a }: { frame: number; a: HeroA }) {
  const bone = useBoneMaterial("#c2b59c");
  const sac = useBoneMaterial("#bba886");
  const fem = useBoneMaterial("#565046");
  const ang = (2 * Math.PI * frame) / DECK.hero;       // frame N == frame 0 -> seamless
  const bob = 0.04 * Math.sin(ang);                      // periodic too
  return (
    <>
      <CameraPath t={0} keys={[{ t: 0, pos: [0, 1.7, -7.3], look: [0, -0.12, 0], fov: 36 }]} />
      <RimLights />
      <Glow position={[0, 0.2, 7]} scale={[22, 14]} opacity={0.5} />
      <Glow position={[0, -2.35, 0]} scale={[7, 7]} rotation={[-Math.PI / 2, 0, 0]} opacity={0.35}
        inner="rgba(110,170,255,0.45)" outer="rgba(40,80,170,0.10)" />
      <group rotation={[0, ang, 0]} position={[0, bob, 0]}>
        <Bone geo={a.hipL} material={bone} />
        <Bone geo={a.hipR} material={bone} />
        <Bone geo={a.sacrum} material={sac} />
        <Bone geo={a.femurs} material={fem} opacity={0.9} />
      </group>
    </>
  );
}

export const DeckHero = () => {
  const { frame } = useT();
  const hipL = useObj("data/hero/hip_left.obj");
  const hipR = useObj("data/hero/hip_right.obj");
  const sacrum = useObj("data/hero/sacrum.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const ready = hipL && hipR && sacrum && femurs;
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && <Stage bloom={0.8}><HeroScene frame={frame} a={{ hipL, hipR, sacrum, femurs }} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// shared rig geometry (AP: source at -z, detector at +z; LAT: source at +x, detector at -x)
// =====================================================================================
type RigA = { meta: Meta; pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry; spine: THREE.BufferGeometry; ap: THREE.Texture; lat: THREE.Texture };

const rigGeom = (meta: Meta) => {
  const box = boxOf(meta);
  return {
    box,
    apZsrc: box.mn[2] - 1.4, apZdet: box.mx[2] + 0.3,
    latXsrc: box.mx[0] + 1.4, latXdet: box.mn[0] - 0.3,
  };
};

/** faint edges of the beam volume between a source panel and its detector */
function BeamVolume({ o, meta }: { o: number; meta: Meta }) {
  const { box, apZsrc, apZdet, latXsrc, latXdet } = rigGeom(meta);
  const [cx, cy, cz] = box.c;
  if (o <= 0.001) return null;
  const [aw, ah] = meta.ap_size, [lw, lh] = meta.lat_size;
  const ap: [number, number, number][][] = [], lat: [number, number, number][][] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    ap.push([[cx + sx * aw / 2, cy + sy * ah / 2, apZsrc], [cx + sx * aw / 2, cy + sy * ah / 2, apZdet]]);
    lat.push([[latXsrc, cy + sy * lh / 2, cz + sx * lw / 2], [latXdet, cy + sy * lh / 2, cz + sx * lw / 2]]);
  }
  return (
    <>
      <Rays segs={ap} opacity={o} color={[0.5, 0.8, 1.3]} />
      <Rays segs={lat} opacity={o} color={[0.5, 0.8, 1.3]} />
    </>
  );
}

// =====================================================================================
// 2. DeckRig: both views fire, beams scan, detectors fill with the real DRRs
// =====================================================================================
function RigScene({ t, a }: { t: number; a: RigA }) {
  const bone = useBoneMaterial(BONE);
  const { box, apZsrc, apZdet, latXsrc, latXdet } = rigGeom(a.meta);
  const [cx, cy, cz] = box.c;

  const rigIn = ramp(t, 0, 0.9);
  const apFire = window01(t, 0.7, 4.6, 0.4, 0.8), latFire = window01(t, 3.6, 7.5, 0.4, 0.8);
  const apScan = ramp(t, 1.0, 4.2, (x) => x);
  const latScan = ramp(t, 3.9, 7.1, (x) => x);
  const ghost = ramp(t, 0.9, 1.9);
  const vol = 0.14 * ramp(t, 7.0, 8.4);
  const srcO = (fire: number) => rigIn * (0.28 + 0.72 * fire);

  const keys: CamKey[] = [
    { t: 0, pos: orbit(0.5, 9.4, 2.4), look: [0, -0.1, 0.2] },
    { t: 10, pos: orbit(0.86, 8.9, 4.3), look: [-0.45, -0.25, 0.35] },
  ];

  return (
    <>
      <CameraPath t={t} keys={keys} />
      <RimLights k={0.7} />
      <Floor y={-2.2} opacity={0.75} />
      <Bone geo={a.pelvis} material={bone} opacity={1 - 0.42 * ghost} />
      <Bone geo={a.femurs} material={bone} opacity={0.9 - 0.55 * ghost} />
      <Bone geo={a.spine} material={bone} opacity={0.9 - 0.55 * ghost} />

      <group position={[cx, cy, 0]}>
        <SourcePanel size={a.meta.ap_size} position={[0, 0, apZsrc]} opacity={srcO(apFire)} />
        <ScanBeam axis="ap" progress={apScan} opacity={1} w={a.meta.ap_size[0]} h={a.meta.ap_size[1]}
          src={apZsrc} det={apZdet} bodyIn={box.mn[2]} bodyOut={box.mx[2]} />
        <Detector tex={a.ap} progress={apScan} opacity={rigIn} size={a.meta.ap_size}
          position={[0, 0, apZdet]} rotation={[0, Math.PI, 0]} />
      </group>
      <group position={[0, cy, cz]}>
        <SourcePanel size={a.meta.lat_size} position={[latXsrc, 0, 0]} rotation={[0, Math.PI / 2, 0]} opacity={srcO(latFire)} />
        <ScanBeam axis="lat" progress={latScan} opacity={1} w={a.meta.lat_size[0]} h={a.meta.lat_size[1]}
          src={latXsrc} det={latXdet} bodyIn={box.mx[0]} bodyOut={box.mn[0]} />
        <Detector tex={a.lat} progress={latScan} opacity={rigIn} size={a.meta.lat_size}
          position={[latXdet, 0, 0]} rotation={[0, Math.PI / 2, 0]} flipX />
      </group>
      <BeamVolume o={vol} meta={a.meta} />
    </>
  );
}

function useRigAssets() {
  const meta = useJson<Meta>("data/hero/meta.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const spine = useObj("data/hero/spine.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const lat = useTex("data/hero/drr_lat.png");
  return meta && pelvis && femurs && spine && ap && lat ? { meta, pelvis, femurs, spine, ap, lat } : null;
}

export const DeckRig = () => {
  const { t } = useT();
  const a = useRigAssets();
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {a && <Stage bloom={1.0}><RigScene t={t} a={a} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// 3. DeckAmbiguity: one ray, one pixel, many possible depths
// =====================================================================================
const SHAPES = ["sphere", "box", "octa", "ico"] as const;

function AmbShape({ kind, position, s, rot, o, glow }: {
  kind: (typeof SHAPES)[number]; position: [number, number, number]; s: number; rot: number; o: number; glow: number;
}) {
  const mat = useMemo(() => new THREE.MeshPhysicalMaterial({ color: new THREE.Color(C.warm), roughness: 0.3, clearcoat: 0.6,
    emissive: new THREE.Color(C.warm), transparent: true }), []);
  mat.opacity = o;
  mat.emissiveIntensity = 0.35 + 1.6 * glow;
  if (o <= 0.001 || s <= 0.001) return null;
  return (
    <mesh position={position} scale={s} rotation={[rot * 0.7, rot, rot * 0.3]} material={mat}>
      {kind === "sphere" && <sphereGeometry args={[0.13, 40, 28]} />}
      {kind === "box" && <boxGeometry args={[0.2, 0.2, 0.2]} />}
      {kind === "octa" && <octahedronGeometry args={[0.16, 0]} />}
      {kind === "ico" && <icosahedronGeometry args={[0.15, 0]} />}
    </mesh>
  );
}

function AmbScene({ t, a }: { t: number; a: RigA }) {
  const bone = useBoneMaterial(BONE);
  const { box, apZsrc, apZdet } = rigGeom(a.meta);
  const [cx, cy, cz] = box.c;
  const rx = cx + 0.55, ry = cy + 0.35;                 // x > 0: through the patient's right ilium
  const zs = [-0.78, -0.26, 0.26, 0.74];                 // four depths inside the pelvis box
  const born = [1.4, 2.0, 2.6, 3.2];

  const inO = 1;                                        // visible from frame 0 (no black lead-in)
  const rayGrow = ramp(t, 0.35, 1.2);
  const srcO = 0.7 * (1 - 0.85 * window01(t, 3.9, 6.4, 0.6, 0.9));
  const zEnd = lerp(apZsrc, apZdet, rayGrow);
  // pixel brightness: on once the ray lands, plus a flash each time a shape's pulse arrives
  const pulseDur = 0.45;
  let flash = 0;
  born.forEach((b, i) => {
    const arrive = b + 0.25 + pulseDur * (apZdet - zs[i]) / (apZdet - apZsrc) * 2;
    flash = Math.max(flash, Math.exp(-Math.pow((t - arrive) / 0.18, 2)));
  });
  const pix = ramp(t, 1.1, 1.4) * (0.75 + 0.9 * flash);

  const keys: CamKey[] = [
    { t: 0, pos: [5.8, 2.6, -6.6], look: [cx + 0.2, ry - 0.2, 0] },
    { t: 3.5, pos: [7.4, ry + 1.4, -1.4], look: [rx - 0.3, ry - 0.1, 0], fov: 36 },
    // beam's-eye: from behind the source, slightly off-axis -> the four shapes stack onto the pixel
    { t: 4.8, pos: [rx + 0.95, ry + 0.62, apZsrc - 6.6], look: [cx + 0.2, cy + 0.05, apZdet], fov: 29 },
    { t: 5.4, pos: [rx + 0.9, ry + 0.6, apZsrc - 6.4], look: [cx + 0.2, cy + 0.05, apZdet], fov: 29 },
    { t: 7.3, pos: [6.6, ry + 1.7, -2.6], look: [rx - 0.6, ry - 0.3, 0.05], fov: 35 },
  ];

  const segs: [number, number, number][][] = rayGrow > 0 ? [[[rx, ry, apZsrc], [rx, ry, zEnd]]] : [];
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <RimLights k={0.6} />
      <Floor y={-2.2} opacity={0.6 * inO} />
      <Bone geo={a.pelvis} material={bone} opacity={0.2 * inO} />
      <Bone geo={a.femurs} material={bone} opacity={0.14 * inO} />
      <group position={[cx, cy, 0]}>
        <SourcePanel size={a.meta.ap_size} position={[0, 0, apZsrc]} opacity={srcO} />
        <Detector tex={a.ap} progress={1} opacity={0.9 * inO} size={a.meta.ap_size} position={[0, 0, apZdet]} rotation={[0, Math.PI, 0]} />
      </group>
      {/* the ray, drawn twice for a thicker glowing core */}
      <Rays segs={segs} opacity={inO} color={[2.2, 3.0, 4.2]} />
      <Rays segs={segs.map(([p, q]) => [[p[0] + 0.006, p[1], p[2]], [q[0] + 0.006, q[1], q[2]]])} opacity={0.6 * inO} color={[1.2, 1.8, 2.8]} />
      {/* ray tip */}
      {rayGrow > 0 && rayGrow < 1 && (
        <mesh position={[rx, ry, zEnd]}><sphereGeometry args={[0.045, 16, 12]} />
          <meshBasicMaterial color={new THREE.Color(2.5, 3.4, 5)} toneMapped={false} /></mesh>
      )}
      {/* the detector pixel */}
      {pix > 0 && (
        <group position={[rx, ry, apZdet - 0.012]} rotation={[0, Math.PI, 0]}>
          <mesh><planeGeometry args={[0.13, 0.13]} />
            <meshBasicMaterial color={new THREE.Color(3.2, 2.3, 0.9)} transparent opacity={Math.min(1, pix)} toneMapped={false} side={THREE.DoubleSide} /></mesh>
          <mesh position={[0, 0, -0.002]}><ringGeometry args={[0.12, 0.135, 4, 1, Math.PI / 4]} />
            <meshBasicMaterial color={new THREE.Color(2.4, 1.8, 0.8)} transparent opacity={0.4 + 0.6 * flash} toneMapped={false} side={THREE.DoubleSide} /></mesh>
        </group>
      )}
      {/* candidate shapes along the ray, each sending a pulse to the same pixel */}
      {zs.map((z, i) => {
        const k = ramp(t, born[i], born[i] + 0.35);
        const pop = k * (1 + 0.25 * Math.sin(Math.PI * k));
        const p0 = born[i] + 0.25, p1 = p0 + pulseDur * (apZdet - z) / (apZdet - apZsrc) * 2;
        const pk = (t - p0) / (p1 - p0);
        const glow = Math.exp(-Math.pow((t - born[i] - 0.2) / 0.25, 2));
        return (
          <group key={i}>
            <AmbShape kind={SHAPES[i]} position={[rx, ry, z]} s={1.3 * pop} rot={t * 0.6 + i * 1.3} o={k} glow={glow} />
            {pk > 0 && pk < 1 && (
              <mesh position={[rx, ry, lerp(z, apZdet, pk)]}><sphereGeometry args={[0.035, 12, 10]} />
                <meshBasicMaterial color={new THREE.Color(3.4, 2.5, 1.0)} toneMapped={false} /></mesh>
            )}
          </group>
        );
      })}
    </>
  );
}

export const DeckAmbiguity = () => {
  const { t } = useT();
  const a = useRigAssets();
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {a && <Stage bloom={1.1}><AmbScene t={t} a={a} /></Stage>}
    </AbsoluteFill>
  );
};

/** occupancy cubes (scene-space centres) built bottom-up by `reveal`; `size` shrinks every cube */
function OccCubes({ centers, cell, reveal, size = 1, color, emissive }: {
  centers: number[]; cell: number; reveal: number; size?: number; color: string; emissive: number;
}) {
  const count = centers.length / 3;
  const { mesh, ymin, ymax } = useMemo(() => {
    const m = new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.05,
      emissive: new THREE.Color(color), emissiveIntensity: emissive });
    const im = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), m, count);
    let a = Infinity, b = -Infinity;
    for (let i = 1; i < centers.length; i += 3) { a = Math.min(a, centers[i]); b = Math.max(b, centers[i]); }
    return { mesh: im, ymin: a, ymax: b };
  }, [centers, count, color, emissive]);
  const m4 = new THREE.Matrix4();
  const cut = ymin + (ymax - ymin + cell) * reveal;
  let shown = 0;
  for (let i = 0; i < count; i++) {
    const y = centers[3 * i + 1];
    if (y > cut) continue;
    const s = cell * 0.86 * size * Math.min(1, (cut - y) / (cell * 3));
    if (s <= 1e-4) continue;
    m4.makeScale(s, s, s).setPosition(centers[3 * i], y, centers[3 * i + 2]);
    mesh.setMatrixAt(shown++, m4);
  }
  mesh.count = shown;
  mesh.instanceMatrix.needsUpdate = true;
  if (shown === 0) return null;
  return <primitive object={mesh} />;
}

// =====================================================================================
// 4. DeckVoxels: the 64^3 occupancy grid builds bottom-up, then becomes the marching-cubes mesh
// =====================================================================================
function VoxScene({ t, occ, mesh }: { t: number; occ: Occ; mesh: THREE.BufferGeometry }) {
  const bone = useBoneMaterial(BONE);
  const cube = occ.cell * occ.n;
  const build = ramp(t, 0.05, 4.4, (x) => x * x * (3 - 2 * x));
  // cubes pull inside the surface first (no z-fighting), the mesh fades in, then the cubes vanish
  const shrink = 1 - 0.3 * ramp(t, 4.5, 5.1) - 0.7 * ramp(t, 5.6, 6.5);
  const morph = ramp(t, 4.8, 6.0);
  const boxO = 0.5 * (1 - 0.6 * morph);
  // voxel extent in y (for the sweep plane)
  const { ymin, ymax } = useMemo(() => {
    let a = Infinity, b = -Infinity;
    for (let i = 1; i < occ.centers.length; i += 3) { a = Math.min(a, occ.centers[i]); b = Math.max(b, occ.centers[i]); }
    return { ymin: a, ymax: b };
  }, [occ]);
  const cut = ymin + (ymax - ymin + occ.cell) * build;
  const sweepO = window01(t, 0.0, 4.6, 0.25, 0.5);
  const ang = lerp(-0.75, 0.75, t / 8);
  const keys: CamKey[] = [{ t: 0, pos: orbit(ang, 7.6, 2.0), look: [0, -0.1, 0], fov: 36 }];
  return (
    <>
      <CameraPath t={0} keys={keys} />
      <RimLights k={0.8} />
      <Floor y={-2.2} opacity={0.55} />
      <WireBox size={[cube, cube, cube]} opacity={boxO} />
      <OccCubes centers={occ.centers} cell={occ.cell} reveal={build} size={shrink} color="#79bff2" emissive={0.1} />
      {sweepO > 0 && (
        <group position={[0, cut - occ.cell / 2, 0]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[cube, cube]} />
            <meshBasicMaterial color={new THREE.Color(0.3, 0.6, 1.2)} transparent opacity={0.09 * sweepO}
              blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} side={THREE.DoubleSide} /></mesh>
          <WireBox size={[cube, 0.001, cube]} opacity={0.9 * sweepO} color="#9fdcff" />
        </group>
      )}
      <Bone geo={mesh} material={bone} opacity={morph} />
    </>
  );
}

export const DeckVoxels = () => {
  const { t } = useT();
  const occ = useJson<Occ>("data/hero/occ64.json");
  const mesh = useObj("data/v05/mc64.obj");
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {occ && mesh && <Stage bloom={1.0}><VoxScene t={t} occ={occ} mesh={mesh} /></Stage>}
    </AbsoluteFill>
  );
};

// =====================================================================================
// 5. DeckHull: AP + LAT silhouettes extruded and intersected = the visual hull (real, from v07 assets)
// =====================================================================================
const DET = 2.05;

/** the true pelvis drawn over everything (depthTest off): an "X-ray vision" ghost inside the hull */
function GhostOverlay({ geo, o }: { geo: THREE.BufferGeometry; o: number }) {
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: new THREE.Color(C.bone), emissive: new THREE.Color("#ffe2b0"),
    emissiveIntensity: 0.08, roughness: 0.6, transparent: true, depthTest: true, depthWrite: true }), []);
  mat.opacity = o;
  if (o <= 0.001) return null;
  // drawn last; clearing depth first means only the ghost's own front surface shows (no stacked layers)
  return <mesh geometry={geo} material={mat} renderOrder={10} onBeforeRender={(r) => r.clearDepth()} />;
}

function HullScene({ t, g, pelvis }: { t: number; g: Vox; pelvis: THREE.BufferGeometry }) {
  const bone = useBoneMaterial(BONE);
  const silAp = useMaskTexture(g.ap, g.n, [127, 208, 255]);
  const silLat = useMaskTexture(g.lat, g.n, [255, 207, 127]);
  const hullIdx = useMemo(() => [...g.truth, ...g.extra], [g]);

  const detO = ramp(t, 0.3, 1.2);
  const pelvisO = 1 - 0.88 * ramp(t, 1.0, 1.8);
  const apEx = ramp(t, 1.4, 3.2);
  const latEx = ramp(t, 2.6, 4.4);
  const exO = 1 - ramp(t, 5.0, 6.0);
  const hullIn = ramp(t, 4.2, 5.4);
  const cutS = Math.floor(lerp(-1, g.n, hullIn));
  const compare = ramp(t, 5.8, 7.0);              // hull goes glassy, the true pelvis shows inside

  const keys: CamKey[] = [
    { t: 0, pos: [6.1, 3.3, -5.9], look: [-0.1, 0.05, 0.1] },
    { t: 4.4, pos: [5.3, 2.9, -6.6], look: [-0.1, 0, 0.1] },
    { t: 8, pos: [3.9, 2.3, -7.4], look: [-0.1, -0.05, 0.1] },
  ];
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <RimLights k={0.7} />
      <Floor y={-2.2} opacity={0.6} />
      <Bone geo={pelvis} material={bone} opacity={pelvisO * (1 - hullIn)} />
      <GhostOverlay geo={pelvis} o={0.5 * compare} />
      {silAp && silLat && (
        <>
          <ViewPlane tex={silAp} axis="ap" size={g.cube} at={DET} opacity={0.9 * detO} />
          <ViewPlane tex={silLat} axis="lat" size={g.cube} at={-DET} opacity={0.9 * detO} />
          <PlaneStack tex={silAp} axis="ap" size={g.cube} progress={apEx} opacity={0.06 * exO} count={30} additive={false} />
          <PlaneStack tex={silLat} axis="lat" size={g.cube} progress={latEx} opacity={0.06 * exO} count={30} additive={false} />
        </>
      )}
      <GridVox g={g} idx={hullIdx} color="#6fb6ee" opacity={hullIn * (1 - 0.25 * compare)} cutS={cutS}
        grow={1} gap={0.12} emissive={0.06} depthWrite />
    </>
  );
}

export const DeckHull = () => {
  const { t } = useT();
  const vox = useJson<Vox>("data/v07/vox.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {vox && pelvis && <Stage bloom={1.0}><HullScene t={t} g={vox} pelvis={pelvis} /></Stage>}
    </AbsoluteFill>
  );
};
