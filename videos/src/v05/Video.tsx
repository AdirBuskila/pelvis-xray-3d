// Video 5 · Representing 3D shape: voxels, point cloud, mesh, SDF; resolution trade-off;
// marching squares -> marching cubes; crop + resample to 64^3. Structure follows v00.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { ReactElement, ReactNode, useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v05/durations.json";
import { buildTimeline, Script } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, Voxels, WireBox } from "../lib/xray";
import { Captions, Credit, Label, Narration, Opening, Panel, RecapCard, tex, Title, YourTurn } from "../lib/Overlays";
import { PointCloud, SdfPlane, Surface, VoxelSet, Wire } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const seg = (id: string) => S[id];

type Occ = { n: number; cell: number; count: number; centers: number[] };
type Res = { n: number; cells: number; occupied: number; cell_mm: number; mb_float32: number; verts: number; faces: number; mesh_err_mm: number };
type Stats = {
  ct_shape: number[]; ct_voxels: number; cube_mm: number; cube_center_scene: [number, number, number];
  sdf_plane: { cx: number; cy: number; z: number; w: number; h: number; range_mm: number };
  res: Record<string, Res>; crop: { bbox: number[]; cube: number[]; full_px: number[] };
};
type Grid = { n: number; v: number[][] };
type McCase = { corners: number[][]; v: number[]; f: number[] };

// screen x of a world x on the row layouts (camera at z = -9.5, fov 38): patient right (+x) is on the left
const rowX = (x: number) => 960 - (x / 5.815) * 960;
const ROW4 = [-4.05, -1.35, 1.35, 4.05];
const ROW3 = [-3.0, 0, 3.0];
const LTR = ({ children }: { children: ReactNode }) => <span style={{ direction: "ltr", display: "inline-block" }}>{children}</span>;

// ---------------- the single marching cube ----------------
function McCube({ o, inside, tri, triO }: { o: number; inside: boolean[]; tri: McCase | null; triO: number }) {
  const s = 1.8;
  const geo = useMemo(() => {
    if (!tri) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(tri.v.map((x) => (x - 0.5) * s), 3));
    g.setIndex(tri.f); g.computeVertexNormals();
    return g;
  }, [tri]);
  const triMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#ffcf7f", emissive: new THREE.Color("#ff9f40"),
    emissiveIntensity: 0.35, side: THREE.DoubleSide, transparent: true, roughness: 0.4 }), []);
  triMat.opacity = o * triO * 0.92;
  if (o <= 0.001) return null;
  const corners: [number, number, number][] = [];
  for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) for (let z = 0; z < 2; z++) corners.push([x, y, z]);
  return (
    <group>
      <WireBox size={[s, s, s]} opacity={o * 0.9} color="#7fd0ff" />
      {corners.map((c, i) => {
        const on = inside[c[0] * 4 + c[1] * 2 + c[2]];
        return (
          <mesh key={i} position={[(c[0] - 0.5) * s, (c[1] - 0.5) * s, (c[2] - 0.5) * s]}>
            <sphereGeometry args={[on ? 0.11 : 0.075, 24, 16]} />
            <meshBasicMaterial color={new THREE.Color(...((on ? [1.9, 1.4, 0.55] : [0.25, 0.4, 0.6]) as [number, number, number]))}
              transparent opacity={o} toneMapped={false} />
          </mesh>
        );
      })}
      {geo && triO > 0.001 && <mesh geometry={geo} material={triMat} />}
    </group>
  );
}

// ---------------- 3D scene ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const st = a.stats;
  const cc = st.cube_center_scene;
  const cubeS = st.cube_mm * 0.01;

  // ---- visibility ----
  const heroOn = window01(t, 2.4, seg("vox1")[0] + 1.2, 1.0, 1.0) + window01(t, seg("recap")[0] - 0.4, tl.total + 1, 1.0, 0.1) * 0.55;
  const voxOn = window01(t, seg("vox1")[0] + 0.3, seg("pts1")[0] + 0.8, 0.8, 0.9);
  const voxBuild = ramp(t, seg("vox1")[0] + 0.6, seg("vox1")[1] - 0.3);
  const boxOn = window01(t, seg("vox1")[0] + 0.3, seg("vox2")[1] + 0.4, 0.8, 0.8);
  const ptsOn = window01(t, seg("pts1")[0] + 0.2, seg("mesh1")[0] + 0.8, 0.8, 0.9);
  const ptsReveal = ramp(t, seg("pts1")[0] + 0.3, seg("pts1")[0] + 4.0);
  const mesh1On = window01(t, seg("mesh1")[0] + 0.3, seg("mesh2")[0] + 1.6, 0.9, 1.0);
  const wireOn = window01(t, seg("mesh1")[0] + 1.2, seg("mesh2")[0] + 1.0, 0.8, 1.0);
  const mesh2On = window01(t, seg("mesh2")[0] + 0.6, seg("sdf1")[0] + 1.0, 1.0, 0.8);
  const sdfBone = window01(t, seg("sdf1")[0] + 0.2, seg("cmp")[0] + 0.4, 0.8, 0.8);
  const sdfOn = window01(t, seg("sdf1")[0] + 1.0, seg("cmp")[0] + 0.4, 1.0, 0.8);
  const sdfLevel = 9 * Math.sin((t - seg("sdf3")[0] - 3.0) * 0.9) * ramp(t, seg("sdf3")[0] + 3.0, seg("sdf3")[0] + 4.0);
  const cmpOn = window01(t, seg("cmp")[0] + 0.2, seg("cmp")[1] + 0.5, 0.9, 0.7);
  const res1On = window01(t, seg("res1")[0], seg("res3")[0] + 0.5, 0.8, 0.6);
  const res3On = window01(t, seg("res3")[0] + 0.3, seg("res4")[1] + 0.5, 0.8, 0.7);
  const cubeOn = window01(t, seg("mc1")[0], seg("mc2")[1] + 0.4, 0.8, 0.6);
  const sweepOn = window01(t, seg("mc3")[0] - 0.2, seg("mc4")[1] + 0.5, 0.8, 0.8);
  const sweep = ramp(t, seg("mc3")[0] + 1.0, seg("mc3")[1] + 0.6, (x) => x);
  const crop4On = window01(t, seg("crop4")[0] + 0.3, seg("crop4")[1] + 0.5, 0.8, 0.6);

  // single cube: binary counting in mc1, the named cases in mc2
  const names = ["one", "edge", "face", "three", "diag"];
  const [m2a, m2b] = seg("mc2");
  const slot = Math.min(names.length - 1, Math.max(0, Math.floor(((t - m2a) / (m2b - m2a)) * names.length)));
  const inMc2 = t >= m2a - 0.2;
  const pattern = Math.floor(Math.max(0, t - seg("mc1")[0] - 0.5) * 6) % 256;
  const cs = a.cases[names[slot]];
  const inside = Array.from({ length: 8 }, (_, i) => {
    if (!inMc2) return ((pattern >> i) & 1) === 1;
    return cs.corners.some((c) => c[0] * 4 + c[1] * 2 + c[2] === i);
  });
  const triO = inMc2 ? ramp(t, m2a + 0.2 + slot * (m2b - m2a) / 5, m2a + 0.9 + slot * (m2b - m2a) / 5) : 0;

  // sweep: faces are sorted bottom-up, so the draw range is a sweep; find the current cut height
  const pos = a.mc[2].getAttribute("position") as THREE.BufferAttribute;
  const F = pos.count / 3;
  const k = Math.floor(F * sweep);
  const kk = Math.min(F - 1, k);
  const cutY = (pos.getY(3 * kk) + pos.getY(3 * kk + 1) + pos.getY(3 * kk + 2)) / 3;
  const occ = a.occ[2];

  const keys: CamKey[] = [
    { t: 0, pos: orbit(-0.5, 8.8, 0.9) },
    { t: seg("intro2")[1], pos: orbit(0.45, 8.4, 1.0) },
    { t: seg("vox1")[0] + 1.0, pos: orbit(0.6, 8.2, 1.8) },
    { t: seg("vox2")[1], pos: orbit(-0.55, 7.8, 1.3) },
    { t: seg("pts2")[1], pos: orbit(-0.1, 7.2, 0.6) },
    { t: seg("mesh1")[0] + 2.5, pos: [2.4, 1.5, -5.6], look: [0.5, 0.2, 0] },
    { t: seg("mesh1")[1], pos: [1.8, 1.3, -5.4], look: [0.5, 0.2, 0] },
    { t: seg("mesh2")[1], pos: orbit(-0.45, 8.0, 1.0) },
    { t: seg("sdf1")[0] + 1.5, pos: [0, 0.1, -8.0] },
    { t: seg("sdf3")[1], pos: [0, 0.1, -7.6] },
    { t: seg("cmp")[0] + 0.4, pos: [0, 0.3, -9.5] },
    { t: seg("res4")[1], pos: [0, 0.3, -9.5] },
    { t: seg("mc1")[0] + 0.4, pos: orbit(0.7, 6.2, 2.4), look: [0, 0.55, 0] },
    { t: seg("mc2")[1], pos: orbit(-0.5, 6.0, 2.1), look: [0, 0.55, 0] },
    { t: seg("mc3")[0] + 0.3, pos: orbit(0.55, 8.0, 1.6) },
    { t: seg("mc3")[1], pos: orbit(-0.2, 7.8, 1.3) },
    { t: seg("mc4")[1], pos: orbit(-0.7, 7.6, 1.1) },
    { t: seg("crop4")[0], pos: orbit(0.4, 8.6, 1.6) },
    { t: seg("crop4")[1], pos: orbit(-0.3, 8.4, 1.4) },
    { t: tl.total, pos: orbit(1.0, 9.5, 1.0) },
  ];

  const sp = st.sdf_plane;
  const rowScale = 0.5;
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.1} opacity={0.7 * (1 - cmpOn) * (1 - res1On) * (1 - res3On)} />

      {/* hero pelvis (full resolution mesh) */}
      <Bone geo={a.pelvis} material={bone} opacity={heroOn + mesh2On} />

      {/* 1 · voxels */}
      <Voxels centers={a.occ[2].centers} cell={a.occ[2].cell} reveal={voxBuild} opacity={voxOn} />
      <WireBox size={[cubeS, cubeS, cubeS]} position={cc} opacity={boxOn * 0.5} />

      {/* 2 · point cloud */}
      <PointCloud p={a.points} opacity={ptsOn} reveal={ptsReveal} />

      {/* 3 · mesh: the 64^3 marching-cubes mesh, triangles visible */}
      <Surface geo={a.mc[2]} opacity={mesh1On} flat color="#6c665c" />
      <Wire geo={a.mc[2]} opacity={wireOn * 0.85} color={[0.7, 1.4, 2.6]} />

      {/* 4 · SDF slice */}
      <Bone geo={a.pelvis} material={bone} opacity={sdfBone * 0.3 * (1 - ramp(t, seg("sdf3")[0] + 1.5, seg("sdf3")[0] + 3))} />
      <SdfPlane tex={a.sdf} R={sp.range_mm} level={sdfLevel} opacity={sdfOn} size={[sp.w, sp.h]} position={[sp.cx, sp.cy, sp.z]} />

      {/* the four side by side */}
      {cmpOn > 0 && (
        <>
          <group position={[ROW4[0], 0, 0]} scale={rowScale}><VoxelSet centers={a.occ[2].centers} cell={a.occ[2].cell} opacity={cmpOn} /></group>
          <group position={[ROW4[1], 0, 0]} scale={rowScale}><PointCloud p={a.points} opacity={cmpOn} size={0.02} /></group>
          <group position={[ROW4[2], 0, 0]} scale={rowScale}><Bone geo={a.pelvis} material={bone} opacity={cmpOn} /></group>
          <group position={[ROW4[3], 0, 0]} scale={rowScale}>
            <SdfPlane tex={a.sdf} R={sp.range_mm} opacity={cmpOn} size={[sp.w, sp.h]} position={[sp.cx, sp.cy, sp.z]} />
          </group>
        </>
      )}

      {/* resolution: voxels 16/32/64, then meshes 16/32/64/128 */}
      {res1On > 0 && a.occ.map((o, i) => (
        <group key={i} position={[ROW3[i], -0.1, 0]} scale={0.62} rotation={[0, 0.25 * Math.sin((t - seg("res1")[0]) * 0.4), 0]}>
          <VoxelSet centers={o.centers} cell={o.cell} opacity={res1On * ramp(t, seg("res1")[0] + 0.3 + i * 0.6, seg("res1")[0] + 1.0 + i * 0.6)} gap={0.1} />
        </group>
      ))}
      {res3On > 0 && a.mc.map((g, i) => (
        <group key={i} position={[ROW4[i], -0.1, 0]} scale={0.5} rotation={[0, 0.3 * Math.sin((t - seg("res3")[0]) * 0.4), 0]}>
          <Surface geo={g} opacity={res3On * ramp(t, seg("res3")[0] + 0.5 + i * 0.5, seg("res3")[0] + 1.2 + i * 0.5)} />
        </group>
      ))}

      {/* marching cubes: one cube */}
      <group position={[0, 0.65, 0]}><McCube o={cubeOn} inside={inside} tri={inMc2 ? cs : null} triO={triO} /></group>

      {/* the sweep through our 64^3 grid */}
      {sweepOn > 0 && (
        <>
          <VoxelSet centers={occ.centers} cell={occ.cell} opacity={sweepOn * 0.22 * (1 - ramp(t, seg("mc3")[1], seg("mc3")[1] + 1))}
            keep={(_, y) => y > cutY + occ.cell * 0.5} gap={0.2} />
          <VoxelSet centers={occ.centers} cell={occ.cell} color="#ffcf7f" opacity={sweepOn * (sweep > 0 && sweep < 1 ? 0.9 : 0)}
            keep={(_, y) => Math.abs(y - cutY) < occ.cell * 0.5} gap={0.2} emissive={0.5} />
          <Surface geo={a.mc[2]} opacity={sweepOn} drawCount={sweep >= 1 ? Infinity : 3 * k} />
          {sweep > 0 && sweep < 1 && (
            <mesh position={[cc[0], cutY, cc[2]]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[cubeS, cubeS]} />
              <meshBasicMaterial color={new THREE.Color(0.3, 0.6, 1.2)} transparent opacity={0.12 * sweepOn}
                side={THREE.DoubleSide} depthWrite={false} toneMapped={false} blending={THREE.AdditiveBlending} />
            </mesh>
          )}
          <WireBox size={[cubeS, cubeS, cubeS]} position={cc} opacity={sweepOn * 0.35} />
        </>
      )}
      {/* crop4: the 64^3 target in its cube */}
      <Voxels centers={occ.centers} cell={occ.cell} opacity={crop4On} />
      <WireBox size={[cubeS, cubeS, cubeS]} position={cc} opacity={crop4On * 0.6} />
    </>
  );
}

// ---------------- 2D: marching squares ----------------
const CELL = 10; // px per grid cell
function msSegments(v: number[][], r: number, c: number, lv = 0.5): [number, number, number, number][] {
  const a = v[r][c], b = v[r][c + 1], d = v[r + 1][c], e = v[r + 1][c + 1]; // tl, tr, bl, br
  const pts: [number, number][] = [];
  const cross = (p: number, q: number) => (lv - p) / (q - p);
  if ((a > lv) !== (b > lv)) pts.push([c + cross(a, b), r]);           // top
  if ((b > lv) !== (e > lv)) pts.push([c + 1, r + cross(b, e)]);       // right
  if ((d > lv) !== (e > lv)) pts.push([c + cross(d, e), r + 1]);       // bottom
  if ((a > lv) !== (d > lv)) pts.push([c, r + cross(a, d)]);           // left
  const out: [number, number, number, number][] = [];
  if (pts.length === 2) out.push([...pts[0], ...pts[1]]);
  if (pts.length === 4) { out.push([...pts[0], ...pts[3]]); out.push([...pts[1], ...pts[2]]); }
  return out;
}

function caseIcon(bits: number, size: number, hl: boolean, key: number) {
  // corners: bit0 = tl, bit1 = tr, bit2 = br, bit3 = bl
  const cor: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const on = cor.map((_, i) => ((bits >> i) & 1) === 1);
  const v = [[on[0] ? 1 : 0, on[1] ? 1 : 0], [on[3] ? 1 : 0, on[2] ? 1 : 0]];
  const segs = msSegments(v, 0, 0);
  const p = 14, s = size - 2 * p;
  return (
    <svg key={key} width={size} height={size} style={{ background: hl ? "rgba(255,207,127,0.22)" : "rgba(8,14,24,0.7)",
      border: `2px solid ${hl ? C.warm : C.line}`, borderRadius: 10 }}>
      <rect x={p} y={p} width={s} height={s} fill="none" stroke="#3b5d7d" strokeWidth={2} />
      {segs.map(([x1, y1, x2, y2], i) => <line key={i} x1={p + x1 * s} y1={p + y1 * s} x2={p + x2 * s} y2={p + y2 * s} stroke={C.warm} strokeWidth={4} />)}
      {cor.map(([x, y], i) => <circle key={i} cx={p + x * s} cy={p + y * s} r={on[i] ? 8 : 6} fill={on[i] ? C.bone : "#22384f"} stroke={on[i] ? C.warm : "#5d7f9f"} strokeWidth={2} />)}
    </svg>
  );
}

function MarchingSquares({ t, g }: { t: number; g: Grid }) {
  const v = g.v, n = v.length;
  const cells = useMemo(() => {
    const out: ReactElement[] = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const x = v[r][c];
      if (x < 0.02) continue;
      const k = Math.min(1, x);
      out.push(<rect key={r * n + c} x={c * CELL} y={r * CELL} width={CELL} height={CELL}
        fill={`rgba(${Math.round(80 + 160 * k)},${Math.round(110 + 120 * k)},${Math.round(150 + 60 * k)},${0.25 + 0.7 * k})`} />);
    }
    return out;
  }, [g]);
  const o = window01(t, S.ms1[0] - 0.2, S.ms3[1] + 0.5, 0.7, 0.7);
  if (o <= 0) return null;
  const prog = ramp(t, S.ms2[0] + 0.6, S.ms3[1] - 0.6, (x) => x);
  const idx = Math.floor(prog * (n - 1) * (n - 1));
  const cr = Math.floor(idx / (n - 1)), cc = idx % (n - 1);
  const segs: [number, number, number, number][] = [];
  const doneAll = prog >= 1;
  for (let r = 0; r < n - 1; r++) for (let c = 0; c < n - 1; c++) {
    if (!doneAll && (r > cr || (r === cr && c > cc))) continue;
    segs.push(...msSegments(v, r, c));
  }
  // the case of the current square: bit0 tl, bit1 tr, bit2 br, bit3 bl
  const curCase = cr < n - 1 && !doneAll && prog > 0
    ? (v[cr][cc] > 0.5 ? 1 : 0) | (v[cr][cc + 1] > 0.5 ? 2 : 0) | (v[cr + 1][cc + 1] > 0.5 ? 4 : 0) | (v[cr + 1][cc] > 0.5 ? 8 : 0)
    : -1;
  const tableO = window01(t, S.ms2[0] + 0.4, S.ms2[1] + 0.6, 0.6, 0.6);
  const interpO = window01(t, S.ms3[0] + 0.2, S.ms3[1] + 0.5, 0.6, 0.6);
  const W = n * CELL;
  const f = ramp(t, S.ms3[0] + 1.2, S.ms3[0] + 3.0) * 0.6, ix = 60 + 440 * f;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <div style={{ position: "absolute", left: 170, top: 150, width: W, height: W, border: `1.5px solid ${C.line}`, background: "rgba(5,10,18,0.85)" }}>
        <svg width={W} height={W} style={{ position: "absolute", inset: 0 }}>
          {cells}
          {Array.from({ length: n + 1 }, (_, i) => (
            <g key={i}>
              <line x1={i * CELL} y1={0} x2={i * CELL} y2={W} stroke="rgba(120,190,255,0.07)" />
              <line x1={0} y1={i * CELL} x2={W} y2={i * CELL} stroke="rgba(120,190,255,0.07)" />
            </g>
          ))}
          {segs.map(([x1, y1, x2, y2], i) => (
            <line key={i} x1={(x1 + 0.5) * CELL} y1={(y1 + 0.5) * CELL} x2={(x2 + 0.5) * CELL} y2={(y2 + 0.5) * CELL}
              stroke={C.warm} strokeWidth={2.6} strokeLinecap="round" />
          ))}
          {curCase >= 0 && (
            <rect x={(cc + 0.5) * CELL - 4} y={(cr + 0.5) * CELL - 4} width={CELL + 8} height={CELL + 8} fill="none" stroke="white" strokeWidth={3} />
          )}
        </svg>
      </div>
      <Label x={170 + W / 2} y={150 + W + 34} text={<span>שכבה אחת מהקובייה: 64×64 ערכים · <span style={{ color: C.warm }}>קו = 0.5</span></span>} size={28} color={C.dim} />
      {/* the 16 cases */}
      <Panel o={tableO} style={{ left: 1010, top: 150, width: 640, padding: "24px 30px" }}>
        <div style={{ fontSize: 34, fontWeight: 700, color: C.xray, marginBottom: 6 }}>16 מקרים (2⁴)</div>
        <div style={{ fontSize: 24, color: C.dim, marginBottom: 16 }}>● בהיר = פינה בתוך העצם (ערך &gt; 0.5) · מסגרת = הריבוע הנוכחי</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 120px)", gap: 14, direction: "ltr", justifyContent: "center" }}>
          {Array.from({ length: 16 }, (_, b) => caseIcon(b, 120, b === curCase, b))}
        </div>
      </Panel>
      {/* linear interpolation along one edge */}
      <Panel o={interpO} style={{ left: 1010, top: 250, width: 640 }}>
        <div style={{ fontSize: 34, fontWeight: 700, color: C.xray, marginBottom: 20 }}>איפה הקו חוצה את הצלע?</div>
        <svg width={560} height={150} style={{ direction: "ltr" }}>
          <line x1={60} y1={70} x2={500} y2={70} stroke="#5d7f9f" strokeWidth={4} />
          <circle cx={60} cy={70} r={16} fill={C.bone} stroke={C.warm} strokeWidth={3} />
          <circle cx={500} cy={70} r={14} fill="#22384f" stroke="#5d7f9f" strokeWidth={3} />
          <text x={60} y={128} fill="white" fontSize={30} textAnchor="middle" fontFamily={FONT}>v₀ = 0.8</text>
          <text x={500} y={128} fill="white" fontSize={30} textAnchor="middle" fontFamily={FONT}>v₁ = 0.3</text>
          <line x1={ix} y1={34} x2={ix} y2={106} stroke={C.warm} strokeWidth={4} />
          <text x={ix} y={26} fill={C.warm} fontSize={26} textAnchor="middle" fontFamily={FONT}>0.5</text>
        </svg>
        <div style={{ fontSize: 40, textAlign: "center", marginTop: 10 }}>{tex("s=\\frac{0.5-v_0}{v_1-v_0}=\\frac{0.5-0.8}{0.3-0.8}=0.6")}</div>
      </Panel>
    </div>
  );
}

// ---------------- 2D: crop + resample ----------------
function CropResample({ t, st }: { t: number; st: Stats }) {
  const o = window01(t, S.crop1[0] - 0.2, S.crop4[0] + 0.6, 0.7, 0.7);
  if (o <= 0) return null;
  const [fw, fh] = st.crop.full_px;
  const H = 640, W = (H * fw) / fh, L = 200, T = 170;
  const bbO = ramp(t, S.crop2[0] + 0.8, S.crop2[0] + 1.6);
  const cubeO = ramp(t, S.crop2[0] + 3.0, S.crop2[0] + 4.2);
  const rs = ramp(t, S.crop3[0] + 0.4, S.crop3[0] + 1.6);
  const maskO = ramp(t, S.crop3[1] - 3.2, S.crop3[1] - 2.2);
  const [bx, by, bw, bh] = st.crop.bbox, [qx, qy, qw, qh] = st.crop.cube;
  const R = 500, RL = 1180, RT = 240;
  const cellMm = st.cube_mm / 64;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <div style={{ position: "absolute", left: L, top: T, width: W, height: H }}>
        <Img src={staticFile("data/v05/ct_coronal_full.png")} style={{ width: W, height: H, borderRadius: 6, border: `1px solid ${C.line}` }} />
        <div style={{ position: "absolute", left: bx * W, top: by * H, width: bw * W, height: bh * H, opacity: bbO,
          border: `3px dashed ${C.xray}`, boxShadow: "0 0 20px rgba(127,208,255,0.4)" }} />
        <div style={{ position: "absolute", left: qx * W, top: qy * H, width: qw * W, height: qh * H, opacity: cubeO,
          border: `3px solid ${C.warm}`, boxShadow: "0 0 26px rgba(255,207,127,0.45)" }} />
      </div>
      <Label x={L + W / 2} y={T - 36} text={<LTR>{st.ct_shape.join(" × ")} = {(st.ct_voxels / 1e6).toFixed(1)}M voxels</LTR>}
        size={30} color={C.dim} o={ramp(t, S.crop1[0] + 1.5, S.crop1[0] + 2.2)} />
      <Label x={L + qx * W + (qw * W) / 2} y={T + qy * H - 24} text={`${st.cube_mm.toFixed(1)} מ"מ`} color={C.warm} size={30} o={cubeO} />
      <Label x={L + bx * W + (bw * W) / 2} y={T + by * H + bh * H - 24} text="התיבה של האגן" color={C.xray} size={26} o={bbO * (1 - cubeO * 0.7)} />
      {/* resampled 64x64 */}
      <div style={{ position: "absolute", left: RL - 130, top: RT + R / 2 - 40, fontSize: 64, color: C.dim, opacity: rs }}>←</div>
      <div style={{ position: "absolute", left: RL, top: RT, width: R, height: R, opacity: rs, border: `3px solid ${C.warm}`, borderRadius: 4, overflow: "hidden" }}>
        <Img src={staticFile("data/v05/ct64_coronal.png")} style={{ width: R, height: R, imageRendering: "pixelated", position: "absolute" }} />
        <Img src={staticFile("data/v05/mask64_coronal.png")} style={{ width: R, height: R, imageRendering: "pixelated", position: "absolute",
          opacity: maskO * 0.8, mixBlendMode: "screen", filter: "sepia(1) saturate(5) hue-rotate(-20deg)" }} />
        <svg width={R} height={R} style={{ position: "absolute", inset: 0 }}>
          {Array.from({ length: 65 }, (_, i) => (
            <g key={i}><line x1={(i * R) / 64} y1={0} x2={(i * R) / 64} y2={R} stroke="rgba(0,0,0,0.35)" />
              <line x1={0} y1={(i * R) / 64} x2={R} y2={(i * R) / 64} stroke="rgba(0,0,0,0.35)" /></g>
          ))}
        </svg>
      </div>
      <Label x={RL + R / 2} y={RT - 36} text={<LTR>64 × 64 · {cellMm.toFixed(2)} mm / cell</LTR>} color={C.warm} size={32} o={rs} />
      <Label x={RL + R / 2} y={RT + R + 36} text={maskO > 0.5 ? "המסכה: תאים עם ערך > 0.5" : "ערך במרכז כל תא (אינטרפולציה)"} color={C.dim} size={28} o={rs} />
    </div>
  );
}

// ---------------- 2D: resolution charts ----------------
function ResCharts({ t, st }: { t: number; st: Stats }) {
  const o = window01(t, S.res2[0], S.res4[1] + 0.5, 0.7, 0.7);
  if (o <= 0) return null;
  const ns = ["16", "32", "64", "128"];
  const errO = ramp(t, S.res3[0] + 0.4, S.res3[0] + 1.2);
  const bar = (val: number, max: number, color: string, k: number) => (
    <div style={{ height: 150, display: "flex", alignItems: "flex-end", justifyContent: "center" }}>
      <div style={{ width: 54, height: Math.max(3, (150 * val) / max) * k, background: color, borderRadius: 6, boxShadow: `0 0 18px ${color}` }} />
    </div>
  );
  const memLabel = (mb: number) => (mb < 1 ? `${Math.round(mb * 1024)} KB` : `${mb} MB`);
  return (
    <Panel o={o} style={{ left: 240, right: 240, top: 60, padding: "18px 34px", display: "flex", gap: 70, justifyContent: "center" }}>
      <div>
        <div style={{ fontSize: 28, color: C.xray, fontWeight: 700, marginBottom: 6 }}>זיכרון (float32): גדל כמו n³</div>
        <div style={{ display: "flex", gap: 30, direction: "ltr" }}>
          {ns.map((n, i) => {
            const r = st.res[n]; const k = ramp(t, S.res2[0] + 0.6 + i * 0.7, S.res2[0] + 1.2 + i * 0.7);
            return <div key={n} style={{ textAlign: "center", width: 100 }}>
              <div style={{ fontSize: 22, color: "white", opacity: k }}>{memLabel(r.mb_float32)}</div>
              {bar(r.mb_float32, 8, C.xray, k)}
              <div style={{ fontSize: 24, color: C.dim }}>{n}³</div>
            </div>;
          })}
        </div>
      </div>
      <div style={{ opacity: errO }}>
        <div style={{ fontSize: 28, color: C.warm, fontWeight: 700, marginBottom: 6 }}>מרחק ממוצע מהמשטח המדויק</div>
        <div style={{ display: "flex", gap: 30, direction: "ltr" }}>
          {ns.map((n, i) => {
            const r = st.res[n]; const k = ramp(t, S.res3[0] + 1.6 + i * 1.6, S.res3[0] + 2.2 + i * 1.6);
            return <div key={n} style={{ textAlign: "center", width: 100 }}>
              <div style={{ fontSize: 22, color: "white", opacity: k }}>{r.mesh_err_mm < 1 ? r.mesh_err_mm.toFixed(2) : r.mesh_err_mm.toFixed(1)} mm</div>
              {bar(r.mesh_err_mm, 30, C.warm, k)}
              <div style={{ fontSize: 24, color: C.dim }}>{n}³</div>
            </div>;
          })}
        </div>
      </div>
    </Panel>
  );
}

type Assets = {
  stats: Stats; pelvis: THREE.BufferGeometry; occ: Occ[]; points: number[]; mc: THREE.BufferGeometry[];
  sdf: THREE.Texture; cases: Record<string, McCase>;
};

export const V05 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const stats = useJson<Stats>("data/v05/stats.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const o16 = useJson<Occ>("data/hero/occ16.json");
  const o32 = useJson<Occ>("data/hero/occ32.json");
  const o64 = useJson<Occ>("data/hero/occ64.json");
  const pts = useJson<{ p: number[] }>("data/v05/points.json");
  const m16 = useObj("data/v05/mc16.obj");
  const m32 = useObj("data/v05/mc32.obj");
  const m64 = useObj("data/v05/mc64.obj");
  const m128 = useObj("data/v05/mc128.obj");
  const sdf = useTex("data/v05/sdf_coronal.png", false);
  const grid = useJson<Grid>("data/v05/grid64_coronal.json");
  const cases = useJson<Record<string, McCase>>("data/v05/mc_cases.json");
  const ready = stats && pelvis && o16 && o32 && o64 && pts && m16 && m32 && m64 && m128 && sdf && grid && cases;

  const r = stats ? stats.res : null;
  const lvl = 9 * Math.sin((t - S.sdf3[0] - 3.0) * 0.9) * ramp(t, S.sdf3[0] + 3.0, S.sdf3[0] + 4.0);
  const m2 = Math.min(4, Math.max(0, Math.floor(((t - S.mc2[0]) / (S.mc2[1] - S.mc2[0])) * 5)));
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ stats, pelvis, occ: [o16, o32, o64], points: pts.p, mc: [m16, m32, m64, m128], sdf, cases }} />
        </Stage>
      )}
      {/* dim the 3D behind 2D-heavy sections */}
      <AbsoluteFill style={{ background: C.bg, pointerEvents: "none",
        opacity: 0.6 * window01(t, S.recap[0] - 0.3, tl.total, 0.8, 0.1) + 0.9 * window01(t, S.ms1[0] - 0.4, S.ms3[1] + 0.4, 0.6, 0.6)
          + 0.9 * window01(t, S.crop1[0] - 0.4, S.crop4[0] + 0.4, 0.6, 0.6) }} />

      <Opening n={5} title="איך מייצגים צורה בתלת-ממד" active={[3, 4]} t={t} end={2.9} />

      {/* representation names */}
      <Title text="1 · voxels" sub="occupancy: 1 = עצם, 0 = אין" t={t} a={S.vox1[0]} b={S.vox2[1] + 0.3} size={64} top={50} />
      <Title text="2 · point cloud" sub="רק (x, y, z) לכל נקודה" t={t} a={S.pts1[0]} b={S.pts2[1] + 0.3} size={64} top={50} />
      <Title text="3 · mesh" sub="vertices + triangles" t={t} a={S.mesh1[0]} b={S.mesh2[1] + 0.3} size={64} top={50} />
      <Title text="4 · SDF" sub="signed distance field: המרחק למשטח, עם סימן" t={t} a={S.sdf1[0]} b={S.sdf3[1] + 0.3} size={64} top={50} />
      {stats && r && grid && (
        <>
          <Label x={1600} y={780} color={C.warm} size={34} o={window01(t, S.vox2[0] + 0.5, S.vox2[1] + 0.3)}
            text={<LTR>{r["64"].occupied.toLocaleString("en-US")} / {r["64"].cells.toLocaleString("en-US")} cells</LTR>} />
          <Label x={1600} y={780} color={C.warm} size={34} o={window01(t, S.pts1[0] + 2.5, S.pts2[1] + 0.3)} text="7,000 נקודות" />
          <Label x={1560} y={780} color={C.warm} size={32} o={window01(t, S.mesh2[0] + 2.5, S.mesh2[1] + 0.3)}
            text={<LTR>128,653 vertices · 257,346 triangles</LTR>} />
          <Label x={1580} y={330} color={C.dim} size={30} o={window01(t, S.sdf2[0] + 0.5, S.sdf3[1] + 0.3)}
            text={<span><span style={{ color: "#7fb0ff" }}>■ בפנים (&lt;0)</span>   <span style={{ color: "#ff9f80" }}>■ בחוץ (&gt;0)</span></span>} />
          <Label x={1580} y={385} color={C.warm} size={32} o={window01(t, S.sdf3[0] + 3.0, S.sdf3[1] + 0.3)}
            text={<LTR>contour: SDF = {lvl >= 0 ? "+" : "−"}{Math.abs(lvl).toFixed(0)} mm</LTR>} />
          {/* the four, side by side */}
          {["voxels", "point cloud", "mesh", "SDF"].map((n, i) => (
            <Label key={n} x={rowX(ROW4[i])} y={735} text={n} size={34} color={i === 0 || i === 2 ? C.warm : C.text}
              o={window01(t, S.cmp[0] + 0.5, S.cmp[1] + 0.4)} />
          ))}
          <Label x={rowX(ROW4[0])} y={785} text="הרשת חוזה" size={26} color={C.warm} o={window01(t, S.cmp[1] - 4.5, S.cmp[1] + 0.4)} />
          <Label x={rowX(ROW4[2])} y={785} text="תצוגה ומדידה" size={26} color={C.warm} o={window01(t, S.cmp[1] - 3.0, S.cmp[1] + 0.4)} />
          {/* resolution labels */}
          {["16", "32", "64"].map((n, i) => (
            <Label key={n} x={rowX(ROW3[i])} y={790} size={32} color={C.text} o={window01(t, S.res1[0] + 0.5 + i * 0.6, S.res3[0] + 0.4)}
              text={<LTR>{n}³ · {r[n].cell_mm.toFixed(1)} mm/cell</LTR>} />
          ))}
          {["16", "32", "64", "128"].map((n, i) => (
            <Label key={n} x={rowX(ROW4[i])} y={790} size={32} color={C.text} o={window01(t, S.res3[0] + 0.6 + i * 0.5, S.res4[1] + 0.4)}
              text={<LTR>{n}³ mesh</LTR>} />
          ))}
          <ResCharts t={t} st={stats} />
          {/* marching cubes labels */}
          <Label x={1640} y={250} size={44} color={C.warm} o={window01(t, S.mc1[0] + 1.5, S.mc1[1] + 0.3)} text={<LTR>2⁸ = 256 cases</LTR>} />
          <Label x={1640} y={315} size={30} color={C.dim} o={window01(t, S.mc1[0] + 1.5, S.mc1[1] + 0.3)}
            text={<LTR>case #{Math.floor(Math.max(0, t - S.mc1[0] - 0.5) * 6) % 256}</LTR>} />
          <Label x={1600} y={180} size={40} color={C.warm} o={window01(t, S.mc2[0] + 0.2, S.mc2[1] + 0.3)}
            text={["פינה אחת ← משולש", "שתי פינות על צלע ← מלבן", "פאה שלמה ← מישור", "שלוש פינות ← 3 משולשים", "שתי פינות מנוגדות ← 2 משולשים"][m2]} />
          <Label x={1600} y={240} size={30} color={C.dim} o={window01(t, S.mc2[1] - 5.0, S.mc2[1] + 0.3)}
            text={<span>בסימטריה: <LTR>256 → 15</LTR> תבניות</span>} />
          <Label x={960} y={130} size={34} color={C.dim} o={window01(t, S.mc4[0] + 0.4, S.mc4[1] + 0.3)}
            text={<LTR>{r["64"].verts.toLocaleString("en-US")} vertices · {r["64"].faces.toLocaleString("en-US")} triangles</LTR>} />
          <Panel o={window01(t, S.mc4[0] + 4.0, S.mc4[1] + 0.4)} style={{ left: 60, top: 770, width: 820, padding: "16px 26px", direction: "ltr" }}>
            <div style={{ fontFamily: "monospace", fontSize: 22, color: C.text, textAlign: "left", lineHeight: 1.5 }}>
              <span style={{ color: C.xray }}>from</span> skimage.measure <span style={{ color: C.xray }}>import</span> marching_cubes<br />
              verts, faces, normals, _ = marching_cubes(occ, level=<span style={{ color: C.warm }}>0.5</span>)
            </div>
          </Panel>
          <MarchingSquares t={t} g={grid} />
          <CropResample t={t} st={stats} />
          <Label x={960} y={130} size={40} color={C.warm} o={window01(t, S.crop4[0] + 0.5, S.crop4[1] + 0.3)}
            text={<span>המטרה של הרשת: <LTR>64³ occupancy</LTR></span>} />
        </>
      )}
      <Title text="Marching squares" sub="קודם בדו-ממד" t={t} a={S.ms1[0]} b={S.ms1[1] + 0.2} size={50} top={34} />
      <Title text="Marching cubes" t={t} a={S.mc1[0]} b={S.mc1[1] + 0.2} size={64} top={50} />
      <Title text="Crop + resample" sub="מה-CT לקובייה של 64³" t={t} a={S.crop1[0]} b={S.crop1[0] + 3.5} size={50} top={34} />

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "ארבעה ייצוגים: voxels · point cloud · mesh · SDF",
        "הזיכרון גדל כמו n³; 64³ = הפשרה (~5 מ\"מ לתא)",
        "marching cubes: 8 פינות ← 256 מקרים ← משולשים",
        "crop + resample: מה-CT לקובייה שהרשת חוזה",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/02-shape-and-metrics/notebook.ipynb" todos={[
        "crop + resample: האגן לקובייה של 64³ (אינטרפולציה, סף 0.5)",
        "marching cubes על הקובייה (skimage)",
        "ייצוא ה-mesh לקובץ (.obj / .stl)",
        "פתיחה וצפייה בתצוגה תלת-ממדית",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v05" />
    </AbsoluteFill>
  );
};
