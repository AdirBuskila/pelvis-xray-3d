// Video 6 · Measuring a reconstruction: Dice/IoU, Hausdorff/HD95/ASSD, Chamfer, NSD, why they disagree,
// a clinical angle, sanity tests. Every number shown comes from prep/v06_assets.py (real masks, scipy EDT).
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { CSSProperties, ReactElement, ReactNode, useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v06/durations.json";
import { buildTimeline, Script } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage } from "../lib/Stage";
import { Rays } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, RecapCard, tex, Title, YourTurn } from "../lib/Overlays";
import { geoFromJson, heat, MeshJson, paint, PointCloud, Surface, VoxelSet } from "../v05/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const seg = (id: string) => S[id];

type M = { dice: number; iou: number; tp: number; fp: number; fn: number; hd: number | null; hd95: number | null;
  assd: number | null; nsd: number; chamfer: number | null };
type Metrics = Record<"gt" | "other" | "wing" | "shift" | "rim" | "empty", M> & {
  cubes: { dice: number; inter: number };
  other_hist: { counts: number[]; edges: number[]; p95: number };
  info: { cell_mm: number; wing_mm: number; shift_mm: number; nsd_tau_mm: number; gt_cm3: number;
    ce: { gt: number; rim: number; rim_removed_cm3: number; centre_px: number[]; edge_gt_px: number[]; edge_rim_px: number[]; r_px: number } };
};
type Ovl = { cell: number; tp: number[]; fp: number[]; fn: number[] };
type Slice = { code: number[][]; surf_gt: number[][]; surf_pred: number[][]; edt: number[][]; counts: { tp: number; fp: number; fn: number } };

const LTR = ({ children }: { children: ReactNode }) => <span style={{ direction: "ltr", display: "inline-block" }}>{children}</span>;
const f2 = (x: number | null) => (x === null ? "—" : x.toFixed(2));
const f1 = (x: number | null) => (x === null ? "—" : x.toFixed(1));
const COL = { tp: "#cfe2f5", fp: "#ff9f40", fn: "#ff4f63" };
const DMAX = 25; // mm: top of the heat scale

/** thin glowing sticks between point pairs (lines are 1 px in WebGL; these read on video) */
function Sticks({ segs, r = 0.007, opacity = 1 }: { segs: [number, number, number][][]; r?: number; opacity?: number }) {
  const mesh = useMemo(() => {
    const m = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(3.2, 2.3, 0.8), transparent: true, toneMapped: false }), Math.max(1, segs.length));
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    segs.forEach(([a, b], i) => {
      const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), d = B.clone().sub(A), L = d.length();
      q.setFromUnitVectors(up, d.clone().normalize());
      m4.compose(A.clone().add(B).multiplyScalar(0.5), q, new THREE.Vector3(r, L, r));
      m.setMatrixAt(i, m4);
    });
    m.count = segs.length;
    return m;
  }, [segs, r]);
  (mesh.material as THREE.MeshBasicMaterial).opacity = opacity;
  if (opacity <= 0.001 || segs.length === 0) return null;
  return <primitive object={mesh} />;
}

// ---------------- 3D scene ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const geos = useMemo(() => {
    const g = {
      gt: geoFromJson(a.gt), gtWing: geoFromJson(a.gt), other: geoFromJson(a.other), otherNsd: geoFromJson(a.other), shift: geoFromJson(a.shift),
    };
    paint(g.gt, null, () => [1, 1, 1]);
    paint(g.gtWing, a.gt.d.wing, (d) => heat(d, DMAX));
    paint(g.other, a.other.d.gt, (d) => heat(d, DMAX));
    paint(g.shift, null, () => [1, 1, 1]);
    return g;
  }, [a]);
  // point clouds (every 2nd vertex) and nearest-neighbour links from a subset of the prediction
  const clouds = useMemo(() => {
    const sub = (v: number[], step: number) => { const o: number[] = []; for (let i = 0; i < v.length / 3; i += step) o.push(v[3 * i], v[3 * i + 1], v[3 * i + 2]); return o; };
    const A = sub(a.gt.v, 2), B = sub(a.other.v, 2);
    const links: [number, number, number][][] = [], far: [number, number, number][][] = [];
    for (let i = 0; i < B.length / 3; i += 7) {
      let best = 1e9, bj = 0;
      for (let j = 0; j < A.length / 3; j++) {
        const dx = A[3 * j] - B[3 * i], dy = A[3 * j + 1] - B[3 * i + 1], dz = A[3 * j + 2] - B[3 * i + 2];
        const d = dx * dx + dy * dy + dz * dz; if (d < best) { best = d; bj = j; }
      }
      const l: [number, number, number][] = [[B[3 * i], B[3 * i + 1], B[3 * i + 2]], [A[3 * bj], A[3 * bj + 1], A[3 * bj + 2]]];
      if (i % 35 === 0) links.push(l);
      if (best > 0.08 * 0.08) far.push(l);       // links longer than 8 mm: visible when zoomed in
    }
    return { A, B, links, far };
  }, [a]);

  const introOn = window01(t, 2.4, seg("dice1")[0] + 0.6, 1.0, 0.8);
  const predIn = ramp(t, seg("intro2")[0] + 3.0, seg("intro2")[0] + 4.5);
  const ovlOn = window01(t, seg("dice3")[0] - 0.2, seg("dice4")[1] + 0.4, 0.8, 0.8) * (1 - 0.7 * window01(t, seg("dice4")[0], seg("dice4")[1] + 0.4, 0.6, 0.4));
  const surfOn = window01(t, seg("surf1")[0], seg("surf2")[0] + 1.2, 0.8, 0.8);
  const linkOn = window01(t, seg("surf1")[0] + 2.5, seg("surf2")[0] + 1.0, 0.8, 0.8);
  const heatOn = window01(t, seg("surf2")[0] + 0.3, seg("edt1")[0] + 0.6, 0.9, 0.6) + window01(t, seg("hist1")[0] - 0.2, seg("assd1")[1] + 0.4, 0.8, 0.8);
  const chamOn = window01(t, seg("cham1")[0], seg("cham2")[1] + 0.4, 0.8, 0.8);
  const chamLinks = window01(t, seg("cham1")[0] + 4.0, seg("cham2")[1] + 0.4, 0.8, 0.8);
  const nsdOn = window01(t, seg("nsd1")[0], seg("nsd1")[1] + 0.6, 0.8, 0.6);
  const tau = ramp(t, seg("nsd1")[0] + 5.5, seg("nsd1")[1] - 3.0) * a.m.info.nsd_tau_mm;
  const wingOn = window01(t, seg("dis1")[0] + 0.3, seg("dis2")[1] + 0.4, 0.8, 0.8);
  const wingCut = ramp(t, seg("dis1")[0] + 4.0, seg("dis1")[1] + 0.2);
  const shiftOn = window01(t, seg("dis3")[0] + 0.2, seg("dis3")[1] + 0.6, 0.8, 0.8);
  const shiftK = ramp(t, seg("dis3")[0] + 2.5, seg("dis3")[0] + 4.0);
  const tailOn = window01(t, seg("recap")[0] - 0.4, tl.total + 1, 1.0, 0.1);

  if (nsdOn > 0) paint(geos.otherNsd, a.other.d.gt, (d) => (d <= tau ? [0.35, 1.0, 0.55] : [1.0, 0.3, 0.32]));

  // right-of-screen placement (patient right is +x = screen left, so negative x moves things to the right)
  const side = -2.3;
  const sideK = Math.max(heatOn > 0 && t > seg("hist1")[0] - 0.4 ? 1 : 0, 0);
  const hx = side * sideK;

  const keys: CamKey[] = [
    { t: 0, pos: orbit(-0.5, 8.6, 0.9) },
    { t: seg("intro2")[1], pos: orbit(0.4, 8.0, 1.0) },
    { t: seg("dice3")[0] + 0.3, pos: [-0.5 + 8.8 * Math.sin(0.3), 1.6, -8.8 * Math.cos(0.3)], look: [-0.5, 0, 0] },
    { t: seg("dice4")[1], pos: [-0.5 - 8.8 * Math.sin(0.3), 1.2, -8.8 * Math.cos(0.3)], look: [-0.5, 0, 0] },
    { t: seg("surf1")[0] + 2.0, pos: [1.2, 1.5, -5.8], look: [0.3, 0.3, 0] },
    { t: seg("surf1")[1], pos: [0.4, 1.3, -5.6], look: [0.1, 0.3, 0] },
    { t: seg("surf2")[1], pos: orbit(-0.2, 7.4, 0.9) },
    { t: seg("hist1")[0], pos: [0, 0.4, -8.0] },
    { t: seg("assd1")[1], pos: [0, 0.4, -8.0] },
    { t: seg("cham1")[0] + 0.5, pos: [-1.6 + 7.4 * Math.sin(0.3), 1.0, -7.4 * Math.cos(0.3)], look: [-1.6, 0, 0] },
    { t: seg("cham2")[1], pos: [-0.9, 0.9, -5.0], look: [-1.3, 0.1, 0] },
    { t: seg("nsd1")[0] + 0.5, pos: orbit(0.3, 7.6, 1.0) },
    { t: seg("nsd1")[1], pos: orbit(-0.4, 7.4, 1.0) },
    { t: seg("dis1")[0] + 0.6, pos: [0, 0.6, -7.8] },
    { t: seg("dis2")[1], pos: orbit(-0.35, 7.6, 1.2), look: [0.2, 0.2, 0] },
    { t: seg("dis3")[0] + 0.6, pos: orbit(0.3, 7.8, 0.7) },
    { t: seg("dis3")[1], pos: orbit(0.15, 7.6, 0.6) },
    { t: seg("recap")[0], pos: orbit(0.3, 9.2, 1.0) },
    { t: tl.total, pos: orbit(1.2, 9.2, 1.0) },
  ];

  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.1} opacity={0.6} />
      {/* intro: GT, then the other person's pelvis on top */}
      <Surface geo={geos.gt} opacity={(introOn * (1 - 0.55 * predIn)) + surfOn * 0.35 + tailOn * 0.6} vertexColors />
      {introOn * predIn > 0 && (
        <mesh geometry={geos.otherNsd}>
          <meshStandardMaterial color="#7fd0ff" transparent opacity={0.55 * introOn * predIn} depthWrite={false} roughness={0.5} />
        </mesh>
      )}

      {/* overlap voxels (dice3/dice4) */}
      {ovlOn > 0 && (
        <>
          <VoxelSet centers={a.ovlOther.tp} cell={a.ovlOther.cell} color={COL.tp} opacity={ovlOn * 0.3} emissive={0.05} />
          <VoxelSet centers={a.ovlOther.fp} cell={a.ovlOther.cell} color={COL.fp} opacity={ovlOn * 0.95} emissive={0.35} />
          <VoxelSet centers={a.ovlOther.fn} cell={a.ovlOther.cell} color={COL.fn} opacity={ovlOn * 0.95} emissive={0.35} />
        </>
      )}

      {/* surface distances: prediction surface + links to the nearest GT point */}
      {surfOn > 0 && (
        <mesh geometry={geos.otherNsd}>
          <meshStandardMaterial color="#7fd0ff" transparent opacity={0.6 * surfOn * (1 - heatOn)} depthWrite={false} roughness={0.5} />
        </mesh>
      )}
      <Sticks segs={clouds.far} opacity={linkOn} />
      <group position={[hx, 0, 0]}>
        <Surface geo={geos.other} opacity={heatOn} vertexColors />
      </group>

      {/* Chamfer: two point clouds and nearest-neighbour links */}
      <group position={[-1.6, 0, 0]}>
        <PointCloud p={clouds.A} opacity={chamOn} size={0.03} color={[2.0, 1.8, 1.4]} />
        <PointCloud p={clouds.B} opacity={chamOn} size={0.03} color={[0.6, 1.7, 3.0]} />
        <Rays segs={clouds.links} opacity={chamLinks} color={[4.0, 3.0, 1.2]} />
      </group>

      {/* NSD: inside / outside the tolerance */}
      <Surface geo={geos.otherNsd} opacity={nsdOn} vertexColors />

      {/* disagreement 1: the missing wing, GT coloured by distance to the prediction */}
      {wingOn > 0 && (
        <>
          <Surface geo={geos.gt} opacity={wingOn * (1 - wingCut)} vertexColors />
          <Surface geo={geos.gtWing} opacity={wingOn * wingCut} vertexColors />
        </>
      )}
      {/* disagreement 2: the same shape shifted 1 cm up */}
      {shiftOn > 0 && (
        <>
          <Surface geo={geos.gt} opacity={shiftOn * 0.9} vertexColors />
          <group position={[0, shiftK * a.m.info.shift_mm * 0.01, 0]}>
            <mesh geometry={geos.shift}>
              <meshStandardMaterial color="#7fd0ff" transparent opacity={0.6 * shiftOn} depthWrite={false} roughness={0.5} />
            </mesh>
          </group>
        </>
      )}
    </>
  );
}

// ---------------- 2D: the overlap slice ----------------
const SC = 9; // px per cell
function OverlapSlice({ t, sl, mode, o, left = 150, top = 170 }: { t: number; sl: Slice; mode: "ovl" | "edt"; o: number; left?: number; top?: number }) {
  const n = sl.code.length, W = n * SC;
  const show = [ramp(t, S.dice1[0] + 2.5, S.dice1[0] + 3.2), ramp(t, S.dice1[0] + 5.5, S.dice1[0] + 6.2), ramp(t, S.dice1[0] + 8.0, S.dice1[0] + 8.7)];
  const rects = useMemo(() => {
    const out: ReactElement[] = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      if (mode === "ovl") {
        const k = sl.code[r][c]; if (!k) continue;
        out.push(<rect key={r * n + c} x={c * SC} y={r * SC} width={SC - 1} height={SC - 1} data-k={k} />);
      } else {
        const d = sl.edt[r][c];
        const [R, G, B] = heat(d, 60);
        const dim = 0.55;
        out.push(<rect key={r * n + c} x={c * SC} y={r * SC} width={SC} height={SC}
          fill={`rgb(${Math.round(R * 255 * dim)},${Math.round(G * 255 * dim)},${Math.round(B * 255 * dim)})`} />);
        if (sl.surf_gt[r][c]) out.push(<rect key={`g${r * n + c}`} x={c * SC} y={r * SC} width={SC} height={SC} fill="white" />);
        if (sl.surf_pred[r][c]) out.push(<rect key={`p${r * n + c}`} x={c * SC + 1.5} y={r * SC + 1.5} width={SC - 3} height={SC - 3} fill="none" stroke={COL.fp} strokeWidth={2} />);
      }
    }
    return out;
  }, [sl, mode]);
  if (o <= 0) return null;
  const fills = [COL.tp, COL.fp, COL.fn];
  return (
    <div style={{ position: "absolute", left, top, width: W, height: W, opacity: o, border: `1.5px solid ${C.line}`, background: "rgba(5,10,18,0.9)" }}>
      <svg width={W} height={W}>
        {mode === "ovl"
          ? rects.map((e) => { const k = (e.props as { "data-k": number })["data-k"]; return <rect key={e.key} {...(e.props as object)} fill={fills[k - 1]} opacity={show[k - 1]} />; })
          : rects}
      </svg>
    </div>
  );
}

function Legend({ items, x, y, o }: { items: [string, string][]; x: number; y: number; o: number }) {
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, opacity: o, display: "flex", gap: 28, fontFamily: FONT, fontSize: 28, color: C.text, direction: "rtl" }}>
      {items.map(([c, s]) => <div key={s} style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ width: 22, height: 22, background: c, borderRadius: 4, display: "inline-block" }} /> {s}</div>)}
    </div>
  );
}

function MetricCard({ title, rows, o, style, children }: { title: string; rows: [ReactNode, string, ("good" | "bad" | "")?][]; o: number; style?: CSSProperties; children?: ReactNode }) {
  return (
    <Panel o={o} style={{ right: 110, top: 220, width: 520, padding: "26px 34px", ...style }}>
      <div style={{ fontSize: 34, fontWeight: 700, color: C.xray, marginBottom: 14 }}>{title}</div>
      {rows.map(([k, v, tone], i) => (
        <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 40, lineHeight: 1.5,
          color: tone === "good" ? C.good : tone === "bad" ? C.bad : C.text }}>
          <span>{k}</span><LTR>{v}</LTR>
        </div>
      ))}
      {children}
    </Panel>
  );
}

// ---------------- 2D: histogram of surface distances ----------------
function Histogram({ t, m }: { t: number; m: Metrics }) {
  const o = window01(t, S.hist1[0], S.assd1[1] + 0.4, 0.7, 0.7);
  if (o <= 0) return null;
  const h = m.other_hist, maxC = Math.max(...h.counts);
  const W = 720, H = 360, x0 = 60, mmToX = (mm: number) => x0 + (mm / 60) * W;
  const bw = W / h.counts.length;
  const mk = (mm: number, label: string, color: string, a0: number, ty = 44) => {
    const k = ramp(t, a0, a0 + 0.6);
    if (k <= 0) return null;
    return <g opacity={k}><line x1={mmToX(mm)} y1={20} x2={mmToX(mm)} y2={H + 10} stroke={color} strokeWidth={4} strokeDasharray="8 6" />
      <text x={mmToX(mm) + 8} y={ty} fill={color} fontSize={28} fontFamily={FONT}>{label}</text></g>;
  };
  return (
    <Panel o={o} style={{ left: 90, top: 170, width: 860, padding: "22px 30px" }}>
      <div style={{ fontSize: 30, color: C.xray, fontWeight: 700 }}>מרחקי משטח: אגן של אדם אחר מול האמת</div>
      <svg width={W + 90} height={H + 70} style={{ direction: "ltr" }}>
        {h.counts.map((c, i) => (
          <rect key={i} x={x0 + i * bw + 1} y={H + 10 - (c / maxC) * (H - 40)} width={bw - 2} height={(c / maxC) * (H - 40)}
            fill="#5fa8e8" opacity={0.85} />
        ))}
        <line x1={x0} y1={H + 10} x2={x0 + W} y2={H + 10} stroke={C.dim} strokeWidth={2} />
        {[0, 10, 20, 30, 40, 50, 60].map((v) => <text key={v} x={mmToX(v)} y={H + 44} fill={C.dim} fontSize={22} textAnchor="middle" fontFamily={FONT}>{v}</text>)}
        <text x={x0 + W + 10} y={H + 44} fill={C.dim} fontSize={22} fontFamily={FONT}>mm</text>
        {mk(m.other.hd!, `max = ${f1(m.other.hd)}`, C.bad, S.hd1[0] + 0.8)}
        {mk(m.other.hd95!, `p95 = ${f1(m.other.hd95)}`, C.warm, S.hd2[0] + 2.5, 84)}
        {mk(m.other.assd!, `mean = ${f1(m.other.assd)}`, C.good, S.assd1[0] + 1.5, 124)}
      </svg>
    </Panel>
  );
}

// ---------------- 2D: CE angle ----------------
function CeAngle({ t, m }: { t: number; m: Metrics }) {
  const o = window01(t, S.clin1[0] + 0.2, S.clin3[1] + 0.5, 0.7, 0.7);
  if (o <= 0) return null;
  const ce = m.info.ce, P = 560;
  const rimO = ramp(t, S.clin2[0] + 6.5, S.clin2[0] + 7.5);
  const pic = (src: string, edge: number[], ang: number, x: number, lab: string, k: number, color: string) => {
    if (k <= 0) return null;
    const [cx, cy] = ce.centre_px.map((v) => v * P), [ex, ey] = edge.map((v) => v * P);
    const R = 0.75 * Math.hypot(ex - cx, ey - cy);
    const ux = (ex - cx) / Math.hypot(ex - cx, ey - cy), uy = (ey - cy) / Math.hypot(ex - cx, ey - cy);
    return (
      <div style={{ position: "absolute", left: x, top: 170, width: P, height: P, opacity: k, borderRadius: 8, overflow: "hidden", border: `1.5px solid ${C.line}` }}>
        <Img src={staticFile(src)} style={{ width: P, height: P, imageRendering: "pixelated", opacity: 0.9 }} />
        <svg width={P} height={P} style={{ position: "absolute", inset: 0 }}>
          <circle cx={cx} cy={cy} r={ce.r_px * P} fill="none" stroke="#9fd6ff" strokeWidth={2} strokeDasharray="6 6" />
          <line x1={cx} y1={cy} x2={cx} y2={cy - 1.25 * Math.hypot(ex - cx, ey - cy)} stroke={C.xray} strokeWidth={4} />
          <line x1={cx} y1={cy} x2={cx + ux * 1.25 * Math.hypot(ex - cx, ey - cy)} y2={cy + uy * 1.25 * Math.hypot(ex - cx, ey - cy)} stroke={color} strokeWidth={4} />
          <path d={`M ${cx} ${cy - R} A ${R} ${R} 0 0 1 ${cx + ux * R} ${cy + uy * R}`} fill="none" stroke={color} strokeWidth={3} />
          <circle cx={cx} cy={cy} r={6} fill="white" /><circle cx={ex} cy={ey} r={7} fill={color} />
        </svg>
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 12, textAlign: "center", fontFamily: FONT, fontSize: 40, fontWeight: 700, color }}>
          {lab}: <LTR>CE = {ang.toFixed(0)}°</LTR></div>
      </div>
    );
  };
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {pic("data/v06/ce_gt.png", ce.edge_gt_px, ce.gt, 1180, "האמת", ramp(t, S.clin1[0] + 0.6, S.clin1[0] + 1.4), C.warm)}
      {pic("data/v06/ce_rim.png", ce.edge_rim_px, ce.rim, 520, "שפה מגולחת", rimO, C.bad)}
      <Label x={1460} y={752} text="ירך שמאל · פרוסה דרך מרכז ראש הירך" size={24} color={C.dim} o={ramp(t, S.clin1[0] + 1.0, S.clin1[0] + 1.6)} />
    </div>
  );
}

type Assets = { m: Metrics; gt: MeshJson; other: MeshJson; shift: MeshJson; ovlOther: Ovl };

export const V06 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const m = useJson<Metrics>("data/v06/metrics.json");
  const gt = useJson<MeshJson>("data/v06/mesh_gt.json");
  const other = useJson<MeshJson>("data/v06/mesh_other.json");
  const shift = useJson<MeshJson>("data/v06/mesh_shift.json");
  const ovlOther = useJson<Ovl>("data/v06/ovl_other.json");
  const sl = useJson<Slice>("data/v06/slice.json");
  const ready = m && gt && other && shift && ovlOther && sl;

  const sliceO = window01(t, S.dice1[0] + 0.2, S.iou1[1] + 0.5, 0.7, 0.7);
  const edtO = window01(t, S.edt1[0] + 0.2, S.edt1[1] + 0.5, 0.7, 0.7);
  const tau = ramp(t, S.nsd1[0] + 5.5, S.nsd1[1] - 3.0) * (m ? m.info.nsd_tau_mm : 5);
  const sc = sl ? sl.counts : { tp: 0, fp: 0, fn: 0 };
  const sliceDice = (2 * sc.tp) / (2 * sc.tp + sc.fp + sc.fn);
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ m, gt, other, shift, ovlOther }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, pointerEvents: "none",
        opacity: 0.6 * window01(t, S.recap[0] - 0.3, tl.total, 0.8, 0.1) + 0.92 * window01(t, S.dice1[0], S.iou1[1] + 0.4, 0.6, 0.6)
          + 0.92 * window01(t, S.edt1[0] - 0.2, S.edt1[1] + 0.4, 0.6, 0.6) + 0.92 * window01(t, S.clin1[0] - 0.2, S.test3[1] + 0.4, 0.6, 0.6)
          + 0.5 * window01(t, S.dis4[0] - 0.2, S.dis4[1] + 0.4, 0.6, 0.6) }} />

      <Opening n={6} title="איך מודדים שחזור" active={[5]} t={t} end={2.9} />
      <Label x={1500} y={300} text="האמת (GT)" color={C.bone} size={36} o={window01(t, S.intro2[0] + 0.5, S.intro2[1] + 0.3)} />
      <Label x={1500} y={350} text="חיזוי: אגן של אדם אחר" color={C.xray} size={36} o={window01(t, S.intro2[0] + 3.2, S.intro2[1] + 0.3)} />

      {ready && (
        <>
          {/* Dice / IoU on one slice */}
          <OverlapSlice t={t} sl={sl} mode="ovl" o={sliceO} />
          <Legend x={150} y={780} o={sliceO} items={[[COL.tp, `TP (${sc.tp})`], [COL.fp, `FP (${sc.fp})`], [COL.fn, `FN (${sc.fn})`]]} />
          <Eq t={t} a={S.dice2[0]} b={S.iou1[1] + 0.4} size={44} style={{ left: 880, top: 190, width: 920 }}
            tex={`\\text{Dice}=\\frac{2\\,\\hl{i}{|A\\cap B|}}{\\hl{a}{|A|}+\\hl{b}{|B|}}=\\frac{2\\cdot ${sc.tp}}{${sc.tp + sc.fn}+${sc.tp + sc.fp}}=${sliceDice.toFixed(2)}`}
            terms={{ i: [S.dice2[0] + 0.5, S.dice2[0] + 2.6], a: [S.dice2[0] + 2.6, S.dice2[0] + 4.6], b: [S.dice2[0] + 2.6, S.dice2[0] + 4.6] }}
            note="A = האמת · B = החיזוי" />
          <Eq t={t} a={S.iou1[0] + 0.3} b={S.iou1[1] + 0.4} size={44} style={{ left: 880, top: 590, width: 920 }}
            tex={`\\text{IoU}=\\frac{|A\\cap B|}{|A\\cup B|}\\qquad \\text{Dice}=\\frac{2\\,\\text{IoU}}{1+\\text{IoU}}`} />
          <MetricCard title="כל הקובייה (64³)" o={window01(t, S.dice3[0] + 0.6, S.dice4[1] + 0.3)} style={{ top: 160 }}
            rows={[["Dice", f2(m.other.dice)], ["IoU", f2(m.other.iou)]]}>
            <div style={{ display: "flex", gap: 24, fontSize: 28, marginTop: 14 }}>
              <span><span style={{ color: COL.fp }}>■</span> FP עודף</span><span><span style={{ color: COL.fn }}>■</span> FN חסר</span></div>
          </MetricCard>
          <BlindSpot t={t} />

          {/* surface distances */}
          <Title text="מרחקי משטח" sub="surface distance: לכל נקודה, המרחק לנקודה הקרובה במשטח השני" t={t} a={S.surf1[0]} b={S.surf1[1] + 0.2} size={56} top={40} />
          <HeatBar o={window01(t, S.surf2[0] + 0.5, S.assd1[1] + 0.4)} />
          <OverlapSlice t={t} sl={sl} mode="edt" o={edtO} left={170} top={160} />
          <Panel o={edtO} style={{ left: 930, top: 220, width: 830 }}>
            <div style={{ fontSize: 34, fontWeight: 700, color: C.xray, marginBottom: 14 }}>distance transform</div>
            <div style={{ fontSize: 30, lineHeight: 1.6 }}>צבע = המרחק (מ"מ) מכל תא למשטח האמיתי (לבן).<br />
              קוראים את הערך בתאי משטח החיזוי (<span style={{ color: COL.fp }}>מסגרת כתומה</span>).</div>
            <div style={{ fontFamily: "monospace", fontSize: 21, marginTop: 18, direction: "ltr", textAlign: "left", color: C.text }}>
              dt = distance_transform_edt(~surf_gt, sampling=5.17)<br />d = dt[surf_pred]</div>
          </Panel>
          <Histogram t={t} m={m} />
          <Eq t={t} a={S.hd1[0] + 0.3} b={S.assd1[1] + 0.4} size={36} style={{ left: 90, top: 690, width: 860, padding: "14px 30px" }}
            tex={`\\hl{h}{\\text{HD}=\\max d}\\quad\\ \\hl{p}{\\text{HD95}=P_{95}(d)}\\quad\\ \\hl{m}{\\text{ASSD}=\\overline{d}}`}
            terms={{ h: [S.hd1[0], S.hd1[1] + 0.3], p: [S.hd2[0], S.hd2[1] + 0.3], m: [S.assd1[0], S.assd1[1] + 0.4] }} />

          {/* Chamfer */}
          <Eq t={t} a={S.cham1[0] + 0.8} b={S.cham2[1] + 0.4} size={36} style={{ left: 60, top: 170, width: 820 }}
            tex={`\\text{CD}=\\tfrac12\\Big(\\tfrac{1}{|P|}\\sum_{p\\in P}\\min_{q\\in Q}\\|p-q\\|+\\tfrac{1}{|Q|}\\sum_{q\\in Q}\\min_{p\\in P}\\|q-p\\|\\Big)`}
            note={`כאן: ${f1(m.other.chamfer)} מ"מ (ASSD: ${f1(m.other.assd)})`} />
          <Legend x={1300} y={120} o={window01(t, S.cham1[0] + 1.0, S.cham2[1] + 0.3)} items={[["#f0e0c0", "P: האמת"], ["#6fb8ff", "Q: החיזוי"]]} />

          {/* NSD */}
          <MetricCard title="NSD" o={window01(t, S.nsd1[0] + 1.5, S.nsd1[1] + 0.5)} style={{ top: 200 }}
            rows={[[<span>τ</span>, `${tau.toFixed(1)} mm`], ["בתוך τ", tau >= m.info.nsd_tau_mm - 0.01 ? `${Math.round(m.other.nsd * 100)}%` : "…", "good"]]} />
          <Legend x={1300} y={460} o={window01(t, S.nsd1[0] + 4.0, S.nsd1[1] + 0.5)} items={[["#5aff8c", "בתוך τ"], ["#ff4f52", "מחוץ ל-τ"]]} />

          {/* disagreement */}
          <Title text="המדדים לא מסכימים" t={t} a={S.dis1[0]} b={S.dis1[0] + 4.0} size={60} top={50} />
          <MetricCard title="כנף חסרה (7.5 ס״מ)" o={window01(t, S.dis2[0], S.dis2[1] + 0.4)}
            rows={[["Dice", f2(m.wing.dice), "good"], ["HD95", `${f1(m.wing.hd95)} mm`, "bad"]]} />
          <MetricCard title="הזזה של 1 ס״מ" o={window01(t, S.dis3[0] + 4.0, S.dis3[1] + 0.5)}
            rows={[["Dice", f2(m.shift.dice), "bad"], ["HD95", `${f1(m.shift.hd95)} mm`, "good"]]} />
          <CompareTable t={t} m={m} />

          {/* clinical angle */}
          <Title text="זווית קלינית" sub="CE (Wiberg) angle, גרסה פשוטה" t={t} a={S.clin1[0]} b={S.clin1[1] + 0.2} size={56} top={36} />
          <CeAngle t={t} m={m} />
          <Panel o={window01(t, S.clin3[0], S.clin3[1] + 0.4)} style={{ left: 420, right: 420, top: 795, padding: "12px 30px", textAlign: "center", fontSize: 40 }}>
            <LTR>Dice = {m.rim.dice.toFixed(3)}</LTR>   ·   <LTR>HD95 = {f1(m.rim.hd95)} mm</LTR>   ·   <span style={{ color: C.bad }}><LTR>ΔCE = {(m.info.ce.gt - m.info.ce.rim).toFixed(1)}°</LTR></span>
          </Panel>
          <Paper t={t} />

          {/* sanity tests */}
          <Tests t={t} m={m} />
        </>
      )}

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "Dice / IoU: חפיפה של תאים (לא רואה כמה רחוק)",
        "HD = הגרוע ביותר · HD95 = אחוזון 95 · ASSD/Chamfer = ממוצע",
        "NSD: כמה מהמשטח בתוך סבולת τ",
        "המדדים לא מסכימים, ואף אחד מהם ≠ דיוק קליני",
        "בדיקות יחידה: GT מול GT, חיזוי ריק, קוביות",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/02-shape-and-metrics/notebook.ipynb" todos={[
        <span><LTR>dice(a, b)</LTR> (וגם IoU)</span>,
        <span><LTR>hd95(a, b, spacing)</LTR> בעזרת distance transform</span>,
        <span><LTR>chamfer(P, Q)</LTR> על ענני נקודות (לכתוב איזו הגדרה)</span>,
        "בדיקות יחידה: GT מול GT, חיזוי ריק, שתי הקוביות (0.75)",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v06" />
    </AbsoluteFill>
  );
};

function HeatBar({ o }: { o: number }) {
  if (o <= 0) return null;
  const stops = [0, 0.25, 0.55, 1].map((k) => { const [r, g, b] = heat(k * DMAX, DMAX); return `rgb(${r * 255},${g * 255},${b * 255}) ${k * 100}%`; });
  return (
    <div style={{ position: "absolute", left: 1500, top: 880, width: 330, opacity: o, fontFamily: FONT, direction: "ltr" }}>
      <div style={{ height: 16, borderRadius: 8, background: `linear-gradient(to right, ${stops.join(",")})` }} />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 22, color: C.dim, marginTop: 4 }}><span>0</span><span>{DMAX}+ mm</span></div>
    </div>
  );
}

function BlindSpot({ t }: { t: number }) {
  const o = window01(t, S.dice4[0] + 0.3, S.dice4[1] + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const cell = 30;
  const box = (far: boolean) => (
    <div style={{ textAlign: "center" }}>
      <svg width={cell * 12} height={cell * 7} style={{ direction: "ltr" }}>
        {Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => (
          <rect key={`${r}${c}`} x={(c + 1) * cell} y={(r + 2) * cell} width={cell - 3} height={cell - 3} fill={COL.tp} />)))}
        <rect x={(far ? 10.5 : 5) * cell} y={3 * cell} width={cell - 3} height={cell - 3} fill={COL.fp} />
        {far && <line x1={5 * cell} y1={3.5 * cell} x2={10.4 * cell} y2={3.5 * cell} stroke={COL.fp} strokeWidth={2} strokeDasharray="5 5" />}
      </svg>
      <div style={{ fontSize: 28, color: COL.fp }}>{far ? "טעות רחוקה (10 ס״מ)" : "טעות צמודה (1 מ״מ)"}</div>
      <div style={{ fontSize: 26, color: C.dim }}>תא FP אחד</div>
    </div>
  );
  return (
    <Panel o={o} style={{ right: 90, top: 420, width: 860, padding: "20px 30px" }}>
      <div style={{ display: "flex", gap: 40, justifyContent: "center", direction: "ltr" }}>{box(false)}{box(true)}</div>
      <div style={{ fontSize: 32, textAlign: "center", marginTop: 10, color: C.warm }}>Dice זהה בשני המקרים</div>
    </Panel>
  );
}

function CompareTable({ t, m }: { t: number; m: Metrics }) {
  const o = window01(t, S.dis4[0] - 0.3, S.dis4[1] + 0.6, 0.6, 0.6);
  if (o <= 0) return null;
  const cols: ["other" | "wing" | "shift", string][] = [["other", "אדם אחר"], ["wing", "כנף חסרה"], ["shift", "הזזה 1 ס״מ"]];
  const rows: [string, (x: M) => string, string][] = [
    ["Dice ↑", (x) => f2(x.dice), ""], ["IoU ↑", (x) => f2(x.iou), ""], ["HD (mm) ↓", (x) => f1(x.hd), ""],
    ["HD95 (mm) ↓", (x) => f1(x.hd95), ""], ["ASSD (mm) ↓", (x) => f1(x.assd), ""], ["Chamfer (mm) ↓", (x) => f1(x.chamfer), ""],
    [`NSD@${m.info.nsd_tau_mm.toFixed(1)}mm ↑`, (x) => `${Math.round(x.nsd * 100)}%`, ""]];
  const hl = (k: string, r: number) => (k === "wing" && r === 0) || (k === "shift" && r === 3) ? C.good : (k === "wing" && r === 3) || (k === "shift" && r === 0) ? C.bad : C.text;
  return (
    <Panel o={o} style={{ left: 330, right: 330, top: 150, padding: "26px 40px" }}>
      <table style={{ width: "100%", fontSize: 34, borderCollapse: "collapse", textAlign: "center" }}>
        <thead><tr><th style={{ textAlign: "right", color: C.dim, fontWeight: 500 }}>מדד</th>
          {cols.map(([k, n]) => <th key={k} style={{ color: C.xray, padding: "6px 10px" }}>{n}</th>)}</tr></thead>
        <tbody>{rows.map(([n, fn], r) => (
          <tr key={n} style={{ borderTop: `1px solid rgba(120,190,255,0.2)` }}>
            <td style={{ textAlign: "right", direction: "ltr", padding: "6px 0", color: C.dim }}>{n}</td>
            {cols.map(([k]) => <td key={k} style={{ color: hl(k, r), direction: "ltr" }}>{fn(m[k])}</td>)}
          </tr>))}</tbody>
      </table>
    </Panel>
  );
}

function Paper({ t }: { t: number }) {
  const o = window01(t, S.clin4[0] + 0.2, S.clin4[1] + 0.5, 0.7, 0.7);
  if (o <= 0) return null;
  return (
    <Panel o={o} style={{ left: 280, right: 280, top: 230, padding: "34px 44px" }}>
      <div style={{ fontSize: 28, color: C.dim, direction: "ltr", textAlign: "left" }}>Shakya &amp; Khanal · NeurIPS 2023 (Datasets &amp; Benchmarks)</div>
      <div style={{ fontSize: 34, color: "white", direction: "ltr", textAlign: "left", marginTop: 8, fontWeight: 500 }}>
        Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Shape Reconstruction</div>
      <div style={{ fontSize: 36, color: C.warm, marginTop: 26, lineHeight: 1.5, direction: "ltr", textAlign: "left" }}>
        “…the dice score improvement does not always bring a corresponding improvement…”</div>
      <div style={{ color: C.dim, fontSize: 28, marginTop: 4 }}>(בהערכה אוטומטית של פרמטרים קליניים)</div>
      <div style={{ fontSize: 32, marginTop: 26, lineHeight: 1.5, opacity: ramp(t, S.clin4[1] - 6, S.clin4[1] - 5) }}>
        בפרויקט: זווית קלינית אחת מה-mesh, ופיזור של Dice מול הזווית.</div>
    </Panel>
  );
}

function Tests({ t, m }: { t: number; m: Metrics }) {
  const o = window01(t, S.test1[0] + 0.2, S.test3[1] + 0.5, 0.7, 0.7);
  if (o <= 0) return null;
  const line = (a0: number, code: string, res: string, ok = true) => {
    const k = ramp(t, a0, a0 + 0.5), r = ramp(t, a0 + 1.2, a0 + 1.6);
    return <div style={{ opacity: k, display: "flex", justifyContent: "space-between", gap: 20 }}>
      <span>{code}</span><span style={{ opacity: r, color: ok ? C.good : C.bad }}>{res}</span></div>;
  };
  const g = m.gt;
  const cubesO = ramp(t, S.test3[0] + 0.5, S.test3[0] + 1.2);
  return (
    <>
      <Panel o={o} style={{ left: 200, right: 200, top: 150, padding: "30px 44px", direction: "ltr" }}>
        <div style={{ fontFamily: "monospace", fontSize: 30, lineHeight: 1.75, color: C.text, textAlign: "left" }}>
          <div style={{ color: C.dim }}># test_metrics.py</div>
          {line(S.test1[0] + 2.5, "assert dice(gt, gt) == 1.0", `✓ ${g.dice.toFixed(1)}`)}
          {line(S.test1[0] + 3.5, "assert hd95(gt, gt, spacing) == 0.0", `✓ ${g.hd95!.toFixed(1)}`)}
          {line(S.test1[0] + 4.5, "assert chamfer(P, P) == 0.0", `✓ ${g.chamfer!.toFixed(1)}`)}
          {line(S.test2[0] + 1.5, "assert dice(empty, gt) == 0.0", `✓ ${m.empty.dice.toFixed(1)}`)}
          {line(S.test2[0] + 5.0, "assert hd95(empty, gt, spacing) == inf   # our convention", "✓ undefined → inf")}
          {line(S.test3[0] + 6.0, "assert dice(cube, shift(cube, 1)) == 0.75", `✓ ${m.cubes.dice.toFixed(2)}`)}
        </div>
      </Panel>
      <div style={{ position: "absolute", left: 420, top: 600, opacity: o * cubesO }}>
        <svg width={1100} height={190} style={{ direction: "ltr" }}>
          {[0, 1].map((s) => Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => (
            <rect key={`${s}${r}${c}`} x={20 + (c + s) * 40 + s * 4} y={20 + r * 40 + s * 4} width={36} height={36}
              fill={s === 0 ? "rgba(207,226,245,0.45)" : "rgba(255,159,64,0.35)"} stroke={s === 0 ? COL.tp : COL.fp} strokeWidth={2} />))))}
          <text x={280} y={80} fill="white" fontSize={32} fontFamily={FONT}>A = 4×4×4 = 64,  B = A shifted by 1 cell</text>
          <text x={280} y={130} fill={C.warm} fontSize={32} fontFamily={FONT}>|A∩B| = 3×4×4 = 48  →  Dice = 2·48 / (64+64) = 0.75</text>
        </svg>
      </div>
    </>
  );
}
