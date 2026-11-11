// Video 7 · From two views back to 3D: back-projection, silhouettes and the visual hull (computed for real
// on the hero pelvis in prep/v07_assets.py), what it gets wrong, more views, CT, and the 2-view ambiguity.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v07/durations.json";
import metrics from "../../public/data/v07/metrics.json";
import { buildTimeline, Script } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, Rays } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { GridVox, gridPos, Grid, PlaneStack, useMaskTexture, ViewPlane } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const seg = (id: string) => S[id];

type Vox = Grid & { truth: number[]; extra: number[]; alt: number[]; extra4: number[]; extra180: number[]; ap: number[]; lat: number[] };
type Assets = { vox: Vox; pelvis: THREE.BufferGeometry; drrAp: THREE.Texture; drrLat: THREE.Texture };

const COL = { hull: "#9fd6ff", truth: C.bone, extra: "#ff6e78", alt: C.warm, ok: "#7dffb0" };
const DET = 2.05;        // detector distance from the centre (scene units)
const K = metrics.slices;

// ---------------- 3D ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const g = a.vox;
  const silAp = useMaskTexture(g.ap, g.n, [127, 208, 255]);
  const silLat = useMaskTexture(g.lat, g.n, [255, 207, 127]);
  const hullIdx = useMemo(() => [...g.truth, ...g.extra], [g]);

  // two test voxels for sil2: one inside both silhouettes, one whose AP ray misses
  const test = useMemo(() => {
    const n = g.n;
    let best: number[] = [0, 0, 0], bd = 1e9;
    for (let i = 0; i < g.truth.length; i += 3) {
      const d = Math.abs(g.truth[i] - 40) + Math.abs(g.truth[i + 1] - 34) + Math.abs(g.truth[i + 2] - 40);
      if (d < bd) { bd = d; best = [g.truth[i], g.truth[i + 1], g.truth[i + 2]]; }
    }
    let miss: number[] = [32, 32, 24]; bd = 1e9;
    for (let r = 0; r < n; r++) for (let s = 0; s < n; s++) {
      if (g.ap[r * n + s]) continue;
      for (let aa = 0; aa < n; aa++) if (g.lat[aa * n + s]) {
        const d = Math.abs(r - 24) + Math.abs(aa - 30) + Math.abs(s - 28);
        if (d < bd) { bd = d; miss = [r, aa, s]; }
      }
    }
    return { hit: gridPos(g, best[0], best[1], best[2]), miss: gridPos(g, miss[0], miss[1], miss[2]) };
  }, [g]);

  // --- visibility ---
  const end = (id: string) => seg(id)[1];
  const st = (id: string) => seg(id)[0];
  const pelvisO = ramp(t, 3.0, 4.2) * (1 - 0.75 * ramp(t, st("bp1") + 0.5, st("bp2"))) * (1 - ramp(t, end("bp4"), end("bp4") + 0.6));
  const detO = window01(t, st("intro1") + 0.5, end("bp4") + 0.6, 0.8, 0.6);
  const apSm = ramp(t, st("bp2") + 0.3, end("bp2") - 0.5);
  const latSm = ramp(t, st("bp3") + 0.2, end("bp3"));
  const smO = 1 - ramp(t, end("bp4"), end("bp4") + 0.6);
  const sil2O = window01(t, st("sil2"), end("eq1") + 0.5, 0.6, 0.6);
  const sil3 = [st("sil3"), end("sil3")];
  const apEx = ramp(t, sil3[0] + 0.3, sil3[0] + (sil3[1] - sil3[0]) * 0.35);
  const latEx = ramp(t, sil3[0] + (sil3[1] - sil3[0]) * 0.35, sil3[0] + (sil3[1] - sil3[0]) * 0.7);
  const exO = window01(t, sil3[0], sil3[1] + 0.5, 0.5, 0.8);
  const hullIn = ramp(t, sil3[0] + (sil3[1] - sil3[0]) * 0.72, sil3[1]);
  const side = window01(t, st("sil4") - 0.2, end("sil4") + 0.4, 0.8, 0.6);       // side-by-side layout
  const hullPhase = window01(t, sil3[0] + (sil3[1] - sil3[0]) * 0.72, end("ghost1") + 0.4, 0.4, 0.6);
  const red = ramp(t, st("wrong2") + 0.3, st("wrong2") + 1.6) * (1 - ramp(t, st("wrong6") - 0.3, st("wrong6") + 0.4))
    + ramp(t, st("ghost1") - 0.3, st("ghost1") + 0.4);
  // cut level for the slice views
  let cut = 999;
  if (t > st("wrong3") - 0.2 && t < st("wrong5") - 0.2) cut = K.wings.k;
  else if (t >= st("wrong5") - 0.2 && t < st("wrong6") - 0.2) cut = K.acetab.k;
  else if (t >= st("ghost1") - 0.3) cut = K.pubis.k;
  const mvO = window01(t, st("mv1"), end("mv3") + 0.5, 0.7, 0.7);
  const mvN = t < st("mv2") + 2.8 ? 2 : t < st("mv2") + 6.0 ? 4 : 180;
  const ambO = window01(t, st("amb1"), end("amb2") + 0.4, 0.8, 0.7);
  const resO = window01(t, st("res1"), tl.total, 0.8, 0.6);
  const spin = (t - st("amb1")) * 0.22;

  // --- camera ---
  const keys: CamKey[] = [
    { t: 0, pos: [8.3, 3.8, -9.1], look: [0, 0.35, 0] },
    { t: end("intro2"), pos: [9.1, 3.1, -7.2], look: [0, 0.35, 0] },
    { t: st("bp2") + 0.5, pos: [7.9, 4.3, -8.3], look: [0, 0.35, 0] },
    { t: end("bp3"), pos: [8.6, 5.3, -6.7], look: [0, 0.35, 0] },
    { t: end("bp4"), pos: [6.0, 7.4, -7.9], look: [0, 0.35, 0] },
    { t: st("sil2") + 0.6, pos: [7.6, 3.9, -7.2], look: [0, 0.3, 0.2] },
    { t: end("eq1"), pos: [7.2, 3.3, -7.8], look: [0, 0.9, 0.2] },
    { t: st("sil3") + 1.0, pos: [6.8, 3.8, -6.8], look: [0, 0, 0] },
    { t: end("sil3"), pos: [5.2, 3.0, -8.2], look: [0, 0, 0] },
    { t: st("sil4") + 0.8, pos: [0, 0.8, -9.0], look: [0, -0.1, 0] },
    { t: end("sil4"), pos: [0.3, 1.0, -8.8], look: [0, -0.1, 0] },
    { t: st("wrong1") + 0.8, pos: [3.6, 2.6, -7.8], look: [0, 0, 0] },
    { t: end("wrong2"), pos: [2.8, 2.4, -8.2], look: [0, 0, 0] },
    // top views of the cut: object on the left of the screen, panel on the right
    { t: st("wrong3") + 1.0, pos: [-1.0, 7.4, -4.6], look: [-1.9, -0.6, 0.1] },
    { t: end("wrong5"), pos: [-1.0, 7.0, -4.8], look: [-1.9, -0.8, 0.1] },
    { t: st("wrong6") + 0.9, pos: [0, 0.1, -8.6], look: [0, -0.1, 0] },
    { t: end("wrong6"), pos: [0, 0.1, -8.3], look: [0, -0.1, 0] },
    { t: st("ghost1") + 0.9, pos: [-1.0, 7.4, -4.4], look: [-1.9, -1.2, 0.1] },
    { t: end("ghost1") + 0.5, pos: [-1.0, 7.4, -4.4], look: [-1.9, -1.2, 0.1] },
    { t: st("mv1") + 0.5, pos: [-0.6, 7.2, -4.8], look: [-1.9, -0.6, 0.1] },
    { t: end("mv3"), pos: [-0.8, 7.0, -5.0], look: [-1.9, -0.6, 0.1] },
    { t: st("amb1") + 0.6, pos: [0, 1.0, -10.8], look: [0, 0, 0] },
    { t: end("amb2"), pos: [0, 1.2, -10.4], look: [0, 0, 0] },
    { t: st("res1") + 1.0, pos: [0, 1.4, -10.0], look: [0, 0, 0] },
    { t: tl.total, pos: [0, 1.6, -10.4], look: [0, 0, 0] },
  ];

  const hullX = 2.1 * side, truthX = -2.1 * side;
  const sc = 1 - 0.25 * side;

  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.2} opacity={0.7 * (1 - mvO)} />

      {/* intro + back-projection */}
      <Bone geo={a.pelvis} material={bone} opacity={pelvisO} />
      <ViewPlane tex={a.drrAp} axis="ap" size={g.cube} at={DET} opacity={0.8 * detO} />
      <ViewPlane tex={a.drrLat} axis="lat" size={g.cube} at={-DET} opacity={0.8 * detO} />
      <PlaneStack tex={a.drrAp} axis="ap" size={g.cube} progress={apSm} opacity={0.032 * smO} count={28} color="#9fd8ff" />
      <PlaneStack tex={a.drrLat} axis="lat" size={g.cube} progress={latSm} opacity={0.032 * smO} count={28} color="#ffd59a" />

      {/* sil2: silhouettes on the detectors, two test voxels and their rays */}
      {silAp && silLat && sil2O > 0 && (
        <group>
          <ViewPlane tex={silAp} axis="ap" size={g.cube} at={DET} opacity={sil2O * 0.9} />
          <ViewPlane tex={silLat} axis="lat" size={g.cube} at={-DET} opacity={sil2O * 0.9} />
          {[test.hit, test.miss].map((p, i) => {
            const o = sil2O * ramp(t, st("sil2") + 1 + i * 3.2, st("sil2") + 1.6 + i * 3.2);
            const col = i === 0 ? COL.ok : COL.extra;
            return (
              <group key={i}>
                <Rays segs={[[[p[0], p[1], -DET - 0.4], [p[0], p[1], DET]]]} opacity={o} color={[1.4, 2.1, 3.0]} />
                <Rays segs={[[[DET + 0.4, p[1], p[2]], [-DET, p[1], p[2]]]]} opacity={o} color={[3.0, 2.2, 1.2]} />
                <mesh position={p}><boxGeometry args={[g.cell * 2.2, g.cell * 2.2, g.cell * 2.2]} />
                  <meshBasicMaterial color={new THREE.Color(col).multiplyScalar(2.2)} transparent opacity={o} toneMapped={false} /></mesh>
                <mesh position={[p[0], p[1], DET - 0.01]}><circleGeometry args={[0.07, 24]} />
                  <meshBasicMaterial color={new THREE.Color(col).multiplyScalar(2.5)} transparent opacity={o} toneMapped={false} side={THREE.DoubleSide} /></mesh>
                <mesh position={[-DET + 0.01, p[1], p[2]]} rotation={[0, Math.PI / 2, 0]}><circleGeometry args={[0.07, 24]} />
                  <meshBasicMaterial color={new THREE.Color(col).multiplyScalar(2.5)} transparent opacity={o} toneMapped={false} side={THREE.DoubleSide} /></mesh>
              </group>
            );
          })}
        </group>
      )}

      {/* sil3: the two extrusions */}
      {silAp && silLat && (
        <>
          <PlaneStack tex={silAp} axis="ap" size={g.cube} progress={apEx} opacity={0.06 * exO} count={30} additive={false} />
          <PlaneStack tex={silLat} axis="lat" size={g.cube} progress={latEx} opacity={0.06 * exO} count={30} additive={false} />
        </>
      )}

      {/* the hull: truth + extra cells (same colour at first, then the extras turn red) */}
      {hullPhase > 0 && (
        <group>
          {red < 0.02 ? (
            <GridVox g={g} idx={hullIdx} color={COL.hull} opacity={hullPhase * hullIn} cutS={cut} position={[hullX, 0, 0]} scale={sc} grow={hullIn} />
          ) : (
            <>
              <GridVox g={g} idx={g.truth} color={COL.truth} opacity={hullPhase} cutS={cut} emissive={0.05} />
              <GridVox g={g} idx={g.extra} color={COL.extra} opacity={hullPhase * (cut < 999 ? 1 : 0.22 + 0.78 * (1 - red))}
                cutS={cut} depthWrite={cut < 999} emissive={0.2} />
            </>
          )}
          <GridVox g={g} idx={g.truth} color={COL.truth} opacity={side} position={[truthX, 0, 0]} scale={sc} emissive={0.05} />
        </group>
      )}

      {/* more views: truth + the extras of an n-view hull, cut at wing level */}
      {mvO > 0 && (
        <group>
          <GridVox g={g} idx={g.truth} color={COL.truth} opacity={mvO} cutS={K.wings.k} emissive={0.05} />
          <GridVox g={g} idx={mvN === 2 ? g.extra : mvN === 4 ? g.extra4 : g.extra180} color={COL.extra} opacity={mvO}
            cutS={K.wings.k} emissive={0.2} />
        </group>
      )}

      {/* ambiguity: another shape with the same two silhouettes */}
      {ambO > 0 && (
        <>
          <GridVox g={g} idx={g.alt} color={COL.alt} opacity={ambO * ramp(t, st("amb1") + 3.0, st("amb1") + 4.2)} position={[2.1, 0, 0]} scale={0.75} rotationY={spin} emissive={0.25} gap={0.0} />
          <GridVox g={g} idx={g.truth} color={COL.truth} opacity={ambO} position={[-2.1, 0, 0]} scale={0.75} rotationY={spin} emissive={0.05} />
        </>
      )}

      {/* results: hull (translucent) around the truth */}
      {resO > 0 && (
        <group position={[-2.5, 0, 0]} rotation={[0, (t - st("res1")) * 0.12, 0]}>
          <GridVox g={g} idx={g.truth} color={COL.truth} opacity={resO} emissive={0.05} />
          <GridVox g={g} idx={g.extra} color={COL.extra} opacity={resO * 0.12} depthWrite={false} emissive={0.2} />
        </group>
      )}
    </>
  );
}

// ---------------- 2D pieces ----------------
const tile = (src: string, label: string, o: number, w = 330, color = C.dim) => (
  <div style={{ opacity: o, transform: `translateY(${(1 - o) * 16}px)`, textAlign: "center" }}>
    <Img src={staticFile(src)} style={{ width: w, height: w, borderRadius: 10, border: `1.5px solid ${C.line}`, display: "block", imageRendering: "pixelated" }} />
    <div style={{ marginTop: 12, fontSize: 28, color, direction: "rtl" }}>{label}</div>
  </div>
);

function BPPanel({ t }: { t: number }) {
  const a = S.bp5[0], b = S.bp6[1] + 0.5;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const k = (x: number) => ramp(t, x, x + 0.6);
  return (
    <div style={{ position: "absolute", top: 240, width: "100%", display: "flex", justifyContent: "center", gap: 40, opacity: o, fontFamily: FONT, direction: "ltr" }}>
      {tile("data/v07/ct_slice_bone.png", "החתך האמיתי (עצם)", k(a + 0.3))}
      {tile("data/v07/bp_ap.png", "מריחה מלפנים ↕", k(a + 5.0), 330, C.xray)}
      {tile("data/v07/bp_lat.png", "מריחה מהצד ↔", k(S.bp6[0] + 0.3), 330, C.warm)}
      {tile("data/v07/bp_sum.png", "הסכום", k(S.bp6[0] + 3.2), 330, "white")}
      <div style={{ position: "absolute", top: -50, width: "100%", textAlign: "center", fontSize: 24, color: C.dim, direction: "rtl" }}>
        מבט מלמעלה · חזית האגן למטה · בגובה כנפי האגן
      </div>
    </div>
  );
}

function SilPanel({ t }: { t: number }) {
  const a = S.sil1[0], b = S.sil1[1] + 0.5;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const k = (x: number) => ramp(t, x, x + 0.6);
  const arrow = (x: number) => <div style={{ fontSize: 54, color: C.dim, alignSelf: "center", marginBottom: 50, opacity: k(x) }}>→</div>;
  const mid = a + (b - a) * 0.45;
  return (
    <div style={{ position: "absolute", top: 250, width: "100%", display: "flex", justifyContent: "center", gap: 28, opacity: o, fontFamily: FONT, direction: "ltr" }}>
      {tile("data/v07/drr_ap.png", "DRR מלפנים", k(a + 0.2), 320)}{arrow(mid)}{tile("data/v07/sil_ap.png", "צללית AP", k(mid), 320, C.xray)}
      <div style={{ width: 60 }} />
      {tile("data/v07/drr_lat.png", "DRR מהצד", k(a + 0.5), 320)}{arrow(mid + 0.4)}{tile("data/v07/sil_lat.png", "צללית LAT", k(mid + 0.4), 320, C.warm)}
    </div>
  );
}

/** axial cell images, seen from above; optional AP-row bar (top) and LAT-row bar (right) on the last one */
function AxialPanel({ t, a, b, imgs, labels, bars, title }: {
  t: number; a: number; b: number; imgs: string[]; labels: string[]; bars?: { ap: number[]; lat: number[]; at: number }; title: string;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const W = 330, n = 64, c = W / n;
  const barO = bars ? ramp(t, bars.at, bars.at + 0.8) : 0;
  return (
    <Panel o={o} style={{ right: 70, top: 130, width: imgs.length * (W + 40) + 30, padding: "26px 30px" }}>
      <div style={{ fontSize: 32, fontWeight: 700, color: C.xray, marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 22, color: C.dim, marginBottom: 30 }}>מבט מלמעלה · חזית האגן למטה</div>
      <div style={{ display: "flex", gap: 40, direction: "ltr", justifyContent: "center" }}>
        {imgs.map((src, i) => (
          <div key={src} style={{ position: "relative", textAlign: "center" }}>
            <Img src={staticFile(src)} style={{ width: W, height: W, imageRendering: "pixelated", borderRadius: 6, display: "block" }} />
            <div style={{ fontSize: 26, marginTop: 10, color: i === imgs.length - 1 ? COL.extra : C.bone }}>{labels[i]}</div>
            {bars && i === imgs.length - 1 && barO > 0 && (
              <>
                {bars.ap.map((v, j) => v ? <div key={`a${j}`} style={{ position: "absolute", top: -18, left: (n - 1 - j) * c, width: c, height: 10, background: C.xray, opacity: barO }} /> : null)}
                {bars.lat.map((v, j) => v ? <div key={`l${j}`} style={{ position: "absolute", top: j * c, left: W + 8, width: 10, height: c, background: C.warm, opacity: barO }} /> : null)}
              </>
            )}
          </div>
        ))}
      </div>
    </Panel>
  );
}

/** two true points, their AP (vertical) and LAT (horizontal) rays, four intersections */
function GhostDiagram({ t }: { t: number }) {
  const a = S.ghost2[0], b = S.ghost3[1] + 0.5;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const P = [[260, 420], [470, 230]];
  const raysO = ramp(t, a + 1.5, a + 3.0);
  const ghostO = ramp(t, a + 5.5, a + 6.5) * (1 - 0.85 * ramp(t, S.ghost3[0] + 4.0, S.ghost3[0] + 5.0));
  const pairO = ramp(t, S.ghost3[0] + 0.5, S.ghost3[0] + 1.5);
  const pulse = 0.6 + 0.4 * Math.sin(t * 5);
  const cols = [C.xray, C.warm];
  return (
    <div style={{ position: "absolute", left: 610, top: 150, opacity: o, fontFamily: FONT }}>
      <svg width={700} height={660} viewBox="0 0 700 660">
        <rect x={120} y={110} width={480} height={460} fill="rgba(20,40,70,0.35)" stroke={C.line} strokeWidth={1.5} rx={8} />
        <rect x={120} y={60} width={480} height={14} fill="rgba(127,208,255,0.25)" stroke={C.xray} />
        <rect x={640} y={110} width={14} height={460} fill="rgba(255,207,127,0.25)" stroke={C.warm} />
        <text x={360} y={44} fill={C.xray} fontSize={24} textAnchor="middle">AP צילום</text>
        <text x={672} y={95} fill={C.warm} fontSize={24} textAnchor="middle">LAT</text>
        {P.map(([x, y], i) => (
          <g key={i} opacity={raysO}>
            <line x1={x} y1={600} x2={x} y2={74} stroke={C.xray} strokeWidth={2} strokeDasharray="6 6" />
            <line x1={90} y1={y} x2={640} y2={y} stroke={C.warm} strokeWidth={2} strokeDasharray="6 6" />
            <circle cx={x} cy={67} r={10} fill={pairO > 0.5 ? cols[i] : "white"} />
            <circle cx={647} cy={y} r={10} fill={pairO > 0.5 ? cols[i] : "white"} />
          </g>
        ))}
        {[[P[0][0], P[1][1]], [P[1][0], P[0][1]]].map(([x, y], i) => (
          <g key={`g${i}`} opacity={ghostO}>
            <circle cx={x} cy={y} r={22} fill="none" stroke={COL.extra} strokeWidth={3} opacity={pulse} />
            <text x={x} y={y + 8} fill={COL.extra} fontSize={24} textAnchor="middle">?</text>
          </g>
        ))}
        {P.map(([x, y], i) => (
          <circle key={`p${i}`} cx={x} cy={y} r={18} fill={pairO > 0.5 ? cols[i] : C.bone} stroke="white" strokeWidth={2} />
        ))}
        <text x={360} y={630} fill={C.dim} fontSize={22} textAnchor="middle">מבט מלמעלה על חתך אחד</text>
      </svg>
      <div style={{ position: "absolute", left: 0, top: 250, width: 110, textAlign: "center", fontSize: 26, color: COL.extra, opacity: ghostO, direction: "rtl" }}>
        רוחות<br />רפאים
      </div>
    </div>
  );
}

function ViewsChart({ t }: { t: number }) {
  const a = S.mv1[0] + 0.6, b = S.mv3[1] + 0.5;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const mv = metrics.multiview;
  const W = 640, H = 400, x0 = 70, y0 = 30;
  const X = (n: number) => x0 + (Math.log(n) / Math.log(180)) * (W - x0 - 30);
  const Y = (d: number) => y0 + (1 - d / 0.7) * (H - y0 - 50);
  const draw = ramp(t, a + 0.8, S.mv2[1] - 0.5);
  const shown = mv.filter((_, i) => i / (mv.length - 1) <= draw + 1e-6);
  const asym = ramp(t, S.mv3[0] + 0.3, S.mv3[0] + 1.3);
  const lim = mv[mv.length - 1].dice;
  const hl = (n: number, at: number) => {
    const p = mv.find((m) => m.views === n)!;
    const k = ramp(t, at, at + 0.5);
    return <g opacity={k}><circle cx={X(n)} cy={Y(p.dice)} r={11} fill="none" stroke={C.warm} strokeWidth={3} />
      <text x={X(n)} y={Y(p.dice) - 22} fill={C.warm} fontSize={24} textAnchor="middle">{p.dice.toFixed(2)}</text></g>;
  };
  return (
    <Panel o={o} style={{ right: 70, top: 150, width: W + 20, padding: "24px 26px" }}>
      <div style={{ fontSize: 30, fontWeight: 700, color: C.xray, marginBottom: 8 }}>Dice מול מספר התצפיות (האגן שלנו)</div>
      <svg width={W} height={H} style={{ direction: "ltr" }}>
        <line x1={x0} y1={Y(0)} x2={W - 20} y2={Y(0)} stroke={C.dim} />
        <line x1={x0} y1={Y(0)} x2={x0} y2={Y(0.7)} stroke={C.dim} />
        {[0, 0.2, 0.4, 0.6].map((d) => <text key={d} x={x0 - 10} y={Y(d) + 8} fill={C.dim} fontSize={20} textAnchor="end">{d.toFixed(1)}</text>)}
        {[1, 2, 4, 8, 18, 36, 90, 180].map((n) => <text key={n} x={X(n)} y={Y(0) + 30} fill={C.dim} fontSize={20} textAnchor="middle">{n}</text>)}
        <line x1={x0} y1={Y(lim)} x2={W - 20} y2={Y(lim)} stroke={C.warm} strokeDasharray="8 6" opacity={asym} />
        <text x={x0 + 10} y={Y(lim) - 14} fill={C.warm} fontSize={20} opacity={asym}>convex hull of each slice</text>
        <polyline points={shown.map((m) => `${X(m.views)},${Y(m.dice)}`).join(" ")} fill="none" stroke={C.xray} strokeWidth={3} />
        {shown.map((m) => <circle key={m.views} cx={X(m.views)} cy={Y(m.dice)} r={6} fill={C.xray} />)}
        {hl(2, S.mv2[0] + 0.4)}{hl(4, S.mv2[0] + 3.0)}{hl(180, S.mv2[0] + 6.2)}
      </svg>
      <div style={{ fontSize: 22, color: C.dim, textAlign: "center" }}>מספר תצפיות סביב הגוף (סקאלה לוגריתמית)</div>
    </Panel>
  );
}

function CTPanel({ t }: { t: number }) {
  const a = S.ct1[0] + 0.3, b = S.ct2[1] + 0.5;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const k = (x: number) => ramp(t, x, x + 0.6);
  const c = S.ct2[0];
  const step = (S.ct2[1] - c - 2) / 4;
  return (
    <div style={{ position: "absolute", top: 300, width: "100%", display: "flex", justifyContent: "center", gap: 26, opacity: o, fontFamily: FONT, direction: "ltr" }}>
      {tile("data/v07/ct_slice.png", "החתך המקורי", k(a), 300, "white")}
      <div style={{ width: 20 }} />
      {[2, 8, 32, 180].map((n, i) => tile(`data/v07/fbp_${n}.png`, `${n} הטלות`, k(c + 0.3 + i * step), 300, n === 180 ? C.good : C.dim))}
    </div>
  );
}

function SameShadows({ t }: { t: number }) {
  const a = S.amb1[0] + 4.0, b = S.amb2[1] + 0.4;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", top: 760, width: "100%", display: "flex", justifyContent: "center", gap: 22, opacity: o, fontFamily: FONT, direction: "ltr" }}>
      {tile("data/v07/sil_ap.png", "AP", 1, 140, C.xray)}{tile("data/v07/sil_lat.png", "LAT", 1, 140, C.warm)}
      <div style={{ position: "absolute", top: -48, fontSize: 28, color: "white", direction: "rtl" }}>אותן שתי צלליות בדיוק, לשתי הצורות</div>
    </div>
  );
}

function ResultsPanel({ t }: { t: number }) {
  const a = S.res1[0], b = S.res4[1] + 0.5;
  const o = window01(t, a, b, 0.7, 0.6);
  if (o <= 0) return null;
  const all = metrics.all27;
  const row = (cells: string[], at: number, color = C.text) => (
    <div style={{ display: "flex", gap: 20, fontSize: 32, padding: "14px 0", borderBottom: `1px solid ${C.line}`, opacity: ramp(t, at, at + 0.5), color }}>
      <div style={{ width: 380 }}>{cells[0]}</div>
      <div style={{ width: 200, direction: "ltr", textAlign: "center" }}>{cells[1]}</div>
      <div style={{ width: 240, direction: "ltr", textAlign: "center" }}>{cells[2]}</div>
    </div>
  );
  return (
    <Panel o={o} style={{ left: 120, top: 190, width: 880 }}>
      <div style={{ fontSize: 40, fontWeight: 700, color: C.xray, marginBottom: 16 }}>visual hull משני צילומים</div>
      {row(["", "Dice", "HD95"], a, C.dim)}
      {row(["האגן שלנו (s1390)", metrics.hull_dice.toFixed(3), `${metrics.hull_hd95_mm.toFixed(1)} mm`], a + 1.0)}
      {row([`${all.n} נבדקים, ממוצע ± ס"ת`, `${all.dice_mean.toFixed(2)} ± ${all.dice_std.toFixed(2)}`, `${all.hd95_mean.toFixed(1)} ± ${all.hd95_std.toFixed(1)} mm`], S.res2[0] + 0.5)}
      <div style={{ fontSize: 26, color: C.dim, marginTop: 18, opacity: ramp(t, S.res2[0] + 6, S.res2[0] + 6.6) }}>
        רשת 64³ (תא של 5.5 מ"מ) · וקסלים אמיתיים שהוחמצו: {metrics.missed_vox} · הדגמה קטנה, לא הניסוי הסופי
      </div>
      <div style={{ fontSize: 28, color: C.warm, marginTop: 16, opacity: ramp(t, S.res3[0] + 0.3, S.res3[0] + 0.9) }}>
        ⚠ הצלליות נלקחו מהמסכה האמיתית: תוצאה אופטימית
      </div>
    </Panel>
  );
}

export const V07 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const vox = useJson<Vox>("data/v07/vox.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const drrAp = useTex("data/v07/drr_ap_scene.png");
  const drrLat = useTex("data/v07/drr_lat_scene.png");
  const ready = vox && pelvis && drrAp && drrLat;

  const dim = Math.max(
    window01(t, S.bp5[0] - 0.3, S.sil1[1] + 0.5, 0.6, 0.6),
    window01(t, S.ghost2[0] - 0.3, S.ghost3[1] + 0.5, 0.6, 0.6),
    window01(t, S.ct1[0], S.ct2[1] + 0.5, 0.6, 0.6),
    window01(t, S.recap[0] - 0.3, tl.total, 0.8, 0.1),
  );
  const m = metrics;

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ vox, pelvis, drrAp, drrLat }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.82 * dim, pointerEvents: "none" }} />

      <Opening n={7} title="משתי תצפיות חזרה לתלת-ממד" active={[1, 3]} t={t} end={2.9} />

      <Title text="הטלה לאחור" sub="back-projection: למרוח כל צילום חזרה לאורך הקרניים" t={t} a={S.bp1[0]} b={S.bp4[1] + 0.3} />
      <Label x={560} y={250} text="AP: מלפנים" color={C.xray} o={window01(t, S.intro1[0] + 1, S.bp4[1], 0.6, 0.6)} />
      <Label x={1400} y={250} text="LAT: מהצד" color={C.warm} o={window01(t, S.intro1[0] + 1, S.bp4[1], 0.6, 0.6)} />
      <Title text="הטלה לאחור, בחתך אחד" t={t} a={S.bp5[0]} b={S.bp6[1] + 0.3} top={60} size={56} />
      <BPPanel t={t} />
      <Title text="צלליות" sub="silhouette: האם הקרן פגשה את האגן? כן / לא" t={t} a={S.sil1[0]} b={S.sil1[1] + 0.3} top={60} size={60} />
      <SilPanel t={t} />
      <Label x={960} y={130} text="visual hull: נשאר רק מה שבתוך שתי הצלליות" size={40} o={window01(t, S.sil2[0] + 0.3, S.sil2[1] + 0.2, 0.6, 0.4)} />
      <Label x={1420} y={760} text="✓ בתוך שתיהן: נשאר" color={COL.ok} o={window01(t, S.sil2[0] + 1.5, S.eq1[1], 0.5, 0.5)} />
      <Label x={1420} y={810} text="✗ קרן AP מחוץ לצללית: נחצב" color={COL.extra} o={window01(t, S.sil2[0] + 4.7, S.eq1[1], 0.5, 0.5)} />
      <Eq t={t} a={S.eq1[0]} b={S.eq1[1] + 0.4} size={56} style={{ left: 470, top: 60, width: 980 }}
        tex={"\\hl{h}{H(r,a,s)} = \\hl{ap}{S_{AP}(r,s)} \\cdot \\hl{lat}{S_{LAT}(a,s)}"}
        note="r = ימין · a = קדימה · s = למעלה"
        terms={{ h: [S.eq1[0] + 0.5, S.eq1[0] + 4.5], ap: [S.eq1[0] + 4.5, S.eq1[0] + 8.5], lat: [S.eq1[0] + 8.5, S.eq1[1]] }} />
      <Label x={960} y={140} text="AP נמשכת לעומק · LAT נמשכת לרוחב · שומרים את החיתוך" size={36}
        o={window01(t, S.sil3[0] + 0.3, S.sil3[1], 0.6, 0.4)} />
      <Label x={560} y={260} text="visual hull" color={COL.hull} size={44} o={window01(t, S.sil4[0] + 0.6, S.sil4[1] + 0.3)} />
      <Label x={1360} y={260} text="האגן האמיתי" color={C.bone} size={44} o={window01(t, S.sil4[0] + 0.6, S.sil4[1] + 0.3)} />

      <Title text="ה-hull תמיד גדול מדי" t={t} a={S.wrong1[0]} b={S.wrong2[1] + 0.3} top={60} size={60} />
      <Label x={960} y={190} size={36} o={window01(t, S.wrong2[0] + 0.3, S.wrong2[1] + 0.3)}
        text={<span>אמת: <b style={{ color: C.bone }}>{m.true_vox.toLocaleString("en-US")}</b> · hull: <b style={{ color: COL.hull }}>{m.hull_vox.toLocaleString("en-US")}</b> · <span style={{ color: COL.extra }}>נוסף: {m.extra_vox.toLocaleString("en-US")}</span></span>} />
      <AxialPanel t={t} a={S.wrong3[0] + 0.4} b={S.wrong4[1] + 0.3} title="חתך בגובה כנפי האגן"
        imgs={["data/v07/axial_wings_true.png", "data/v07/axial_wings.png"]} labels={["האמת", "visual hull"]}
        bars={{ ap: K.wings.ap_row, lat: K.wings.lat_row, at: S.wrong4[0] + 3.0 }} />
      <AxialPanel t={t} a={S.wrong5[0]} b={S.wrong5[1] + 0.3} title="חתך בגובה האצטבולום"
        imgs={["data/v07/axial_acetab_true.png", "data/v07/axial_acetab.png"]} labels={["האמת", "visual hull"]} />
      <Label x={960} y={150} size={34} text="החור האובטורטורי: הצילום מלפנים רואה דרכו, אז הוא נחצב לכל העומק" color={C.warm}
        o={window01(t, S.wrong6[0] + 1.0, S.wrong6[1] + 0.2)} />
      <AxialPanel t={t} a={S.ghost1[0] + 0.4} b={S.ghost1[1] + 0.4} title="חתך בגובה עצם הערווה"
        imgs={["data/v07/axial_pubis_true.png", "data/v07/axial_pubis.png"]} labels={["האמת", "visual hull"]} />
      <Title text="רוחות רפאים" sub="ghosts: התאמה לא ידועה בין שני הצילומים" t={t} a={S.ghost2[0]} b={S.ghost3[1] + 0.3} top={40} size={56} />
      <GhostDiagram t={t} />
      <Label x={960} y={890} size={30} color={C.good} text="עם התאמה ידועה: טריאנגולציה (triangulation)" o={window01(t, S.ghost3[0] + 1.5, S.ghost3[1] + 0.3)} />

      <Title text="עוד תצפיות?" t={t} a={S.mv1[0]} b={S.mv1[1]} top={60} size={60} />
      <ViewsChart t={t} />
      <Label x={560} y={880} size={34} color={COL.extra}
        text={(t < S.mv2[0] + 2.8 ? "2" : t < S.mv2[0] + 6.0 ? "4" : "180") + " תצפיות · חתך בגובה הכנפיים"}
        o={window01(t, S.mv1[0] + 0.8, S.mv3[1] + 0.3)} />
      <Title text="מה CT עושה אחרת" sub="filtered back-projection: מאות זוויות + העוצמות עצמן" t={t} a={S.ct1[0]} b={S.ct2[1] + 0.3} top={80} size={60} />
      <CTPanel t={t} />

      <Title text="שתי תצפיות, הרבה צורות" t={t} a={S.amb1[0]} b={S.amb2[1] + 0.3} top={60} size={60} />
      <Label x={560} y={200} size={34} color={C.warm} text={`צורה אחרת: ${m.alt_vox.toLocaleString("en-US")} וקסלים`} o={window01(t, S.amb1[0] + 3.2, S.amb2[1] + 0.3)} />
      <Label x={1360} y={200} size={34} color={C.bone} text={`האגן: ${m.true_vox.toLocaleString("en-US")} וקסלים`} o={window01(t, S.amb1[0] + 1.0, S.amb2[1] + 0.3)} />
      <SameShadows t={t} />

      <ResultsPanel t={t} />
      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "back-projection: כל צילום נמרח חזרה לאורך הקרניים שלו",
        "visual hull: רק מה שבתוך שתי הצלליות; אף פעם לא מפספס עצם",
        "אבל ממלא כל שקע שאף צילום לא רואה דרכו, ויוצר רוחות רפאים",
        "עוד צלליות עוזרות עד גבול; CT משתמש במאות זוויות ובעוצמות",
        `האגן שלנו: Dice ${m.hull_dice.toFixed(2)}, HD95 ≈ ${Math.round(m.hull_hd95_mm)} mm`,
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/03-baselines/notebook.ipynb" todos={[
        "silhouettes(mask): צללית AP וצללית LAT מתוך מסכת 64³",
        "visual_hull(ap_mask, lat_mask): החיתוך של שתי המנסרות",
        "Dice ו-HD95 מול המסכה האמיתית (הפונקציות מיחידה 02)",
        "בדיקה: ה-hull מכיל את כל הווקסלים האמיתיים",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v07" />
    </AbsoluteFill>
  );
};
