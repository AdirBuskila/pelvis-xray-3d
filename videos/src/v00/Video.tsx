// Video 0 · The big picture. Reference implementation for the series: narration-driven timeline,
// DOM-level asset loading, one <Stage> whose contents switch by segment, 2D overlays on top.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v00/durations.json";
import { buildTimeline, Script, span } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Detector, Meta, Rays, ScanBeam, SlicePlane, SourcePanel, Voxels } from "../lib/xray";
import { Bullets, Captions, Credit, Label, Narration, Opening, PipelineMap, RecapCard, Title, YourTurn } from "../lib/Overlays";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const seg = (id: string) => S[id];

type Occ = { n: number; cell: number; count: number; centers: number[] };

// ---------------- ambiguity scene: shapes + their shadows on two walls ----------------
function ShadowRig({ x, front, side, shape, o, sideO, bicyl, see = 0 }: {
  x: number; front: "circle"; side: "circle" | "rect" | "tri"; shape: "sphere" | "cyl" | "cone" | "bicyl";
  o: number; sideO: number; bicyl: THREE.BufferGeometry | null; see?: number;
}) {
  const mat = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#d9ecff", roughness: 0.35, clearcoat: 0.6, transparent: true }), []);
  mat.opacity = o * (1 - 0.7 * see);
  mat.depthWrite = see < 0.3;
  const wall = (opacity: number) => (
    <meshBasicMaterial color={new THREE.Color(0.10, 0.22, 0.38)} transparent opacity={0.9 * opacity} side={THREE.DoubleSide} toneMapped={false} />
  );
  const shadow = (opacity: number) => (
    <meshBasicMaterial color={new THREE.Color(0.01, 0.02, 0.04)} transparent opacity={opacity} side={THREE.DoubleSide} toneMapped={false} />
  );
  const R = 0.75;
  if (o <= 0.001) return null;
  return (
    <group position={[x, 0, 0]}>
      {shape === "sphere" && <mesh material={mat}><sphereGeometry args={[R, 64, 48]} /></mesh>}
      {shape === "cyl" && <mesh material={mat} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[R, R, 1.9, 64]} /></mesh>}
      {shape === "cone" && <mesh material={mat} rotation={[-Math.PI / 2, 0, 0]}><coneGeometry args={[R, 1.9, 64]} /></mesh>}
      {shape === "bicyl" && bicyl && <mesh material={mat} geometry={bicyl} scale={R} />}
      {/* front wall (behind, +z) shows the AP shadow */}
      <mesh position={[0, 0, 1.45]}><planeGeometry args={[2.1, 2.1]} />{wall(o)}</mesh>
      <mesh position={[0, 0, 1.44]}><circleGeometry args={[R, 64]} />{shadow(o)}</mesh>
      {/* side wall (-x) shows the LAT shadow */}
      {sideO > 0.001 && (
        <group position={[-1.25, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
          <mesh><planeGeometry args={[2.3, 2.1]} />{wall(sideO)}</mesh>
          <group position={[0, 0, 0.01]}>
            {side === "circle" && <mesh><circleGeometry args={[R, 64]} />{shadow(sideO)}</mesh>}
            {side === "rect" && <mesh><planeGeometry args={[1.9, 2 * R]} />{shadow(sideO)}</mesh>}
            {side === "tri" && (
              <mesh rotation={[0, 0, Math.PI / 2]}>
                <bufferGeometry onUpdate={(g) => g.computeVertexNormals()}>
                  <bufferAttribute attach="attributes-position" args={[new Float32Array([0, 0.95, 0, -R, -0.95, 0, R, -0.95, 0]), 3]} />
                </bufferGeometry>
                {shadow(sideO)}
              </mesh>
            )}
          </group>
        </group>
      )}
    </group>
  );
}

// ---------------- the network sketch (2D, pipe3) ----------------
function NetworkSketch({ o }: { o: number }) {
  if (o <= 0) return null;
  const block = (x: number, h: number, w: number, label?: string, color = C.xray) => (
    <div style={{ position: "absolute", left: x, top: 540 - h / 2, width: w, height: h, borderRadius: 10,
      border: `2px solid ${color}`, background: "rgba(127,208,255,0.10)", boxShadow: `0 0 30px rgba(127,208,255,0.25)` }}>
      {label && <div style={{ position: "absolute", bottom: -44, width: "100%", textAlign: "center", fontSize: 24, color: C.dim }}>{label}</div>}
    </div>
  );
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT, direction: "rtl" }}>
      <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 250, top: 330, height: 190, borderRadius: 8, border: `1px solid ${C.line}` }} />
      <Img src={staticFile("data/hero/drr_lat.png")} style={{ position: "absolute", left: 270, top: 560, height: 190, borderRadius: 8, border: `1px solid ${C.line}` }} />
      <div style={{ position: "absolute", left: 470, top: 520, fontSize: 40, color: C.dim }}>→</div>
      {block(540, 300, 60)}{block(620, 220, 70)}{block(710, 150, 80)}{block(810, 90, 90, "encoder 2D")}
      <div style={{ position: "absolute", left: 930, top: 520, fontSize: 40, color: C.dim }}>→</div>
      {block(1000, 90, 90, undefined, C.warm)}{block(1110, 150, 90, undefined, C.warm)}{block(1220, 220, 90, "decoder 3D", C.warm)}
      <div style={{ position: "absolute", left: 1335, top: 520, fontSize: 40, color: C.dim }}>→</div>
      <div style={{ position: "absolute", left: 1400, top: 430, width: 220, height: 220, perspective: 600 }}>
        <div style={{ width: 160, height: 160, margin: 30, border: `2px solid ${C.warm}`, transform: "rotateX(-20deg) rotateY(35deg)",
          background: "rgba(255,207,127,0.12)", boxShadow: "0 0 40px rgba(255,207,127,0.3)" }} />
        <div style={{ textAlign: "center", fontSize: 26, color: C.warm, direction: "ltr" }}>64 × 64 × 64</div>
      </div>
    </div>
  );
}

// ---------------- 3D scene ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const box = boxOf(a.meta);
  const [cx, cy, cz] = box.c;
  const apZsrc = box.mn[2] - 1.4, apZdet = box.mx[2] + 0.3;
  const latXsrc = box.mx[0] + 1.4, latXdet = box.mn[0] - 0.3;

  // --- visibility by section ---
  const heroOn = window01(t, 2.6, seg("amb1")[0] + 0.2, 1.0, 0.8);
  const ghost = ramp(t, seg("xr1")[0] + 0.8, seg("xr1")[0] + 2.2) * (1 - ramp(t, seg("depth2")[1], seg("amb1")[0]));
  const apOn = window01(t, seg("xr1")[0] + 0.6, seg("amb1")[0] + 0.1, 0.9, 0.6);
  const apScan = ramp(t, seg("xr1")[0] + 1.8, seg("xr1")[1] - 0.2);
  const latOn = window01(t, seg("xr2")[0], seg("depth1")[0] + 0.3, 0.9, 0.6);
  const latScan = ramp(t, seg("xr2")[0] + 0.6, seg("xr2")[1] + 0.2);
  const depthOn = window01(t, seg("depth1")[0] + 0.4, seg("depth2")[1] + 0.2, 0.8, 0.5);

  const shapesOn = window01(t, seg("amb1")[0], seg("amb5")[1] + 0.4, 0.8, 0.8);
  const sideOn = ramp(t, seg("amb3")[0] + 0.3, seg("amb3")[0] + 1.6);
  const swap = ramp(t, seg("amb4")[0] + 0.2, seg("amb4")[0] + 1.4);
  const see = window01(t, seg("amb1")[0] + 4.0, seg("amb2")[1] + 0.4, 0.8, 0.8);

  const priorOn = window01(t, seg("prior1")[0] - 0.2, seg("prior2")[1] + 0.4, 0.9, 0.8);
  const [p1, p2] = span(tl, "pipe0", "pipe6");
  const pipeOn = window01(t, p1 + 0.2, p2 + 0.4, 0.8, 0.8);
  const ctOn = window01(t, seg("pipe1")[0], seg("pipe2")[0] + 0.6, 0.8, 0.6);
  const drrOn = window01(t, seg("pipe2")[0] + 0.2, seg("pipe3")[0] + 0.6, 0.8, 0.6);
  const voxOn = window01(t, seg("pipe4")[0], seg("pipe6")[1] + 0.4, 0.6, 0.8);
  const voxBuild = ramp(t, seg("pipe4")[0] + 0.3, seg("pipe4")[1] - 0.5);
  const meshOn = ramp(t, seg("pipe5")[0] + 0.2, seg("pipe5")[1]) * (1 - ramp(t, seg("pipe6")[1], seg("pipe6")[1] + 0.6));
  const pipeBone = ctOn * 0.35 + drrOn * 0.35;
  const tailOn = window01(t, seg("honest1")[0], tl.total, 1.0, 1.0);

  // --- the depth ray: one ray through the sacrum region to one detector pixel ---
  const ry = cy + 0.25, rx = cx + 0.15;
  const pts = [0.18, 0.5, 0.82].map((f) => apZsrc + (apZdet - apZsrc) * (0.25 + f * 0.5));
  const slide = Math.sin((t - seg("depth2")[0]) * 1.6);

  // --- camera ---
  const front = (r: number, y = 0.5): [number, number, number] => [cx, y, -r];
  const keys: CamKey[] = [
    { t: 0, pos: orbit(-0.9, 9.2, 0.7) },
    { t: seg("hero2")[1], pos: orbit(0.55, 8.6, 0.9) },
    { t: seg("xr1")[0] + 1.2, pos: [8.6, 2.4, -3.6], look: [0, 0, 0.1] },
    { t: seg("xr1")[1], pos: [8.2, 1.8, -4.4], look: [0, 0, 0.2] },
    { t: seg("xr2")[1], pos: [6.6, 5.2, -7.8], look: [-0.4, -0.2, 0.3] },
    { t: seg("depth1")[0] + 1.5, pos: [5.0, ry + 1.0, -2.2], look: [cx, ry, cz] },
    { t: seg("depth2")[1], pos: [4.4, ry + 0.8, -1.4], look: [cx, ry, cz + 0.4] },
    // shapes: start oblique, go straight-on (all circles), then to the side
    { t: seg("amb1")[0] + 0.6, pos: [5.5, 3.6, -9.5], look: [0, 0, 0.4] },
    { t: seg("amb2")[0], pos: [0, 0.2, -11.5], look: [0, 0, 1.4], fov: 32 },
    { t: seg("amb2")[1] + 0.3, pos: [0, 0.2, -11.5], look: [0, 0, 1.4], fov: 32 },
    { t: seg("amb3")[1], pos: [9.5, 3.2, -6.5], look: [-0.6, 0, 0.4] },
    { t: seg("amb4")[1], pos: [7.0, 2.6, -8.0], look: [-0.4, 0, 0.4] },
    { t: seg("amb5")[1], pos: [6.2, 2.8, -8.6], look: [-0.4, 0, 0.4] },
    { t: seg("prior1")[0] + 1.2, pos: [0, 1.2, -11.5], look: [0, 0, 0] },
    { t: seg("prior2")[1], pos: [0, 0.6, -10.0], look: [0, 0, 0] },
    { t: seg("pipe1")[0] + 1.0, pos: orbit(0.7, 8.8, 2.0), look: [cx, cy, cz] },
    { t: seg("pipe2")[1], pos: [6.6, 4.4, -7.6], look: [-0.3, -0.1, 0.2] },
    { t: seg("pipe4")[0] + 0.5, pos: orbit(-0.4, 7.4, 1.6) },
    { t: seg("pipe6")[1], pos: orbit(0.8, 7.0, 1.2) },
    { t: tl.total, pos: orbit(2.0, 9.5, 1.0) },
  ];

  // shapes row layout
  const xs = [-3.1, 0, 3.1];
  const pelvises = [a.pelvis, ...a.priors];
  const spin = t * 0.25;

  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.1} opacity={1 - shapesOn * 0.6} />

      {/* hero bones */}
      <group>
        <Bone geo={a.pelvis} material={bone} opacity={heroOn * (1 - 0.6 * ghost) + pipeBone + meshOn * 0.95 + tailOn * 0.9} />
        <Bone geo={a.femurs} material={bone} opacity={heroOn * (1 - 0.7 * ghost) + pipeBone * 0.8 + tailOn * 0.8} />
        <Bone geo={a.spine} material={bone} opacity={heroOn * (1 - 0.7 * ghost) + pipeBone * 0.8 + tailOn * 0.8} />
      </group>

      {/* AP rig */}
      <group position={[cx, cy, 0]}>
        <SourcePanel size={a.meta.ap_size} position={[0, 0, apZsrc]} opacity={apOn + drrOn} />
        <ScanBeam axis="ap" progress={apScan} opacity={apOn} w={a.meta.ap_size[0]} h={a.meta.ap_size[1]}
          src={apZsrc} det={apZdet} bodyIn={box.mn[2]} bodyOut={box.mx[2]} />
        <Detector tex={a.ap} progress={Math.max(apScan, drrOn)} opacity={apOn + drrOn} size={a.meta.ap_size}
          position={[0, 0, apZdet]} rotation={[0, Math.PI, 0]} />
      </group>
      {/* LAT rig */}
      <group position={[0, cy, cz]}>
        <SourcePanel size={a.meta.lat_size} position={[latXsrc, 0, 0]} rotation={[0, Math.PI / 2, 0]} opacity={latOn + drrOn} />
        <ScanBeam axis="lat" progress={latScan} opacity={latOn} w={a.meta.lat_size[0]} h={a.meta.lat_size[1]}
          src={latXsrc} det={latXdet} bodyIn={box.mx[0]} bodyOut={box.mn[0]} />
        <Detector tex={a.lat} progress={Math.max(latScan, drrOn)} opacity={latOn + drrOn} size={a.meta.lat_size}
          position={[latXdet, 0, 0]} rotation={[0, Math.PI / 2, 0]} flipX />
      </group>

      {/* depth: one ray, three possible bone positions, one pixel */}
      {depthOn > 0 && (
        <group>
          <Rays segs={[[[rx, ry, apZsrc], [rx, ry, apZdet]]]} opacity={depthOn} />
          {pts.map((z0, i) => {
            const z = z0 + slide * 0.25 * (i === 1 ? 1 : 0);
            const on = i === Math.floor(((t - seg("depth2")[0]) / 1.6) % 3 + 3) % 3 || t < seg("depth2")[0];
            return (
              <mesh key={i} position={[rx, ry, z]}>
                <sphereGeometry args={[0.09, 24, 16]} />
                <meshBasicMaterial color={new THREE.Color(...(on ? [3.2, 2.4, 1.0] : [0.8, 0.7, 0.4]) as [number, number, number])}
                  transparent opacity={depthOn} toneMapped={false} />
              </mesh>
            );
          })}
          <mesh position={[rx, ry, apZdet - 0.02]}>
            <planeGeometry args={[0.16, 0.16]} />
            <meshBasicMaterial color={new THREE.Color(3, 2.2, 0.9)} transparent opacity={depthOn} toneMapped={false} side={THREE.DoubleSide} />
          </mesh>
        </group>
      )}

      {/* ambiguity shapes */}
      <group position={[0, 0, 0]}>
        <ShadowRig x={xs[0]} front="circle" side="circle" shape="sphere" o={shapesOn} sideO={shapesOn * sideOn} bicyl={a.bicyl} see={see} />
        <ShadowRig x={xs[1]} front="circle" side="rect" shape="cyl" o={shapesOn * (1 - swap)} sideO={shapesOn * sideOn * (1 - swap)} bicyl={a.bicyl} see={see} />
        <ShadowRig x={xs[2]} front="circle" side="tri" shape="cone" o={shapesOn * (1 - swap)} sideO={shapesOn * sideOn * (1 - swap)} bicyl={a.bicyl} see={see} />
        <ShadowRig x={xs[1] + 1.5} front="circle" side="circle" shape="bicyl" o={shapesOn * swap} sideO={shapesOn * swap} bicyl={a.bicyl} />
      </group>

      {/* priors: four different people's pelvises */}
      {priorOn > 0 && pelvises.map((g, i) => (
        <Bone key={i} geo={g} material={bone} opacity={priorOn * ramp(t, seg("prior1")[0] + i * 0.35, seg("prior1")[0] + 0.6 + i * 0.35)}
          position={[(i - 1.5) * 3.3, 0, 0]} scale={0.78} rotation={[0, spin + i * 0.4, 0]} />
      ))}

      {/* pipeline visuals */}
      <SlicePlane tex={a.coronal} size={[box.s[0], box.s[1]]} position={[cx, cy, 0]} rotation={[0, Math.PI, 0]} opacity={ctOn * 0.9} />
      <group position={[0, cy, cz]} scale={[-1, 1, 1]}>
        <SlicePlane tex={a.sagittal} size={[box.s[2], box.s[1]]} position={[0, 0, 0]} rotation={[0, -Math.PI / 2, 0]} opacity={ctOn * 0.9} />
      </group>
      <group position={[cx, 0, cz]} scale={[-1, 1, 1]}>
        <SlicePlane tex={a.axial} size={[box.s[0], box.s[2]]} position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]} opacity={ctOn * 0.9} />
      </group>
      {a.occ64 && <Voxels centers={a.occ64.centers} cell={a.occ64.cell} reveal={voxBuild} opacity={voxOn * (1 - meshOn)} />}
    </>
  );
}

type Assets = {
  meta: Meta; pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry; spine: THREE.BufferGeometry;
  ap: THREE.Texture; lat: THREE.Texture; coronal: THREE.Texture; sagittal: THREE.Texture; axial: THREE.Texture;
  priors: (THREE.BufferGeometry | null)[]; bicyl: THREE.BufferGeometry | null; occ64: Occ | null;
};

export const V00 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const meta = useJson<Meta>("data/hero/meta.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const spine = useObj("data/hero/spine.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const lat = useTex("data/hero/drr_lat.png");
  const coronal = useTex("data/hero/slice_coronal.png");
  const sagittal = useTex("data/hero/slice_sagittal.png");
  const axial = useTex("data/hero/slice_axial.png");
  const p1 = useObj("data/v00/pelvis_s1361.obj");
  const p2 = useObj("data/v00/pelvis_s1387.obj");
  const p3 = useObj("data/v00/pelvis_s1400.obj");
  const bicyl = useObj("data/v00/bicylinder.obj");
  const occ64 = useJson<Occ>("data/hero/occ64.json");
  const ready = meta && pelvis && femurs && spine && ap && lat && coronal && sagittal && axial && p1 && p2 && p3 && bicyl && occ64;

  const [p0, p6] = span(tl, "pipe0", "pipe6");
  const pipeIdx = ["pipe1", "pipe2", "pipe3", "pipe4", "pipe5", "pipe6"].findIndex((id) => t >= S[id][0] - 0.2 && t < S[id][1] + 0.4);
  const tail = S.honest1[0];

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ meta, pelvis, femurs, spine, ap, lat, coronal, sagittal, axial, priors: [p1, p2, p3], bicyl, occ64 }} />
        </Stage>
      )}
      {/* dim the 3D behind text-heavy sections */}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.62 * window01(t, tail - 0.3, tl.total, 0.8, 0.1), pointerEvents: "none" }} />

      <Opening n={0} title="התמונה הגדולה" active={[0, 1, 2, 3, 4, 5]} t={t} end={2.9} />

      <Title text="צל מאבד את העומק" t={t} a={S.depth1[0]} b={S.depth2[1] + 0.3} />
      <Title text="בעיה הפוכה" sub="inverse problem: הרבה צורות ← אותן תמונות" t={t} a={S.amb5[0]} b={S.amb5[1] + 0.4} />
      <Title text="ידע מוקדם" sub="prior: איך אגן אמור להיראות" t={t} a={S.prior1[0]} b={S.prior2[1] + 0.3} />
      <Label x={960} y={790} text="× ~450 סריקות במאגר המלא" color={C.warm} size={36}
        o={window01(t, S.prior2[0] + 0.5, S.prior2[1] + 0.3)} />
      <PipelineMap active={pipeIdx >= 0 ? [pipeIdx] : []} t={t} a={p0} b={p6 + 0.4} top={80} label="עכשיו" />
      <NetworkSketch o={window01(t, S.pipe3[0] - 0.1, S.pipe3[1] + 0.3, 0.4, 0.4)} />

      <Bullets t={t} o={window01(t, S.honest1[0], S.honest2[1] + 0.4)} title="מה יהפוך את הפרויקט למעניין"
        items={["השוואה לשיטות בלי רשת: כמה הרשת באמת למדה?", "בדיקה על צילומי רנטגן אמיתיים, לא רק מדומים"]}
        times={[S.honest1[0] + 0.3, S.honest2[0] + 0.2]} />
      <Bullets t={t} o={window01(t, S.plan1[0], S.plan2[1] + 0.4)} title="איך בנויה הסדרה" style={{ width: 900 }}
        items={["1 · תמונות וגאומטריה", "2 · צורות בתלת-ממד", "3 · שיטות בלי רשת", "4 · רשתות נוירונים לראייה", "5 · ניסויים ועולם אמיתי"]}
        times={[0, 1, 2, 3, 4].map((i) => S.plan2[0] + 0.3 + i * (S.plan2[1] - S.plan2[0] - 0.6) / 5)} />
      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "צילום רנטגן = צל: כל פיקסל סוכם קרן שלמה",
        "צל מאבד עומק; גם שני צילומים לא קובעים צורה אחת",
        "לכן צריך ידע מוקדם (prior), והרשת לומדת אותו מהנתונים",
        "הצינור: CT → DRR → רשת → 64³ → mesh → מדדים",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/00-setup/notebook.ipynb" todos={[
        "להריץ את בדיקת הסביבה מקומית",
        "לפתוח את אותה מחברת ב-Colab וב-Kaggle",
        "למלא את TODO הצעצוע: הטלה דו-ממדית של ריבוע",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.hero1[0]} />
      <Narration tl={tl} dir="v00" />
    </AbsoluteFill>
  );
};
