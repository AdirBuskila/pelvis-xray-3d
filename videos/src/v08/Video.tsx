// Video 8 · The memorisation floor: a REAL (small: 27 subjects) nearest-neighbour baseline computed in
// prep/v08_assets.py: L2 on 32x32 AP+LAT DRRs, retrieved pelvis vs the hero's, leave-one-out table.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import script from "./script.json";
import durations from "../../public/audio/v08/durations.json";
import nn from "../../public/data/v08/nn.json";
import m7 from "../../public/data/v07/metrics.json";
import { buildTimeline, Script } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj } from "../lib/assets";
import { CameraPath, CamKey, Floor, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone } from "../lib/xray";
import { Bullets, Captions, Credit, Eq, Label, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Grid, GridVox } from "../v07/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;

type Vox = Grid & { tp: number[]; fp: number[]; fn: number[] };
const H = nn.hero;
const DB = nn.subjects.filter((s) => s !== H.query);                       // original order
const RANK: Record<string, number> = Object.fromEntries(H.ranked.map((r, i) => [r.s, i]));
const BY: Record<string, (typeof H.ranked)[number]> = Object.fromEntries(H.ranked.map((r) => [r.s, r]));
const T = nn.table;
const COL = { tp: "#2fd08a", fp: "#ff5a6a", fn: "#ff9a2e", nn: "#9fd6ff" };

// ---------------- 3D ----------------
function Scene({ t, a }: { t: number; a: { hero: THREE.BufferGeometry; nnGeo: THREE.BufferGeometry; vox: Vox } }) {
  const bone = useBoneMaterial();
  const boneNN = useBoneMaterial("#bfe3ff");
  const introO = window01(t, 3.0, S.idea1[0] + 0.4, 1.0, 0.8);
  const sideO = window01(t, S.q3[0], S.m1[1] + 0.3, 0.8, 0.5);
  const merge = ramp(t, S.m1[0] + 0.3, S.m1[0] + 2.2);
  const voxO = window01(t, S.m2[0] - 0.2, S.m3[1] + 0.5, 0.6, 0.6);
  const meshO = sideO * (1 - ramp(t, S.m2[0] - 0.3, S.m2[0] + 0.3));
  const spin = t * 0.2;
  const keys: CamKey[] = [
    { t: 0, pos: [5.5, 2.6, -7.8], look: [0, 0, 0] },
    { t: S.idea1[0], pos: [7.0, 2.4, -6.0], look: [0, 0, 0] },
    { t: S.q3[0], pos: [0, 0.8, -10.0], look: [0, 0, 0] },
    { t: S.q3[1], pos: [0, 0.9, -9.6], look: [0, 0, 0] },
    { t: S.m1[0] + 2.2, pos: [2.0, 1.6, -7.2], look: [0, 0, 0] },
    { t: S.m2[0] + 1.0, pos: [3.0, 2.4, -7.0], look: [0, 0, 0] },
    { t: S.m2[1] + 0.5, pos: [-3.0, 2.6, -7.0], look: [0, 0, 0] },
    { t: tl.total, pos: [-3.0, 2.6, -7.0], look: [0, 0, 0] },
  ];
  const dx = 1.95 * (1 - merge);
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.0} opacity={0.8} />
      <Bone geo={a.hero} material={bone} opacity={introO} rotation={[0, spin, 0]} />
      {/* query truth (left of screen = +x) and the retrieved pelvis */}
      <Bone geo={a.hero} material={bone} opacity={meshO * (merge > 0.5 ? 0.55 : 1)} position={[dx, 0, 0]} scale={0.85} rotation={[0, (t - S.q3[0]) * 0.15, 0]} />
      <Bone geo={a.nnGeo} material={boneNN} opacity={meshO * (merge > 0.5 ? 0.55 : 1)} position={[-dx, 0, 0]} scale={0.85} rotation={[0, (t - S.q3[0]) * 0.15, 0]} />
      {voxO > 0 && (
        <group rotation={[0, (t - S.m2[0]) * 0.18, 0]}>
          <GridVox g={a.vox} idx={a.vox.tp} color={COL.tp} opacity={voxO} emissive={0.08} />
          <GridVox g={a.vox} idx={a.vox.fp} color={COL.fp} opacity={voxO} emissive={0.25} />
          <GridVox g={a.vox} idx={a.vox.fn} color={COL.fn} opacity={voxO * 0.3} depthWrite={false} emissive={0.25} />
        </group>
      )}
    </>
  );
}

// ---------------- 2D: the gallery ----------------
const CW = 196, CH = 122, X0 = 70, Y0 = 175, DXC = 212, DYC = 136, COLS = 6;
const slot = (i: number) => {
  const c = i % COLS, r = Math.floor(i / COLS);
  return [X0 + (COLS - 1 - c) * DXC, Y0 + r * DYC];          // RTL: first slot top-right
};

function Gallery({ t }: { t: number }) {
  const o = Math.max(window01(t, S.idea1[0], S.demo1[1] + 0.5, 0.7, 0.6), window01(t, S.q1[0], S.q2[1] + 0.6, 0.6, 0.6),
    window01(t, S.cav1[0], S.cav1[1] + 0.6, 0.6, 0.6));
  if (o <= 0) return null;
  const sortK = Math.max(ramp(t, S.q2[0] + 2.5, S.q2[0] + 4.5), window01(t, S.cav1[0] - 1, S.cav1[1] + 2, 0.01, 0.01));
  const distO = Math.max(ramp(t, S.q2[0] + 0.6, S.q2[0] + 2.2) * (t < S.q2[1] + 1 ? 1 : 0), t > S.cav1[0] - 1 ? 1 : 0);
  const diceO = ramp(t, S.cav1[0] + 3.0, S.cav1[0] + 4.0);
  const scan = t > S.idea2[0] + 2.0 && t < S.idea2[1] ? Math.floor((t - S.idea2[0] - 2.0) * 7) % DB.length : -1;
  const bestO = Math.max(ramp(t, S.idea2[1] - 3.5, S.idea2[1] - 3.0) * (t < S.idea2[1] + 0.5 ? 1 : 0), ramp(t, S.q2[0] + 4.5, S.q2[0] + 5.2) * (t < S.q2[1] + 1 ? 1 : 0), t > S.cav1[0] - 1 ? 1 : 0);
  const oracleO = ramp(t, S.cav1[0] + 6.5, S.cav1[0] + 7.3);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      {DB.map((s, i) => {
        const appear = ramp(t, S.idea1[0] + 0.3 + i * 0.08, S.idea1[0] + 0.7 + i * 0.08);
        const [x1, y1] = slot(i), [x2, y2] = slot(RANK[s]);
        const x = lerp(x1, x2, sortK), y = lerp(y1, y2, sortK);
        const isNN = s === H.nn, isOr = s === H.oracle;
        const lit = scan === i;
        const border = isNN && bestO > 0.5 ? C.warm : isOr && oracleO > 0.5 ? C.good : lit ? C.xray : C.line;
        return (
          <div key={s} style={{ position: "absolute", left: x, top: y, width: CW, opacity: appear,
            border: `2px solid ${border}`, borderRadius: 10, padding: 4, background: "rgba(8,14,24,0.75)",
            boxShadow: (isNN && bestO > 0.5) || lit ? `0 0 26px ${border}` : "none" }}>
            <div style={{ display: "flex", gap: 4, direction: "ltr" }}>
              <Img src={staticFile(`data/v08/g/ap_${s}.png`)} style={{ width: 92, height: 92 }} />
              <Img src={staticFile(`data/v08/g/lat_${s}.png`)} style={{ width: 92, height: 92 }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 17, color: C.dim, direction: "ltr", marginTop: 2 }}>
              <span>{s}</span>
              <span style={{ color: C.xray, opacity: distO }}>d={BY[s].dist.toFixed(0)}</span>
              <span style={{ color: isOr ? C.good : C.warm, opacity: diceO }}>{BY[s].dice.toFixed(2)}</span>
            </div>
          </div>
        );
      })}
      <div style={{ position: "absolute", left: X0, top: Y0 - 50, width: COLS * DXC, textAlign: "center", fontSize: 30, color: C.dim, direction: "rtl" }}>
        {t < S.q1[0] ? "מאגר: לכל מטופל AP + LAT, ואגן אמיתי" : sortK > 0.5 ? "ממוינים לפי המרחק לשאילתה (d), מהקרוב לרחוק →" : "המאגר: 26 נבדקים"}
      </div>
    </div>
  );
}

function QueryCard({ t }: { t: number }) {
  const o = Math.max(window01(t, S.idea2[0] + 0.6, S.idea2[1] + 0.5, 0.6, 0.6), window01(t, S.q1[0], S.q2[1] + 0.6, 0.6, 0.6),
    window01(t, S.cav1[0], S.cav1[1] + 0.6, 0.6, 0.6));
  if (o <= 0) return null;
  const named = t > S.q1[0] - 0.5;
  return (
    <Panel o={o} style={{ left: 1420, top: 230, width: 420, padding: "22px 24px", border: `2px solid ${C.warm}` }}>
      <div style={{ fontSize: 32, fontWeight: 700, color: C.warm, marginBottom: 14 }}>שאילתה{named ? `: ${H.query}` : ""}</div>
      <div style={{ display: "flex", gap: 8, direction: "ltr", justifyContent: "center" }}>
        <Img src={staticFile(`data/v08/g/ap_${H.query}.png`)} style={{ width: 180, height: 180, borderRadius: 6 }} />
        <Img src={staticFile(`data/v08/g/lat_${H.query}.png`)} style={{ width: 180, height: 180, borderRadius: 6 }} />
      </div>
      <div style={{ fontSize: 24, color: C.dim, marginTop: 12 }}>
        {named ? "הוצא מהמאגר: לא ימצא את עצמו" : "שני צילומים חדשים · האגן לא ידוע"}
      </div>
    </Panel>
  );
}

function DownPanel({ t }: { t: number }) {
  const a = S.dist1[0], b = S.eq1[1] + 0.5;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const k = (x: number) => ramp(t, x, x + 0.6);
  const eqOn = t > S.eq1[0] - 0.2;
  const im = (src: string, label: string, kk: number, w = 230, color = C.dim) => (
    <div style={{ textAlign: "center", opacity: kk }}>
      <Img src={staticFile(src)} style={{ width: w, height: w, imageRendering: "pixelated", borderRadius: 8, border: `1.5px solid ${C.line}`, display: "block" }} />
      <div style={{ fontSize: 24, color, marginTop: 8, direction: "rtl" }}>{label}</div>
    </div>
  );
  return (
    <div style={{ position: "absolute", top: eqOn ? 560 : 300, right: 110, display: "flex", gap: 26, alignItems: "center", opacity: o, fontFamily: FONT, direction: "ltr" }}>
      {!eqOn && im(`data/v08/g/ap_${H.query}.png`, "128 × 128", k(a + 0.3), 300)}
      {!eqOn && <div style={{ fontSize: 50, color: C.dim, opacity: k(a + 2.5) }}>→</div>}
      {im(`data/v08/g/ap32_${H.query}.png`, "שאילתה 32×32 · AP", k(a + 2.5), eqOn ? 200 : 300, C.warm)}
      {im(`data/v08/g/lat32_${H.query}.png`, "LAT", k(S.dist2[0] + 3.0), eqOn ? 200 : 300, C.warm)}
      {eqOn && <div style={{ fontSize: 44, color: C.dim }}>−</div>}
      {eqOn && im(`data/v08/g/ap32_${H.nn}.png`, `${H.nn} · AP`, 1, 200, C.xray)}
      {eqOn && im(`data/v08/g/lat32_${H.nn}.png`, "LAT", 1, 200, C.xray)}
      {eqOn && <div style={{ fontSize: 44, color: C.dim }}>=</div>}
      {eqOn && im("data/v08/diff_ap.png", "|הפרש|", ramp(t, S.eq1[0] + 3, S.eq1[0] + 3.6), 200, "white")}
      {eqOn && im("data/v08/diff_lat.png", "", ramp(t, S.eq1[0] + 3, S.eq1[0] + 3.6), 200, "white")}
    </div>
  );
}

function Scatter({ t }: { t: number }) {
  const a = S.cav2[0], b = S.cav2[1] + 0.6;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const W = 900, Hh = 520, x0 = 90, y0 = 30, yb = Hh - 70;
  const X = (d: number) => x0 + (d / 130) * (W - x0 - 30);
  const Y = (v: number) => yb - ((v - 0.35) / 0.4) * (yb - y0);
  return (
    <Panel o={o} style={{ left: 510, top: 170, width: W + 20, padding: "22px 26px" }}>
      <div style={{ fontSize: 30, fontWeight: 700, color: C.xray, marginBottom: 6 }}>מרחק בתמונה מול Dice אמיתי · שאילתה {H.query}</div>
      <svg width={W} height={Hh} style={{ direction: "ltr" }}>
        <line x1={x0} y1={yb} x2={W - 20} y2={yb} stroke={C.dim} />
        <line x1={x0} y1={yb} x2={x0} y2={y0} stroke={C.dim} />
        {[0.4, 0.5, 0.6, 0.7].map((v) => <text key={v} x={x0 - 12} y={Y(v) + 7} fill={C.dim} fontSize={20} textAnchor="end">{v.toFixed(1)}</text>)}
        {[0, 25, 50, 75, 100, 125].map((d) => <text key={d} x={X(d)} y={yb + 30} fill={C.dim} fontSize={20} textAnchor="middle">{d}</text>)}
        <text x={(x0 + W) / 2} y={yb + 60} fill={C.dim} fontSize={22} textAnchor="middle">L2 distance (32×32 AP+LAT)</text>
        <text x={24} y={(y0 + yb) / 2} fill={C.dim} fontSize={22} textAnchor="middle" transform={`rotate(-90 24 ${(y0 + yb) / 2})`}>Dice vs {H.query}</text>
        {H.ranked.map((r, i) => {
          const k = ramp(t, a + 0.5 + i * 0.08, a + 0.9 + i * 0.08);
          const col = r.s === H.nn ? C.warm : r.s === H.oracle ? C.good : C.xray;
          const big = r.s === H.nn || r.s === H.oracle;
          return <g key={r.s} opacity={k}>
            <circle cx={X(r.dist)} cy={Y(r.dice)} r={big ? 11 : 7} fill={col} />
            {big && <text x={X(r.dist) + 16} y={Y(r.dice) + 7} fill={col} fontSize={22}>{r.s === H.nn ? `NN ${r.s}` : `oracle ${r.s}`}</text>}
          </g>;
        })}
        <text x={W - 30} y={y0 + 20} fill={C.dim} fontSize={22} textAnchor="end" opacity={ramp(t, a + 4, a + 4.6)}>Spearman ρ = {H.spearman_dist_dice.toFixed(2)}</text>
      </svg>
    </Panel>
  );
}

function Table({ t }: { t: number }) {
  const a = S.tab1[0] + 0.5, b = S.tab4[1] + 0.6;
  const o = window01(t, a, b, 0.7, 0.6);
  if (o <= 0) return null;
  const f = (r: { dice_mean: number; dice_std: number; hd95_mean: number; hd95_std: number }) =>
    [`${r.dice_mean.toFixed(2)} ± ${r.dice_std.toFixed(2)}`, `${r.hd95_mean.toFixed(1)} ± ${r.hd95_std.toFixed(1)}`];
  const rows: { name: string; cells: string[]; at: number; color?: string; hl?: boolean }[] = [
    { name: "אגן אקראי של מטופל אחר (ממוצע)", cells: f(T.random), at: S.tab2[0] + 0.5 },
    { name: "השכן הקרוב (L2 על DRR)", cells: f(T.nn), at: S.tab2[0] + 7.0, color: C.warm, hl: true },
    { name: "השכן הטוב ביותר בדיעבד (oracle)", cells: f(T.oracle), at: S.tab3[0] + 0.5, color: C.good },
    { name: "visual hull (סרטון 7)", cells: f(T.hull), at: S.tab3[1] - 4.0, color: COL.fp },
    { name: "הרשת שלנו (U-Net)", cells: ["נמדוד", "נמדוד"], at: S.tab4[0] + 6.0, color: C.dim },
  ];
  return (
    <Panel o={o} style={{ left: 170, right: 170, top: 150, padding: "30px 40px" }}>
      <div style={{ fontSize: 42, fontWeight: 700, color: C.xray }}>טבלת התוצאות הראשונה</div>
      <div style={{ fontSize: 24, color: C.dim, marginBottom: 18 }}>
        leave-one-out על {nn.n} נבדקים · רשת 64³ (תא 5.5 מ"מ), מיושר לפי מרכז מסה · ממוצע ± ס"ת · הדגמה קטנה, לא הניסוי הסופי
      </div>
      <div style={{ display: "flex", fontSize: 28, color: C.dim, borderBottom: `1.5px solid ${C.line}`, paddingBottom: 10 }}>
        <div style={{ flex: 1 }}>שיטה</div>
        <div style={{ width: 260, textAlign: "center", direction: "ltr" }}>Dice ↑</div>
        <div style={{ width: 300, textAlign: "center", direction: "ltr" }}>HD95 (mm) ↓</div>
      </div>
      {rows.map((r) => {
        const k = ramp(t, r.at, r.at + 0.5);
        return (
          <div key={r.name} style={{ display: "flex", fontSize: 32, padding: "15px 0", borderBottom: `1px solid rgba(120,190,255,0.18)`,
            opacity: k, color: r.color ?? C.text, background: r.hl ? "rgba(255,207,127,0.07)" : "none" }}>
            <div style={{ flex: 1 }}>{r.name}</div>
            <div style={{ width: 260, textAlign: "center", direction: "ltr" }}>{r.cells[0]}</div>
            <div style={{ width: 300, textAlign: "center", direction: "ltr" }}>{r.cells[1]}</div>
          </div>
        );
      })}
      <div style={{ fontSize: 24, color: C.dim, marginTop: 16, opacity: ramp(t, S.tab3[0] + 7, S.tab3[0] + 7.6) }}>
        ה-NN בחר את ה-oracle ב-{nn.loo_nn_equals_oracle} מתוך {nn.n} שאילתות
      </div>
    </Panel>
  );
}

function PaperCard({ t }: { t: number }) {
  const a = S.tat1[0], b = S.tat2[1] + 0.5;
  const o = window01(t, a, b, 0.7, 0.6);
  if (o <= 0) return null;
  const lane = (y: number, mid: string, col: string, at: number) => {
    const k = ramp(t, at, at + 0.7);
    return (
      <div style={{ position: "absolute", top: y, left: 40, right: 40, display: "flex", alignItems: "center", gap: 26, opacity: k, direction: "ltr", fontSize: 30 }}>
        <div style={{ padding: "14px 22px", border: `2px solid ${C.line}`, borderRadius: 12 }}>image</div>
        <div style={{ color: C.dim }}>→</div>
        <div style={{ padding: "14px 22px", border: `2px solid ${col}`, borderRadius: 12, color: col, minWidth: 330, textAlign: "center" }}>{mid}</div>
        <div style={{ color: C.dim }}>→</div>
        <div style={{ padding: "14px 22px", border: `2px solid ${C.line}`, borderRadius: 12 }}>3D shape</div>
      </div>
    );
  };
  return (
    <Panel o={o} style={{ left: 260, right: 260, top: 150, height: 560 }}>
      <div style={{ fontSize: 30, color: C.dim, direction: "ltr", textAlign: "left" }}>Tatarchenko, Richter, Ranftl, Li, Koltun, Brox · CVPR 2019</div>
      <div style={{ fontSize: 40, fontWeight: 700, color: "white", direction: "ltr", textAlign: "left", marginTop: 8 }}>What Do Single-view 3D Reconstruction Networks Learn?</div>
      {lane(250, "deep network", C.xray, S.tat2[0] + 1.0)}
      {lane(370, "retrieval / clustering", C.warm, S.tat2[0] + 6.0)}
      <div style={{ position: "absolute", top: 330, right: 40, fontSize: 40, color: C.good, opacity: ramp(t, S.tat2[0] + 8.5, S.tat2[0] + 9.2) }}>≈</div>
      <div style={{ position: "absolute", bottom: 34, left: 40, right: 40, fontSize: 30, color: C.warm, opacity: ramp(t, S.tat2[0] + 11, S.tat2[0] + 11.6) }}>
        במידה רבה: זיהוי של סוג הצורה, ולא שחזור (על ShapeNet)
      </div>
    </Panel>
  );
}

/** the Dice "floor": where our real baselines sit, and what a network result would mean */
function FloorLine({ t }: { t: number }) {
  const a = S.tat3[0], b = S.tat3[1] + 0.6;
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const x = (v: number) => 260 + ((v - 0.3) / 0.5) * 1400;
  const marks = [
    { v: T.hull.dice_mean, n: "visual hull", c: COL.fp }, { v: T.random.dice_mean, n: "אקראי", c: C.dim },
    { v: T.nn.dice_mean, n: "NN", c: C.warm }, { v: T.oracle.dice_mean, n: "oracle", c: C.good },
  ];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <div style={{ position: "absolute", left: x(0.3), width: x(0.8) - x(0.3), top: 520, height: 4, background: C.line }} />
      {[0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map((v) => (
        <div key={v} style={{ position: "absolute", left: x(v) - 30, width: 60, top: 540, textAlign: "center", color: C.dim, fontSize: 22 }}>{v.toFixed(1)}</div>
      ))}
      {marks.map((m, i) => (
        <div key={m.n} style={{ position: "absolute", left: x(m.v) - 80, width: 160, top: i % 2 ? 420 : 380, textAlign: "center", color: m.c, fontSize: 28, direction: "rtl" }}>
          {m.n}<div style={{ fontSize: 22, direction: "ltr" }}>{m.v.toFixed(2)}</div>
          <div style={{ margin: "6px auto 0", width: 3, height: i % 2 ? 50 : 90, background: m.c }} />
        </div>
      ))}
      <div style={{ position: "absolute", left: x(0.5), width: x(T.nn.dice_mean) - x(0.5), top: 600, height: 60, background: "rgba(255,110,120,0.15)",
        border: `1.5px solid ${COL.fp}`, borderRadius: 8, opacity: ramp(t, a + 1.5, a + 2.1) }} />
      <div style={{ position: "absolute", left: x(0.5) - 200, width: x(T.nn.dice_mean) - x(0.5) + 400, top: 680, textAlign: "center", color: COL.fp, fontSize: 28, direction: "rtl", opacity: ramp(t, a + 1.5, a + 2.1) }}>
        רשת כאן ≈ שינון
      </div>
      <div style={{ position: "absolute", left: x(T.oracle.dice_mean), width: x(0.8) - x(T.oracle.dice_mean), top: 600, height: 60, background: "rgba(125,255,176,0.15)",
        border: `1.5px solid ${C.good}`, borderRadius: 8, opacity: ramp(t, a + 6, a + 6.6) }} />
      <div style={{ position: "absolute", left: x(T.oracle.dice_mean) - 100, width: x(0.8) - x(T.oracle.dice_mean) + 200, top: 680, textAlign: "center", color: C.good, fontSize: 28, direction: "rtl", opacity: ramp(t, a + 6, a + 6.6) }}>
        כאן: באמת משחזרת
      </div>
      <div style={{ position: "absolute", top: 780, width: "100%", textAlign: "center", color: C.dim, fontSize: 24, direction: "rtl" }}>
        ממוצעי Dice מההדגמה הקטנה (27 נבדקים, leave-one-out)
      </div>
    </div>
  );
}

export const V08 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const hero = useObj("data/hero/pelvis.obj");
  const nnGeo = useObj("data/v08/pelvis_nn.obj");
  const vox = useJson<Vox>("data/v08/vox.json");
  const ready = hero && nnGeo && vox;
  const dim = Math.max(
    window01(t, S.idea1[0] - 0.3, S.q2[1] + 0.6, 0.6, 0.6),
    window01(t, S.cav1[0] - 0.3, S.tat3[1] + 0.6, 0.6, 0.6),
    window01(t, S.recap[0] - 0.3, tl.total, 0.8, 0.1),
  );

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ hero, nnGeo, vox }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.86 * dim, pointerEvents: "none" }} />

      <Opening n={8} title="רצפת השינון" active={[1, 3, 5]} t={t} end={2.9} />
      <Title text="האם הרשת משחזרת, או נזכרת?" t={t} a={S.intro2[0]} b={S.intro2[1] + 0.4} top={90} size={64} />

      <Title text="השכן הקרוב" sub="nearest neighbour: לשלוף את המטופל הכי דומה" t={t} a={S.idea2[0] + 0.2} b={S.idea2[1] + 0.3} top={20} size={44} />
      <Gallery t={t} />
      <QueryCard t={t} />
      <Label x={1630} y={700} size={28} color={C.dim} o={window01(t, S.demo1[0] + 0.5, S.demo1[1] + 0.4)}
        text={<span>27 נבדקים · AP + LAT<br />ריבוע של 352 מ"מ סביב מרכז האגן</span>} />

      <Title text="מרחק בין תמונות" t={t} a={S.dist1[0]} b={S.eq1[1] + 0.3} top={60} size={60} />
      <DownPanel t={t} />
      <Label x={960} y={760} size={36} color={C.warm} text={<span style={{ direction: "ltr", display: "inline-block" }}>2 × 32 × 32 = 2,048</span>}
        o={window01(t, S.dist2[0] + 5.0, S.dist2[1] + 0.3)} />
      <Eq t={t} a={S.eq1[0]} b={S.eq1[1] + 0.4} size={54} style={{ left: 560, top: 150, width: 800 }}
        tex={"d(\\hl{q}{q}, \\hl{x}{x^{(i)}}) = \\hl{sq}{\\sqrt{\\hl{sum}{\\sum_{p}} \\hl{df}{(q_p - x^{(i)}_p)^2}}}"}
        note="p רץ על כל הפיקסלים של AP ושל LAT"
        terms={{ q: [S.eq1[0] + 2.5, S.eq1[0] + 5], x: [S.eq1[0] + 2.5, S.eq1[0] + 5], df: [S.eq1[0] + 0.5, S.eq1[0] + 6.5],
          sum: [S.eq1[0] + 6.5, S.eq1[0] + 9.5], sq: [S.eq1[0] + 9.5, S.eq1[1]] }} />

      {/* the retrieved pelvis */}
      <Label x={560} y={220} size={40} color={C.bone} text={`האמת: ${H.query}`} o={window01(t, S.q3[0] + 0.5, S.m1[0] + 1.2)} />
      <Label x={1360} y={220} size={40} color={COL.nn} text={`השכן הקרוב: ${H.nn}`} o={window01(t, S.q3[0] + 0.5, S.m1[0] + 1.2)} />
      <Panel o={window01(t, S.m1[0] + 3.0, S.m3[1] + 0.4)} style={{ right: 90, top: 200, width: 520 }}>
        <div style={{ fontSize: 30, color: C.dim, marginBottom: 10 }}>{H.nn} מול {H.query}</div>
        <div style={{ fontSize: 46, direction: "ltr", textAlign: "right" }}>Dice = <b style={{ color: C.warm }}>{H.nn_dice.toFixed(3)}</b></div>
        <div style={{ fontSize: 46, direction: "ltr", textAlign: "right" }}>HD95 ≈ <b style={{ color: C.warm }}>{H.nn_hd95_mm.toFixed(1)} mm</b></div>
        <div style={{ fontSize: 22, color: C.dim, marginTop: 10 }}>מיושר לפי מרכז מסה · רשת 64³</div>
        <div style={{ opacity: ramp(t, S.m3[0] + 0.5, S.m3[0] + 1.1), marginTop: 22, fontSize: 30 }}>
          {[{ n: "השכן הקרוב", v: H.nn_dice, c: C.warm }, { n: "visual hull", v: m7.hull_dice, c: COL.fp }].map((r) => (
            <div key={r.n} style={{ display: "flex", alignItems: "center", gap: 12, margin: "8px 0" }}>
              <div style={{ width: 170 }}>{r.n}</div>
              <div style={{ height: 26, width: r.v * 400, background: r.c, borderRadius: 6 }} />
              <div style={{ direction: "ltr" }}>{r.v.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </Panel>
      <div style={{ position: "absolute", left: 110, top: 240, fontFamily: FONT, fontSize: 30, direction: "rtl", lineHeight: 1.7,
        opacity: window01(t, S.m2[0] + 0.5, S.m2[1] + 0.4) }}>
        <div style={{ color: COL.tp }}>■ משותף ({(vox?.tp.length ?? 0) / 3} וקסלים)</div>
        <div style={{ color: COL.fp }}>■ עודף ({(vox?.fp.length ?? 0) / 3})</div>
        <div style={{ color: COL.fn }}>■ חסר ({(vox?.fn.length ?? 0) / 3})</div>
      </div>

      <Title text="הצילום הכי דומה ≠ האגן הכי דומה" t={t} a={S.cav1[0]} b={S.cav1[1] + 0.3} top={20} size={44} />
      <Label x={1630} y={720} size={26} color={C.good} o={window01(t, S.cav1[0] + 6.5, S.cav1[1] + 0.4)}
        text={<span>Dice הכי גבוה: {H.oracle} ({H.oracle_dice.toFixed(2)})<br />מקום {H.oracle_rank} במרחק</span>} />
      <Scatter t={t} />
      <Bullets t={t} o={window01(t, S.cav3[0], S.cav3[1] + 0.5)} title="מה עוד נכנס למרחק בפיקסלים" style={{ right: 110, top: 220, width: 820 }}
        items={["כמות הרקמה הרכה", "הירכיים ועמוד השדרה", "איפה הסריקה נגמרת (טווח ה-CT)", "שיפור אפשרי: מרחק על features"]}
        times={[S.cav3[0] + 3.0, S.cav3[0] + 4.3, S.cav3[0] + 5.6, S.cav3[0] + 9.0]} />
      <div style={{ position: "absolute", left: 150, top: 260, display: "flex", gap: 20, opacity: window01(t, S.cav3[0] + 5.6, S.cav3[1] + 0.5), fontFamily: FONT, direction: "ltr" }}>
        {["s1358", "s0250"].map((s) => (
          <div key={s} style={{ textAlign: "center" }}>
            <Img src={staticFile(`data/v08/g/ap_${s}.png`)} style={{ width: 300, height: 300, borderRadius: 8, border: `1.5px solid ${C.line}` }} />
            <div style={{ color: C.dim, fontSize: 22 }}>{s}: הסריקה נגמרת בתוך התמונה</div>
          </div>
        ))}
      </div>

      <Title text="המחיר" t={t} a={S.cost1[0]} b={S.cost2[1] + 0.3} top={60} size={60} />
      <Bullets t={t} o={window01(t, S.cost1[0], S.cost2[1] + 0.5)} title="שכן קרוב" style={{ right: 260, left: 260, width: "auto", top: 220 }}
        items={["שומרים את כל המאגר: צילומים + אגנים", "כל שאילתה: השוואה לכל מטופל (26 כאן, מאות בניסוי)", "אבל: לא יכול להחזיר אגן שלא ראה. רק בוחר, לא מרכיב"]}
        times={[S.cost1[0] + 0.4, S.cost1[0] + 3.5, S.cost2[0] + 0.4]} />
      <Table t={t} />
      <PaperCard t={t} />
      <Title text="רצפת השינון" t={t} a={S.tat3[0]} b={S.tat3[1] + 0.3} top={120} size={60} />
      <FloorLine t={t} />

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "NN: משווים את הצילומים לכל המאגר, ומחזירים את האגן של הדומה ביותר",
        `מרחק L2 על DRR מוקטנים (32×32); על ${H.query}: Dice ${H.nn_dice.toFixed(2)}`,
        `LOO על 27: NN ${T.nn.dice_mean.toFixed(2)}, אקראי ${T.random.dice_mean.toFixed(2)}, oracle ${T.oracle.dice_mean.toFixed(2)}, hull ${T.hull.dice_mean.toFixed(2)}`,
        "Tatarchenko 2019: רשתות מתנהגות הרבה פעמים כמו שליפה",
        "הרשת שלנו צריכה לעבור את הרצפה הזאת בבירור",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/03-baselines/notebook.ipynb" todos={[
        "downsample + flatten: וקטור AP+LAT לכל מטופל",
        "nn_baseline(query, train): השכן הקרוב ב-L2, ומחזירים את המסכה שלו",
        "Dice + HD95 על סט הבדיקה (מיושר לפי מרכז מסה)",
        "טבלת התוצאות הראשונה: NN · אגן אקראי · visual hull",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro1[0]} />
      <Narration tl={tl} dir="v08" />
    </AbsoluteFill>
  );
};
