// Video 12 · A controlled experiment. One view vs two, variance over seeds, the second comparison,
// and how to present a results table honestly. All seed numbers here are SIMULATED and labelled so.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import script from "./script.json";
import durations from "../../public/audio/v12/durations.json";
import { buildTimeline, Script, span } from "../lib/timeline";
import { ramp, window01, lerp } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Detector, Meta, Rays, SourcePanel } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Box, IllusBadge, Table, Txt } from "./parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const A = (id: string) => S[id][0];
const B = (id: string) => S[id][1];

// ---------------- SIMULATED seed results (illustration only, never project results) ----------------
const ARM2 = [0.795, 0.823, 0.811]; // "AP + LAT", seeds 0,1,2
const ARM1 = [0.781, 0.802, 0.768]; // "AP only",  seeds 0,1,2
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const std = (a: number[]) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
// simulated per-patient paired differences (AP+LAT minus AP only), 20 test patients
const PAIRED = Array.from({ length: 20 }, (_, i) => 0.02 + 0.026 * Math.sin(i * 12.9898 + 1.7) * Math.cos(i * 4.1414));

// ---------------- 3D: the two-view rig (intro, and the LAT tilt in sec2) ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const box = boxOf(a.meta);
  const [cx, cy, cz] = box.c;
  const apZsrc = box.mn[2] - 1.4, apZdet = box.mx[2] + 0.3;
  const latXsrc = box.mx[0] + 1.4, latXdet = box.mn[0] - 0.3;
  const introOn = window01(t, 2.4, B("intro2") + 0.5, 1.0, 0.7);
  const latDim = ramp(t, A("intro2") + 1.0, A("intro2") + 2.2);
  const secOn = window01(t, A("sec2") - 0.2, B("sec2") + 0.6, 0.8, 0.6);
  const on = Math.max(introOn, secOn);
  const tilt = secOn > 0 ? ramp(t, A("sec2") + 4.0, B("sec2") - 0.8) : 0;
  const ang = (tilt * 10 * Math.PI) / 180;
  const latTex = tilt > 0.5 ? a.lat100 : a.lat90;
  const latSize: [number, number] = [2.05 * 1.5, 1.96 * 1.5]; // v12 LAT DRRs: 205 x 196 px at 1.5 mm
  const keys: CamKey[] = [
    { t: 0, pos: orbit(-0.8, 9.4, 1.4), look: [0, 0, 0.2] },
    { t: B("intro2"), pos: orbit(0.5, 9.0, 1.8), look: [0, 0, 0.2] },
    { t: A("sec2"), pos: [3.0, 9.5, -5.5], look: [cx, cy - 0.3, cz] },
    { t: B("sec2") + 0.6, pos: [1.5, 10.5, -4.0], look: [cx, cy - 0.3, cz] },
  ];
  if (on <= 0.001) return <CameraPath t={t} keys={keys} />;
  const latO = introOn * (1 - 0.82 * latDim) + secOn;
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.1} opacity={on * 0.8} />
      <Bone geo={a.pelvis} material={bone} opacity={on} />
      <Bone geo={a.femurs} material={bone} opacity={on * 0.85} />
      {/* AP rig */}
      <group position={[cx, cy, 0]}>
        <SourcePanel size={a.meta.ap_size} position={[0, 0, apZsrc]} opacity={on} />
        <Detector tex={a.ap} opacity={on} size={a.meta.ap_size} position={[0, 0, apZdet]} rotation={[0, Math.PI, 0]} />
      </group>
      <Rays segs={[[[cx, cy, apZsrc], [cx, cy, apZdet]]]} opacity={secOn} color={[1.4, 2.1, 3.0]} />
      {/* LAT rig, rotated about the vertical axis through the pelvis by the extra tilt */}
      <group position={[cx, 0, cz]} rotation={[0, -ang, 0]}>
        <group position={[-cx, cy, -cz]}>
          <SourcePanel size={secOn > 0 ? latSize : a.meta.lat_size} position={[latXsrc, 0, cz]} rotation={[0, Math.PI / 2, 0]} opacity={latO} />
          {secOn > 0
            ? <Detector tex={latTex} opacity={secOn} size={latSize} position={[latXdet, 0, cz]} rotation={[0, Math.PI / 2, 0]} flipX />
            : <Detector tex={a.lat} opacity={latO} size={a.meta.lat_size} position={[latXdet, 0, cz]} rotation={[0, Math.PI / 2, 0]} flipX />}
          <Rays segs={[[[latXsrc, 0, cz], [latXdet, 0, cz]]]} opacity={secOn} color={[3.0, 2.2, 0.9]} />
        </group>
      </group>
    </>
  );
}

type Assets = {
  meta: Meta; pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry;
  ap: THREE.Texture; lat: THREE.Texture; lat90: THREE.Texture; lat100: THREE.Texture;
};

// ---------------- 2D: the two arms ----------------
function Arms({ t }: { t: number }) {
  const [a0, a1] = span(tl, "ctrl1", "ctrl5");
  const o = window01(t, a0 - 0.2, a1 + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const zeroHl = window01(t, A("ctrl3"), B("ctrl3") + 0.3, 0.4, 0.4);
  const conf = ramp(t, A("ctrl4") + 1.8, A("ctrl4") + 2.6) * (1 - ramp(t, B("ctrl5") + 0.1, B("ctrl5") + 0.5));
  const knobs = ["ארכיטקטורה ומספר פרמטרים", "נתוני אימון ופיצול", "מספר צעדי אימון", "seeds", "augmentation ומדדים"];
  const knobT = (i: number) => A("ctrl2") + 3.0 + i * ((B("ctrl2") - A("ctrl2") - 3.4) / knobs.length);
  const arm = (x: number, title: string, two: boolean) => (
    <Box x={x} y={190} w={600} h={520} o={1} border={two ? C.xray : C.warm}>
      <div style={{ padding: "22px 30px", fontSize: 38, fontWeight: 700, color: two ? C.xray : C.warm }}>{title}</div>
      <div style={{ position: "absolute", top: 100, right: 40, display: "flex", gap: 22, direction: "rtl" }}>
        <div style={{ textAlign: "center" }}>
          <Img src={staticFile("data/hero/drr_ap.png")} style={{ height: 190, borderRadius: 8, border: `1px solid ${C.line}` }} />
          <div style={{ fontSize: 26, color: C.dim }}>AP</div>
        </div>
        <div style={{ textAlign: "center" }}>
          {two
            ? <Img src={staticFile("data/hero/drr_lat.png")} style={{ height: 190, borderRadius: 8, border: `1px solid ${C.line}` }} />
            : <div style={{ height: 190, width: 118, borderRadius: 8, background: "#000", border: `2px solid ${zeroHl > 0.3 ? C.warm : C.line}`,
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: 54, color: C.warm,
                boxShadow: `0 0 ${40 * zeroHl}px rgba(255,207,127,0.6)` }}>0</div>}
          <div style={{ fontSize: 26, color: two ? C.dim : C.warm }}>{two ? "LAT" : "LAT = אפסים"}</div>
        </div>
      </div>
      <div style={{ position: "absolute", top: 360, left: 40, right: 40, display: "flex", alignItems: "center", gap: 16, direction: "ltr", justifyContent: "center" }}>
        <div style={{ padding: "14px 22px", borderRadius: 12, border: `2px solid ${C.line}`, fontSize: 28 }}>U-Net 2D→3D</div>
        <div style={{ fontSize: 30, color: C.dim }}>→</div>
        <div style={{ padding: "14px 22px", borderRadius: 12, border: `2px solid ${C.line}`, fontSize: 28 }}>64³</div>
        <div style={{ fontSize: 30, color: C.dim }}>→</div>
        <div style={{ padding: "14px 22px", borderRadius: 12, border: `2px solid ${C.line}`, fontSize: 28 }}>Dice</div>
      </div>
      {two && conf > 0 && (
        <div style={{ position: "absolute", top: 445, left: 0, right: 0, textAlign: "center", fontSize: 30, color: C.bad, opacity: conf }}>
          + אימון ארוך פי 2 ⚠
        </div>
      )}
    </Box>
  );
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {arm(1180, "זרוע 1 · שני צילומים", true)}
      {arm(140, "זרוע 2 · צילום קדמי בלבד", false)}
      {/* the locked knobs between the arms */}
      <div style={{ position: "absolute", left: 760, top: 200, width: 400, fontFamily: FONT, direction: "rtl" }}>
        <div style={{ fontSize: 30, color: C.dim, textAlign: "center", marginBottom: 12, opacity: ramp(t, A("ctrl2"), A("ctrl2") + 0.6) }}>זהה בשתיהן</div>
        {knobs.map((k, i) => {
          const kk = ramp(t, knobT(i), knobT(i) + 0.5);
          const broken = i === 2 ? conf : 0;
          return (
            <div key={k} style={{ opacity: kk, display: "flex", gap: 12, alignItems: "center", fontSize: 27, color: broken > 0.5 ? C.bad : C.text,
              padding: "10px 16px", marginBottom: 10, borderRadius: 12, border: `1.5px solid ${broken > 0.5 ? C.bad : "rgba(125,255,176,0.5)"}`,
              background: "rgba(8,14,24,0.7)" }}>
              <span style={{ color: broken > 0.5 ? C.bad : C.good }}>{broken > 0.5 ? "✗" : "="}</span>{k}
            </div>
          );
        })}
        <div style={{ marginTop: 18, textAlign: "center", fontSize: 30, color: C.warm, opacity: ramp(t, A("ctrl2") + 1.0, A("ctrl2") + 1.6) }}>
          שונה: רק ה-LAT
        </div>
      </div>
      {/* the confound: two causes into one effect */}
      {conf > 0 && (
        <div style={{ position: "absolute", left: 560, top: 760, width: 800, height: 130, opacity: window01(t, A("ctrl4") + 3.0, B("ctrl5") + 0.5, 0.6, 0.4), fontFamily: FONT, direction: "rtl" }}>
          <div style={{ position: "absolute", right: 0, top: 0, padding: "12px 20px", border: `2px solid ${C.xray}`, borderRadius: 12, fontSize: 28, color: C.xray }}>צילום LAT</div>
          <div style={{ position: "absolute", right: 0, top: 70, padding: "12px 20px", border: `2px solid ${C.bad}`, borderRadius: 12, fontSize: 28, color: C.bad }}>אימון ×2</div>
          <svg style={{ position: "absolute", left: 190, top: 0 }} width={400} height={130}>
            <path d="M 400 28 C 250 28 200 65 60 65" stroke={C.xray} strokeWidth={3} fill="none" />
            <path d="M 400 98 C 250 98 200 65 60 65" stroke={C.bad} strokeWidth={3} fill="none" />
          </svg>
          <div style={{ position: "absolute", left: 0, top: 36, padding: "12px 20px", border: `2px solid ${C.warm}`, borderRadius: 12, fontSize: 30, color: C.warm, direction: "ltr" }}>Dice ↑ ?</div>
        </div>
      )}
    </div>
  );
}

// ---------------- 2D: seeds, dot plot ----------------
const X0 = 0.74, X1 = 0.85;
function DotPlot({ t }: { t: number }) {
  const [v0, v1] = span(tl, "var1", "var6");
  const o = window01(t, A("var1") - 0.2, v1 + 0.4, 0.6, 0.6) * (1 - window01(t, A("var2") - 0.2, B("var2") + 0.5, 0.4, 0.4));
  if (o <= 0 || v0 < 0) return null;
  const L = 300, R = 1560, px = (v: number) => L + ((v - X0) / (X1 - X0)) * (R - L);
  const rows = [{ y: 450, name: "AP + LAT", vals: ARM2, col: C.xray }, { y: 610, name: "AP בלבד", vals: ARM1, col: C.warm }];
  const all = ramp(t, A("var3") + 2.5, A("var3") + 4.0);
  const two = ramp(t, A("var1") + 1.5, A("var1") + 2.5);
  const pick = window01(t, A("var4") + 0.3, B("var4") + 0.4, 0.5, 0.5);
  const stat = ramp(t, A("var5") + 0.5, A("var5") + 2.0);
  const shown = (r: number, i: number) => (r === 0 && i < 2 ? Math.max(two, all) : all);
  const isPick = (r: number, i: number) => (r === 0 && i === 0) || (r === 1 && i === 1);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <line x1={L} y1={710} x2={R} y2={710} stroke={C.line} strokeWidth={2} />
        {[0.74, 0.76, 0.78, 0.80, 0.82, 0.84].map((v) => (
          <g key={v}>
            <line x1={px(v)} y1={706} x2={px(v)} y2={720} stroke={C.line} strokeWidth={2} />
            <text x={px(v)} y={755} fill={C.dim} fontSize={26} textAnchor="middle" fontFamily={FONT}>{v.toFixed(2)}</text>
            <line x1={px(v)} y1={380} x2={px(v)} y2={700} stroke="rgba(120,190,255,0.08)" strokeWidth={1} />
          </g>
        ))}
        <text x={(L + R) / 2} y={795} fill={C.dim} fontSize={28} textAnchor="middle" fontFamily={FONT}>Dice (מדומה)</text>
        {rows.map((row, r) => {
          const m = mean(row.vals), s = std(row.vals);
          return (
            <g key={r}>
              <text x={R + 40} y={row.y + 10} fill={row.col} fontSize={32} fontFamily={FONT} textAnchor="start">{row.name}</text>
              {stat > 0 && (
                <g opacity={stat}>
                  <rect x={px(m - s)} y={row.y - 46} width={px(m + s) - px(m - s)} height={92} fill={row.col} opacity={0.12} rx={10} />
                  <line x1={px(m - s)} y1={row.y} x2={px(m + s)} y2={row.y} stroke={row.col} strokeWidth={3} />
                  <line x1={px(m - s)} y1={row.y - 20} x2={px(m - s)} y2={row.y + 20} stroke={row.col} strokeWidth={3} />
                  <line x1={px(m + s)} y1={row.y - 20} x2={px(m + s)} y2={row.y + 20} stroke={row.col} strokeWidth={3} />
                  <line x1={px(m)} y1={row.y - 40} x2={px(m)} y2={row.y + 40} stroke="white" strokeWidth={4} />
                  <text x={px(m)} y={row.y - 58} fill="white" fontSize={28} textAnchor="middle" fontFamily={FONT}>
                    {m.toFixed(3)} ± {s.toFixed(3)}
                  </text>
                </g>
              )}
              {row.vals.map((v, i) => {
                const k = shown(r, i);
                if (k <= 0) return null;
                const dim = pick > 0 && !isPick(r, i) ? 1 - 0.75 * pick : 1;
                const big = isPick(r, i) ? 1 + 0.4 * pick : 1;
                return (
                  <g key={i} opacity={k * dim}>
                    <circle cx={px(v)} cy={row.y} r={16 * big} fill={row.col} stroke="white" strokeWidth={isPick(r, i) ? 3 * pick : 0} />
                    <text x={px(v)} y={row.y + 48} fill={C.dim} fontSize={22} textAnchor="middle" fontFamily={FONT}>seed {i}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        {pick > 0 && <line x1={px(ARM2[0])} y1={450} x2={px(ARM1[1])} y2={610} stroke={C.bad} strokeWidth={3} strokeDasharray="10 8" opacity={pick} />}
      </svg>
      {pick > 0 && <Txt x={460} y={300} w={1000} o={pick} align="center" size={36} color={C.bad}>
        ריצה אחת מכל זרוע: <span style={{ direction: "ltr", display: "inline-block" }}>AP only ({ARM1[1].toFixed(3)}) &gt; AP + LAT ({ARM2[0].toFixed(3)}) ?!</span></Txt>}
      <Txt x={260} y={300} w={1400} o={window01(t, A("var1") + 0.3, B("var1") + 0.4)} align="center" size={34} color={C.dim}>
        אותה זרוע, אותו קוד, seed שונה ← Dice שונה</Txt>
    </div>
  );
}

function PairedBars({ t }: { t: number }) {
  const o = window01(t, A("var7") + 0.2, B("var7") + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const L = 360, R = 1560, y0 = 700, sc = 5200;
  const w = (R - L) / PAIRED.length;
  const pos = PAIRED.filter((d) => d > 0).length;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <line x1={L - 20} y1={y0} x2={R + 20} y2={y0} stroke={C.line} strokeWidth={2} />
        {PAIRED.map((d, i) => {
          const k = ramp(t, A("var7") + 3 + i * 0.12, A("var7") + 3.4 + i * 0.12);
          const h = d * sc * k;
          return <rect key={i} x={L + i * w + 8} y={d > 0 ? y0 - h : y0} width={w - 16} height={Math.abs(h)} rx={6}
            fill={d > 0 ? C.good : C.bad} opacity={0.85} />;
        })}
        <text x={R + 30} y={y0 + 8} fill={C.dim} fontSize={26} fontFamily={FONT}>0</text>
        <text x={(L + R) / 2} y={y0 + 60} fill={C.dim} fontSize={28} textAnchor="middle" fontFamily={FONT}>מטופלי בדיקה 1 … 20</text>
      </svg>
      <Txt x={360} y={330} w={1200} align="center" size={32}>
        הפרש Dice לכל מטופל: <span style={{ direction: "ltr", display: "inline-block" }}>(AP + LAT) − (AP בלבד)</span></Txt>
      <Txt x={560} y={810} w={800} o={ramp(t, A("var7") + 6.5, A("var7") + 7.3)} align="center" size={34} color={C.good}>
        {pos} מתוך {PAIRED.length} מטופלים: הפרש חיובי</Txt>
    </div>
  );
}

// ---------------- 2D: the second comparison ----------------
function SecondCmp({ t }: { t: number }) {
  const o1 = window01(t, A("sec1"), B("sec1") + 0.4, 0.5, 0.5);
  const o3 = window01(t, A("sec3") - 0.1, B("sec4") + 0.4, 0.6, 0.6);
  const o5 = window01(t, A("sec5"), B("sec6") + 0.5, 0.6, 0.6);
  const card = (x: number, title: string, q: string, k: number) => (
    <Box x={x} y={330} w={720} h={360} o={k} border={C.xray}>
      <div style={{ padding: "34px 40px" }}>
        <div style={{ fontSize: 44, fontWeight: 700, color: C.xray }}>{title}</div>
        <div style={{ fontSize: 34, color: C.text, marginTop: 30, lineHeight: 1.5 }}>{q}</div>
      </div>
    </Box>
  );
  const angs = [90, 95, 100];
  const q4 = ramp(t, A("sec4"), A("sec4") + 0.8);
  const grid = (x: number, n: number, k: number, label: string) => {
    const cols = 20, s = 22;
    return (
      <div style={{ position: "absolute", left: x, top: 260, width: cols * s, opacity: k }}>
        <div style={{ fontSize: 40, color: C.warm, fontWeight: 700, textAlign: "center", marginBottom: 16, fontFamily: FONT, direction: "rtl" }}>{label}</div>
        <div style={{ display: "flex", flexWrap: "wrap", width: cols * s, direction: "ltr" }}>
          {Array.from({ length: n }, (_, i) => (
            <div key={i} style={{ width: s - 5, height: s - 5, margin: 2.5, borderRadius: 4, background: "rgba(127,208,255,0.55)",
              opacity: ramp(t, A("sec5") + 1 + (i / n) * 2.5, A("sec5") + 1.3 + (i / n) * 2.5) }} />
          ))}
        </div>
      </div>
    );
  };
  return (
    <>
      {o1 > 0 && <div style={{ position: "absolute", inset: 0, opacity: o1 }}>
        {card(1020, "1 · זווית ה-LAT", "LAT ב-91° עד 100° במקום 90° בדיוק", ramp(t, A("sec1") + 1.5, A("sec1") + 2.1))}
        {card(180, "2 · כמות הנתונים", "100 מול 400 נפחי אימון", ramp(t, A("sec1") + 2.3, A("sec1") + 2.9))}
      </div>}
      {o3 > 0 && <div style={{ position: "absolute", inset: 0, opacity: o3 }}>
        {angs.map((a, i) => (
          <div key={a} style={{ position: "absolute", left: 1270 - i * 520, top: 200, opacity: ramp(t, A("sec3") + 1 + i * 0.8, A("sec3") + 1.6 + i * 0.8) }}>
            <Img src={staticFile(`data/v12/lat_${a}.png`)} style={{ height: 460, display: "block", borderRadius: 10, border: `2px solid ${C.line}` }} />
            <div style={{ position: "absolute", left: 110, top: 245, width: 200, height: 170, borderRadius: "50%",
              border: `3px solid ${C.warm}`, opacity: ramp(t, A("sec3") + 5.5, A("sec3") + 6.3) }} />
            <div style={{ textAlign: "center", fontSize: 40, color: C.xray, marginTop: 10, fontFamily: FONT }}>{a}°</div>
          </div>
        ))}
        <Txt x={260} y={790} w={1400} align="center" o={q4} size={36} color={C.warm}>
          אימון: 90° בדיוק ← בדיקה: 95°, 100°. כמה הדיוק יורד?</Txt>
      </div>}
      {o5 > 0 && <div style={{ position: "absolute", inset: 0, opacity: o5 }}>
        {grid(1240, 100, ramp(t, A("sec5") + 0.6, A("sec5") + 1.2), "100 נפחים")}
        {grid(240, 400, ramp(t, A("sec5") + 0.6, A("sec5") + 1.2), "400 נפחים")}
        <Txt x={740} y={300} w={440} align="center" o={ramp(t, A("sec5") + 4, A("sec5") + 4.6)} size={30} color={C.dim}>
          קבועים: קבוצת הבדיקה · הרשת · מספר הצעדים</Txt>
        <Scenarios t={t} />
      </div>}
    </>
  );
}

/** two qualitative outcomes for 100 vs 400 (no numbers: the shape of the answer only) */
function Scenarios({ t }: { t: number }) {
  const k = ramp(t, A("sec6") + 0.3, A("sec6") + 1.2);
  if (k <= 0) return null;
  const x0 = 1020, w = 380, y0 = 860, h = 170;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: k }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <line x1={x0} y1={y0} x2={x0 + w} y2={y0} stroke={C.line} strokeWidth={2} />
        <line x1={x0} y1={y0} x2={x0} y2={y0 - h} stroke={C.line} strokeWidth={2} />
        <text x={x0 + 40} y={y0 + 32} fill={C.dim} fontSize={24} fontFamily={FONT} textAnchor="middle">100</text>
        <text x={x0 + w - 40} y={y0 + 32} fill={C.dim} fontSize={24} fontFamily={FONT} textAnchor="middle">400</text>
        <text x={x0 + w + 16} y={y0 - h + 20} fill={C.dim} fontSize={24} fontFamily={FONT} textAnchor="start">Dice</text>
        <line x1={x0 + 40} y1={y0 - 50} x2={x0 + w - 40} y2={y0 - 150} stroke={C.good} strokeWidth={4} strokeDasharray="12 8" />
        <line x1={x0 + 40} y1={y0 - 60} x2={x0 + w - 40} y2={y0 - 66} stroke={C.warm} strokeWidth={4} strokeDasharray="12 8" />
      </svg>
      <Txt x={560} y={y0 - 175} w={440} size={28} color={C.good} align="right">עולה ← הרשת מוגבלת בנתונים</Txt>
      <Txt x={560} y={y0 - 95} w={440} size={28} color={C.warm} align="right">שטוח ← עוד נתונים לא יעזרו</Txt>
      <Txt x={560} y={y0 - 30} w={440} size={22} color={C.dim} align="right">(צורה של תשובה, לא תוצאה)</Txt>
    </div>
  );
}

// ---------------- 2D: the results table ----------------
function ResultsTable({ t }: { t: number }) {
  const [t0, t1] = span(tl, "tab1", "tab5");
  const o = window01(t, t0 - 0.1, t1 + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const cellK = window01(t, A("tab2") + 0.5, B("tab2") + 0.3, 0.4, 0.4);
  const rows = ["AP + LAT (90°)", "AP בלבד", "השוואה 2: ___"];
  const cols = ["Dice ↑", "HD95 (mm) ↓", "ASSD (mm) ↓"];
  const cells = rows.map(() => cols.map(() => "— ± —"));
  const fixedO = ramp(t, A("tab5") + 0.3, A("tab5") + 1.0);
  const rules = [
    { k: ramp(t, A("tab3") + 0.2, A("tab3") + 0.8), s: "מדגישים רק הבדל גדול מהפיזור" },
    { k: ramp(t, A("tab3") + 3.2, A("tab3") + 3.8), s: "כל המדדים, גם כשהם לא לטובתכם" },
    { k: ramp(t, A("tab4") + 0.2, A("tab4") + 0.8), s: "test נפתח פעם אחת, בסוף" },
    { k: ramp(t, A("tab4") + 3.4, A("tab4") + 4.0), s: "כל בחירה: על validation" },
  ];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Table x={700} y={210} w={1100} cols={cols} rows={rows} cells={cells} colW={330}
        rowO={[1, 1, ramp(t, A("tab1") + 2.5, A("tab1") + 3.2)]}
        hl={[0, 1, 2].map((c) => ({ r: 0, c, k: cellK }))}
        title={<span style={{ opacity: ramp(t, A("tab2") + 3, A("tab2") + 3.8) }}>ממוצע ± סטיית תקן על פני 3 seeds · n = ___ מטופלי בדיקה (אותם מטופלים בכל השורות)</span>}
        foot={<span style={{ opacity: fixedO }}>קבוע בכל השורות: אותה ארכיטקטורה (___ פרמטרים) · אותו פיצול · ___ צעדי אימון · seeds 0, 1, 2 · הסף נבחר על validation</span>} />
      {cellK > 0 && <Txt x={700} y={660} w={1100} o={cellK} align="center" size={34} color={C.warm}>
        בכל תא: <span style={{ direction: "ltr", display: "inline-block" }}>mean ± std</span> (כאן יופיעו המספרים שלכם)</Txt>}
      <Box x={110} y={250} w={520} o={ramp(t, A("tab3"), A("tab3") + 0.6)}>
        <div style={{ padding: "26px 30px" }}>
          <div style={{ fontSize: 34, fontWeight: 700, color: C.xray, marginBottom: 14 }}>כללי יושר</div>
          {rules.map((r, i) => (
            <div key={i} style={{ fontSize: 28, lineHeight: 1.5, marginBottom: 12, opacity: r.k, display: "flex", gap: 12 }}>
              <span style={{ color: C.good }}>✓</span>{r.s}</div>
          ))}
        </div>
      </Box>
      <SplitBar t={t} />
    </div>
  );
}

function SplitBar({ t }: { t: number }) {
  const k = window01(t, A("tab4") + 0.4, B("tab4") + 0.6, 0.5, 0.5);
  if (k <= 0) return null;
  const x = 700, y = 770, w = 1100;
  const seg = (from: number, to: number, label: string, col: string, lock = false) => (
    <div style={{ position: "absolute", left: x + w * (1 - to), top: y, width: w * (to - from) - 6, height: 70, borderRadius: 12,
      border: `2px solid ${col}`, background: `${col}22`, display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 28, color: col, fontFamily: FONT, direction: "rtl", gap: 10 }}>{lock && "🔒"} {label}</div>
  );
  return (
    <div style={{ position: "absolute", inset: 0, opacity: k }}>
      {seg(0, 0.7, "train 70%", C.xray)}
      {seg(0.7, 0.8, "val 10%", C.good)}
      {seg(0.8, 1.0, "test 20%", C.warm, true)}
    </div>
  );
}

/** var2: the sources of randomness */
function Randomness({ t }: { t: number }) {
  const o = window01(t, A("var2"), B("var2") + 0.4, 0.5, 0.5);
  if (o <= 0) return null;
  const items = ["משקולות התחלתיות", "סדר הדוגמאות בכל epoch", "augmentation", "dropout"];
  const step = (B("var2") - A("var2") - 4) / items.length;
  return (
    <div style={{ position: "absolute", left: 200, right: 200, top: 400, opacity: o, display: "flex", gap: 24, justifyContent: "center", direction: "rtl" }}>
      {items.map((s, i) => (
        <div key={s} style={{ opacity: ramp(t, A("var2") + 0.8 + i * step, A("var2") + 1.3 + i * step), padding: "22px 26px", borderRadius: 16,
          border: `2px solid ${C.line}`, background: C.panel, fontFamily: FONT, fontSize: 32, color: C.text }}>{s}</div>
      ))}
      <div style={{ position: "absolute", top: 150, left: 0, right: 0, textAlign: "center", fontFamily: FONT, fontSize: 38, color: C.warm,
        opacity: ramp(t, B("var2") - 3.5, B("var2") - 2.8) }}>
        seed = 0, 1, 2 ← כל אחד נותן אימון קצת אחר</div>
    </div>
  );
}

export const V12 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const meta = useJson<Meta>("data/hero/meta.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const lat = useTex("data/hero/drr_lat.png");
  const lat90 = useTex("data/v12/lat_90.png");
  const lat100 = useTex("data/v12/lat_100.png");
  const ready = meta && pelvis && femurs && ap && lat && lat90 && lat100;
  const dim2d = Math.max(
    window01(t, A("ctrl1") - 0.3, B("sec1") + 0.4, 0.6, 0.6),
    window01(t, A("sec3") - 0.3, tl.total + 1, 0.6, 0.1),
  );
  const varTerms: Record<string, [number, number]> = {
    m: [A("var5") + 0.4, A("var5") + 3.5], s: [A("var5") + 3.5, B("var5")], n: [A("var6") + 0.3, A("var6") + 4.5],
  };

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ meta, pelvis, femurs, ap, lat, lat90, lat100 }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.9 * dim2d, pointerEvents: "none" }} />

      <Opening n={12} title="ניסוי מבוקר" active={[5]} t={t} end={2.9} />

      <Title text="כמה שווה הצילום השני?" sub="תצפית אחת (AP) מול שתיים (AP + LAT)" t={t} a={A("intro2")} b={B("intro2") + 0.5} />
      <Label x={1450} y={640} text="?" color={C.warm} size={120} o={window01(t, A("intro2") + 1.6, B("intro2") + 0.5)} />

      <Title text="משנים משתנה אחד" t={t} a={A("ctrl1")} b={B("ctrl3") + 0.3} top={60} size={60} />
      <Title text="משתנה מבלבל" sub="confound: שני שינויים יחד, ההשפעות מתערבבות" t={t} a={A("ctrl4")} b={B("ctrl5") + 0.4} top={40} size={56} />
      <Arms t={t} />

      <Title text="ריצה אחת לא מוכיחה כלום" t={t} a={A("var1")} b={B("var4") + 0.3} top={70} size={58} />
      <Title text="ממוצע ± סטיית תקן" t={t} a={A("var5")} b={B("var6") + 0.3} top={70} size={58} />
      <IllusBadge o={window01(t, A("var3"), B("var7") + 0.4)} y={225} />
      <Randomness t={t} />
      <DotPlot t={t} />
      <Eq t={t} a={A("var6") - 0.2} b={B("var6") + 0.3} size={38} style={{ left: 560, top: 815, width: 800, padding: "0px 30px" }}
        tex={String.raw`\hl{m}{\bar d = \tfrac{1}{n}\textstyle\sum_i d_i}\qquad \hl{s}{s} = \sqrt{\tfrac{1}{\hl{n}{n-1}}\textstyle\sum_i (d_i-\bar d)^2}`}
        terms={varTerms} />
      <Title text="השוואה בזוגות" sub="אותם מטופלי בדיקה בשתי הזרועות" t={t} a={A("var7")} b={B("var7") + 0.4} top={70} size={58} />
      <PairedBars t={t} />

      <Title text="ההשוואה השנייה" t={t} a={A("sec1")} b={B("sec1") + 0.3} top={130} />
      <SecondCmp t={t} />
      <Label x={960} y={170} size={64} color={C.warm} o={window01(t, A("sec2") + 3.6, B("sec2") + 0.5)}
        text={<span style={{ direction: "ltr", display: "inline-block" }}>{`${Math.round(lerp(90, 100, ramp(t, A("sec2") + 4.0, B("sec2") - 0.8)))}°`}</span>} />
      <Label x={960} y={90} size={34} color={C.dim} o={window01(t, A("sec2") + 0.6, B("sec2") + 0.5)} text="הזווית בין קרן ה-AP לקרן ה-LAT" />
      <Title text="הזווית של ה-LAT" t={t} a={A("sec3")} b={B("sec4") + 0.3} top={60} size={56} />
      <Title text="100 מול 400 נפחי אימון" t={t} a={A("sec5")} b={B("sec6") + 0.3} top={70} size={56} />

      <Title text="טבלת תוצאות, ביושר" t={t} a={A("tab1")} b={B("tab5") + 0.3} top={50} size={58} />
      <ResultsTable t={t} />

      <RecapCard t={t} a={A("recap")} b={B("recap") + 0.3} items={[
        "משנים משתנה אחד; כל השאר קבוע (אחרת: משתנה מבלבל)",
        "ריצה אחת לא מוכיחה כלום: 3 seeds, ממוצע ± סטיית תקן",
        "השוואה בזוגות: אותם מטופלים, הפרש לכל מטופל",
        "ההשוואה השנייה: זווית LAT 91–100°, או 100 מול 400 נפחים",
        "בוחרים על validation; test נפתח פעם אחת",
      ]} />
      <YourTurn t={t} a={A("turn")} b={tl.total} notebook="units/06-experiments/notebook.ipynb" todos={[
        "להריץ AP + LAT ו-AP בלבד (LAT = אפסים), 3 seeds לכל זרוע",
        "לחשב ממוצע ± std והפרש בזוגות לכל מטופל",
        "לבחור ולהריץ את ההשוואה השנייה (זווית LAT או 100/400)",
        "למלא את טבלת התוצאות, עם מה שהיה קבוע",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={A("intro1")} />
      <Narration tl={tl} dir="v12" />
    </AbsoluteFill>
  );
};
