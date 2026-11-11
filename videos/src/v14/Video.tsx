// Video 14 · Putting it together. Pipeline fly-through on the hero pelvis (GT only: the students'
// prediction does not exist yet, and the video says so), the results story as a PLAN with empty slots,
// the report, the class demo, and a recap of the whole series. No result numbers appear here.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { ReactNode } from "react";
import script from "./script.json";
import durations from "../../public/audio/v14/durations.json";
import { buildTimeline, Script, span } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, orbit, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Detector, Meta, ScanBeam, SlicePlane, SourcePanel, Voxels, WireBox } from "../lib/xray";
import { Captions, Credit, Label, Narration, Opening, PipelineMap, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Box, Table, Txt } from "../v12/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const A = (id: string) => S[id][0];
const B = (id: string) => S[id][1];

type Occ = { n: number; cell: number; count: number; centers: number[] };

// ---------------- 3D: the fly-through ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const box = boxOf(a.meta);
  const [cx, cy, cz] = box.c;
  const apZsrc = box.mn[2] - 1.4, apZdet = box.mx[2] + 0.3;
  const latXsrc = box.mx[0] + 1.4, latXdet = box.mn[0] - 0.3;
  const [f0, f6] = span(tl, "fly1", "fly6");
  const introOn = window01(t, 2.4, B("intro2") + 0.6, 1.0, 0.8);
  const ctOn = window01(t, A("fly1"), A("fly2") + 0.8, 0.8, 0.6);
  const rigOn = window01(t, A("fly2"), B("fly3") + 0.4, 0.8, 0.6);
  const apScan = ramp(t, A("fly2") + 0.4, A("fly2") + 2.4);
  const latScan = ramp(t, A("fly2") + 2.0, B("fly2"));
  const voxOn = window01(t, A("fly4"), B("fly6") + 0.6, 0.6, 0.8);
  const voxBuild = ramp(t, A("fly4") + 0.3, B("fly4") - 0.5);
  const meshK = ramp(t, A("fly6") + 0.3, A("fly6") + 2.3);
  const slot = window01(t, A("fly5") + 1.5, B("fly6") + 0.6, 0.8, 0.8);
  const boneO = introOn + ctOn * 0.4 + rigOn * 0.35;
  const sep = ramp(t, A("fly6") + 0.3, A("fly6") + 2.3) * 2.3;
  const keys: CamKey[] = [
    { t: 0, pos: orbit(-0.9, 9.2, 0.9) },
    { t: B("intro2"), pos: orbit(0.9, 9.6, 1.0) },
    { t: A("fly1") + 1.0, pos: orbit(0.7, 8.6, 2.0), look: [cx, cy, cz] },
    { t: A("fly2") + 1.0, pos: [6.6, 4.4, -7.6], look: [-0.3, -0.1, 0.2] },
    { t: B("fly2"), pos: [7.2, 3.6, -7.0], look: [-0.3, -0.1, 0.2] },
    { t: A("fly4") + 0.5, pos: orbit(-0.4, 7.4, 1.6) },
    { t: B("fly5"), pos: orbit(0.4, 8.0, 1.4) },
    { t: B("fly6"), pos: orbit(0.0, 10.5, 1.4), look: [0, 0, 0] },
    { t: f6 + 1, pos: orbit(0.0, 10.5, 1.4), look: [0, 0, 0] },
  ];
  if (t > f6 + 1.2) return <CameraPath t={t} keys={keys} />;
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.1} opacity={0.9} />
      <Bone geo={a.pelvis} material={bone} opacity={Math.min(1, boneO) * (1 - voxOn)} />
      <Bone geo={a.femurs} material={bone} opacity={Math.min(0.85, boneO) * (1 - voxOn)} />
      <SlicePlane tex={a.coronal} size={[box.s[0], box.s[1]]} position={[cx, cy, 0]} rotation={[0, Math.PI, 0]} opacity={ctOn * 0.9} />
      <group position={[0, cy, cz]} scale={[-1, 1, 1]}>
        <SlicePlane tex={a.sagittal} size={[box.s[2], box.s[1]]} position={[0, 0, 0]} rotation={[0, -Math.PI / 2, 0]} opacity={ctOn * 0.9} />
      </group>
      <group position={[cx, cy, 0]}>
        <SourcePanel size={a.meta.ap_size} position={[0, 0, apZsrc]} opacity={rigOn} />
        <ScanBeam axis="ap" progress={apScan} opacity={rigOn} w={a.meta.ap_size[0]} h={a.meta.ap_size[1]} src={apZsrc} det={apZdet} bodyIn={box.mn[2]} bodyOut={box.mx[2]} />
        <Detector tex={a.ap} progress={apScan} opacity={rigOn} size={a.meta.ap_size} position={[0, 0, apZdet]} rotation={[0, Math.PI, 0]} />
      </group>
      <group position={[0, cy, cz]}>
        <SourcePanel size={a.meta.lat_size} position={[latXsrc, 0, 0]} rotation={[0, Math.PI / 2, 0]} opacity={rigOn} />
        <ScanBeam axis="lat" progress={latScan} opacity={rigOn} w={a.meta.lat_size[0]} h={a.meta.lat_size[1]} src={latXsrc} det={latXdet} bodyIn={box.mx[0]} bodyOut={box.mn[0]} />
        <Detector tex={a.lat} progress={latScan} opacity={rigOn} size={a.meta.lat_size} position={[latXdet, 0, 0]} rotation={[0, Math.PI / 2, 0]} flipX />
      </group>
      <group position={[sep, 0, 0]}>
        <Voxels centers={a.occ64.centers} cell={a.occ64.cell} reveal={voxBuild} opacity={voxOn} />
      </group>
      {/* GT mesh (left) next to the empty slot where the prediction will go (right) */}
      <Bone geo={a.pelvis} material={bone} opacity={voxOn * meshK} position={[-sep, 0, 0]} />
      <WireBox size={[3.3, 3.3, 3.3]} position={[sep, 0, 0]} opacity={slot * 0.9} color="#ffcf7f" />
    </>
  );
}

type Assets = {
  meta: Meta; pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry; ap: THREE.Texture; lat: THREE.Texture;
  coronal: THREE.Texture; sagittal: THREE.Texture; occ64: Occ;
};

// ---------------- 2D: network sketch (fly3) ----------------
function NetSketch({ t }: { t: number }) {
  const o = window01(t, A("fly3") - 0.1, B("fly3") + 0.3, 0.5, 0.4);
  if (o <= 0) return null;
  const k = (i: number) => ramp(t, A("fly3") + 0.4 + i * 0.5, A("fly3") + 0.8 + i * 0.5);
  const block = (x: number, h: number, w: number, i: number, color: string, label?: string) => (
    <div style={{ position: "absolute", left: x, top: 540 - h / 2, width: w, height: h, borderRadius: 10, opacity: k(i),
      border: `2px solid ${color}`, background: `${color}1f`, boxShadow: `0 0 30px ${color}40` }}>
      {label && <div style={{ position: "absolute", bottom: -46, width: 300, left: w / 2 - 150, textAlign: "center", fontSize: 26, color: C.dim }}>{label}</div>}
    </div>
  );
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT, background: "rgba(5,7,11,0.82)" }}>
      <Img src={staticFile("data/hero/drr_ap.png")} style={{ position: "absolute", left: 160, top: 320, height: 190, borderRadius: 8, border: `1px solid ${C.line}` }} />
      <Img src={staticFile("data/hero/drr_lat.png")} style={{ position: "absolute", left: 180, top: 560, height: 190, borderRadius: 8, border: `1px solid ${C.line}` }} />
      {block(440, 300, 50, 0, C.xray)}{block(510, 230, 60, 1, C.xray)}{block(590, 160, 70, 2, C.xray, "מקודד 2D")}
      <div style={{ position: "absolute", left: 690, top: 515, fontSize: 40, color: C.dim, opacity: k(3) }}>→</div>
      <div style={{ position: "absolute", left: 750, top: 430, width: 220, height: 220, perspective: 600, opacity: k(3) }}>
        <div style={{ width: 140, height: 140, margin: 40, border: `2px solid ${C.good}`, transform: "rotateX(-20deg) rotateY(35deg)", background: "rgba(125,255,176,0.12)" }} />
        <div style={{ textAlign: "center", fontSize: 26, color: C.good, direction: "rtl", marginTop: -20 }}>הרמה לתלת-ממד</div>
      </div>
      <div style={{ position: "absolute", left: 990, top: 515, fontSize: 40, color: C.dim, opacity: k(4) }}>→</div>
      {block(1060, 90, 90, 4, C.warm)}{block(1170, 160, 90, 5, C.warm)}{block(1280, 230, 90, 6, C.warm, "מפענח 3D")}
      <div style={{ position: "absolute", left: 1395, top: 515, fontSize: 40, color: C.dim, opacity: k(7) }}>→</div>
      <div style={{ position: "absolute", left: 1460, top: 430, width: 240, height: 240, perspective: 600, opacity: k(7) }}>
        <div style={{ width: 160, height: 160, margin: 40, border: `2px solid ${C.warm}`, transform: "rotateX(-20deg) rotateY(35deg)", background: "rgba(255,207,127,0.14)" }} />
        <div style={{ textAlign: "center", fontSize: 28, color: C.warm, direction: "ltr", marginTop: -20 }}>64 × 64 × 64</div>
      </div>
    </div>
  );
}

// ---------------- 2D: the results story (plan, empty slots) ----------------
const CH = [
  { id: "res2", t: "1 · כמה הרשת למדה", s: "baselines מול הרשת" },
  { id: "res3", t: "2 · כמה שווה ה-LAT", s: "1 מול 2 תצפיות, 3 seeds" },
  { id: "res4", t: "3 · Dice מול זווית קלינית", s: "האם מסכימים?" },
  { id: "res5", t: "4 · מדומה מול אמיתי", s: "הפער על DeepFluoro" },
];
function Story({ t }: { t: number }) {
  const [r0, r1] = span(tl, "res1", "res6");
  const o = window01(t, r0 - 0.1, r1 + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const cur = CH.findIndex((c) => t >= A(c.id) - 0.3 && t < B(c.id) + 0.4);
  const vis = (id: string) => window01(t, A(id) - 0.1, B(id) + 0.5, 0.5, 0.4);
  const slot = (x: number, y: number, w: number, h: number, label: ReactNode, col: string) => (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, border: `3px dashed ${col}`, borderRadius: 10,
      display: "flex", alignItems: "center", justifyContent: "center", color: col, fontSize: 40, background: `${col}10` }}>{label}</div>
  );
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      {/* chapter list (right) */}
      <div style={{ position: "absolute", right: 90, top: 230, width: 520, direction: "rtl" }}>
        {CH.map((c, i) => {
          const on = i === cur;
          return <div key={c.id} style={{ marginBottom: 18, padding: "18px 26px", borderRadius: 16, border: `2px solid ${on ? C.warm : C.line}`,
            background: on ? "rgba(255,207,127,0.10)" : C.panel, opacity: ramp(t, A("res1b") + 1.0 + i * 0.6, A("res1b") + 1.5 + i * 0.6) }}>
            <div style={{ fontSize: 34, fontWeight: 700, color: on ? C.warm : C.text }}>{c.t}</div>
            <div style={{ fontSize: 26, color: C.dim }}>{c.s}</div>
          </div>;
        })}
      </div>
      <Txt x={90} y={420} w={1180} align="center" size={52} color={C.xray} o={window01(t, A("res1b") + 0.5, B("res1b") + 0.3)}>
        שאלה ← ניסוי ← מספר</Txt>
      <div style={{ position: "absolute", left: 90, top: 230, width: 1180, height: 640 }}>
        {/* ch1: bars with empty slots */}
        <div style={{ position: "absolute", inset: 0, opacity: vis("res2") }}>
          {["nearest neighbour", "visual hull", "U-Net (AP + LAT)"].map((l, i) => (
            <div key={l}>
              {slot(160 + i * 330, 120, 200, 380, "?", i === 2 ? C.xray : C.dim)}
              <div style={{ position: "absolute", left: 100 + i * 330, top: 520, width: 320, textAlign: "center", fontSize: 28, color: C.text }}>{l}</div>
            </div>
          ))}
          <div style={{ position: "absolute", left: 0, top: 40, width: 1180, textAlign: "center", fontSize: 30, color: C.dim, direction: "rtl" }}>Dice על קבוצת הבדיקה, אותם מטופלים</div>
        </div>
        {/* ch2: the two arms + the second comparison */}
        <div style={{ position: "absolute", inset: 0, opacity: vis("res3") }}>
          <Table x={40} y={120} w={1100} cols={["Dice", "HD95 (mm)"]} rows={["AP + LAT", "AP בלבד", "השוואה 2"]}
            cells={[["— ± —", "— ± —"], ["— ± —", "— ± —"], ["— ± —", "— ± —"]]} colW={360} title="ממוצע ± std על פני 3 seeds" />
        </div>
        {/* ch3: empty scatter */}
        <div style={{ position: "absolute", inset: 0, opacity: vis("res4") }}>
          <svg width={1180} height={640} style={{ position: "absolute", inset: 0 }}>
            <line x1={200} y1={540} x2={1080} y2={540} stroke={C.line} strokeWidth={3} />
            <line x1={200} y1={540} x2={200} y2={80} stroke={C.line} strokeWidth={3} />
            <text x={640} y={600} fill={C.dim} fontSize={30} textAnchor="middle" fontFamily={FONT}>Dice</text>
            <text x={170} y={310} fill={C.dim} fontSize={28} textAnchor="end" fontFamily={FONT}>שגיאת זווית</text>
          </svg>
          {slot(420, 200, 440, 220, "נקודה לכל מטופל", C.dim)}
        </div>
        {/* ch4: sim vs real */}
        <div style={{ position: "absolute", inset: 0, opacity: vis("res5") }}>
          {slot(300, 120, 220, 380, "?", C.xray)}
          {slot(680, 120, 220, 380, "?", C.warm)}
          <div style={{ position: "absolute", left: 260, top: 520, width: 300, textAlign: "center", fontSize: 30, color: C.xray }}>DRR תואם</div>
          <div style={{ position: "absolute", left: 640, top: 520, width: 300, textAlign: "center", fontSize: 30, color: C.warm, direction: "rtl" }}>צילום אמיתי</div>
          <div style={{ position: "absolute", left: 0, top: 40, width: 1180, textAlign: "center", fontSize: 32, color: C.text, direction: "ltr" }}>gap = Dice(DRR) − Dice(real) = ?</div>
        </div>
        {/* res6: every outcome is a finding */}
        <div style={{ position: "absolute", inset: 0, opacity: vis("res6") }}>
          <Box x={90} y={120} w={1000} h={400} border={C.warm}>
            <div style={{ padding: "50px 50px", fontSize: 44, lineHeight: 1.6 }}>
              <div style={{ color: C.warm, fontWeight: 700 }}>כל תוצאה היא ממצא</div>
              <div style={{ fontSize: 34 }}>הרשת לא מנצחת את השליפה? מדווחים, ומסבירים.</div>
              <div style={{ fontSize: 34, color: C.bad }}>אסור: לבחור את הסיפור אחרי שרואים את המספרים.</div>
            </div>
          </Box>
        </div>
      </div>
    </div>
  );
}

// ---------------- 2D: the report ----------------
function Report({ t }: { t: number }) {
  const [p0, p1] = span(tl, "rep1", "rep5");
  const o = window01(t, p0 - 0.1, p1 + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const secs = ["שאלת המחקר", "נתונים", "שיטה", "ניסויים", "תוצאות", "ניתוח שגיאות", "מגבלות", "רישיונות", "תרומה אישית"];
  const hlIdx = t >= A("rep2") - 0.2 && t < A("rep3") - 0.2 ? 7 : t >= A("rep3") - 0.2 && t < A("rep4") - 0.2 ? 8 : t >= A("rep5") - 0.2 ? 6 : -1;
  const step = (B("rep1") - A("rep1") - 3) / 7;
  const lic = window01(t, A("rep2") + 0.3, B("rep2") + 0.4, 0.5, 0.5);
  const log = window01(t, A("rep3") + 0.3, B("rep3") + 0.4, 0.5, 0.5);
  const nb = window01(t, A("rep4") + 0.3, B("rep4") + 0.5, 0.5, 0.5);
  const lim = window01(t, A("rep5") + 0.3, B("rep5") + 0.5, 0.5, 0.5);
  const limT = (i: number) => ramp(t, A("rep5") + 4 + i * 3, A("rep5") + 4.5 + i * 3);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      {/* the document */}
      <div style={{ position: "absolute", right: 140, top: 200, width: 500, height: 690, background: "#e9eef6", borderRadius: 14, padding: "30px 34px",
        boxSizing: "border-box", direction: "rtl", boxShadow: "0 0 60px rgba(60,140,255,0.2)" }}>
        <div style={{ fontSize: 30, fontWeight: 700, color: "#0b1626", marginBottom: 6 }}>שחזור אגן תלת-ממדי משני צילומי רנטגן</div>
        <div style={{ fontSize: 20, color: "#4a5a70", marginBottom: 18 }}>אדיר · נועה</div>
        {secs.map((s, i) => {
          const k = i < 7 ? ramp(t, A("rep1") + 0.8 + i * step, A("rep1") + 1.2 + i * step) : ramp(t, A(i === 7 ? "rep2" : "rep3"), A(i === 7 ? "rep2" : "rep3") + 0.5);
          const on = i === hlIdx;
          return <div key={s} style={{ opacity: k, marginBottom: 4, padding: "3px 10px", borderRadius: 8, background: on ? "rgba(255,180,60,0.35)" : "transparent" }}>
            <div style={{ fontSize: 25, fontWeight: 700, color: "#0b1626" }}>{i + 1}. {s}</div>
            <div style={{ height: 7, width: `${70 + 25 * Math.sin(i * 3.1)}%`, background: "#b8c3d3", borderRadius: 4, marginTop: 6 }} />
          </div>;
        })}
      </div>
      {lic > 0 && <Table x={120} y={300} w={1100} o={lic} colW={300} rowH={86} size={26} cols={["רישיון", "מה מותר"]}
        rows={["TotalSegmentator", "DeepFluoro", "DiffDRR (קוד)"]} 
        cells={[["CC BY 4.0", "לשתף, עם ייחוס"], ["CC BY-NC-ND 4.0", "לבדיקה; לא מפיצים נגזרות"], ["MIT", "שימוש חופשי, עם הודעת הרישיון"]]}
        title="סעיף רישיונות" />}
      {log > 0 && <Box x={180} y={300} w={1000} o={log} border={C.line} bg="rgba(8,14,24,0.92)">
        <div style={{ padding: "26px 34px", fontFamily: "monospace", direction: "ltr", fontSize: 24, lineHeight: 1.6, color: C.text }}>
          <div style={{ color: C.dim }}>units/03-baselines/LOG.md</div>
          <div><span style={{ color: C.warm }}>TODO 1</span> nn_baseline · driver: Adir · hints: tier 1</div>
          <div><span style={{ color: C.warm }}>TODO 2</span> visual_hull · driver: Noa · hints: tier 1, 2</div>
          <div><span style={{ color: C.warm }}>explain-back</span> Noa → Adir: why the hull over-fills the inlet</div>
        </div>
        <div style={{ padding: "0 34px 22px", fontSize: 24, color: C.dim }}>(דוגמה למבנה; השורות האמיתיות נכתבות אצלכם)</div>
      </Box>}
      {lim > 0 && <Box x={180} y={300} w={1000} o={lim} border={C.line}>
        <div style={{ padding: "30px 40px" }}>
          <div style={{ fontSize: 38, fontWeight: 700, color: C.xray, marginBottom: 18 }}>מגבלות, בגלוי</div>
          {["אימון על צילומים מדומים (DRR) בלבד", "רק 6 גופות ב-DeepFluoro: מדגם קטן", "המסכות של TotalSegmentator (האמת שלנו) לא מושלמות"].map((l, i) => (
            <div key={i} style={{ fontSize: 32, lineHeight: 1.6, marginBottom: 10, opacity: limT(i) }}><span style={{ color: C.warm }}>• </span>{l}</div>
          ))}
        </div>
      </Box>}
      {nb > 0 && <Box x={180} y={280} w={1000} o={nb} border={C.good} bg="rgba(8,14,24,0.92)">
        <div style={{ padding: "24px 34px", fontFamily: "monospace", direction: "ltr", fontSize: 23, lineHeight: 1.7 }}>
          {["SPLIT = load('splits/split.json')   # fixed, never regenerated", "set_seed(0); set_seed(1); set_seed(2)", "pip install -r requirements.txt   # pinned versions",
            "train → evaluate → results_table.csv", "▶ Run all: Colab ✓  Kaggle ✓  local ✓"].map((l, i) => (
            <div key={i} style={{ opacity: ramp(t, A("rep4") + 1 + i * 1.2, A("rep4") + 1.4 + i * 1.2), color: i === 4 ? C.good : C.text }}>
              <span style={{ color: C.dim }}>[{i + 1}] </span>{l}</div>
          ))}
        </div>
      </Box>}
    </div>
  );
}

// ---------------- 2D: the class demo ----------------
function Demo({ t }: { t: number }) {
  const o = window01(t, A("demo1") - 0.1, B("demo3") + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const step = (B("demo1") - A("demo1") - 3) / 4;
  const items: { l: string; el: ReactNode }[] = [
    { l: "1 · שני צילומים", el: <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
      <Img src={staticFile("data/hero/drr_ap.png")} style={{ height: 170, borderRadius: 6 }} /><Img src={staticFile("data/hero/drr_lat.png")} style={{ height: 170, borderRadius: 6 }} /></div> },
    { l: "2 · הנפח המשוחזר", el: <div style={{ width: 150, height: 150, margin: "10px auto", border: `2px solid ${C.warm}`, transform: "rotateX(-20deg) rotateY(35deg)", background: "rgba(255,207,127,0.14)" }} /> },
    { l: "3 · mesh מול האמת", el: <div style={{ fontSize: 90, textAlign: "center", color: C.bone, lineHeight: "170px" }}>⟳</div> },
    { l: "4 · צילום אמיתי, והפער", el: <div style={{ fontSize: 44, textAlign: "center", color: C.warm, lineHeight: "170px", direction: "ltr" }}>gap = ?</div> },
  ];
  const tip = ramp(t, A("demo2") + 0.3, A("demo2") + 1.0);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      {items.map((it, i) => (
        <Box key={i} x={1440 - i * 420} y={300} w={360} h={300} o={ramp(t, A("demo1") + 1.5 + i * step, A("demo1") + 2 + i * step)} border={i === 3 ? C.warm : C.line}>
          <div style={{ padding: "18px 10px 0", fontSize: 30, fontWeight: 700, textAlign: "center" }}>{it.l}</div>
          <div style={{ marginTop: 14 }}>{it.el}</div>
        </Box>
      ))}
      <Txt x={360} y={680} w={1200} o={tip} align="center" size={34} color={C.warm}>
        מטופל שנבחר מראש · ומקרה כושל אחד, מוסבר</Txt>
      <Txt x={360} y={750} w={1200} o={ramp(t, A("demo3") + 0.3, A("demo3") + 1.0)} align="center" size={34} color={C.xray}>
        שקף = שאלה אחת · מספר = עם הפיזור שלו</Txt>
    </div>
  );
}

// ---------------- 2D: the whole series ----------------
const TITLES = ["התמונה הגדולה", "מהי תמונה", "מאיפה באים מספרי ה-CT", "מצלמות והטלה", "ה-DRR: הדמיית רנטגן", "ייצוג צורה בתלת-ממד",
  "מדידת שחזור", "משתי תצפיות לתלת-ממד", "רצפת השינון", "קונבולוציה מאפס", "מקודד-מפענח ו-U-Net", "לאמן את הרשת", "ניסוי מבוקר", "מדומה מול אמיתי"];
const SEG_OF = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]; // which all* segment lights each title
function Series({ t }: { t: number }) {
  const [s0, s1] = span(tl, "all1", "all3");
  const o = window01(t, s0 - 0.1, s1 + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const segs = ["all1", "all2", "all3"];
  const counts = [5, 4, 5];
  return (
    <div style={{ position: "absolute", left: 140, right: 140, top: 230, display: "flex", flexWrap: "wrap", gap: 20, justifyContent: "center", direction: "rtl", opacity: o, fontFamily: FONT }}>
      {TITLES.map((ti, i) => {
        const g = SEG_OF[i], j = i - SEG_OF.indexOf(g);
        const a0 = A(segs[g]) + (g === 0 ? 0.6 : 0.4), d = (B(segs[g]) - a0 - 0.5) / counts[g];
        const k = ramp(t, a0 + j * d, a0 + j * d + 0.5);
        return <div key={i} style={{ width: 300, padding: "18px 20px", borderRadius: 16, border: `2px solid ${k > 0.5 ? C.xray : C.line}`,
          background: C.panel, opacity: 0.25 + 0.75 * k, boxShadow: k > 0.5 ? "0 0 30px rgba(127,208,255,0.25)" : "none" }}>
          <div style={{ fontSize: 24, color: C.dim }}>סרטון {i}</div>
          <div style={{ fontSize: 30, color: C.text, fontWeight: 500 }}>{ti}</div>
        </div>;
      })}
    </div>
  );
}

export const V14 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const meta = useJson<Meta>("data/hero/meta.json");
  const pelvis = useObj("data/hero/pelvis.obj");
  const femurs = useObj("data/hero/femurs.obj");
  const ap = useTex("data/hero/drr_ap.png");
  const lat = useTex("data/hero/drr_lat.png");
  const coronal = useTex("data/hero/slice_coronal.png");
  const sagittal = useTex("data/hero/slice_sagittal.png");
  const occ64 = useJson<Occ>("data/hero/occ64.json");
  const ready = meta && pelvis && femurs && ap && lat && coronal && sagittal && occ64;
  const [f0, f6] = span(tl, "fly1", "fly6");
  const stageIdx = t >= A("fly6") - 0.2 ? [4, 5] : t >= A("fly4") - 0.2 ? [3] : t >= A("fly3") - 0.2 ? [2] : t >= A("fly2") - 0.2 ? [1] : [0];
  const dim2d = window01(t, A("res1") - 0.3, tl.total + 1, 0.6, 0.1);

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ meta, pelvis, femurs, ap, lat, coronal, sagittal, occ64 }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.92 * dim2d, pointerEvents: "none" }} />

      <Opening n={14} title="מחברים הכול" active={[0, 1, 2, 3, 4, 5]} t={t} end={2.9} />

      <PipelineMap active={[0, 1, 2, 3, 4, 5]} t={t} a={A("intro2")} b={B("intro2") + 0.4} top={140} label="הכול" />
      <PipelineMap active={stageIdx} t={t} a={f0} b={f6 + 0.4} top={60} label="עכשיו" />
      <Label x={960} y={190} size={30} color={C.dim} o={window01(t, A("fly1"), B("fly1") + 0.3)} text="נניח: מטופל מקבוצת הבדיקה" />
      <NetSketch t={t} />
      <Label x={960} y={185} size={34} color={C.warm} o={window01(t, A("fly5"), B("fly6") + 0.5)}
        text="מה שרואים = האמת (GT). התחזית שלכם עוד לא קיימת" />
      <Label x={960 - 370 * ramp(t, A("fly6") + 0.3, A("fly6") + 2.3)} y={250} size={30} color={C.warm} o={window01(t, A("fly5") + 1.5, B("fly6") + 0.5)}
        text="64³: כאן תיכנס התחזית שלכם" />
      <Label x={1330} y={250} size={30} color={C.bone} o={window01(t, A("fly6") + 1.5, B("fly6") + 0.5)} text="mesh (marching cubes) של האמת" />
      <Box x={560} y={870} w={800} o={window01(t, A("fly6") + 3, B("fly6") + 0.5)} border={C.warm}>
        <div style={{ padding: "14px 24px", fontSize: 28, direction: "ltr", textAlign: "center", lineHeight: 1.5 }}>
          Dice = ? · HD95 = ? · ASSD = ? · angle = ?</div>
      </Box>

      <Title text="סיפור התוצאות" sub="התוכנית. המשבצות ריקות עד שתמלאו אותן" t={t} a={A("res1")} b={B("res6") + 0.3} top={60} size={58} />
      <Story t={t} />

      <Title text="הדוח" t={t} a={A("rep1")} b={B("rep5") + 0.3} top={60} size={58} />
      <Report t={t} />

      <Title text="ההדגמה בכיתה" t={t} a={A("demo1")} b={B("demo3") + 0.3} top={110} size={60} />
      <Demo t={t} />

      <Title text="כל מה שעברנו" t={t} a={A("all1")} b={B("all3") + 0.3} top={70} size={60} />
      <Series t={t} />

      <RecapCard t={t} a={A("recap")} b={B("recap") + 0.3} items={[
        "הצינור: CT → DRR (AP + LAT) → רשת → 64³ → mesh → מדדים",
        "התוצאות: baselines, 1 מול 2 תצפיות, Dice מול זווית, מדומה מול אמיתי",
        "הדוח: רישיונות, תרומה אישית, מחברת שאפשר לשחזר",
        "ההדגמה: שני צילומים → נפח → mesh → צילום אמיתי והפער",
      ]} />
      <YourTurn t={t} a={A("turn")} b={tl.total} notebook="units/08-report/notebook.ipynb" todos={[
        "למלא את טבלת התוצאות המלאה (כל ארבעת הפרקים)",
        "לכתוב את הדוח, כולל סעיף רישיונות ותרומה אישית",
        "להריץ את המחברת מההתחלה עד הסוף, נקי, ב-Colab או ב-Kaggle",
        "להכין את ההדגמה: מטופל שנבחר מראש + מקרה כושל אחד",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={A("intro1")} />
      <Narration tl={tl} dir="v14" />
    </AbsoluteFill>
  );
};
