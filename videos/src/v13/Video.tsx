// Video 13 · Simulated vs real. LOCAL RENDER ONLY: uses DeepFluoro (CC BY-NC-ND 4.0); its derived assets
// live in public/data/df_v13/ (gitignored), built by prep/v13_assets.py.
// Real: the DeepFluoro frame, our DRR of the same CT, the FOV match, noise, the profile, the realism chain.
// ILLUSTRATIVE (labelled on screen): the deformed "prediction" for re-projection, the MC-dropout maps.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import { useMemo } from "react";
import script from "./script.json";
import durations from "../../public/audio/v13/durations.json";
import { buildTimeline, Script } from "../lib/timeline";
import { ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson, useObj, useTex } from "../lib/assets";
import { CameraPath, CamKey, Floor, Stage, useBoneMaterial } from "../lib/Stage";
import { Bone, boxOf, Detector, Meta, PointSource, Rays } from "../lib/xray";
import { Captions, Credit, Eq, Label, Narration, Opening, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Box, ExtraCredit, IllusBadge, Thumb, Txt } from "../v12/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
const A = (id: string) => S[id][0];
const B = (id: string) => S[id][1];
const D = "data/df_v13";

type Info = {
  fov_box: [number, number, number, number]; fov_mm: number; magnification: number; mm_per_px_at_patient: number;
  det_mm: number; sdd_mm: number; drr_shape: [number, number]; hp_std_real: number; hp_std_drr: number;
  profile_row: number; profile_real: number[]; profile_drr: number[]; zoom_box: [number, number, number, number]; zoom_mm: number;
  illus_sil_iou: number; match_ncc: number;
};

// ---------------- 3D: re-projection of the (illustrative) predicted shape ----------------
function Scene({ t, a }: { t: number; a: Assets }) {
  const bone = useBoneMaterial();
  const warm = useBoneMaterial("#ffc46b");
  const box = boxOf(a.meta);
  const [cx, cy, cz] = box.c;
  const zSrc = box.mn[2] - 2.6, zDet = box.mx[2] + 0.35;
  const on = window01(t, A("rep1") - 0.3, B("rep2") + 0.6, 0.9, 0.7) + window01(t, A("rep4") - 0.3, B("rep5") + 0.6, 0.9, 0.7);
  const proj = ramp(t, A("rep1") + 2.0, B("rep1") - 0.3);
  const gtO = window01(t, A("rep2") - 0.2, B("rep2") + 0.6, 0.6, 0.6);
  const keys: CamKey[] = [
    { t: A("rep1") - 0.3, pos: [7.4, 3.0, -9.6], look: [cx, cy, cz + 0.8] },
    { t: B("rep1"), pos: [6.2, 2.4, -10.4], look: [cx, cy, cz + 1.0] },
    { t: B("rep2"), pos: [4.6, 1.8, -11.0], look: [cx - 1.2, cy, cz + 1.0] },
    { t: A("rep4"), pos: [-6.6, 3.0, -8.8], look: [cx - 1.0, cy - 0.3, cz + 0.8] },
    { t: B("rep5") + 0.6, pos: [-5.4, 3.6, -9.6], look: [cx - 1.0, cy - 0.3, cz + 0.8] },
  ];
  // a fan of rays from the point source through the shape's outline (deterministic)
  const fan = useMemo(() => {
    const out: [number, number, number][][] = [];
    for (let i = 0; i < 26; i++) {
      const ang = (i / 26) * Math.PI * 2;
      const r = 1.25 + 0.25 * Math.sin(i * 12.9898);
      const p: [number, number, number] = [cx + r * Math.cos(ang) * 1.2, cy + r * Math.sin(ang) * 0.9, zDet];
      out.push([[cx, cy, zSrc], p]);
    }
    return out;
  }, [cx, cy, zSrc, zDet]);
  if (on <= 0.001) return <CameraPath t={t} keys={keys} />;
  const mag = (zDet - zSrc) / (cz - zSrc);
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Floor y={-2.1} opacity={on * 0.8} />
      <Bone geo={a.pred} material={warm} opacity={on} />
      <Bone geo={a.pelvis} material={bone} opacity={gtO * 0.35} />
      <PointSource position={[cx, cy, zSrc]} opacity={on} />
      <Rays segs={fan} opacity={on * 0.5 * proj} />
      <Detector tex={a.diff} progress={proj} opacity={on} size={[a.meta.ap_size[0] * mag, a.meta.ap_size[1] * mag]}
        position={[cx, cy, zDet]} rotation={[0, Math.PI, 0]} />
    </>
  );
}

type Assets = { meta: Meta; pelvis: THREE.BufferGeometry; pred: THREE.BufferGeometry; diff: THREE.Texture };

// ---------------- 2D: real vs DRR side by side ----------------
const RX = 120, RY = 190, RS = 640;            // real frame (left of screen)
const DH = 560;                                 // DRR height
function SideBySide({ t, info }: { t: number; info: Info }) {
  const o = window01(t, A("side1") - 0.2, B("side5") + 0.5, 0.7, 0.6);
  if (o <= 0) return null;
  const [dh, dw] = info.drr_shape;
  const DW = (DH * dw) / dh, DX = 1800 - DW, DY = 230;
  const [fx, fy, fw, fh] = info.fov_box;
  const fov = window01(t, A("side2") + 2.5, B("side3") + 0.3, 0.6, 0.5);
  const zoomO = window01(t, A("side4") + 1.0, B("side4") + 0.4, 0.6, 0.5);
  const [zx, zy, zw, zh] = info.zoom_box;
  const prof = window01(t, A("side5") + 0.3, B("side5") + 0.4, 0.6, 0.5);
  const fade = 1 - 0.93 * Math.max(zoomO, prof);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <div style={{ position: "absolute", left: RX, top: RY, opacity: fade }}>
        <Img src={staticFile(`${D}/real.png`)} style={{ width: RS, height: RS, display: "block", borderRadius: 10, border: `2px solid ${C.warm}` }} />
        <div style={{ textAlign: "center", fontSize: 30, color: C.warm, marginTop: 10, direction: "rtl" }}>אמיתי · DeepFluoro, גופה 1</div>
        <div style={{ position: "absolute", left: zx * RS, top: zy * RS, width: zw * RS, height: zh * RS, border: `3px solid ${C.warm}`, opacity: zoomO > 0 ? 1 : 0 }} />
      </div>
      <div style={{ position: "absolute", left: DX, top: DY, opacity: fade }}>
        <Img src={staticFile(`${D}/drr_ap.png`)} style={{ width: DW, height: DH, display: "block", borderRadius: 10, border: `2px solid ${C.xray}` }} />
        <div style={{ textAlign: "center", fontSize: 30, color: C.xray, marginTop: 10, direction: "rtl" }}>ה-DRR שלנו · אותו CT</div>
        {fov > 0 && <>
          <div style={{ position: "absolute", left: fx * DW, top: fy * DH, width: fw * DW, height: fh * DH, border: `3px dashed ${C.warm}`,
            boxShadow: "0 0 0 2000px rgba(0,0,0,0.45)", opacity: fov, borderRadius: 4 }} />
          <div style={{ position: "absolute", left: fx * DW, top: fy * DH - 44, width: fw * DW, textAlign: "center", color: C.warm, fontSize: 28, opacity: fov, direction: "rtl" }}>
            ≈ {Math.round(info.fov_mm)} מ"מ</div>
          <div style={{ position: "absolute", left: 0, top: -44, width: DW, textAlign: "center", color: C.dim, fontSize: 24, opacity: fov, direction: "rtl" }}>
            רוחב ה-CT: {dw} מ"מ</div>
        </>}
        {zoomO > 0 && <div style={{ position: "absolute", left: (fx + zx * fw) * DW, top: (fy + zy * fh) * DH, width: zw * fw * DW, height: zh * fh * DH, border: `3px solid ${C.xray}` }} />}
      </div>
      {zoomO > 0 && (
        <div style={{ position: "absolute", inset: 0, opacity: zoomO }}>
          <Thumb src={`${D}/zoom_real.png`} x={330} y={300} h={460} border={C.warm} label={<span>אמיתי: פיקסל 0.15 מ"מ בגוף, רעש קוונטי</span>} />
          <Thumb src={`${D}/zoom_drr.png`} x={1130} y={300} h={460} border={C.xray} label={<span>DRR: ווקסל של 1 מ"מ, חלק</span>} />
          <Txt x={460} y={850} w={1000} align="center" size={28} color={C.dim}>
            אותו אזור, {info.zoom_mm} מ"מ. רכיב התדר-הגבוה: פי {(info.hp_std_real / info.hp_std_drr).toFixed(1)} באמיתי</Txt>
        </div>
      )}
      {prof > 0 && <Profile info={info} o={prof} />}
    </div>
  );
}

function Profile({ info, o }: { info: Info; o: number }) {
  const L = 360, R = 1560, T = 330, H = 380;
  const path = (v: number[]) => v.map((y, i) => `${i ? "L" : "M"} ${L + (i / (v.length - 1)) * (R - L)} ${T + H - y * H}`).join(" ");
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <rect x={L - 30} y={T - 30} width={R - L + 60} height={H + 60} rx={18} fill="rgba(8,14,24,0.85)" stroke={C.line} />
        <path d={path(info.profile_drr)} stroke={C.xray} strokeWidth={4} fill="none" />
        <path d={path(info.profile_real)} stroke={C.warm} strokeWidth={3} fill="none" opacity={0.95} />
      </svg>
      <Txt x={L} y={T + H + 50} w={R - L} align="center" size={28} color={C.dim}>
        בהירות לאורך שורה אחת, באותו מקום בשתי התמונות (מימין לשמאל במסך: אותה אנטומיה)</Txt>
      <Txt x={L} y={T - 90} w={R - L} align="center" size={32}>
        <span style={{ color: C.warm }}>אמיתי: ערפל, ניגודיות אחרת, רעש</span>{"   ·   "}<span style={{ color: C.xray }}>DRR: חלק ונקי</span></Txt>
    </div>
  );
}

function Magnification({ t, info }: { t: number; info: Info }) {
  const o = window01(t, A("side3") + 0.2, B("side3") + 0.3, 0.6, 0.5);
  if (o <= 0) return null;
  // perspective (top) vs parallel (bottom) beam, drawn to a schematic scale
  const sx = 520, dx = 1400, ox = 1400 - (1400 - 520) / info.magnification, y1 = 420, y2 = 790;
  const half = 70, halfDet = half * info.magnification;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, background: "rgba(5,7,11,0.9)", fontFamily: FONT }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <circle cx={sx} cy={y1} r={12} fill={C.xray} />
        <line x1={sx} y1={y1} x2={dx} y2={y1 - halfDet} stroke={C.xray} strokeWidth={2} />
        <line x1={sx} y1={y1} x2={dx} y2={y1 + halfDet} stroke={C.xray} strokeWidth={2} />
        <rect x={ox - 14} y={y1 - half} width={28} height={2 * half} fill={C.bone} rx={6} />
        <line x1={dx} y1={y1 - 150} x2={dx} y2={y1 + 150} stroke={C.line} strokeWidth={8} />
        <line x1={dx + 30} y1={y1 - halfDet} x2={dx + 30} y2={y1 + halfDet} stroke={C.warm} strokeWidth={4} />
        <line x1={sx} y1={y1 + 190} x2={dx} y2={y1 + 190} stroke={C.dim} strokeWidth={1.5} />
        <line x1={sx} y1={y1 + 165} x2={ox} y2={y1 + 165} stroke={C.dim} strokeWidth={1.5} />
        <text x={(sx + dx) / 2} y={y1 + 222} fill={C.dim} fontSize={26} textAnchor="middle" fontFamily={FONT}>SDD = {info.sdd_mm} mm</text>
        <text x={(sx + ox) / 2} y={y1 + 155} fill={C.dim} fontSize={26} textAnchor="middle" fontFamily={FONT}>SOD</text>
        {[-1, -0.5, 0, 0.5, 1].map((k) => <line key={k} x1={sx - 120} y1={y2 + k * half} x2={dx} y2={y2 + k * half} stroke={C.xray} strokeWidth={2} opacity={0.6} />)}
        <rect x={ox - 14} y={y2 - half} width={28} height={2 * half} fill={C.bone} rx={6} />
        <line x1={dx} y1={y2 - 150} x2={dx} y2={y2 + 150} stroke={C.line} strokeWidth={8} />
        <line x1={dx + 30} y1={y2 - half} x2={dx + 30} y2={y2 + half} stroke={C.warm} strokeWidth={4} />
      </svg>
      <Txt x={1460} y={y1 - 30} w={420} size={34} color={C.warm} align="left">
        <span style={{ direction: "ltr", display: "inline-block" }}>{`×${info.magnification.toFixed(2)}`}</span></Txt>
      <Txt x={1460} y={y2 - 30} w={420} size={34} color={C.warm} align="left"><span style={{ direction: "ltr", display: "inline-block" }}>×1</span></Txt>
      <Txt x={40} y={y1 - 30} w={400} size={30} color={C.text}>צילום אמיתי: קרניים מנקודה</Txt>
      <Txt x={40} y={y2 - 30} w={400} size={30} color={C.text}>ה-DRR המקבילי שלנו</Txt>
      <Txt x={460} y={180} w={1000} size={30} align="center" color={C.dim}>
        <span style={{ direction: "ltr", display: "inline-block" }}>M = SDD / SOD</span>
        {` · פיקסל ${info.mm_per_px_at_patient.toFixed(2)} מ"מ בגוף מול 0.194 מ"מ בגלאי (מההתאמה)`}</Txt>
    </div>
  );
}

function Chain({ t }: { t: number }) {
  const o = window01(t, A("side6") - 0.1, B("side6") + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const labels = ["DRR נקי", "+ פיזור (ערפל)", "+ רעש קוונטי", "+ טשטוש גלאי"];
  const step = (B("side6") - A("side6") - 2) / 4;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {labels.map((l, i) => (
        <Thumb key={i} src={`${D}/chain${i}.png`} x={1440 - i * 420} y={300} h={390} label={l}
          o={ramp(t, A("side6") + 0.4 + i * step, A("side6") + 0.9 + i * step)} border={i === 0 ? C.xray : C.line} />
      ))}
      {[0, 1, 2].map((i) => <Label key={i} x={1430 - i * 420} y={495} text="←" size={44} color={C.dim}
        o={ramp(t, A("side6") + 0.9 + (i + 1) * step, A("side6") + 1.2 + (i + 1) * step)} />)}
      <Txt x={260} y={790} w={1400} align="center" size={26} color={C.dim}
        o={ramp(t, A("side6") + 3, A("side6") + 4)}>העיבוד שלנו על ה-DRR (אותו חלון תצוגה בכל השלבים), כדי להמחיש. לא מודל פיזיקלי מלא.</Txt>
    </div>
  );
}

// ---------------- 2D: domain shift ----------------
function Clouds({ t }: { t: number }) {
  const o = window01(t, A("shift1"), B("shift1") + 0.4, 0.6, 0.5);
  if (o <= 0) return null;
  const pts = (n: number, cx: number, cy: number, sx: number, sy: number, s: number) =>
    Array.from({ length: n }, (_, i) => {
      const u = Math.sin(i * 12.9898 + s) * 43758.5453, v = Math.sin(i * 78.233 + s) * 12345.678;
      const a = (u - Math.floor(u)) * 2 * Math.PI, r = Math.sqrt(v - Math.floor(v));
      return [cx + Math.cos(a) * r * sx, cy + Math.sin(a) * r * sy];
    });
  const train = pts(140, 1180, 560, 260, 170, 1), test = pts(60, 760, 620, 200, 140, 7);
  const k2 = ramp(t, A("shift1") + 3.0, A("shift1") + 4.0);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <ellipse cx={1180} cy={560} rx={300} ry={205} fill="rgba(127,208,255,0.06)" stroke={C.xray} strokeDasharray="8 8" strokeWidth={2} />
        {train.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={7} fill={C.xray} opacity={0.75} />)}
        {test.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={8} fill={C.warm} opacity={k2} />)}
      </svg>
      <Txt x={1230} y={330} w={500} size={30} color={C.xray} align="center">אימון: DRR</Txt>
      <Txt x={410} y={800} w={500} size={30} color={C.warm} align="center" o={k2}>בדיקה: רנטגן אמיתי</Txt>
      <Txt x={930} y={800} w={600} size={26} color={C.dim} align="center" o={k2}>(המחשה: כל נקודה = תמונה)</Txt>
    </div>
  );
}

function PsnrBars({ t }: { t: number }) {
  const o = window01(t, A("shift2") + 0.2, B("shift3") + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const vals = [{ v: 29.45, l: "הטלות מדומות", c: C.xray }, { v: 23.22, l: "רנטגן אמיתי", c: C.warm }];
  const y0 = 760, sc = 15;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      {vals.map((b, i) => {
        const k = ramp(t, A("shift2") + 1.5 + i * 3.5, A("shift2") + 2.5 + i * 3.5);
        const h = b.v * sc * k;
        return (
          <div key={i} style={{ position: "absolute", left: 1260 - i * 300, top: y0 - h, width: 200, height: h, background: `${b.c}55`,
            border: `2px solid ${b.c}`, borderRadius: 10 }}>
            <div style={{ position: "absolute", top: -56, width: "100%", textAlign: "center", fontSize: 40, color: b.c, opacity: k }}>{b.v.toFixed(2)} dB</div>
            <div style={{ position: "absolute", bottom: -50, width: "100%", textAlign: "center", fontSize: 28, color: C.text, direction: "rtl", whiteSpace: "nowrap" }}>{b.l}</div>
          </div>
        );
      })}
      <Txt x={880} y={850} w={700} size={24} color={C.dim} align="center">IXGS, Sci. Reports 2025, חוליות. תוצאה של מחקר אחר, לא שלנו.</Txt>
    </div>
  );
}

function Protocol({ t }: { t: number }) {
  const o = window01(t, A("shift4") - 0.1, B("shift4") + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const row = (y: number, src: string, label: string, col: string, out: string, k: number) => (
    <div style={{ position: "absolute", left: 0, top: y, width: 1920, height: 200, opacity: k }}>
      <Thumb src={src} x={1430} y={0} h={180} border={col} />
      <Txt x={1100} y={60} w={310} size={28} color={col} align="center">{label}</Txt>
      <Label x={1060} y={90} text="→" size={40} color={C.dim} />
      <Box x={820} y={50} w={200} h={80} border={C.line}><div style={{ fontSize: 28, textAlign: "center", lineHeight: "76px", direction: "ltr" }}>אותה רשת</div></Box>
      <Label x={780} y={90} text="→" size={40} color={C.dim} />
      <Txt x={460} y={60} w={290} size={32} color={col} align="center"><span style={{ direction: "ltr", display: "inline-block" }}>{out}</span></Txt>
    </div>
  );
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {row(240, `${D}/real.png`, "צילום אמיתי", C.warm, "Dice(real)", ramp(t, A("shift4") + 0.5, A("shift4") + 1.2))}
      {row(480, `${D}/drr_crop.png`, "DRR: אותה תנוחה ומצלמה", C.xray, "Dice(DRR)", ramp(t, A("shift4") + 2.5, A("shift4") + 3.2))}
    </div>
  );
}

// ---------------- 2D: re-projection overlay, MC dropout ----------------
function ReprojOverlay({ t, info }: { t: number; info: Info }) {
  const o = window01(t, A("rep2") + 0.3, B("rep3") + 0.4, 0.6, 0.6);
  if (o <= 0) return null;
  const legend = [{ c: "#78c8ff", s: "חופף" }, { c: "#5affaa", s: "חסר בתחזית" }, { c: "#ff7882", s: "נוסף בתחזית" }];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      <Box x={1180} y={230} w={620} h={600} bg="rgba(5,8,14,0.92)">
        <Img src={staticFile(`${D}/reproj_diff.png`)} style={{ position: "absolute", left: 40, top: 30, width: 540, height: 514 }} />
        <div style={{ position: "absolute", bottom: 14, left: 0, right: 0, textAlign: "center", fontSize: 26, color: C.dim, direction: "ltr" }}>
          silhouette IoU = {info.illus_sil_iou.toFixed(3)} <span style={{ direction: "rtl" }}>(על ההמחשה)</span></div>
      </Box>
      <div style={{ position: "absolute", left: 1180, top: 850, width: 620, display: "flex", gap: 26, justifyContent: "center", direction: "rtl" }}>
        {legend.map((l) => <div key={l.s} style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 26, color: C.text }}>
          <span style={{ width: 22, height: 22, background: l.c, borderRadius: 4, display: "inline-block" }} />{l.s}</div>)}
      </div>
    </div>
  );
}

function McDropout({ t }: { t: number }) {
  const o = window01(t, A("mc2") - 0.1, B("mc4") + 0.5, 0.6, 0.6);
  if (o <= 0) return null;
  const passK = (i: number) => ramp(t, A("mc2") + 2.0 + i * 0.7, A("mc2") + 2.5 + i * 0.7);
  const maps = ramp(t, A("mc3") + 0.3, A("mc3") + 1.2);
  const errK = ramp(t, A("mc4") + 0.3, A("mc4") + 1.2);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, fontFamily: FONT }}>
      {[0, 1, 2, 3].map((i) => (
        <Thumb key={i} src={`${D}/mc_pass${i}.png`} x={1480 - i * 250} y={250} h={220} o={passK(i) * (1 - 0.5 * maps)}
          label={<span style={{ direction: "ltr", display: "inline-block" }}>{`pass ${i + 1}`}</span>} />
      ))}
      <Thumb src={`${D}/mc_var.png`} x={1380} y={540} h={340} o={maps} border={C.warm}
        label={<span>שונות לכל voxel <span style={{ direction: "ltr", display: "inline-block" }}>σ²</span></span>} />
      <Thumb src={`${D}/mc_err.png`} x={840} y={540} h={340} o={errK} border={C.bad}
        label={<span>שגיאה <span style={{ direction: "ltr", display: "inline-block" }}>|μ − GT|</span></span>} />
      <Txt x={1200} y={680} w={170} size={34} color={C.warm} align="center" o={errK}>
        <span style={{ direction: "ltr", display: "inline-block" }}>corr = ?</span></Txt>
    </div>
  );
}

export const V13 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const meta = useJson<Meta>("data/hero/meta.json");
  const info = useJson<Info>(`${D}/info.json`);
  const pelvis = useObj("data/hero/pelvis.obj");
  const pred = useObj(`${D}/pred_illus.obj`);
  const diff = useTex(`${D}/reproj_diff.png`);
  const ready = meta && pelvis && pred && diff && info;
  const in3d = window01(t, A("rep1") - 0.4, B("rep2") + 0.5, 0.7, 0.6) + window01(t, A("rep4") - 0.4, B("rep5") + 0.5, 0.7, 0.6);
  const intro = window01(t, 2.6, B("intro3") + 0.5, 0.8, 0.6);

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {ready && (
        <Stage>
          <Scene t={t} a={{ meta, pelvis, pred, diff }} />
        </Stage>
      )}
      <AbsoluteFill style={{ background: C.bg, opacity: 0.95 * (1 - Math.min(1, in3d)), pointerEvents: "none" }} />

      <Opening n={13} title="מדומה מול אמיתי" active={[1, 5]} t={t} end={2.9} />

      {/* intro: simulated vs real, DeepFluoro */}
      {intro > 0 && <div style={{ position: "absolute", inset: 0, opacity: intro }}>
        <Thumb src={`${D}/drr_crop.png`} x={1100} y={250} h={480} border={C.xray} label="מדומה (DRR)" o={ramp(t, 3.0, 4.0)} />
        <Thumb src={`${D}/real.png`} x={340} y={250} h={480} border={C.warm} label="אמיתי" o={ramp(t, A("intro1") + 5.0, A("intro1") + 6.0) * (1 - ramp(t, A("intro2"), A("intro2") + 0.6))} />
        <Label x={960} y={490} text="?" size={110} color={C.warm} o={window01(t, A("intro1") + 6, A("intro2") + 0.4)} />
        {[0, 1, 2, 3].map((i) => (
          <Thumb key={i} src={`${D}/thumb${i}.png`} x={180 + (i % 2) * 400} y={190 + Math.floor(i / 2) * 360} h={320} border={C.warm}
            o={window01(t, A("intro2") + 0.3 + i * 0.3, B("intro3") + 0.5, 0.5, 0.5)} />
        ))}
        <Txt x={1000} y={790} w={800} align="center" size={30} color={C.dim} o={window01(t, A("intro2") + 4, B("intro3") + 0.5)}>
          <div>6 גופות · פלואורוסקופיה אמיתית + CT</div>
          <div>רישיון <span style={{ direction: "ltr", display: "inline-block" }}>CC BY-NC-ND 4.0</span>: רק בסרטון הזה, רינדור מקומי</div></Txt>
      </div>}
      <Title text="DeepFluoro" sub="צילומים אמיתיים, עם אמת בתלת-ממד" t={t} a={A("intro2")} b={B("intro2") + 0.3} top={80} size={64} />
      <Title text="החידוש: פער מדוד" sub="לא ״נראה סביר״, אלא מספר" t={t} a={A("intro3")} b={B("intro3") + 0.4} top={80} size={64} />

      {info && <>
        <SideBySide t={t} info={info} />
        <Magnification t={t} info={info} />
      </>}
      <Title text="שדה ראייה" t={t} a={A("side2")} b={B("side2") + 0.2} top={50} size={50} />
      <Title text="הגדלה" t={t} a={A("side3")} b={B("side3") + 0.2} top={50} size={50} />
      <Title text="רעש ופרטים" t={t} a={A("side4")} b={B("side4") + 0.2} top={50} size={50} />
      <Title text="פיזור וניגודיות" t={t} a={A("side5")} b={B("side5") + 0.2} top={50} size={50} />
      <Title text="לקצר את המרחק" sub="augmentation שמחקה צילום אמיתי" t={t} a={A("side6")} b={B("side6") + 0.4} top={70} size={58} />
      <Chain t={t} />

      <Title text="שינוי תחום" sub="domain shift" t={t} a={A("shift1")} b={B("shift1") + 0.3} top={80} size={64} />
      <Clouds t={t} />
      <Title text="כמה זה עולה?" t={t} a={A("shift2")} b={B("shift3") + 0.3} top={70} size={60} />
      <PsnrBars t={t} />
      <Eq t={t} a={A("shift3")} b={B("shift3") + 0.4} size={44} style={{ left: 90, top: 300, width: 700 }}
        tex={String.raw`\mathrm{PSNR} = 10\log_{10}\frac{\mathrm{MAX}^2}{\hl{mse}{\mathrm{MSE}}}\\[10pt] \frac{\mathrm{MSE}_{real}}{\mathrm{MSE}_{sim}} = 10^{\hl{d}{6.23}/10} \approx \hl{r}{4.2}`}
        terms={{ mse: [A("shift3") + 0.5, A("shift3") + 3], d: [A("shift3") + 2.5, A("shift3") + 5], r: [A("shift3") + 5, B("shift3") + 0.4] }} />
      <Title text="הפער שלנו" t={t} a={A("shift4")} b={B("shift4") + 0.3} top={70} size={60} />
      <Protocol t={t} />
      <Eq t={t} a={A("shift4") + 5.5} b={B("shift4") + 0.4} size={46} style={{ left: 560, top: 750, width: 800, padding: "4px 30px" }}
        tex={String.raw`\text{gap} = \mathrm{Dice}(\text{DRR}) - \mathrm{Dice}(\text{real}) = \;?`} />

      <Title text="בדיקת הטלה חוזרת" sub="re-projection" t={t} a={A("rep1")} b={B("rep1") + 0.3} top={50} size={58} />
      <IllusBadge o={window01(t, A("rep1") + 0.5, B("rep3") + 0.4)} y={1000 - 60} text="המחשה: צורה שעיוותנו בכוונה, לא תחזית של רשת" />
      <ReprojOverlay t={t} info={info ?? ({ illus_sil_iou: 0 } as Info)} />
      <Title text="אנטומיה מומצאת" sub="hallucination" t={t} a={A("rep3")} b={B("rep3") + 0.3} top={60} size={56} />
      <Title text="רינדור דיפרנציאבילי" sub="DiffDRR" t={t} a={A("rep4")} b={B("rep5") + 0.3} top={50} size={56} />
      <Eq t={t} a={A("rep4") + 3} b={B("rep5") + 0.3} size={42} style={{ left: 90, top: 640, width: 760, padding: "8px 30px" }}
        tex={String.raw`\mathcal{L}(\hl{th}{\theta}) = \textstyle\sum_{\text{pixels}} \big(\hl{sp}{S_\theta} - \hl{so}{S_{obs}}\big)^2\\[8pt] \hl{th}{\theta} \leftarrow \hl{th}{\theta} - \eta\,\hl{g}{\nabla_\theta \mathcal{L}}`}
        terms={{ th: [A("rep4") + 3.5, A("rep4") + 7], sp: [A("rep4") + 5, B("rep4")], so: [A("rep5"), A("rep5") + 3], g: [A("rep5") + 3, B("rep5") + 0.3] }} />

      <Title text="אי-ודאות: MC dropout" t={t} a={A("mc1")} b={B("mc4") + 0.3} top={60} size={56} />
      <IllusBadge o={window01(t, A("mc1") + 0.5, B("mc4") + 0.4)} y={180} text="המחשה: מפות מדומות, לא פלט של רשת" />
      <McDropout t={t} />
      <Eq t={t} a={A("mc3") - 0.2} b={B("mc3") + 0.3} size={34} style={{ left: 60, top: 600, width: 700, padding: "4px 20px" }}
        tex={String.raw`\hl{mu}{\mu} = \tfrac{1}{T}\textstyle\sum_t p_t \qquad \hl{v}{\sigma^2} = \tfrac{1}{T}\textstyle\sum_t (p_t-\mu)^2`}
        terms={{ mu: [A("mc3"), A("mc3") + 3], v: [A("mc3") + 3, B("mc3")] }} />
      <Thumb src={`${D}/mc_pass0.png`} x={790} y={260} h={360} o={window01(t, A("mc1") + 0.5, B("mc1") + 0.4)} border={C.xray}
        label="תשובה אחת של הרשת (המחשה)" />
      <Label x={1290} y={440} size={120} color={C.warm} text="?" o={window01(t, A("mc1") + 2.5, B("mc1") + 0.4)} />
      <Label x={960} y={740} size={34} color={C.dim} o={window01(t, A("mc1") + 4, B("mc1") + 0.4)} text="כמה היא בטוחה בכל voxel?" />

      <RecapCard t={t} a={A("recap")} b={B("recap") + 0.3} items={[
        "אמיתי ≠ DRR: שדה ראייה, הגדלה, רעש, פיזור, ניגודיות",
        "domain shift: אימון על DRR, בדיקה על צילום אמיתי",
        "הפער: אותה רשת, אמיתי מול DRR באותה תנוחה ומצלמה",
        "re-projection + DiffDRR: תופסים אנטומיה מומצאת, ואפשר גם לתקן",
        "MC dropout: שונות לכל voxel; לבדוק שהיא מתואמת עם השגיאה",
      ]} />
      <YourTurn t={t} a={A("turn")} b={tl.total} notebook="units/07-real-xray/notebook.ipynb" todos={[
        "לטעון צילומי DeepFluoro עם התנוחות ופרמטרי המצלמה שלהם",
        "לרנדר DRR תואם עם DiffDRR, ולהריץ את הרשת על שניהם",
        "לחשב Dice מול ה-CT בשניהם, ואת הפער",
        "בדיקת הטלה חוזרת: IoU של צלליות",
        "אם יש זמן: MC dropout ומתאם בין שונות לשגיאה",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={A("intro1")} />
      <ExtraCredit o={ramp(t, A("intro1"), A("intro1") + 0.8)} text="Real fluoroscopy + CT: DeepFluoro (Grupp et al., 2020), CC BY-NC-ND 4.0" />
      <Narration tl={tl} dir="v13" />
    </AbsoluteFill>
  );
};
