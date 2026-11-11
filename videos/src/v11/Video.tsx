// Video 11 · Training it. 2D diagrams in the DOM layer; one 3D scene (the hero 64^3 grid, mostly empty).
// Real numbers (occupied fraction, "all empty" accuracy) come from prep/v11_assets.py. Loss curves are illustrations.
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import script from "./script.json";
import durations from "../../public/audio/v11/durations.json";
import d11 from "../../public/data/v11/v11.json";
import { buildTimeline, Script } from "../lib/timeline";
import { lerp, ramp, window01 } from "../lib/anim";
import { C, FONT } from "../lib/style";
import { useJson } from "../lib/assets";
import { CameraPath, CamKey, orbit, Stage } from "../lib/Stage";
import { Voxels, WireBox } from "../lib/xray";
import { Captions, Credit, Eq, Narration, Opening, Panel, RecapCard, Title, YourTurn } from "../lib/Overlays";
import { Block, CellGrid, Code, Illustrative, Plot, Svg, T, Tile, fmt } from "../v09/parts";

export const tl = buildTimeline(script as Script, durations);
const S = tl.seg;
type Occ = { n: number; cell: number; count: number; centers: number[] };
const win = (t: number, a: string, b: string, fin = 0.6, fout = 0.6, pad = 0.4) => window01(t, S[a][0] - 0.1, S[b][1] + pad, fin, fout);
const frac = (id: string, f: number) => S[id][0] + f * (S[id][1] - S[id][0]);
const sig = (x: number) => 1 / (1 + Math.exp(-x));
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

// ---------------- intro ----------------
function Intro({ t }: { t: number }) {
  const o = win(t, "intro", "intro", 0.8, 0.6, 0.6);
  if (o <= 0) return null;
  const items = ["פלט: הסתברויות", "הפסד: BCE + Dice", "נתונים: חלוקה ואוגמנטציה", "אימון: Colab / Kaggle"];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      {items.map((it, i) => {
        const k = ramp(t, frac("intro", 0.2 + i * 0.17), frac("intro", 0.28 + i * 0.17));
        return <Block key={i} x={1460 - i * 420} y={420} w={360} h={170} o={k} color={i % 2 ? C.warm : C.xray}
          inner={<span style={{ fontSize: 32, direction: "rtl", padding: 10 }}>{it}</span>} />;
      })}
    </div>
  );
}

// ---------------- logits and sigmoid ----------------
function SigmoidPlot({ t }: { t: number }) {
  const o = win(t, "logit1", "logit2");
  if (o <= 0) return null;
  const X0 = 300, Y0 = 250, W = 760, H = 460;
  const px = (x: number) => X0 + ((x + 8) / 16) * W, py = (p: number) => Y0 + H - p * H;
  const draw = ramp(t, S.logit1[0] + 2.0, frac("logit1", 0.85));
  const pts = Array.from({ length: 161 }, (_, i) => -8 + i * 0.1).filter((x) => (x + 8) / 16 <= draw + 1e-6);
  const d = pts.map((x, i) => `${i ? "L" : "M"}${px(x).toFixed(1)},${py(sig(x)).toFixed(1)}`).join(" ");
  const L = [-6, -2, 0, 2, 6];
  const dots = ramp(t, S.logit2[0], S.logit2[0] + 1.0);
  const thr = ramp(t, frac("logit2", 0.72), frac("logit2", 0.82));
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Svg>
        <line x1={X0} y1={Y0 + H} x2={X0 + W} y2={Y0 + H} stroke={C.dim} strokeWidth={2} />
        <line x1={px(0)} y1={Y0} x2={px(0)} y2={Y0 + H} stroke={C.dim} strokeWidth={1.5} strokeDasharray="6 6" />
        {[0, 0.5, 1].map((p) => <g key={p}><line x1={X0} y1={py(p)} x2={X0 + W} y2={py(p)} stroke="rgba(143,166,196,0.15)" />
          <text x={X0 - 14} y={py(p) + 8} fill={C.dim} fontSize={24} textAnchor="end">{p}</text></g>)}
        {[-8, -4, 0, 4, 8].map((x) => <text key={x} x={px(x)} y={Y0 + H + 34} fill={C.dim} fontSize={24} textAnchor="middle">{x}</text>)}
        <path d={d} fill="none" stroke={C.xray} strokeWidth={5} style={{ filter: `drop-shadow(0 0 8px ${C.xray})` }} />
        {thr > 0 && <line x1={X0} y1={py(0.5)} x2={X0 + W} y2={py(0.5)} stroke={C.warm} strokeWidth={3} strokeDasharray="12 8" opacity={thr} />}
        {L.map((x) => <circle key={x} cx={px(x)} cy={py(sig(x))} r={11} fill={sig(x) > 0.5 ? C.warm : C.xray} opacity={dots} />)}
      </Svg>
      <T x={X0 + W / 2} y={Y0 + H + 80} size={28} color={C.dim}>logit z</T>
      <T x={X0 - 10} y={Y0 - 30} size={28} color={C.dim}>p = σ(z)</T>
      <T x={px(5.5)} y={py(0.5) - 30} size={24} color={C.warm} o={thr}>p &gt; 0.5 → עצם</T>
      <Eq tex={"\\sigma(z) = \\frac{1}{1 + e^{-z}}"} t={t} a={S.logit1[0] + 1.0} b={S.logit2[1] + 0.4} size={60}
        style={{ left: 1180, width: 600, top: 240 }} />
      <div style={{ position: "absolute", left: 1180, top: 520, width: 600, opacity: dots, fontFamily: FONT, direction: "ltr" }}>
        <div style={{ display: "flex", fontSize: 26, color: C.dim, marginBottom: 8 }}><span style={{ width: 200 }}>logit</span><span style={{ width: 200 }}>p</span><span>mask</span></div>
        {L.map((x) => (
          <div key={x} style={{ display: "flex", fontSize: 30, color: C.text, marginBottom: 4 }}>
            <span style={{ width: 200 }}>{x}</span><span style={{ width: 200 }}>{sig(x).toFixed(3)}</span>
            <span style={{ color: sig(x) > 0.5 ? C.warm : C.dim, opacity: thr }}>{sig(x) > 0.5 ? 1 : 0}</span></div>
        ))}
      </div>
    </div>
  );
}

// ---------------- BCE ----------------
function BCEPlot({ t }: { t: number }) {
  const o = win(t, "bce1", "bce2");
  if (o <= 0) return null;
  const X0 = 1060, Y0 = 250, W = 700, H = 440, yMax = 4.6;
  const px = (p: number) => X0 + p * W, py = (v: number) => Y0 + H - Math.min(v, yMax) / yMax * H;
  const ps = Array.from({ length: 199 }, (_, i) => 0.005 + i * 0.005);
  const curve = (f: (p: number) => number) => ps.map((p, i) => `${i ? "L" : "M"}${px(p).toFixed(1)},${py(f(p)).toFixed(1)}`).join(" ");
  const one = ramp(t, frac("bce1", 0.3), frac("bce1", 0.45));
  const m9 = ramp(t, frac("bce1", 0.55), frac("bce1", 0.62)), m1 = ramp(t, frac("bce1", 0.7), frac("bce1", 0.78));
  const zero = ramp(t, S.bce2[0] + 0.2, S.bce2[0] + 1.2);
  const B = d11.bce_examples as Record<string, number>;
  const db = S.bce1[1] - S.bce1[0];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Svg>
        <line x1={X0} y1={Y0 + H} x2={X0 + W} y2={Y0 + H} stroke={C.dim} strokeWidth={2} />
        <line x1={X0} y1={Y0} x2={X0} y2={Y0 + H} stroke={C.dim} strokeWidth={2} />
        {[0, 0.5, 1].map((p) => <text key={p} x={px(p)} y={Y0 + H + 34} fill={C.dim} fontSize={24} textAnchor="middle">{p}</text>)}
        {[0, 1, 2, 3, 4].map((v) => <text key={v} x={X0 - 14} y={py(v) + 8} fill={C.dim} fontSize={22} textAnchor="end">{v}</text>)}
        <path d={curve((p) => -Math.log(p))} fill="none" stroke={C.xray} strokeWidth={5} opacity={one} />
        <path d={curve((p) => -Math.log(1 - p))} fill="none" stroke={C.warm} strokeWidth={5} opacity={zero} />
        <circle cx={px(0.9)} cy={py(B["0.9"])} r={11} fill={C.good} opacity={m9} />
        <circle cx={px(0.1)} cy={py(B["0.1"])} r={11} fill={C.bad} opacity={m1} />
      </Svg>
      <T x={px(0.9)} y={py(B["0.9"]) - 34} size={26} color={C.good} o={m9}>{`p=0.9 → ${B["0.9"].toFixed(2)}`}</T>
      <T x={px(0.1) + 30} y={py(B["0.1"]) - 30} size={26} color={C.bad} o={m1} align="left">{`p=0.1 → ${B["0.1"].toFixed(2)}`}</T>
      <T x={X0 + W / 2} y={Y0 + H + 80} size={28} color={C.dim}>p (ההסתברות שהרשת נתנה לעצם)</T>
      <T x={px(0.25)} y={Y0 + 20} size={26} color={C.xray} o={one} rtl>תא עם עצם: −log p</T>
      <T x={px(0.75)} y={Y0 + 20} size={26} color={C.warm} o={zero} rtl>תא ריק: −log(1−p)</T>
      <Eq tex={"\\text{BCE} = -\\frac{1}{N}\\sum_i \\big[\\hl{a}{y_i \\log p_i} + \\hl{b}{(1-y_i)\\log(1-p_i)}\\big]"} t={t}
        a={S.bce1[0] + 0.8} b={S.bce2[1] + 0.4} size={40} style={{ left: 90, width: 880, top: 260 }}
        terms={{ a: [S.bce1[0] + db * 0.3, S.bce1[1]], b: [S.bce2[0], frac("bce2", 0.45)] }}
        note="y = 1 עצם · y = 0 ריק · ממוצע על כל N התאים" />
      <Code x={110} y={600} w={840} size={24} o={ramp(t, frac("bce2", 0.5), frac("bce2", 0.6))} lines={[
        "loss_fn = nn.BCEWithLogitsLoss()   # sigmoid + BCE, numerically stable",
        "loss = loss_fn(logits, target)     # logits: 4x1x64x64x64",
      ]} />
    </div>
  );
}

// ---------------- class imbalance ----------------
function Imbalance({ t }: { t: number }) {
  const o = win(t, "imb1", "imb2");
  if (o <= 0) return null;
  const bar = ramp(t, frac("imb1", 0.6), frac("imb1", 0.8));
  const empty = ramp(t, S.imb2[0] + 0.2, S.imb2[0] + 1.2);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Panel style={{ left: 90, top: 230, width: 640, padding: "28px 34px" }}>
        <div style={{ fontSize: 30, color: C.dim }}>מפת התפוסה של האגן שלנו</div>
        <div style={{ fontSize: 40, direction: "ltr", textAlign: "left", marginTop: 10 }}>
          <span style={{ color: C.warm, fontWeight: 700 }}>{fmt(d11.occupied)}</span> / {fmt(d11.cells)}</div>
        <div style={{ fontSize: 30, marginTop: 4 }}>תאי עצם: <b style={{ color: C.warm }}>{pct(d11.frac)}</b></div>
        <div style={{ position: "relative", height: 46, marginTop: 22, borderRadius: 8, background: "rgba(127,208,255,0.18)", opacity: bar, direction: "ltr" }}>
          <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${100 * d11.frac}%`, background: C.warm, borderRadius: 8 }} />
          <div style={{ position: "absolute", right: 12, top: 8, fontSize: 22, color: C.dim, direction: "rtl" }}>ריק: {pct(1 - d11.frac)}</div>
        </div>
      </Panel>
      <Panel o={empty} style={{ left: 90, top: 600, width: 640, padding: "24px 34px", border: `1.5px solid ${C.bad}` }}>
        <div style={{ fontSize: 30, color: C.bad, fontWeight: 700 }}>ניחוש: "ריק בכל מקום"</div>
        <div style={{ fontSize: 30, marginTop: 10 }}>תאים נכונים: <b>{pct(d11.acc_empty)}</b></div>
        <div style={{ fontSize: 30 }}>תאי עצם שנמצאו: <b>0</b></div>
      </Panel>
    </div>
  );
}
function ImbScene({ t, occ }: { t: number; occ: Occ }) {
  const on = win(t, "imb1", "imb2", 0.8, 0.8, 0.5);
  const cube = occ.n * occ.cell;
  const c = useCenter(occ);
  const keys: CamKey[] = [
    { t: S.imb1[0] - 1, pos: orbit(-0.45, 7.6, 2.4), look: [c[0] + 1.9, c[1], c[2]] },
    { t: S.imb2[1] + 1, pos: orbit(-0.75, 7.6, 2.9), look: [c[0] + 1.9, c[1], c[2]] },
  ];
  return (
    <>
      <CameraPath t={t} keys={keys} />
      <Voxels centers={occ.centers} cell={occ.cell} opacity={on} gap={0.1} color="#ffcf7f" />
      <WireBox size={[cube, cube, cube]} position={c} opacity={on * 0.8} />
    </>
  );
}
function useCenter(occ: Occ): [number, number, number] {
  const n = occ.centers.length / 3;
  const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) { const v = occ.centers[3 * i + a]; lo[a] = Math.min(lo[a], v); hi[a] = Math.max(hi[a], v); }
  return [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2) as [number, number, number];
}

// ---------------- soft Dice ----------------
const G = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0];
const PR = [0.1, 0.05, 0.3, 0.7, 0.9, 0.85, 0.8, 0.4, 0.2, 0.05, 0, 0.1];
function DiceDemo({ t }: { t: number }) {
  const o = win(t, "dice1", "dice3");
  if (o <= 0) return null;
  const strip = ramp(t, frac("dice1", 0.25), frac("dice1", 0.35));
  const emp = window01(t, S.dice2[0] + 0.3, S.dice2[1] + 0.3, 0.5, 0.4);
  const P = emp > 0.5 ? PR.map(() => 0) : PR;
  const inter = P.reduce((s, p, i) => s + p * G[i], 0), sp = P.reduce((s, p) => s + p, 0), sg = G.reduce((s, g) => s + g, 0);
  const dice = (2 * inter + 1e-6) / (sp + sg + 1e-6);
  const comb = ramp(t, S.dice3[0] + 0.3, S.dice3[0] + 1.2);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Eq tex={"\\text{Dice}_{soft} = \\frac{2\\,\\hl{i}{\\sum_i p_i\\, g_i} + \\hl{e}{\\varepsilon}}{\\hl{p}{\\sum_i p_i} + \\hl{g}{\\sum_i g_i} + \\hl{e}{\\varepsilon}}"} t={t}
        a={S.dice1[0] + 0.5} b={S.dice3[0] + 0.6} size={46} style={{ left: 560, width: 800, top: 170, padding: "18px 30px" }}
        terms={{ i: [frac("dice1", 0.3), frac("dice1", 0.5)], p: [frac("dice1", 0.48), frac("dice1", 0.62)], g: [frac("dice1", 0.58), frac("dice1", 0.7)], e: [frac("dice1", 0.7), frac("dice1", 0.84)] }} />
      <div style={{ opacity: strip }}>
        <T x={300} y={520} size={28} color={C.dim} rtl>אמת g</T>
        <T x={300} y={600} size={28} color={C.dim} rtl>ניחוש p</T>
        <T x={300} y={680} size={28} color={C.dim} rtl>p · g</T>
        <CellGrid x={400} y={490} rows={1} cols={12} size={80} font={24} cell={(_, j) => ({ bg: G[j] ? "rgba(255,207,127,0.75)" : "rgba(30,45,65,0.8)", text: G[j], color: G[j] ? C.bg : C.dim })} />
        <CellGrid x={400} y={570} rows={1} cols={12} size={80} font={24} cell={(_, j) => ({ bg: `rgba(127,208,255,${0.1 + 0.7 * P[j]})`, text: P[j].toFixed(2) })} />
        <CellGrid x={400} y={650} rows={1} cols={12} size={80} font={24} cell={(_, j) => ({ bg: P[j] * G[j] > 0 ? "rgba(125,255,176,0.45)" : "rgba(20,30,45,0.6)", text: (P[j] * G[j]).toFixed(2), color: P[j] * G[j] > 0 ? "white" : C.dim })} />
        <T x={960} y={790} size={36} color={emp > 0.5 ? C.bad : C.warm} weight={700}>{`Dice = 2·${inter.toFixed(2)} / (${sp.toFixed(2)} + ${sg}) = ${dice.toFixed(2)}`}</T>
        <T x={960} y={838} size={24} color={C.dim} rtl o={1 - emp}>דוגמה קטנה (12 תאים), להמחשה</T>
        <T x={960} y={838} size={24} color={C.bad} rtl o={emp}>{`הניחוש "ריק בכל מקום": Dice = 0 (ולא ${pct(d11.acc_empty)})`}</T>
      </div>
      <Panel o={comb} style={{ left: 560, right: 560, top: 170, padding: "18px 30px", textAlign: "center", border: `1.5px solid ${C.warm}` }}>
        <div style={{ fontSize: 26, color: C.dim }}>ההפסד שלנו</div>
        <div style={{ fontSize: 50, direction: "ltr", color: "white" }}>L = BCE + (1 − Dice<sub style={{ fontSize: 26 }}>soft</sub>)</div>
      </Panel>
    </div>
  );
}

// ---------------- split by patient (leakage) ----------------
const NP = 10, PER = 2;
const PCOL = ["#7fd0ff", "#ffcf7f", "#7dffb0", "#ff7f8a", "#c79bff", "#ffa66b", "#6bd6c9", "#f2e27a", "#9fb7ff", "#ff9bd2"];
const hash = (i: number) => { const s = Math.sin(i * 91.17 + 7.3) * 43758.5453; return s - Math.floor(s); };
function Split({ t }: { t: number }) {
  const o = win(t, "split1", "split3");
  if (o <= 0) return null;
  const bySample = ramp(t, S.split2[0] + 0.3, S.split2[0] + 2.0);
  const byPatient = ramp(t, S.split3[0] + 0.2, S.split3[0] + 2.0);
  const leakOn = window01(t, frac("split2", 0.45), S.split3[0] + 0.3, 0.4, 0.3);
  const cards = Array.from({ length: NP * PER }, (_, k) => ({ p: Math.floor(k / PER), s: k % PER }));
  // pool position
  const pool = (k: number) => ({ x: 290 + Math.floor(k / PER) * 140 + (k % PER) * 56, y: 270 });
  // bins: train [0], val [1], test [2]
  const binX = [130, 1130, 1450], binW = [940, 260, 340], binY = 560;
  const slot = (bin: number, idx: number) => ({ x: binX[bin] + 24 + (idx % (bin === 0 ? 13 : 4)) * 68, y: binY + 70 + Math.floor(idx / (bin === 0 ? 13 : 4)) * 92 });
  // sample split: 14 train / 2 val / 4 test chosen by a fixed pseudo-random order
  const order = cards.map((_, k) => k).sort((a, b) => hash(a) - hash(b));
  const sBin = new Map<number, [number, number]>();
  order.forEach((k, r) => sBin.set(k, r < 14 ? [0, r] : r < 16 ? [1, r - 14] : [2, r - 16]));
  // patient split: patients 0-6 train, 7 val, 8-9 test
  const pBin = (k: number): [number, number] => { const p = Math.floor(k / PER); return p < 7 ? [0, k] : p === 7 ? [1, k - 14] : [2, k - 16]; };
  const testP = new Set(cards.filter((_, k) => sBin.get(k)![0] === 2).map((c) => c.p));
  const trainP = new Set(cards.filter((_, k) => sBin.get(k)![0] === 0).map((c) => c.p));
  const leaky = (p: number) => testP.has(p) && trainP.has(p);
  const names = ["אימון 70%", "ולידציה 10%", "בדיקה 20%"];
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <T x={960} y={205} size={26} color={C.dim} rtl>10 נבדקים · מכל CT שני זוגות צילומים (צבע = נבדק)</T>
      {[0, 1, 2].map((b) => (
        <div key={b} style={{ position: "absolute", left: binX[b], top: binY, width: binW[b], height: 260, borderRadius: 18,
          border: `1.5px dashed ${b === 0 ? C.xray : b === 1 ? C.good : C.warm}`, opacity: Math.max(bySample, byPatient) }}>
          <div style={{ fontFamily: FONT, fontSize: 28, color: C.text, textAlign: "center", marginTop: 12, direction: "rtl" }}>{names[b]}</div>
        </div>
      ))}
      {cards.map((c, k) => {
        const a = pool(k), sb = sBin.get(k)!, pb = pBin(k);
        const s1 = slot(sb[0], sb[1]), s2 = slot(pb[0], pb[1]);
        const x = lerp(lerp(a.x, s1.x, bySample), s2.x, byPatient), y = lerp(lerp(a.y, s1.y, bySample), s2.y, byPatient);
        const bad = leaky(c.p) && leakOn > 0 && byPatient < 0.5;
        return (
          <div key={k} style={{ position: "absolute", left: x, top: y, width: 50, height: 70, borderRadius: 8, background: PCOL[c.p],
            opacity: 0.92, border: bad ? `4px solid ${C.bad}` : "2px solid rgba(0,0,0,0.4)", boxShadow: bad ? `0 0 20px ${C.bad}` : "none",
            display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FONT, fontSize: 24, color: C.bg, fontWeight: 700 }}>
            {String.fromCharCode(65 + c.p)}</div>
        );
      })}
      <T x={960} y={880} size={30} color={C.bad} o={leakOn * (1 - byPatient)} rtl>אותו נבדק גם באימון וגם בבדיקה = דליפה</T>
      <T x={960} y={880} size={30} color={C.good} o={byPatient} rtl>כל נבדק בקבוצה אחת בלבד · נשמר ב-splits/split_v1.json</T>
    </div>
  );
}

// ---------------- augmentation ----------------
function Augment({ t }: { t: number }) {
  const o = win(t, "aug1", "aug2");
  if (o <= 0) return null;
  const geo = ramp(t, S.aug2[0] + 0.2, S.aug2[0] + 1.2);
  const tiles = [["data/hero/drr_ap.png", "מקור"], ["data/v11/aug_bright.png", "בהירות"], ["data/v11/aug_contrast.png", "ניגודיות"], ["data/v11/aug_noise.png", "רעש"]];
  const w = 360, h = w * 235 / 248;
  const ok = ramp(t, frac("aug2", 0.3), frac("aug2", 0.4)), bad = ramp(t, frac("aug2", 0.45), frac("aug2", 0.55));
  const tw = 470, th = tw * 235 / 248;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <div style={{ opacity: 1 - geo }}>
        {tiles.map(([src, lab], i) => (
          <div key={src} style={{ opacity: ramp(t, frac("aug1", 0.3 + i * 0.12), frac("aug1", 0.36 + i * 0.12)) }}>
            <div style={{ position: "absolute", left: 110 + i * (w + 50), top: 260, width: w }}>
              <Img src={staticFile(src)} style={{ width: w, display: "block", borderRadius: 8, border: `1px solid ${C.line}` }} />
              <Img src={staticFile("data/v11/target_outline.png")} style={{ width: w, position: "absolute", left: 0, top: 0, opacity: 0.85 }} />
            </div>
            <T x={110 + i * (w + 50) + w / 2} y={260 + h + 36} size={30} color={C.text} rtl>{lab}</T>
          </div>
        ))}
        <T x={960} y={800} size={28} color={C.warm} rtl>קו כתום = המטרה (צללית האגן). בשינויי עוצמה היא לא זזה</T>
      </div>
      <div style={{ opacity: geo }}>
        {[["data/v11/target_outline_shift.png", "הצילום והמטרה זזו יחד ✓", C.good, ok], ["data/v11/target_outline.png", "רק הצילום זז ✗", C.bad, bad]].map(([ol, lab, col, k], i) => (
          <div key={i} style={{ opacity: k as number }}>
            <div style={{ position: "absolute", left: 400 + i * (tw + 180), top: 230, width: tw }}>
              <Img src={staticFile("data/v11/aug_shift.png")} style={{ width: tw, display: "block", borderRadius: 8, border: `2px solid ${col}` }} />
              <Img src={staticFile(ol as string)} style={{ width: tw, position: "absolute", left: 0, top: 0 }} />
            </div>
            <T x={400 + i * (tw + 180) + tw / 2} y={230 + th + 40} size={32} color={col as string} rtl>{lab as string}</T>
          </div>
        ))}
        <T x={960} y={850} size={26} color={C.dim} rtl o={ok}>{`הזזה של ${d11.shift_px} פיקסלים (33 מ"מ) ימינה במסך`}</T>
      </div>
    </div>
  );
}

// ---------------- training loop, overfit test ----------------
const LOOP = [
  "model.train()",
  "for x_ap, x_lat, y in train_loader:      # y: 4x1x64x64x64",
  "    logits = model(x_ap, x_lat)",
  "    loss = bce(logits, y) + (1 - soft_dice(logits, y))",
  "    opt.zero_grad()",
  "    loss.backward()",
  "    opt.step()",
];
const VAL = [
  "model.eval()",
  "with torch.no_grad():",
  "    for x_ap, x_lat, y in val_loader:",
  "        pred = torch.sigmoid(model(x_ap, x_lat)) > 0.5",
  "        dices.append(dice(pred, y))",
  "log(epoch, mean(dices))     # validation Dice per epoch",
];
function Loop({ t }: { t: number }) {
  const o = win(t, "loop1", "loop2");
  if (o <= 0) return null;
  const d1 = S.loop1[1] - S.loop1[0];
  const steps = [[1], [2], [3], [4], [5], [6]];
  const idx = Math.min(5, Math.floor(Math.max(0, t - S.loop1[0] - 1.0) / ((d1 - 1.5) / 6)));
  const hi = t < S.loop1[1] + 0.3 ? steps[idx] : [];
  const v = ramp(t, S.loop2[0] + 0.2, S.loop2[0] + 1.0);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Code x={110} y={200} w={1000} size={27} lines={LOOP} hi={hi} title="train · epoch אחד" />
      <Code x={110} y={570} w={1000} size={27} lines={VAL} o={v} hi={t > frac("loop2", 0.4) ? [3, 4, 5] : []} title="validation · בסוף כל epoch" />
      <div style={{ position: "absolute", left: 1200, top: 220, width: 620, opacity: 1 - 0.5 * v, fontFamily: FONT }}>
        {["forward", "loss", "zero_grad", "backward", "step"].map((s, i) => {
          const on = t < S.loop1[1] + 0.3 && Math.max(0, idx - 1) === i;
          return (
            <div key={s} style={{ display: "flex", alignItems: "center", gap: 18, marginBottom: 16, direction: "ltr" }}>
              <div style={{ width: 50, height: 50, borderRadius: 25, background: on ? C.warm : "rgba(127,208,255,0.2)", color: on ? C.bg : C.text,
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, fontWeight: 700 }}>{i + 1}</div>
              <div style={{ fontSize: 34, color: on ? C.warm : C.text }}>{s}</div>
            </div>
          );
        })}
        <div style={{ fontSize: 28, color: C.dim, direction: "rtl", marginTop: 10 }}>↻ לכל batch</div>
      </div>
    </div>
  );
}
const xs = Array.from({ length: 61 }, (_, i) => i);
function Overfit({ t }: { t: number }) {
  const o = win(t, "overfit", "overfit", 0.6, 0.6, 0.6);
  if (o <= 0) return null;
  const up = ramp(t, S.overfit[0] + 1.0, frac("overfit", 0.7), (x) => x);
  const loss = xs.map((x) => 1.25 * Math.exp(-x / 9) + 0.02 + 0.03 * Math.exp(-x / 4) * Math.sin(x));
  const dice = xs.map((x) => 0.98 - 0.93 * Math.exp(-x / 10));
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Plot x={260} y={290} w={620} h={420} xMax={60} yMax={1.4} upto={up} ticksY={[0, 0.5, 1]} ticksX={[0, 30, 60]}
        series={[{ ys: loss, color: C.xray, label: "loss" }]} xLabel="צעדים על אותו batch" />
      <Plot x={1060} y={290} w={620} h={420} xMax={60} yMax={1} upto={up} ticksY={[0, 0.5, 1]} ticksX={[0, 30, 60]}
        series={[{ ys: dice, color: C.warm, label: "Dice" }]} xLabel="צעדים על אותו batch" />
      <Illustrative x={840} y={200} />
      <T x={960} y={850} size={30} color={C.text} rtl o={ramp(t, frac("overfit", 0.7), frac("overfit", 0.8))}>אם הרשת לא מצליחה לשנן batch אחד ← יש באג</T>
    </div>
  );
}

// ---------------- reading loss curves (illustrations) ----------------
function Curves({ t }: { t: number }) {
  const o = win(t, "curves1", "curves2");
  if (o <= 0) return null;
  const k = (i: number) => (i === 0 ? ramp(t, S.curves1[0] + 2.5, S.curves1[0] + 3.5) : ramp(t, frac("curves2", i === 1 ? 0.05 : 0.45), frac("curves2", i === 1 ? 0.15 : 0.55)));
  const best = ramp(t, frac("curves2", 0.7), frac("curves2", 0.8));
  const C3 = [
    { name: "underfitting", tr: xs.map((x) => 0.72 + 0.2 * Math.exp(-x / 12)), va: xs.map((x) => 0.76 + 0.2 * Math.exp(-x / 12)) },
    { name: "overfitting", tr: xs.map((x) => 0.85 * Math.exp(-x / 13) + 0.05), va: xs.map((x) => 0.85 * Math.exp(-x / 13) + 0.2 + 0.0002 * Math.max(0, x - 14) ** 2) },
    { name: "בריא", tr: xs.map((x) => 0.8 * Math.exp(-x / 16) + 0.14), va: xs.map((x) => 0.8 * Math.exp(-x / 16) + 0.2) },
  ];
  const vmin = (va: number[]) => va.reduce((b, v, i) => (v < va[b] ? i : b), 0);
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <Illustrative x={820} y={175} text="המחשה · עקומות מומצאות, לא תוצאות" />
      {C3.map((c, i) => {
        const x0 = 130 + i * 600, bi = vmin(c.va);
        return (
          <div key={i} style={{ opacity: k(i) }}>
            <T x={x0 + 240} y={290} size={34} color={i === 2 ? C.good : C.warm}>{c.name}</T>
            <Plot x={x0} y={340} w={480} h={340} xMax={60} yMax={1.1} ticksY={[0, 0.5, 1]}
              series={[{ ys: c.tr, color: C.xray, label: "train" }, { ys: c.va, color: C.warm, label: "validation", dash: "10 6" }]}
              xLabel="epoch" yLabel="loss" />
            {i > 0 && <div style={{ position: "absolute", left: x0 + (bi / 60) * 480 - 10, top: 340 + 340 - (c.va[bi] / 1.1) * 340 - 10, width: 20, height: 20,
              borderRadius: 10, background: C.good, opacity: best, boxShadow: `0 0 16px ${C.good}` }} />}
          </div>
        );
      })}
      <T x={960} y={830} size={28} color={C.good} rtl o={best}>● ה-checkpoint הטוב ביותר ב-validation</T>
    </div>
  );
}

// ---------------- Colab / Kaggle in practice ----------------
function Practice({ t }: { t: number }) {
  const o = win(t, "gpu1", "gpu3");
  if (o <= 0) return null;
  const X0 = 200, W = 1520, Y = 330;
  const hx = (h: number) => X0 + (h / 4) * W;
  const ck = ramp(t, S.gpu2[0] + 0.3, frac("gpu2", 0.6));
  const drop = ramp(t, frac("gpu2", 0.62), frac("gpu2", 0.7));
  const seeds = ramp(t, S.gpu3[0] + 0.2, S.gpu3[0] + 1.0);
  const nCk = 12;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o }}>
      <div style={{ opacity: 1 - 0.65 * seeds }}>
        <T x={960} y={215} size={30} color={C.dim} rtl>Colab / Kaggle · GPU (למשל T4) · ריצה אחת</T>
        <Svg>
          <line x1={X0} y1={Y} x2={X0 + W} y2={Y} stroke={C.xray} strokeWidth={10} strokeLinecap="round" opacity={0.7} />
          {[0, 1, 2, 3, 4].map((h) => <g key={h}><line x1={hx(h)} y1={Y + 14} x2={hx(h)} y2={Y + 30} stroke={C.dim} strokeWidth={2} />
            <text x={hx(h)} y={Y + 60} fill={C.dim} fontSize={24} textAnchor="middle">{h}h</text></g>)}
          <line x1={hx(4)} y1={Y - 50} x2={hx(4)} y2={Y + 30} stroke={C.warm} strokeWidth={4} />
          {Array.from({ length: nCk }, (_, i) => (i + 1) / (nCk + 1) * 4).filter((h) => h / 4 <= ck).map((h, i) => (
            <rect key={i} x={hx(h) - 8} y={Y - 8} width={16} height={16} rx={3} fill={C.good} opacity={drop > 0 && h > 2.4 ? 0.25 : 1} />))}
          {drop > 0 && <g opacity={drop}>
            <text x={hx(2.5)} y={Y - 30} fill={C.bad} fontSize={44} textAnchor="middle">⚡</text>
            <path d={`M${hx(2.5)},${Y + 80} C${hx(2.5)},${Y + 150} ${hx(2.15)},${Y + 150} ${hx(2.15)},${Y + 24}`} fill="none" stroke={C.good} strokeWidth={3} strokeDasharray="8 6" markerEnd="url(#ahg)" />
          </g>}
        </Svg>
        <T x={hx(4)} y={Y - 70} size={26} color={C.warm} rtl>כל ריצה מתחת ל-4 שעות</T>
        <T x={hx(1.1)} y={Y - 50} size={24} color={C.good} rtl o={ck}>■ checkpoint ל-Drive כל כמה מאות צעדים</T>
        <T x={hx(2.6)} y={Y + 190} size={24} color={C.good} rtl o={drop}>הסשן נפל ← ממשיכים מה-checkpoint האחרון</T>
        <Code x={200} y={600} w={1520} size={25} o={ck} lines={[
          "torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step},",
          "           '/content/drive/MyDrive/pelvis/ckpt_last.pt')",
        ]} />
      </div>
      <Code x={420} y={420} w={1080} size={28} o={seeds} title="בתחילת כל מחברת" lines={[
        "import random, numpy as np, torch",
        "SEED = 0                        # write it in the run log",
        "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)",
      ]} />
    </div>
  );
}

export const V11 = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const occ = useJson<Occ>("data/hero/occ64.json");
  const in3d = t > S.imb1[0] - 1.2 && t < S.imb2[1] + 1.2;
  return (
    <AbsoluteFill style={{ backgroundColor: C.bg }}>
      {occ && in3d && <Stage><ImbScene t={t} occ={occ} /></Stage>}

      <Opening n={11} title="לאמן את הרשת" active={[2, 3]} t={t} end={2.9} />
      <Intro t={t} />

      <Title text="מ-logit להסתברות: sigmoid" t={t} a={S.logit1[0]} b={S.logit2[1] + 0.4} top={60} size={54} />
      <SigmoidPlot t={t} />
      <Title text="Binary cross-entropy" t={t} a={S.bce1[0]} b={S.bce2[1] + 0.4} top={60} size={54} />
      <BCEPlot t={t} />
      <Title text="הקובייה כמעט ריקה" t={t} a={S.imb1[0]} b={S.imb2[1] + 0.4} top={60} size={54} />
      <Imbalance t={t} />
      <Title text="Soft Dice" t={t} a={S.dice1[0]} b={S.dice2[1] + 0.3} top={50} size={54} />
      <Title text="BCE + Dice" t={t} a={S.dice3[0]} b={S.dice3[1] + 0.4} top={50} size={54} />
      <DiceDemo t={t} />

      <Title text="חלוקה לפי נבדק" t={t} a={S.split1[0]} b={S.split3[1] + 0.4} top={60} size={54} />
      <Split t={t} />
      <Title text="Augmentation שומר על הגאומטריה" t={t} a={S.aug1[0]} b={S.aug2[1] + 0.4} top={60} size={54} />
      <Augment t={t} />

      <Title text="לולאת האימון" t={t} a={S.loop1[0]} b={S.loop2[1] + 0.4} top={60} size={54} />
      <Loop t={t} />
      <Title text="קודם: overfit על batch אחד" t={t} a={S.overfit[0]} b={S.overfit[1] + 0.6} top={60} size={54} />
      <Overfit t={t} />
      <Title text="לקרוא עקומות הפסד" t={t} a={S.curves1[0]} b={S.curves2[1] + 0.4} top={60} size={54} />
      <Curves t={t} />
      <Title text="בפועל: Colab ו-Kaggle" t={t} a={S.gpu1[0]} b={S.gpu2[1] + 0.3} top={60} size={54} />
      <Title text="seeds: לשחזר ריצה" t={t} a={S.gpu3[0]} b={S.gpu3[1] + 0.4} top={60} size={54} />
      <Practice t={t} />

      <RecapCard t={t} a={S.recap[0]} b={S.recap[1] + 0.3} items={[
        "sigmoid הופך logits להסתברות; מסכה: p > 0.5",
        `ההפסד: BCE + (1 − soft Dice), כי רק ${pct(d11.frac)} מהתאים הם עצם`,
        "חלוקה לפי נבדק (70/10/20), קובץ קבוע",
        "augmentation גאומטרי חל גם על המטרה",
        "קודם overfit על batch אחד; Dice על validation כל epoch",
        "ריצות מתחת ל-4 שעות, checkpoints ל-Drive, seed קבוע",
      ]} />
      <YourTurn t={t} a={S.turn[0]} b={tl.total} notebook="units/05-network-training/notebook.ipynb" todos={[
        "צעד אימון: forward, BCE + soft Dice, backward, step",
        "מבחן overfit על batch אחד: loss → ~0, Dice → ~1",
        "ריצה ראשונה ב-Colab: checkpoints ל-Drive, Dice על validation בכל epoch",
      ]} />

      <Captions tl={tl} t={t} />
      <Credit t={t} a={S.intro[0]} />
      <Narration tl={tl} dir="v11" />
    </AbsoluteFill>
  );
};
