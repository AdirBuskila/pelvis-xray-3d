// 2D overlays shared by every video: narration audio, captions, cards, equations, the pipeline map.
import { CSSProperties, ReactNode } from "react";
import { Audio, Sequence, staticFile, useVideoConfig } from "remotion";
import katex from "katex";
import "katex/dist/katex.min.css";
import { C, FONT } from "./style";
import { ramp, window01 } from "./anim";
import { Timeline } from "./timeline";

export function Narration({ tl, dir }: { tl: Timeline; dir: string }) {
  const { fps } = useVideoConfig();
  return (
    <>
      {tl.order.map((id) => (
        <Sequence key={id} from={Math.round(tl.seg[id][0] * fps)} layout="none">
          <Audio src={staticFile(`audio/${dir}/${id}.mp3`)} />
        </Sequence>
      ))}
    </>
  );
}

/** Subtitles: the segment's `caption` (or its narration text) while it plays */
export function Captions({ tl, t, bottom = 56 }: { tl: Timeline; t: number; bottom?: number }) {
  const cur = tl.script.segments.find((s) => t >= tl.seg[s.id][0] - 0.15 && t <= tl.seg[s.id][1] + 0.35);
  if (!cur) return null;
  const [a, b] = tl.seg[cur.id];
  const o = window01(t, a - 0.15, b + 0.35, 0.3, 0.3);
  return (
    <div style={{
      position: "absolute", bottom, left: 220, right: 220, textAlign: "center", direction: "rtl",
      fontFamily: FONT, fontSize: 36, lineHeight: 1.45, color: C.text, opacity: o,
      textShadow: "0 2px 16px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.9)",
    }}>{cur.caption ?? cur.text}</div>
  );
}

export function Panel({ children, style, o = 1 }: { children: ReactNode; style?: CSSProperties; o?: number }) {
  return (
    <div style={{
      position: "absolute", background: C.panel, border: `1.5px solid ${C.line}`, borderRadius: 26,
      boxShadow: "0 0 60px rgba(60,140,255,0.14)", padding: "34px 40px", opacity: o,
      transform: `translateY(${(1 - o) * 24}px)`, fontFamily: FONT, color: C.text, direction: "rtl", ...style,
    }}>{children}</div>
  );
}

export function Title({ text, sub, t, a, b, top = 86, size = 76 }: {
  text: string; sub?: string; t: number; a: number; b: number; top?: number; size?: number;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", top, width: "100%", textAlign: "center", opacity: o, direction: "rtl",
      transform: `translateY(${(1 - o) * -14}px)` }}>
      <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: size, color: "white", letterSpacing: -1,
        textShadow: "0 0 40px rgba(80,160,255,0.35)" }}>{text}</div>
      {sub && <div style={{ fontFamily: FONT, fontSize: size * 0.42, color: C.dim, marginTop: 10 }}>{sub}</div>}
    </div>
  );
}

/** KaTeX equation. `terms` maps a \htmlClass name to the [a,b] seconds it is highlighted. */
export function Eq({ tex, t, a, b, terms = {}, style, size = 64, note, color = C.warm }: {
  tex: string; t: number; a: number; b: number; terms?: Record<string, [number, number]>;
  style?: CSSProperties; size?: number; note?: string; color?: string;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  let src = tex;
  for (const [cls, [ta, tb]] of Object.entries(terms)) {
    const on = t >= ta && t <= tb;
    src = src.replaceAll(`\\hl{${cls}}`, `\\htmlStyle{color:${on ? color : "white"};}`);
  }
  const html = katex.renderToString(src, { trust: true, strict: false, displayMode: true });
  return (
    <Panel o={o} style={{ left: 110, top: 280, width: 780, ...style }}>
      <div style={{ fontSize: size, color: "white", direction: "ltr" }} dangerouslySetInnerHTML={{ __html: html }} />
      {note && <div style={{ fontSize: 28, color: C.dim, textAlign: "center", lineHeight: 1.6 }}>{note}</div>}
    </Panel>
  );
}

/** Inline (non-display) KaTeX for use inside text */
export const tex = (s: string) => (
  <span style={{ direction: "ltr", display: "inline-block" }}
    dangerouslySetInnerHTML={{ __html: katex.renderToString(s, { strict: false }) }} />
);

/** Bullet list where item i appears at times[i] */
export function Bullets({ items, times, t, style, o = 1, title }: {
  items: ReactNode[]; times: number[]; t: number; style?: CSSProperties; o?: number; title?: string;
}) {
  return (
    <Panel o={o} style={{ right: 110, top: 250, width: 760, ...style }}>
      {title && <div style={{ fontSize: 40, fontWeight: 700, marginBottom: 18, color: C.xray }}>{title}</div>}
      {items.map((it, i) => {
        const k = ramp(t, times[i], times[i] + 0.5);
        return (
          <div key={i} style={{ fontSize: 34, lineHeight: 1.5, marginBottom: 14, opacity: k,
            transform: `translateX(${(1 - k) * 30}px)`, display: "flex", gap: 16 }}>
            <span style={{ color: C.xray }}>●</span><span>{it}</span>
          </div>
        );
      })}
    </Panel>
  );
}

export const STAGES = ["CT", "DRR: AP + LAT", "רשת 2D→3D", "occupancy 64³", "mesh", "מדדים"];

/** The project pipeline with the current stages highlighted ("you are here") */
export function PipelineMap({ active, t, a, b, top = 460, label = "אנחנו כאן" }: {
  active: number[]; t: number; a: number; b: number; top?: number; label?: string;
}) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", top, width: "100%", display: "flex", justifyContent: "center",
      gap: 18, direction: "ltr", opacity: o, fontFamily: FONT }}>
      {STAGES.map((s, i) => {
        const on = active.includes(i);
        const k = ramp(t, a + 0.3 + i * 0.18, a + 0.8 + i * 0.18);
        return (
          <div key={s} style={{ display: "flex", alignItems: "center", gap: 18, opacity: k }}>
            <div style={{ position: "relative", padding: "18px 26px", borderRadius: 18, fontSize: 30,
              border: `2px solid ${on ? C.warm : C.line}`, color: on ? "white" : C.dim,
              background: on ? "rgba(255,207,127,0.12)" : "rgba(8,14,24,0.6)",
              boxShadow: on ? "0 0 40px rgba(255,207,127,0.35)" : "none", direction: "rtl" }}>
              {s}
              {on && i === Math.min(...active) && (
                <div style={{ position: "absolute", top: -46, left: 0, right: 0, textAlign: "center",
                  fontSize: 24, color: C.warm, whiteSpace: "nowrap" }}>▼ {label}</div>
              )}
            </div>
            {i < STAGES.length - 1 && <div style={{ color: C.dim, fontSize: 30 }}>→</div>}
          </div>
        );
      })}
    </div>
  );
}

/** Series opening: series title, video number + title, pipeline position */
export function Opening({ n, title, active, t, end }: { n: number; title: string; active: number[]; t: number; end: number }) {
  const o = window01(t, 0, end, 0.7, 0.8);
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", inset: 0, opacity: o, background: `radial-gradient(ellipse at center, rgba(20,40,70,0.55), ${C.bg} 70%)` }}>
      <div style={{ position: "absolute", top: 200, width: "100%", textAlign: "center", fontFamily: FONT, direction: "rtl" }}>
        <div style={{ fontSize: 30, color: C.dim, letterSpacing: 2 }}>משני צילומי רנטגן לאגן תלת-ממדי · סרטון {n}</div>
        <div style={{ fontSize: 92, fontWeight: 700, color: "white", marginTop: 18, letterSpacing: -1,
          textShadow: "0 0 50px rgba(80,160,255,0.4)" }}>{title}</div>
      </div>
      <PipelineMap active={active} t={t} a={0.4} b={end + 1} top={560} />
    </div>
  );
}

export function RecapCard({ items, t, a, b }: { items: ReactNode[]; t: number; a: number; b: number }) {
  const o = window01(t, a, b, 0.6, 0.6);
  if (o <= 0) return null;
  const step = Math.min(1.1, (b - a - 1.5) / Math.max(1, items.length));
  return (
    <Panel o={o} style={{ left: 300, right: 300, top: 170 }}>
      <div style={{ fontSize: 50, fontWeight: 700, color: C.xray, marginBottom: 24 }}>מה למדנו</div>
      {items.map((it, i) => {
        const k = ramp(t, a + 0.6 + i * step, a + 1.0 + i * step);
        return <div key={i} style={{ fontSize: 36, lineHeight: 1.55, opacity: k, marginBottom: 10 }}>
          <span style={{ color: C.good }}>✓ </span>{it}</div>;
      })}
    </Panel>
  );
}

export function YourTurn({ notebook, todos, t, a, b }: { notebook: string; todos: ReactNode[]; t: number; a: number; b: number }) {
  const o = window01(t, a, b, 0.6, 0.8);
  if (o <= 0) return null;
  return (
    <Panel o={o} style={{ left: 300, right: 300, top: 170, border: `1.5px solid ${C.warm}` }}>
      <div style={{ fontSize: 50, fontWeight: 700, color: C.warm, marginBottom: 8 }}>תורכם</div>
      <div style={{ fontSize: 30, color: C.dim, marginBottom: 24, direction: "ltr", textAlign: "right" }}>{notebook}</div>
      {todos.map((it, i) => {
        const k = ramp(t, a + 0.8 + i * 0.7, a + 1.2 + i * 0.7);
        return <div key={i} style={{ fontSize: 34, lineHeight: 1.55, opacity: k, marginBottom: 10 }}>
          <span style={{ color: C.warm, fontFamily: "monospace" }}>TODO {i + 1} · </span>{it}</div>;
      })}
    </Panel>
  );
}

export function Credit({ t, a }: { t: number; a: number }) {
  const o = ramp(t, a, a + 0.8);
  return (
    <div style={{ position: "absolute", bottom: 22, left: 40, fontFamily: FONT, fontSize: 18, color: C.dim, opacity: 0.7 * o, direction: "ltr" }}>
      CT data: TotalSegmentator (Wasserthal et al., 2023), CC BY 4.0
    </div>
  );
}

/** Small label anchored at screen coords */
export function Label({ x, y, text, o = 1, color = C.text, size = 30, align = "center" }: {
  x: number; y: number; text: ReactNode; o?: number; color?: string; size?: number; align?: "center" | "left" | "right";
}) {
  if (o <= 0) return null;
  const tx = align === "center" ? "-50%" : align === "right" ? "-100%" : "0";
  return (
    <div style={{ position: "absolute", left: x, top: y, transform: `translate(${tx}, -50%)`, opacity: o,
      fontFamily: FONT, fontSize: size, color, direction: "rtl", whiteSpace: "nowrap",
      textShadow: "0 2px 12px rgba(0,0,0,0.9)" }}>{text}</div>
  );
}
