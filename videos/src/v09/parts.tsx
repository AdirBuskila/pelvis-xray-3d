// 2D diagram pieces for videos 09-11 (grids of numbers, image tiles, arrows, code, plots, blocks).
// All are pure functions of their props; screen coordinates are 1920x1080 pixels.
import { CSSProperties, ReactNode } from "react";
import { Img, staticFile } from "remotion";
import { C, FONT } from "../lib/style";

/** diverging colour for signed values: negative blue, positive orange; |v|/scale clamped */
export const signed = (v: number, scale: number, alpha = 1) => {
  const k = Math.max(-1, Math.min(1, v / scale));
  return k >= 0
    ? `rgba(${Math.round(40 + 215 * k)},${Math.round(40 + 150 * k)},${Math.round(50 + 40 * k)},${alpha})`
    : `rgba(${Math.round(40 + 30 * -k)},${Math.round(40 + 130 * -k)},${Math.round(50 + 205 * -k)},${alpha})`;
};
export const gray = (v: number, alpha = 1) => `rgba(${v},${v},${v},${alpha})`;

/** a grid of cells with optional numbers; cell(i,j) returns {bg, text, color, border} */
export function CellGrid({ x, y, rows, cols, size, o = 1, cell, gap = 2, font }: {
  x: number; y: number; rows: number; cols: number; size: number; o?: number; gap?: number; font?: number;
  cell: (i: number, j: number) => { bg?: string; text?: ReactNode; color?: string; border?: string; o?: number } | null;
}) {
  if (o <= 0) return null;
  const items: ReactNode[] = [];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const c = cell(i, j);
    if (!c) continue;
    items.push(
      <div key={`${i}-${j}`} style={{
        position: "absolute", left: j * size + gap / 2, top: i * size + gap / 2, width: size - gap, height: size - gap,
        background: c.bg ?? "rgba(20,30,45,0.8)", border: c.border ?? "none", boxSizing: "border-box", borderRadius: 4,
        display: "flex", alignItems: "center", justifyContent: "center", opacity: c.o ?? 1,
        fontFamily: FONT, fontWeight: 500, fontSize: font ?? size * 0.34, color: c.color ?? "white",
        textShadow: "0 1px 4px rgba(0,0,0,0.9)",
      }}>{c.text}</div>);
  }
  return <div style={{ position: "absolute", left: x, top: y, width: cols * size, height: rows * size, opacity: o, direction: "ltr" }}>{items}</div>;
}

/** a rectangle outline (e.g. the kernel window) at screen coords */
export function Frame({ x, y, w, h, o = 1, color = C.xray, width = 4, glow = true, radius = 6 }: {
  x: number; y: number; w: number; h: number; o?: number; color?: string; width?: number; glow?: boolean; radius?: number;
}) {
  if (o <= 0) return null;
  return <div style={{ position: "absolute", left: x, top: y, width: w, height: h, border: `${width}px solid ${color}`,
    borderRadius: radius, boxSizing: "border-box", opacity: o, boxShadow: glow ? `0 0 24px ${color}` : "none" }} />;
}

/** image tile with optional label underneath */
export function Tile({ src, x, y, w, h, o = 1, label, sub, pixelated = false, style, labelColor = C.text }: {
  src: string; x: number; y: number; w: number; h?: number; o?: number; label?: ReactNode; sub?: ReactNode;
  pixelated?: boolean; style?: CSSProperties; labelColor?: string;
}) {
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, opacity: o, fontFamily: FONT }}>
      <Img src={staticFile(src)} style={{ width: w, height: h, display: "block", borderRadius: 8, border: `1px solid ${C.line}`,
        imageRendering: pixelated ? "pixelated" : "auto", ...style }} />
      {label && <div style={{ textAlign: "center", marginTop: 12, fontSize: 30, color: labelColor, direction: "rtl" }}>{label}</div>}
      {sub && <div style={{ textAlign: "center", marginTop: 4, fontSize: 24, color: C.dim, direction: "ltr" }}>{sub}</div>}
    </div>
  );
}

/** SVG overlay covering the frame (for arrows and lines) */
export function Svg({ children, o = 1 }: { children: ReactNode; o?: number }) {
  if (o <= 0) return null;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, opacity: o, overflow: "visible", fontFamily: FONT }}>
      <defs>
        {[["ah", C.dim], ["ahx", C.xray], ["ahw", C.warm], ["ahg", C.good], ["ahb", C.bad]].map(([id, col]) => (
          <marker key={id} id={id} markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={col} />
          </marker>
        ))}
      </defs>
      {children}
    </svg>
  );
}
export function Arrow({ x1, y1, x2, y2, color = C.dim, w = 3, head = "ah", o = 1, dash }: {
  x1: number; y1: number; x2: number; y2: number; color?: string; w?: number; head?: string; o?: number; dash?: string;
}) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={w} markerEnd={`url(#${head})`} opacity={o} strokeDasharray={dash} />;
}

/** monospace code panel; lines appear progressively with `shown` (0..n), line `hi` highlighted */
export function Code({ lines, x, y, w, o = 1, shown, hi = [], size = 26, title }: {
  lines: string[]; x: number; y: number; w: number; o?: number; shown?: number; hi?: number[]; size?: number; title?: string;
}) {
  if (o <= 0) return null;
  const n = shown ?? lines.length;
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, opacity: o, background: "rgba(6,12,22,0.92)",
      border: `1.5px solid ${C.line}`, borderRadius: 18, padding: "22px 28px", direction: "ltr",
      boxShadow: "0 0 50px rgba(60,140,255,0.12)" }}>
      {title && <div style={{ fontFamily: FONT, fontSize: 24, color: C.dim, marginBottom: 10, direction: "rtl" }}>{title}</div>}
      {lines.map((l, i) => {
        const k = Math.max(0, Math.min(1, n - i));
        return (
          <div key={i} style={{ fontFamily: "Consolas, 'Courier New', monospace", fontSize: size, lineHeight: 1.5,
            whiteSpace: "pre", opacity: k, color: hi.includes(i) ? C.warm : l.trim().startsWith("#") ? "#6f8aa8" : "#d8e6ff",
            background: hi.includes(i) ? "rgba(255,207,127,0.10)" : "none", borderRadius: 6 }}>{l || " "}</div>
        );
      })}
    </div>
  );
}

/** labelled block (network layer); label below, inner text centred */
export function Block({ x, y, w, h, o = 1, color = C.xray, label, inner, fill, sub, labelTop = false }: {
  x: number; y: number; w: number; h: number; o?: number; color?: string; label?: ReactNode; inner?: ReactNode;
  fill?: string; sub?: ReactNode; labelTop?: boolean;
}) {
  if (o <= 0) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, opacity: o, fontFamily: FONT }}>
      <div style={{ width: w, height: h, borderRadius: 10, border: `2px solid ${color}`, boxSizing: "border-box",
        background: fill ?? "rgba(127,208,255,0.10)", boxShadow: `0 0 26px ${color}44`, display: "flex",
        alignItems: "center", justifyContent: "center", color: "white", fontSize: 24, textAlign: "center" }}>{inner}</div>
      {label && <div style={{ position: "absolute", [labelTop ? "bottom" : "top"]: h + 10, left: -60, right: -60, textAlign: "center",
        fontSize: 24, color: C.dim, direction: "ltr", lineHeight: 1.25 }}>{label}</div>}
      {sub && <div style={{ position: "absolute", top: h + 44, left: -60, right: -60, textAlign: "center", fontSize: 22, color: C.dim, direction: "rtl" }}>{sub}</div>}
    </div>
  );
}

/** small SVG line plot. series: arrays of y values over x = 0..n-1 (or given xs); draw fraction `upto` */
export function Plot({ x, y, w, h, o = 1, series, xMax, yMin = 0, yMax = 1, upto = 1, xLabel, yLabel, ticksY = [], ticksX = [], tag }: {
  x: number; y: number; w: number; h: number; o?: number; xMax: number; yMin?: number; yMax?: number; upto?: number;
  series: { xs?: number[]; ys: number[]; color: string; label?: string; dash?: string; width?: number }[];
  xLabel?: string; yLabel?: string; ticksY?: number[]; ticksX?: number[]; tag?: ReactNode;
}) {
  if (o <= 0) return null;
  const X = (v: number) => (v / xMax) * w, Y = (v: number) => h - ((v - yMin) / (yMax - yMin)) * h;
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, opacity: o, fontFamily: FONT }}>
      <svg width={w} height={h} style={{ overflow: "visible", position: "absolute", fontFamily: FONT }}>
        <line x1={0} y1={h} x2={w} y2={h} stroke={C.dim} strokeWidth={2} />
        <line x1={0} y1={0} x2={0} y2={h} stroke={C.dim} strokeWidth={2} />
        {ticksY.map((v) => <g key={v}><line x1={-8} y1={Y(v)} x2={w} y2={Y(v)} stroke="rgba(143,166,196,0.15)" />
          <text x={-14} y={Y(v) + 8} fill={C.dim} fontSize={22} textAnchor="end">{v}</text></g>)}
        {ticksX.map((v) => <text key={v} x={X(v)} y={h + 32} fill={C.dim} fontSize={22} textAnchor="middle">{v}</text>)}
        {series.map((s, si) => {
          const xs = s.xs ?? s.ys.map((_, i) => i);
          const n = Math.max(2, Math.ceil(xs.length * upto));
          const d = s.ys.slice(0, n).map((v, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
          return <path key={si} d={d} fill="none" stroke={s.color} strokeWidth={s.width ?? 4} strokeDasharray={s.dash}
            strokeLinejoin="round" style={{ filter: `drop-shadow(0 0 6px ${s.color})` }} />;
        })}
      </svg>
      {xLabel && <div style={{ position: "absolute", top: h + 46, width: w, textAlign: "center", fontSize: 24, color: C.dim, direction: "rtl" }}>{xLabel}</div>}
      {yLabel && <div style={{ position: "absolute", left: -20, top: -46, fontSize: 24, color: C.dim, direction: "rtl", whiteSpace: "nowrap" }}>{yLabel}</div>}
      <div style={{ position: "absolute", right: 0, top: -6, display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
        {series.filter((s) => s.label).map((s) => <div key={s.label} style={{ fontSize: 24, color: s.color, direction: "rtl" }}>— {s.label}</div>)}
      </div>
      {tag}
    </div>
  );
}

/** "illustration, not a result" badge */
export function Illustrative({ x, y, o = 1, text = "המחשה · לא תוצאה" }: { x: number; y: number; o?: number; text?: string }) {
  if (o <= 0) return null;
  return <div style={{ position: "absolute", left: x, top: y, opacity: o, fontFamily: FONT, fontSize: 24, color: C.warm,
    border: `1.5px dashed ${C.warm}`, borderRadius: 10, padding: "4px 14px", direction: "rtl", background: "rgba(40,28,8,0.6)" }}>{text}</div>;
}

/** simple text at screen coords, LTR by default (numbers, shapes) */
export function T({ x, y, children, o = 1, size = 28, color = C.text, align = "center", rtl = false, weight = 400, mono = false, style }: {
  x: number; y: number; children: ReactNode; o?: number; size?: number; color?: string; align?: "center" | "left" | "right";
  rtl?: boolean; weight?: number; mono?: boolean; style?: CSSProperties;
}) {
  if (o <= 0) return null;
  const tx = align === "center" ? "-50%" : align === "right" ? "-100%" : "0";
  return <div style={{ position: "absolute", left: x, top: y, transform: `translate(${tx}, -50%)`, opacity: o, whiteSpace: "nowrap",
    fontFamily: mono ? "Consolas, 'Courier New', monospace" : FONT, fontSize: size, color, fontWeight: weight,
    direction: rtl ? "rtl" : "ltr", textShadow: "0 2px 10px rgba(0,0,0,0.9)", ...style }}>{children}</div>;
}

/** fmt integer with thousands separators */
export const fmt = (n: number) => n.toLocaleString("en-US");
