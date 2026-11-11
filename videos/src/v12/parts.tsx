// Small 2D building blocks shared by videos 12–14 (cards, image thumbs, the "illustration" badge, tables).
import { CSSProperties, ReactNode } from "react";
import { Img, staticFile } from "remotion";
import { C, FONT } from "../lib/style";

export function Box({ x, y, w, h, o = 1, children, border = C.line, bg = C.panel, style }: {
  x: number; y: number; w: number; h?: number; o?: number; children?: ReactNode; border?: string; bg?: string; style?: CSSProperties;
}) {
  if (o <= 0.001) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, opacity: o, borderRadius: 20,
      border: `2px solid ${border}`, background: bg, fontFamily: FONT, color: C.text, direction: "rtl",
      boxShadow: "0 0 40px rgba(60,140,255,0.12)", boxSizing: "border-box", ...style }}>{children}</div>
  );
}

export function Thumb({ src, x, y, h, w, o = 1, label, border = C.line, style, fit = "cover" }: {
  src: string; x: number; y: number; h: number; w?: number; o?: number; label?: ReactNode; border?: string;
  style?: CSSProperties; fit?: "cover" | "contain" | "fill";
}) {
  if (o <= 0.001) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, opacity: o, fontFamily: FONT }}>
      <Img src={staticFile(src)} style={{ height: h, width: w, objectFit: fit, display: "block", borderRadius: 10,
        border: `2px solid ${border}`, background: "#000", ...style }} />
      {label && <div style={{ textAlign: "center", fontSize: 26, color: C.dim, marginTop: 8, direction: "rtl" }}>{label}</div>}
    </div>
  );
}

/** Badge that marks anything that is NOT a project result */
export function IllusBadge({ o, x = 960, y = 210, text = "המחשה: מספרים מדומים" }: { o: number; x?: number; y?: number; text?: string }) {
  if (o <= 0.001) return null;
  return (
    <div style={{ position: "absolute", left: x, top: y, transform: "translate(-50%, -50%)", opacity: o, fontFamily: FONT,
      direction: "rtl", fontSize: 28, fontWeight: 700, color: "#1a0d00", background: C.warm, padding: "8px 22px",
      borderRadius: 12, whiteSpace: "nowrap", boxShadow: "0 0 30px rgba(255,207,127,0.35)" }}>⚠ {text}</div>
  );
}

/** A results table; `cells[r][c]` is shown as-is (use "—" for empty slots) */
export function Table({ x, y, w, cols, rows, cells, o = 1, rowO, hl, title, foot, colW, rowH = 74, size = 32 }: {
  x: number; y: number; w: number; cols: ReactNode[]; rows: ReactNode[]; cells: ReactNode[][]; o?: number;
  rowO?: number[]; hl?: { r: number; c: number; k: number }[]; title?: ReactNode; foot?: ReactNode; colW?: number; rowH?: number; size?: number;
}) {
  if (o <= 0.001) return null;
  const firstW = colW ?? w * 0.3;
  const cw = (w - firstW) / cols.length;
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, opacity: o, fontFamily: FONT, color: C.text }}>
      {title && <div style={{ direction: "rtl", fontSize: 28, color: C.dim, marginBottom: 12, textAlign: "right" }}>{title}</div>}
      <div style={{ border: `2px solid ${C.line}`, borderRadius: 18, overflow: "hidden", background: C.panel }}>
        <div style={{ display: "flex", direction: "rtl", height: rowH, alignItems: "center", background: "rgba(127,208,255,0.10)",
          borderBottom: `2px solid ${C.line}` }}>
          <div style={{ width: firstW, padding: "0 24px", fontSize: size * 0.85, color: C.dim }}>זרוע</div>
          {cols.map((c, i) => <div key={i} style={{ width: cw, textAlign: "center", fontSize: size * 0.9, color: C.xray, fontWeight: 700 }}>{c}</div>)}
        </div>
        {rows.map((r, ri) => (
          <div key={ri} style={{ display: "flex", direction: "rtl", height: rowH, alignItems: "center", opacity: rowO ? rowO[ri] : 1,
            borderBottom: ri < rows.length - 1 ? `1px solid rgba(120,190,255,0.2)` : "none" }}>
            <div style={{ width: firstW, padding: "0 24px", fontSize: size }}>{r}</div>
            {cells[ri].map((v, ci) => {
              const k = hl?.find((h) => h.r === ri && h.c === ci)?.k ?? 0;
              return <div key={ci} style={{ width: cw, textAlign: "center", fontSize: size, direction: "ltr",
                color: k > 0.5 ? C.warm : C.text, background: `rgba(255,207,127,${0.16 * k})`, height: "100%",
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: k > 0 ? `inset 0 0 0 2px rgba(255,207,127,${k})` : "none" }}>{v}</div>;
            })}
          </div>
        ))}
      </div>
      {foot && <div style={{ direction: "rtl", fontSize: 24, color: C.dim, marginTop: 12, textAlign: "right", lineHeight: 1.5 }}>{foot}</div>}
    </div>
  );
}

/** Simple RTL text block at a position */
export function Txt({ x, y, w, o = 1, children, size = 34, color = C.text, align = "right", weight = 400 }: {
  x: number; y: number; w: number; o?: number; children: ReactNode; size?: number; color?: string; align?: "right" | "center" | "left"; weight?: number;
}) {
  if (o <= 0.001) return null;
  return <div style={{ position: "absolute", left: x, top: y, width: w, opacity: o, fontFamily: FONT, fontSize: size, color,
    direction: "rtl", textAlign: align, lineHeight: 1.45, fontWeight: weight, textShadow: "0 2px 12px rgba(0,0,0,0.9)" }}>{children}</div>;
}

/** Credit for the datasets beyond TotalSegmentator (bottom-left, under the shared Credit line) */
export function ExtraCredit({ o, text }: { o: number; text: string }) {
  return <div style={{ position: "absolute", bottom: 4, left: 40, fontFamily: FONT, fontSize: 18, color: C.dim,
    opacity: 0.7 * o, direction: "ltr" }}>{text}</div>;
}
