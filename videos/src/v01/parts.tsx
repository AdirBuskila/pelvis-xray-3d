// Video 1 helpers (also used by video 2): CT volume slice shader, 3D arrows, screen projection of 3D
// points for DOM labels, number grids and an isometric box.
import { CSSProperties, ReactNode, useMemo } from "react";
import { interpolate } from "remotion";
import * as THREE from "three";
import { CamKey } from "../lib/Stage";
import { ease, lerp3 } from "../lib/anim";
import { C, FONT, W, H } from "../lib/style";

// ---------------- camera replica: same math as lib/Stage CameraPath, for projecting labels ----------------
export function camAt(keys: CamKey[], t: number) {
  let i = 0;
  while (i < keys.length - 2 && t > keys[i + 1].t) i++;
  const a = keys[i], b = keys[i + 1] ?? keys[i];
  const k = b.t === a.t ? 1 : interpolate(t, [a.t, b.t], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });
  const cam = new THREE.PerspectiveCamera((a.fov ?? 38) + ((b.fov ?? 38) - (a.fov ?? 38)) * k, W / H, 0.05, 80);
  cam.position.set(...lerp3(a.pos, b.pos, k));
  cam.lookAt(new THREE.Vector3(...lerp3(a.look ?? [0, 0, 0], b.look ?? [0, 0, 0], k)));
  cam.updateMatrixWorld();
  return cam;
}
/** 3D scene point -> screen pixel [x, y] */
export function project(cam: THREE.PerspectiveCamera, p: [number, number, number]): [number, number] {
  const v = new THREE.Vector3(...p).project(cam);
  return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H];
}

// ---------------- CT slice: a plane that samples the uint8 volume (Data3DTexture) where it sits ----------------
const sliceShader = {
  vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `
    precision highp sampler3D;
    uniform sampler3D vol; uniform sampler3D lab; uniform vec3 bmin; uniform vec3 bmax;
    uniform float opacity; uniform float labMix; uniform float gain; uniform vec3 tint; varying vec3 vW;
    vec3 labColor(float id){
      if (id < 0.5) return vec3(0.0);
      if (id < 1.5) return vec3(1.0, 0.55, 0.25);   // hip_left
      if (id < 2.5) return vec3(0.3, 0.75, 1.0);    // hip_right
      if (id < 3.5) return vec3(0.45, 1.0, 0.55);   // sacrum
      if (id < 5.5) return vec3(0.75, 0.7, 0.9);    // femurs
      return vec3(0.9, 0.85, 0.5);                  // spine
    }
    void main(){
      vec3 q = (vW - bmin) / (bmax - bmin);         // scene (x=R, y=S, z=P) -> 0..1 along (i, k, j)
      vec3 uvw = vec3(q.x, q.z, q.y);               // texture axes: x = R (i), y = A (j), z = S (k)
      if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) discard;
      float v = texture(vol, uvw).r;
      vec3 col = vec3(v) * gain * tint;
      if (labMix > 0.0) {
        float id = floor(texture(lab, uvw).r * 255.0 + 0.5);
        vec3 lc = labColor(id);
        if (id > 0.5) col = mix(col, lc * 0.9 + col * 0.3, labMix);
      }
      gl_FragColor = vec4(col, opacity);
    }`,
};

export type VolBox = { bmin: [number, number, number]; bmax: [number, number, number] };

export function VolumeSlice({ vol, lab, box, axis, at, size, opacity = 1, labMix = 0, gain = 0.92, tint = [1, 1, 1] }: {
  vol: THREE.Data3DTexture; lab?: THREE.Data3DTexture | null; box: VolBox; axis: "axial" | "coronal" | "sagittal";
  at: number; size: [number, number]; opacity?: number; labMix?: number; gain?: number; tint?: [number, number, number];
}) {
  const uniforms = useMemo(() => ({
    vol: { value: vol }, lab: { value: lab ?? vol }, bmin: { value: new THREE.Vector3() }, bmax: { value: new THREE.Vector3() },
    opacity: { value: 1 }, labMix: { value: 0 }, gain: { value: gain }, tint: { value: new THREE.Vector3(1, 1, 1) },
  }), [vol, lab]);
  uniforms.bmin.value.set(...box.bmin); uniforms.bmax.value.set(...box.bmax);
  uniforms.opacity.value = opacity; uniforms.labMix.value = labMix; uniforms.gain.value = gain;
  uniforms.tint.value.set(...tint);
  if (opacity <= 0.001) return null;
  const c = box.bmin.map((v, i) => (v + box.bmax[i]) / 2);
  const pos: [number, number, number] = axis === "axial" ? [c[0], at, c[2]] : axis === "coronal" ? [c[0], c[1], at] : [at, c[1], c[2]];
  const rot: [number, number, number] = axis === "axial" ? [-Math.PI / 2, 0, 0] : axis === "coronal" ? [0, 0, 0] : [0, Math.PI / 2, 0];
  return (
    <mesh position={pos} rotation={rot}>
      <planeGeometry args={[size[0] * 1.02, size[1] * 1.02]} />
      <shaderMaterial args={[sliceShader]} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
    </mesh>
  );
}

/** glowing outline of a slice plane */
export function PlaneOutline({ axis, at, box, opacity = 1, color = [0.6, 1.0, 2.2] }: {
  axis: "axial" | "coronal" | "sagittal"; at: number; box: VolBox; opacity?: number; color?: [number, number, number];
}) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  if (opacity <= 0.001) return null;
  const [x0, y0, z0] = box.bmin, [x1, y1, z1] = box.bmax;
  const q: [number, number, number][] = axis === "axial" ? [[x0, at, z0], [x1, at, z0], [x1, at, z1], [x0, at, z1]]
    : axis === "coronal" ? [[x0, y0, at], [x1, y0, at], [x1, y1, at], [x0, y1, at]]
      : [[at, y0, z0], [at, y0, z1], [at, y1, z1], [at, y1, z0]];
  const pos: number[] = [];
  for (let i = 0; i < 4; i++) pos.push(...q[i], ...q[(i + 1) % 4]);
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial color={new THREE.Color(...color)} transparent opacity={opacity} toneMapped={false} />
    </lineSegments>
  );
}

// ---------------- 3D arrow ----------------
export function Arrow3D({ from, dir, len, color, opacity = 1, r = 0.025 }: {
  from: [number, number, number]; dir: [number, number, number]; len: number; color: string; opacity?: number; r?: number;
}) {
  const mat = useMemo(() => new THREE.MeshBasicMaterial({ transparent: true, toneMapped: false }), []);
  mat.color.set(color).multiplyScalar(1.6); mat.opacity = opacity;
  if (opacity <= 0.001 || Math.abs(len) < 1e-4) return null;
  const d = new THREE.Vector3(...dir).normalize();
  const sgn = Math.sign(len), L = Math.abs(len);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().multiplyScalar(sgn));
  const head = 0.16;
  const mid = new THREE.Vector3(...from).addScaledVector(d, sgn * (L - head) / 2);
  const tip = new THREE.Vector3(...from).addScaledVector(d, sgn * (L - head / 2));
  return (
    <group>
      <mesh position={mid} quaternion={q} material={mat}><cylinderGeometry args={[r, r, Math.max(0.01, L - head), 16]} /></mesh>
      <mesh position={tip} quaternion={q} material={mat}><coneGeometry args={[r * 2.8, head, 20]} /></mesh>
    </group>
  );
}

/** an emissive dot */
export function Dot({ p, opacity = 1, r = 0.05, color = [3.2, 2.4, 1.0] }: {
  p: [number, number, number]; opacity?: number; r?: number; color?: [number, number, number];
}) {
  if (opacity <= 0.001) return null;
  return (
    <mesh position={p}>
      <sphereGeometry args={[r, 24, 16]} />
      <meshBasicMaterial color={new THREE.Color(...color)} transparent opacity={opacity} toneMapped={false} />
    </mesh>
  );
}

// ---------------- DOM pieces ----------------
export const mono: CSSProperties = { fontFamily: FONT, fontVariantNumeric: "tabular-nums", direction: "ltr", unicodeBidi: "isolate" };

/** A grid of numbers. cell(r, c) returns {text, bg, fg}; optional row/col index labels. */
export function NumGrid({ rows, cols, cell, x, y, size, o = 1, rowLabels, colLabels, font = 0.34, highlight, gridColor = "rgba(0,0,0,0.55)" }: {
  rows: number; cols: number; cell: (r: number, c: number) => { text: ReactNode; bg: string; fg?: string; o?: number };
  x: number; y: number; size: number; o?: number; rowLabels?: ReactNode[]; colLabels?: ReactNode[]; font?: number;
  highlight?: { r: number; c: number; o: number; color?: string }; gridColor?: string;
}) {
  if (o <= 0) return null;
  const out: ReactNode[] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const v = cell(r, c);
    out.push(
      <div key={`${r}-${c}`} style={{ position: "absolute", left: c * size, top: r * size, width: size, height: size,
        background: v.bg, color: v.fg ?? "white", fontSize: size * font, display: "flex", alignItems: "center",
        justifyContent: "center", boxShadow: `inset 0 0 0 1px ${gridColor}`, opacity: v.o ?? 1, ...mono }}>{v.text}</div>);
  }
  return (
    <div style={{ position: "absolute", left: x, top: y, width: cols * size, height: rows * size, opacity: o }}>
      {out}
      {rowLabels?.map((l, r) => (
        <div key={`r${r}`} style={{ position: "absolute", left: -size * 0.9, top: r * size, width: size * 0.8, height: size,
          display: "flex", alignItems: "center", justifyContent: "flex-end", color: C.dim, fontSize: size * 0.3, ...mono }}>{l}</div>))}
      {colLabels?.map((l, c) => (
        <div key={`c${c}`} style={{ position: "absolute", top: -size * 0.62, left: c * size, width: size, textAlign: "center",
          color: C.dim, fontSize: size * 0.3, ...mono }}>{l}</div>))}
      {highlight && highlight.o > 0 && (
        <div style={{ position: "absolute", left: highlight.c * size - 3, top: highlight.r * size - 3, width: size + 6, height: size + 6,
          border: `4px solid ${highlight.color ?? C.warm}`, borderRadius: 6, opacity: highlight.o,
          boxShadow: `0 0 24px ${highlight.color ?? C.warm}` }} />)}
    </div>
  );
}

/** isometric box (w along x, d along depth, h up), drawn as three faces, in an SVG of its own */
export function IsoBox({ w, d, h, x, y, scale, o = 1, color = "#7fd0ff", labels }: {
  w: number; d: number; h: number; x: number; y: number; scale: number; o?: number; color?: string;
  labels?: { w: string; d: string; h: string };
}) {
  if (o <= 0) return null;
  const c30 = Math.cos(Math.PI / 6), s30 = 0.5;
  const P = (a: number, b: number, z: number): [number, number] => [(a - b) * c30 * scale, ((a + b) * s30 - z) * scale];
  const pts = (q: [number, number][]) => q.map((p) => p.join(",")).join(" ");
  const top = [P(0, 0, h), P(w, 0, h), P(w, d, h), P(0, d, h)];
  const left = [P(0, d, 0), P(w, d, 0), P(w, d, h), P(0, d, h)];
  const right = [P(w, 0, 0), P(w, d, 0), P(w, d, h), P(w, 0, h)];
  const lab = (p: [number, number], s: string, dx = 0, dy = 0) => (
    <text x={p[0] + dx} y={p[1] + dy} fill={C.text} fontSize={30} fontFamily={FONT} textAnchor="middle" direction="ltr">{s}</text>);
  return (
    <svg style={{ position: "absolute", left: x - 400, top: y - 400, width: 800, height: 800, opacity: o, overflow: "visible" }}
      viewBox="-400 -400 800 800">
      <polygon points={pts(left)} fill={color} fillOpacity={0.22} stroke={color} strokeWidth={2.5} />
      <polygon points={pts(right)} fill={color} fillOpacity={0.12} stroke={color} strokeWidth={2.5} />
      <polygon points={pts(top)} fill={color} fillOpacity={0.32} stroke={color} strokeWidth={2.5} />
      {labels && lab(P(w / 2, d, 0), labels.w, -26, 42)}
      {labels && lab(P(w, d / 2, 0), labels.d, 40, 40)}
      {labels && lab(P(0, d, h / 2), labels.h, -70, 0)}
    </svg>
  );
}

/** a simple straight arrow in screen space */
export function Arrow2D({ x1, y1, x2, y2, o = 1, color = C.warm, width = 4 }: {
  x1: number; y1: number; x2: number; y2: number; o?: number; color?: string; width?: number;
}) {
  if (o <= 0) return null;
  const a = Math.atan2(y2 - y1, x2 - x1), hl = 18;
  const p1 = [x2 - hl * Math.cos(a - 0.4), y2 - hl * Math.sin(a - 0.4)], p2 = [x2 - hl * Math.cos(a + 0.4), y2 - hl * Math.sin(a + 0.4)];
  return (
    <svg style={{ position: "absolute", left: 0, top: 0, width: W, height: H, opacity: o, pointerEvents: "none" }}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeLinecap="round" />
      <polygon points={`${x2},${y2} ${p1[0]},${p1[1]} ${p2[0]},${p2[1]}`} fill={color} />
    </svg>
  );
}
