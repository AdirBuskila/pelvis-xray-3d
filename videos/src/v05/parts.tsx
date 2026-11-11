// Reusable pieces for videos 5 and 6 (3D shape representations and metrics).
import { useMemo } from "react";
import * as THREE from "three";

/** instanced cubes with an optional filter on the centre (for slabs / sweeps / coloured sets) */
export function VoxelSet({ centers, cell, opacity = 1, color = "#9fd6ff", gap = 0.14, emissive = 0.18, keep }: {
  centers: number[]; cell: number; opacity?: number; color?: string; gap?: number; emissive?: number;
  keep?: (x: number, y: number, z: number) => boolean;
}) {
  const count = centers.length / 3;
  const mesh = useMemo(() => {
    const m = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05, transparent: true,
      emissive: new THREE.Color(color), emissiveIntensity: emissive });
    return new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), m, Math.max(1, count));
  }, [centers, count, color, emissive]);
  const m4 = new THREE.Matrix4();
  const s = cell * (1 - gap);
  let shown = 0;
  for (let i = 0; i < count; i++) {
    const x = centers[3 * i], y = centers[3 * i + 1], z = centers[3 * i + 2];
    if (keep && !keep(x, y, z)) continue;
    m4.makeScale(s, s, s).setPosition(x, y, z);
    mesh.setMatrixAt(shown++, m4);
  }
  mesh.count = shown;
  mesh.instanceMatrix.needsUpdate = true;
  const mat = mesh.material as THREE.MeshStandardMaterial;
  mat.opacity = opacity;
  mat.depthWrite = opacity > 0.7;
  if (opacity <= 0.001 || shown === 0) return null;
  return <primitive object={mesh} />;
}

/** glowing point cloud; `reveal` 0..1 shows the first fraction of points */
export function PointCloud({ p, opacity = 1, size = 0.028, color = [1.2, 2.0, 3.0], reveal = 1 }: {
  p: number[]; opacity?: number; size?: number; color?: [number, number, number]; reveal?: number;
}) {
  const { geo, mat } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    const m = new THREE.PointsMaterial({ size, sizeAttenuation: true, transparent: true, toneMapped: false,
      color: new THREE.Color(...color), depthWrite: false });
    return { geo: g, mat: m };
  }, [p, size, color[0], color[1], color[2]]);
  mat.opacity = opacity;
  geo.setDrawRange(0, Math.floor((p.length / 3) * reveal));
  if (opacity <= 0.001) return null;
  return <points geometry={geo} material={mat} />;
}

/** wireframe overlay of a geometry (own material per instance) */
export function Wire({ geo, opacity = 1, color = [0.5, 0.9, 1.6] }: {
  geo: THREE.BufferGeometry; opacity?: number; color?: [number, number, number];
}) {
  const mat = useMemo(() => new THREE.MeshBasicMaterial({ wireframe: true, transparent: true, toneMapped: false,
    color: new THREE.Color(...color), depthWrite: false }), [color[0], color[1], color[2]]);
  mat.opacity = opacity;
  if (opacity <= 0.001) return null;
  return <mesh geometry={geo} material={mat} />;
}

/** a simple lit surface with its own material (flat or smooth), optional vertex colours */
export function Surface({ geo, opacity = 1, color = "#cfc5b0", flat = false, vertexColors = false, drawCount, emissive = 0 }: {
  geo: THREE.BufferGeometry; opacity?: number; color?: string; flat?: boolean; vertexColors?: boolean; drawCount?: number; emissive?: number;
}) {
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: vertexColors ? "#ffffff" : color, roughness: 0.55,
    metalness: 0.02, transparent: true, flatShading: flat, vertexColors, side: THREE.DoubleSide,
    emissive: new THREE.Color(vertexColors ? "#000000" : color), emissiveIntensity: emissive }), [color, flat, vertexColors, emissive]);
  mat.opacity = opacity;
  mat.depthWrite = opacity > 0.6;
  geo.setDrawRange(0, drawCount ?? Infinity);
  if (opacity <= 0.001) return null;
  return <mesh geometry={geo} material={mat} />;
}

// ---------------- signed distance field slice ----------------
const sdfShader = {
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    uniform sampler2D map; uniform float level; uniform float opacity; uniform float R; varying vec2 vUv;
    void main(){
      float d = (texture2D(map, vUv).r * 2.0 - 1.0) * R;     // mm, negative inside
      float s = d - level;
      float k = clamp(abs(d) / R, 0.0, 1.0);
      vec3 col = d < 0.0 ? mix(vec3(0.35, 0.62, 1.0), vec3(0.05, 0.12, 0.45), k)
                         : mix(vec3(1.0, 0.62, 0.45), vec3(0.22, 0.04, 0.04), k);
      float f = abs(fract(d / 5.0 + 0.5) - 0.5) * 5.0;        // mm to the nearest 5 mm contour
      float w = max(fwidth(d), 0.05);
      col = mix(col, col * 0.3, (1.0 - smoothstep(0.0, w * 1.3, f)) * 0.85);
      float z = 1.0 - smoothstep(w * 0.6, w * 2.2, abs(s));     // the iso-contour at 'level'
      col = mix(col, vec3(3.2, 2.7, 1.5), z);
      gl_FragColor = vec4(col, opacity);
    }`,
};

export function SdfPlane({ tex, level = 0, opacity = 1, R, size, position }: {
  tex: THREE.Texture; level?: number; opacity?: number; R: number; size: [number, number]; position: [number, number, number];
}) {
  const uniforms = useMemo(() => ({ map: { value: tex }, level: { value: 0 }, opacity: { value: 1 }, R: { value: R } }), [tex, R]);
  uniforms.level.value = level;
  uniforms.opacity.value = opacity;
  if (opacity <= 0.001) return null;
  return (
    <mesh position={position} rotation={[0, Math.PI, 0]}>
      <planeGeometry args={size} />
      <shaderMaterial args={[sdfShader]} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
    </mesh>
  );
}

/** distance (mm) -> colour: blue (0) -> cyan -> yellow -> red (>= max) */
export function heat(d: number, max: number): [number, number, number] {
  const k = Math.min(1, Math.max(0, d / max));
  const stops: [number, [number, number, number]][] = [
    [0, [0.18, 0.42, 1.0]], [0.25, [0.3, 0.9, 1.0]], [0.55, [1.0, 0.9, 0.3]], [1, [1.0, 0.22, 0.2]]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i], [b, cb] = stops[i + 1];
    if (k <= b) { const u = (k - a) / (b - a); return [0, 1, 2].map((j) => ca[j] + (cb[j] - ca[j]) * u) as [number, number, number]; }
  }
  return stops[stops.length - 1][1];
}

export type MeshJson = { v: number[]; n: number[]; f: number[]; d: Record<string, number[]> };

/** indexed geometry from a mesh_*.json (v06), with a colour attribute to be filled per frame */
export function geoFromJson(m: MeshJson) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(m.v, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(m.n, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(m.v.length).fill(1), 3));
  g.setIndex(m.f);
  return g;
}

/** fill the colour attribute from per-vertex distances with a colouring function */
export function paint(g: THREE.BufferGeometry, d: number[] | null, fn: (d: number) => [number, number, number]) {
  const c = g.getAttribute("color") as THREE.BufferAttribute;
  for (let i = 0; i < c.count; i++) {
    const [r, gg, b] = d ? fn(d[i]) : [0.78, 0.74, 0.66];
    c.setXYZ(i, r, gg, b);
  }
  c.needsUpdate = true;
}
