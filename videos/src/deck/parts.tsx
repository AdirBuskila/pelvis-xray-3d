// Copied from src/v07/parts.tsx (kept local so the deck clips don't depend on another video's folder).
// Grid convention (prep/v07_common.py): index (r, a, s) -> scene x = R, y = S, z = -A, cube centred on
// the pelvis centroid (the scene origin).
import { useMemo } from "react";
import * as THREE from "three";

export type Grid = { n: number; cell: number; cube: number };
export const gridPos = (g: Grid, r: number, a: number, s: number): [number, number, number] => [
  (r + 0.5) * g.cell - g.cube / 2, (s + 0.5) * g.cell - g.cube / 2, -((a + 0.5) * g.cell - g.cube / 2),
];

/** Instanced cubes for flat (r,a,s) index triples. Only cells with s <= cutS are drawn. */
export function GridVox({ g, idx, color, opacity = 1, cutS = 999, gap = 0.1, emissive = 0.12, scale = 1,
  position = [0, 0, 0], rotationY = 0, grow = 1, depthWrite }: {
  g: Grid; idx: number[]; color: string; opacity?: number; cutS?: number; gap?: number; emissive?: number;
  scale?: number; position?: [number, number, number]; rotationY?: number; grow?: number; depthWrite?: boolean;
}) {
  const count = idx.length / 3;
  const mesh = useMemo(() => {
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const m = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05, transparent: true,
      emissive: new THREE.Color(color), emissiveIntensity: emissive });
    return new THREE.InstancedMesh(geo, m, Math.max(1, count));
  }, [idx, color, emissive, count]);
  const m4 = useMemo(() => new THREE.Matrix4(), []);
  const s = g.cell * (1 - gap) * grow;
  let shown = 0;
  for (let i = 0; i < count; i++) {
    const k = idx[3 * i + 2];
    if (k > cutS) continue;
    const [x, y, z] = gridPos(g, idx[3 * i], idx[3 * i + 1], k);
    m4.makeScale(s, s, s).setPosition(x, y, z);
    mesh.setMatrixAt(shown++, m4);
  }
  mesh.count = shown;
  mesh.instanceMatrix.needsUpdate = true;
  const mat = mesh.material as THREE.MeshStandardMaterial;
  mat.opacity = opacity;
  mat.depthWrite = depthWrite ?? opacity > 0.6;
  if (opacity <= 0.001 || shown === 0) return null;
  return (
    <group position={position} scale={scale} rotation={[0, rotationY, 0]}>
      <primitive object={mesh} />
    </group>
  );
}

/** 64x64 0/1 mask (flat, (u, s) C-order) -> RGBA texture, texel column = u, row = s (bottom up) */
export function useMaskTexture(mask: number[] | null, n: number, rgb: [number, number, number]) {
  return useMemo(() => {
    if (!mask) return null;
    const data = new Uint8Array(n * n * 4);
    for (let u = 0; u < n; u++) for (let s = 0; s < n; s++) {
      const o = (s * n + u) * 4, on = mask[u * n + s];
      data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2]; data[o + 3] = on ? 255 : 0;
    }
    const t = new THREE.DataTexture(data, n, n, THREE.RGBAFormat);
    t.magFilter = t.minFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  }, [mask, n, rgb[0], rgb[1], rgb[2]]);
}

/**
 * A stack of `count` textured planes filling the cube along the view's depth axis: the image
 * "smeared back" (axis "ap": planes ⟂ z, image u along +x; axis "lat": planes ⟂ x, image u along -z).
 * `progress` 0..1 grows the stack from the detector side (ap: +z, lat: -x) toward the source.
 */
export function PlaneStack({ tex, axis, size, count = 26, progress = 1, opacity = 1, additive = true, color = "#ffffff", from }: {
  tex: THREE.Texture; axis: "ap" | "lat"; size: number; count?: number; progress?: number; opacity?: number;
  additive?: boolean; color?: string; from?: number;
}) {
  const mats = useMemo(() => Array.from({ length: count }, () => new THREE.MeshBasicMaterial({
    map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false, toneMapped: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending, color: new THREE.Color(color),
  })), [tex, count, additive, color]);
  if (opacity <= 0.001 || progress <= 0) return null;
  const half = size / 2;
  const start = from ?? half;
  return (
    <group>
      {mats.map((m, i) => {
        const f = (i + 0.5) / count;                      // 0 at the detector side
        const k = Math.min(1, Math.max(0, (progress * 1.15 - f) / 0.15));
        m.opacity = opacity * k;
        if (k <= 0) return null;
        const d = start - f * (start + half);             // depth coordinate from detector side to source side
        return axis === "ap"
          ? <mesh key={i} material={m} position={[0, 0, d]}><planeGeometry args={[size, size]} /></mesh>
          : <mesh key={i} material={m} position={[-d, 0, 0]} rotation={[0, Math.PI / 2, 0]}><planeGeometry args={[size, size]} /></mesh>;
      })}
    </group>
  );
}

/** a single textured plane at the detector position of a view */
export function ViewPlane({ tex, axis, size, at, opacity = 1, border = true }: {
  tex: THREE.Texture; axis: "ap" | "lat"; size: number; at: number; opacity?: number; border?: boolean;
}) {
  const mat = useMemo(() => new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide,
    depthWrite: false, toneMapped: false }), [tex]);
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.PlaneGeometry(size * 1.01, size * 1.01)), [size]);
  mat.opacity = opacity;
  if (opacity <= 0.001) return null;
  const pos: [number, number, number] = axis === "ap" ? [0, 0, at] : [at, 0, 0];
  const rot: [number, number, number] = axis === "ap" ? [0, 0, 0] : [0, Math.PI / 2, 0];
  return (
    <group position={pos} rotation={rot}>
      <mesh material={mat}><planeGeometry args={[size, size]} /></mesh>
      {border && (
        <lineSegments geometry={edges}>
          <lineBasicMaterial color={new THREE.Color(0.45, 0.75, 1.0)} transparent opacity={0.6 * opacity} toneMapped={false} />
        </lineSegments>
      )}
    </group>
  );
}

// ---------------- detector (same look as lib/xray Detector) ----------------
// The material is built once and its uniforms are written every frame directly on material.uniforms.
// lib's Detector passes `uniforms` as a JSX prop; in a full (sequential) render its progress stayed at
// the first frame's value (a still or a render starting mid-scan looked fine), so the deck uses this one.
const detectorShader = {
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    uniform sampler2D map; uniform float progress; uniform float opacity; uniform float flipX; varying vec2 vUv;
    void main(){
      vec2 uv = vec2(mix(vUv.x, 1.0 - vUv.x, flipX), vUv.y);
      float row = 1.0 - vUv.y;
      vec3 img = texture2D(map, uv).rgb;
      vec3 panel = vec3(0.03, 0.05, 0.08);
      float shown = step(row, progress);
      float edge = smoothstep(0.018, 0.0, abs(row - progress)) * step(progress, 0.999) * step(0.001, progress);
      vec3 col = mix(panel, img, shown) + edge * vec3(0.55, 0.85, 1.6);
      gl_FragColor = vec4(col, opacity);
    }`,
};

export function DeckDetector({ tex, progress = 1, opacity = 1, size, position, rotation = [0, 0, 0], flipX = false }: {
  tex: THREE.Texture; progress?: number; opacity?: number; size: [number, number];
  position: [number, number, number]; rotation?: [number, number, number]; flipX?: boolean;
}) {
  const mat = useMemo(() => new THREE.ShaderMaterial({
    ...detectorShader, transparent: true, side: THREE.DoubleSide, depthWrite: false, toneMapped: false,
    uniforms: { map: { value: tex }, progress: { value: 0 }, opacity: { value: 1 }, flipX: { value: 0 } },
  }), [tex]);
  mat.uniforms.progress.value = progress;
  mat.uniforms.opacity.value = opacity;
  mat.uniforms.flipX.value = flipX ? 1 : 0;
  mat.uniformsNeedUpdate = true;
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.PlaneGeometry(size[0] * 1.02, size[1] * 1.02)), [size[0], size[1]]);
  return (
    <group position={position} rotation={rotation} visible={opacity > 0.001}>
      <mesh material={mat}><planeGeometry args={size} /></mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={new THREE.Color(0.45, 0.75, 1.0)} transparent opacity={0.6 * opacity} toneMapped={false} />
      </lineSegments>
    </group>
  );
}
