// Reusable pieces for videos 7 and 8: voxels on the 64^3 grid (by index), smeared/extruded planes.
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
