// Reusable 3D pieces: X-ray rig (source, beam, detector), bones, voxel grids, CT slice planes, rays.
import { useMemo } from "react";
import * as THREE from "three";

export type Meta = {
  subject: string; spacing_mm: number[]; crop_shape: number[]; vol_dims_xyz: [number, number, number];
  scene_scale: number; box_min: number[]; box_max: number[]; ap_size: [number, number]; lat_size: [number, number];
  occ_cube_mm: number; labels: string[];
};

/** bounding box helpers in scene units (x = R, y = S, z = P) */
export const boxOf = (m: Meta) => {
  const mn = m.box_min.map((v, i) => Math.min(v, m.box_max[i]));
  const mx = m.box_min.map((v, i) => Math.max(v, m.box_max[i]));
  return { mn, mx, c: mn.map((v, i) => (v + mx[i]) / 2), s: mx.map((v, i) => v - mn[i]) };
};

// ---------------- detector with row-by-row reveal ----------------
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

export function Detector({ tex, progress = 1, opacity = 1, size, position, rotation = [0, 0, 0], flipX = false }: {
  tex: THREE.Texture; progress?: number; opacity?: number; size: [number, number];
  position: [number, number, number]; rotation?: [number, number, number]; flipX?: boolean;
}) {
  const uniforms = useMemo(() => ({ map: { value: tex }, progress: { value: 0 }, opacity: { value: 1 }, flipX: { value: 0 } }), [tex]);
  uniforms.progress.value = progress;
  uniforms.opacity.value = opacity;
  uniforms.flipX.value = flipX ? 1 : 0;
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.PlaneGeometry(size[0] * 1.02, size[1] * 1.02)), [size[0], size[1]]);
  if (opacity <= 0.001) return null;
  return (
    <group position={position} rotation={rotation}>
      <mesh>
        <planeGeometry args={size} />
        <shaderMaterial args={[detectorShader]} uniforms={uniforms} transparent side={THREE.DoubleSide} toneMapped={false} depthWrite={false} />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={new THREE.Color(0.45, 0.75, 1.0)} transparent opacity={0.6 * opacity} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

/** Flat emitter outline: reads as "the source", never as a wall */
export function SourcePanel({ size, position, rotation = [0, 0, 0], opacity = 1 }: {
  size: [number, number]; position: [number, number, number]; rotation?: [number, number, number]; opacity?: number;
}) {
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.PlaneGeometry(...size)), [size[0], size[1]]);
  if (opacity <= 0.001) return null;
  return (
    <group position={position} rotation={rotation}>
      <mesh>
        <planeGeometry args={size} />
        <meshBasicMaterial color={new THREE.Color(0.3, 0.55, 1.0)} transparent opacity={0.025 * opacity}
          side={THREE.DoubleSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={new THREE.Color(0.6, 1.0, 2.2)} transparent opacity={0.9 * opacity} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

/** Glowing point source (perspective X-ray tube) */
export function PointSource({ position, opacity = 1, r = 0.07 }: { position: [number, number, number]; opacity?: number; r?: number }) {
  if (opacity <= 0.001) return null;
  return (
    <mesh position={position}>
      <sphereGeometry args={[r, 24, 16]} />
      <meshBasicMaterial color={new THREE.Color(1.6, 2.4, 4.0)} transparent opacity={opacity} toneMapped={false} />
    </mesh>
  );
}

/**
 * Parallel beam: one scan line of rays sweeping top -> bottom.
 * axis "ap": rays along +z from zSrc to zDet across x in [-w/2, w/2]
 * axis "lat": rays along -x from xSrc to xDet across z in [-w/2, w/2]
 * `bodyIn/bodyOut` mark where the body starts/ends along the ray (rays dim after it).
 */
export function ScanBeam({ axis, progress, opacity, w, h, src, det, bodyIn, bodyOut, n = 34 }: {
  axis: "ap" | "lat"; progress: number; opacity: number; w: number; h: number;
  src: number; det: number; bodyIn: number; bodyOut: number; n?: number;
}) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  if (opacity <= 0.001 || progress <= 0 || progress >= 1) return null;
  const y = h / 2 - progress * h;
  const pos: number[] = [], col: number[] = [];
  for (let k = 0; k < n; k++) {
    const u = ((k + 0.5) / n - 0.5) * w;
    const j = Math.sin(k * 12.9898) * 0.012;
    const P = (s: number) => (axis === "ap" ? [u, y + j, s] : [s, y + j, u]);
    pos.push(...P(src), ...P(bodyIn), ...P(bodyIn), ...P(bodyOut), ...P(bodyOut), ...P(det));
    col.push(1.4, 2.1, 3.0, 1.4, 2.1, 3.0, 1.0, 1.5, 2.2, 0.45, 0.7, 1.0, 0.45, 0.7, 1.0, 0.25, 0.4, 0.6);
  }
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent opacity={opacity} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

/** A single glowing ray (or fan of rays) between points */
export function Rays({ segs, opacity = 1, color = [1.4, 2.1, 3.0] }: {
  segs: [number, number, number][][]; opacity?: number; color?: [number, number, number];
}) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  if (opacity <= 0.001 || segs.length === 0) return null;
  const pos: number[] = [], col: number[] = [];
  for (const [a, b] of segs) { pos.push(...a, ...b); col.push(...color, ...color); }
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent opacity={opacity} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

export function Bone({ geo, material, opacity = 1, position, scale = 1, rotation }: {
  geo: THREE.BufferGeometry | null; material: THREE.MeshPhysicalMaterial; opacity?: number;
  position?: [number, number, number]; scale?: number; rotation?: [number, number, number];
}) {
  // own copy per bone: three.js skips uniform refresh when one material is drawn twice in a row,
  // so per-object opacity on a shared material would not apply reliably
  const mat = useMemo(() => material.clone(), [material]);
  mat.opacity = opacity;
  mat.depthWrite = opacity > 0.6;
  if (!geo || opacity <= 0.001) return null;
  return <mesh geometry={geo} material={mat} position={position} scale={scale} rotation={rotation} />;
}

/**
 * Occupancy voxels as instanced cubes. `reveal` 0..1 builds the grid bottom-up (by y),
 * `gap` shrinks each cube so the grid structure reads.
 */
export function Voxels({ centers, cell, reveal = 1, opacity = 1, gap = 0.12, color = "#9fd6ff", emissive = 0.15 }: {
  centers: number[]; cell: number; reveal?: number; opacity?: number; gap?: number; color?: string; emissive?: number;
}) {
  const count = centers.length / 3;
  const { mesh, ys, ymin, ymax } = useMemo(() => {
    const g = new THREE.BoxGeometry(1, 1, 1);
    const m = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05, transparent: true,
      emissive: new THREE.Color(color), emissiveIntensity: emissive });
    const im = new THREE.InstancedMesh(g, m, count);
    const ys = new Float32Array(count);
    for (let i = 0; i < count; i++) ys[i] = centers[3 * i + 1];
    let ymin = Infinity, ymax = -Infinity;
    for (const y of ys) { ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
    return { mesh: im, ys, ymin, ymax };
  }, [centers, count, color, emissive]);
  const m4 = new THREE.Matrix4();
  const cut = ymin + (ymax - ymin + cell) * reveal;
  const s = cell * (1 - gap);
  let shown = 0;
  for (let i = 0; i < count; i++) {
    if (ys[i] > cut) continue;
    const grow = Math.min(1, (cut - ys[i]) / (cell * 3));
    m4.makeScale(s * grow, s * grow, s * grow).setPosition(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]);
    mesh.setMatrixAt(shown++, m4);
  }
  mesh.count = shown;
  mesh.instanceMatrix.needsUpdate = true;
  (mesh.material as THREE.MeshStandardMaterial).opacity = opacity;
  if (opacity <= 0.001) return null;
  return <primitive object={mesh} />;
}

/** One textured slice plane (CT slice PNG) placed in the CT box */
export function SlicePlane({ tex, size, position, rotation, opacity = 1 }: {
  tex: THREE.Texture; size: [number, number]; position: [number, number, number];
  rotation: [number, number, number]; opacity?: number;
}) {
  if (opacity <= 0.001) return null;
  return (
    <mesh position={position} rotation={rotation}>
      <planeGeometry args={size} />
      <meshBasicMaterial map={tex} transparent opacity={opacity} side={THREE.DoubleSide} toneMapped={false} depthWrite={false} />
    </mesh>
  );
}

/** Wireframe box */
export function WireBox({ size, position = [0, 0, 0], opacity = 1, color = "#6fb4ff" }: {
  size: [number, number, number]; position?: [number, number, number]; opacity?: number; color?: string;
}) {
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)), [size[0], size[1], size[2]]);
  if (opacity <= 0.001) return null;
  return (
    <lineSegments geometry={edges} position={position}>
      <lineBasicMaterial color={color} transparent opacity={opacity} toneMapped={false} />
    </lineSegments>
  );
}
