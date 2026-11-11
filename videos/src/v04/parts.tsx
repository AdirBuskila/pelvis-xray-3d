// Video 4 parts: perspective fan beam, screen projection for labels.
import { useMemo } from "react";
import * as THREE from "three";
import { W, H } from "../lib/style";

export type V3 = [number, number, number];

/** Project a scene point to screen pixels for a static camera (same model as CameraPath). */
export function toScreen(p: V3, pos: V3, look: V3, fov = 38): [number, number] {
  const c = new THREE.PerspectiveCamera(fov, W / H, 0.05, 80);
  c.position.set(...pos);
  c.lookAt(new THREE.Vector3(...look));
  c.updateMatrixWorld();
  const v = new THREE.Vector3(...p).project(c);
  return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H];
}

/**
 * Perspective beam: a row of rays from a point source to one detector row, sweeping top -> bottom.
 * The detector lies in the plane z = detZ, centred at (cx, cy), size [w, h]. Rays dim after `bodyOut` (z).
 */
export function FanBeam({ src, detZ, cx, cy, w, h, progress, opacity, bodyIn, bodyOut, n = 34 }: {
  src: V3; detZ: number; cx: number; cy: number; w: number; h: number; progress: number; opacity: number;
  bodyIn: number; bodyOut: number; n?: number;
}) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  if (opacity <= 0.001 || progress <= 0 || progress >= 1) return null;
  const y = cy + h / 2 - progress * h;
  const pos: number[] = [], col: number[] = [];
  const at = (x: number, z: number): V3 => {
    const s = (z - src[2]) / (detZ - src[2]);
    return [src[0] + (x - src[0]) * s, src[1] + (y - src[1]) * s, z];
  };
  for (let k = 0; k < n; k++) {
    const x = cx + ((k + 0.5) / n - 0.5) * w;
    pos.push(...src, ...at(x, bodyIn), ...at(x, bodyIn), ...at(x, bodyOut), ...at(x, bodyOut), ...at(x, detZ));
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
