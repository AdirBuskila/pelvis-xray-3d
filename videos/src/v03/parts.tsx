// Video 3 parts: pinhole box, coordinate triads, screen projection for labels.
import { useMemo } from "react";
import * as THREE from "three";
import { W, H } from "../lib/style";

/** Project a scene point to screen pixels for a static camera (same model as CameraPath). */
export function toScreen(p: [number, number, number], pos: [number, number, number], look: [number, number, number], fov = 38): [number, number] {
  const c = new THREE.PerspectiveCamera(fov, W / H, 0.05, 80);
  c.position.set(...pos);
  c.lookAt(new THREE.Vector3(...look));
  c.updateMatrixWorld();
  const v = new THREE.Vector3(...p).project(c);
  return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H];
}

const glow = (r: number, g: number, b: number) => new THREE.Color(r, g, b);

/** The "object" for the pinhole demo: an upright arrow with a flag on its right (+x) so both flips read. */
export function ArrowObject({ opacity, flat = false, emissive }: { opacity: number; flat?: boolean; emissive?: THREE.Color }) {
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#ffcf7f", roughness: 0.4, transparent: true }), []);
  const fmat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#7dffb0", roughness: 0.4, transparent: true }), []);
  mat.opacity = opacity; fmat.opacity = opacity;
  if (emissive) { mat.emissive = emissive; mat.emissiveIntensity = 1; fmat.emissive = new THREE.Color(0.3, 1.6, 0.8); fmat.emissiveIntensity = 1; }
  if (opacity <= 0.001) return null;
  return (
    <group scale={[1, 1, flat ? 0.02 : 1]}>
      <mesh material={mat} position={[0, -0.1, 0]}><cylinderGeometry args={[0.07, 0.07, 1.6, 24]} /></mesh>
      <mesh material={mat} position={[0, 0.9, 0]}><coneGeometry args={[0.22, 0.4, 32]} /></mesh>
      <mesh material={fmat} position={[0.28, 0.45, 0]}><boxGeometry args={[0.42, 0.26, 0.05]} /></mesh>
    </group>
  );
}

/** Dark box with a pinhole in the front face (z = 0); back wall at z = depth. */
export function PinholeBox({ opacity, depth = 2.2, size = 2.6 }: { opacity: number; depth?: number; size?: number }) {
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(size, size, depth)), [size, depth]);
  const front = useMemo(() => {
    const sh = new THREE.Shape();
    sh.moveTo(-size / 2, -size / 2); sh.lineTo(size / 2, -size / 2); sh.lineTo(size / 2, size / 2); sh.lineTo(-size / 2, size / 2); sh.closePath();
    const hole = new THREE.Path(); hole.absarc(0, 0, 0.06, 0, Math.PI * 2, true); sh.holes.push(hole);
    return new THREE.ShapeGeometry(sh, 32);
  }, [size]);
  if (opacity <= 0.001) return null;
  const wall = (o: number) => (
    <meshBasicMaterial color={glow(0.06, 0.12, 0.22)} transparent opacity={o * opacity} side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
  );
  return (
    <group>
      <lineSegments geometry={edges} position={[0, 0, depth / 2]}>
        <lineBasicMaterial color={glow(0.4, 0.7, 1.2)} transparent opacity={0.8 * opacity} toneMapped={false} />
      </lineSegments>
      {/* back wall: the screen */}
      <mesh position={[0, 0, depth]}><planeGeometry args={[size, size]} />{wall(0.85)}</mesh>
      {/* front wall with a hole: a ring */}
      <mesh position={[0, 0, 0]} geometry={front}>{wall(0.35)}</mesh>
      <mesh position={[0, 0, -0.001]}><ringGeometry args={[0.045, 0.075, 32]} />
        <meshBasicMaterial color={glow(2.5, 2.0, 0.9)} transparent opacity={opacity} toneMapped={false} side={THREE.DoubleSide} />
      </mesh>
      {/* far side wall (-x) and floor only: the camera looks in from the open +x side */}
      <mesh position={[-size / 2, 0, depth / 2]} rotation={[0, Math.PI / 2, 0]}><planeGeometry args={[depth, size]} />{wall(0.4)}</mesh>
      <mesh position={[0, -size / 2, depth / 2]} rotation={[Math.PI / 2, 0, 0]}><planeGeometry args={[size, depth]} />{wall(0.4)}</mesh>
    </group>
  );
}

/** Coordinate triad: x, y, z arrows. */
export function Triad({ position, rotation = [0, 0, 0], scale = 1, opacity = 1, colors = ["#ff7f8a", "#7dffb0", "#7fd0ff"] }: {
  position: [number, number, number]; rotation?: [number, number, number]; scale?: number; opacity?: number; colors?: string[];
}) {
  const mats = useMemo(() => colors.map((c) => new THREE.MeshBasicMaterial({ color: new THREE.Color(c).multiplyScalar(1.6), transparent: true, toneMapped: false, depthTest: false })), [colors.join()]);
  mats.forEach((m) => (m.opacity = opacity));
  if (opacity <= 0.001) return null;
  const axis = (i: number) => {
    const rot: [number, number, number] = i === 0 ? [0, 0, -Math.PI / 2] : i === 1 ? [0, 0, 0] : [Math.PI / 2, 0, 0];
    return (
      <group key={i} rotation={rot}>
        <mesh material={mats[i]} position={[0, 0.45, 0]}><cylinderGeometry args={[0.025, 0.025, 0.9, 12]} /></mesh>
        <mesh material={mats[i]} position={[0, 0.98, 0]}><coneGeometry args={[0.07, 0.18, 16]} /></mesh>
      </group>
    );
  };
  return <group position={position} rotation={rotation} scale={scale}>{[0, 1, 2].map(axis)}</group>;
}

/** Glowing marker sphere */
export function Marker({ position, opacity = 1, r = 0.07, color = [3.2, 2.4, 1.0] }: {
  position: [number, number, number]; opacity?: number; r?: number; color?: [number, number, number];
}) {
  if (opacity <= 0.001) return null;
  return (
    <mesh position={position}>
      <sphereGeometry args={[r, 24, 16]} />
      <meshBasicMaterial color={new THREE.Color(...color)} transparent opacity={opacity} toneMapped={false} />
    </mesh>
  );
}
