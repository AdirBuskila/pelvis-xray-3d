// The 3D stage shared by every video: canvas, lights, environment, post-processing, camera path.
import { ReactNode, useEffect, useMemo } from "react";
import { useVideoConfig } from "remotion";
import { ThreeCanvas } from "@remotion/three";
import { useFrame, useThree } from "@react-three/fiber";
import { Environment, Lightformer } from "@react-three/drei";
import {
  BlendFunction, BloomEffect, EffectComposer, EffectPass, NoiseEffect, RenderPass, VignetteEffect,
} from "postprocessing";
import * as THREE from "three";
import { C } from "./style";
import { ease, lerp3 } from "./anim";
import { interpolate } from "remotion";

// Post-processing is driven manually: @react-three/postprocessing's <EffectComposer> renders an
// empty canvas under Remotion (verified), while a composer rendered inside useFrame works.
function PostFX({ bloom = 1.0 }: { bloom?: number }) {
  const { gl, scene, camera, size } = useThree();
  const composer = useMemo(() => {
    const noise = new NoiseEffect({ blendFunction: BlendFunction.SOFT_LIGHT, premultiply: false });
    noise.blendMode.opacity.value = 0.1;
    const c = new EffectComposer(gl, { frameBufferType: THREE.HalfFloatType });
    c.addPass(new RenderPass(scene, camera));
    c.addPass(new EffectPass(camera,
      new BloomEffect({ mipmapBlur: true, intensity: bloom, luminanceThreshold: 0.62, luminanceSmoothing: 0.2, radius: 0.7 }),
      noise,
      new VignetteEffect({ offset: 0.25, darkness: 0.72 })));
    return c;
  }, [gl, scene, camera, bloom]);
  useEffect(() => composer.setSize(size.width, size.height), [composer, size]);
  useFrame((_, delta) => composer.render(delta), 1);
  return null;
}

export type CamKey = { t: number; pos: [number, number, number]; look?: [number, number, number]; fov?: number };

/** Piecewise camera path; each leg eases from key i to key i+1. `orbit` keys interpolate on a circle. */
export function CameraPath({ t, keys }: { t: number; keys: CamKey[] }) {
  const { camera } = useThree();
  let i = 0;
  while (i < keys.length - 2 && t > keys[i + 1].t) i++;
  const a = keys[i], b = keys[i + 1] ?? keys[i];
  const k = b.t === a.t ? 1 : interpolate(t, [a.t, b.t], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });
  const pos = lerp3(a.pos, b.pos, k);
  const look = lerp3(a.look ?? [0, 0, 0], b.look ?? [0, 0, 0], k);
  camera.position.set(...pos);
  camera.lookAt(new THREE.Vector3(...look));
  const fa = a.fov ?? 38, fb = b.fov ?? 38;
  const cam = camera as THREE.PerspectiveCamera;
  const fov = fa + (fb - fa) * k;
  if (cam.fov !== fov) { cam.fov = fov; cam.updateProjectionMatrix(); }
  return null;
}

/** orbit helper: point on a circle of radius r around the y axis at angle a (0 = in front, -z) */
export const orbit = (a: number, r: number, y: number): [number, number, number] => [r * Math.sin(a), y, -r * Math.cos(a)];

export function Lights() {
  return (
    <>
      <ambientLight intensity={0.45} />
      <hemisphereLight args={["#cfe3ff", "#1a1208", 0.9]} />
      <directionalLight position={[-4, 6, -5]} intensity={2.8} color="#fff4e6" />
      <directionalLight position={[4, 3, -6]} intensity={1.6} color="#ffe9cc" />
      <directionalLight position={[5, 1, 4]} intensity={1.4} color="#7fb6ff" />
      <Environment resolution={256} frames={1}>
        <Lightformer form="rect" intensity={2.5} position={[0, 5, -3]} scale={[8, 3, 1]} />
        <Lightformer form="rect" intensity={1.2} color="#6aa8ff" position={[6, 0, 3]} scale={[3, 6, 1]} />
        <Lightformer form="ring" intensity={0.8} position={[-6, 1, 0]} scale={3} />
      </Environment>
    </>
  );
}

export function Floor({ y = -1.95, opacity = 1 }: { y?: number; opacity?: number }) {
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(30, 60, "#1f4a6e", "#12283c");
    (g.material as THREE.Material).transparent = true;
    return g;
  }, []);
  (grid.material as THREE.Material).opacity = opacity;
  return <primitive object={grid} position={[0, y, 0]} />;
}

export function Stage({ children, bloom, fov = 38 }: { children: ReactNode; bloom?: number; fov?: number }) {
  const { width, height } = useVideoConfig();
  return (
    <ThreeCanvas width={width} height={height} camera={{ fov, near: 0.05, far: 80 }}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      style={{ position: "absolute", inset: 0 }}>
      <color attach="background" args={[C.bg]} />
      <fog attach="fog" args={[C.bg, 10, 24]} />
      <Lights />
      {children}
      <PostFX bloom={bloom} />
    </ThreeCanvas>
  );
}

export function useBoneMaterial(color = C.bone) {
  return useMemo(() => new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(color), roughness: 0.55, clearcoat: 0.35, clearcoatRoughness: 0.4,
    sheen: 0.4, sheenColor: new THREE.Color("#ffd9a8"), transparent: true,
  }), [color]);
}
