// SPIKE (throwaway): 50-second test shot to judge the video quality.
// Real CT (DeepFluoro subject01) -> bone meshes + parallel-beam DRRs, rendered with three.js inside Remotion.
import { useEffect, useMemo, useState } from "react";
import {
  AbsoluteFill, Audio, Easing, Sequence, cancelRender, continueRender, delayRender,
  interpolate, staticFile, useCurrentFrame, useVideoConfig,
} from "remotion";
import { ThreeCanvas } from "@remotion/three";
import { useThree, useFrame } from "@react-three/fiber";
import { Environment, Lightformer } from "@react-three/drei";
import { BloomEffect, EffectComposer as PPComposer, EffectPass, NoiseEffect, RenderPass, VignetteEffect, BlendFunction } from "postprocessing";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import katex from "katex";
import "katex/dist/katex.min.css";
import "@fontsource/rubik/hebrew-400.css";
import "@fontsource/rubik/hebrew-700.css";
import "@fontsource/rubik/latin-400.css";
import "@fontsource/rubik/latin-700.css";
import durations from "../../public/audio/spike/durations.json";

const fontFamily = "Rubik"; // bundled locally via @fontsource, so renders never depend on the network

// ---------- timeline (seconds), driven by the narration lengths ----------
const GAP = 0.5;
const T: Record<string, [number, number]> = {};
{
  let t = 0.6;
  for (const id of ["s1", "s2", "s3", "s4"]) {
    const d = (durations as Record<string, number>)[id];
    T[id] = [t, t + d];
    t += d + GAP;
  }
}
export const SPIKE_DURATION = T.s4[1] + 2.0;

const CAPTIONS: Record<string, string> = {
  s1: "זה האגן של אדם אמיתי, משוחזר מסריקת CT. בפרויקט שלנו ננסה לבנות אותו מחדש משני צילומי רנטגן בלבד.",
  s2: "מקור קרינה שולח קרניים דרך הגוף, וכל קרן נחלשת בדרכה. יותר עצם בדרך: קרן חלשה יותר, פיקסל בהיר יותר.",
  s3: "חוק Beer–Lambert: העוצמה בגלאי = העוצמה ההתחלתית × אקספוננט של מינוס סכום ההנחתה לאורך הקרן.",
  s4: "צילום מלפנים וצילום מהצד. השאלה של הפרויקט: האם שתי תמונות שטוחות מספיקות לשחזור צורה תלת-ממדית?",
};

// pelvis bounding box in scene units (mm * 0.01): x = L-R, y = S, z = P
const W = 3.98, H = 3.41, D = 1.97;
const AP_SRC_Z = -2.9, AP_DET_Z = 1.25;     // AP: beam travels anterior (-z) -> posterior (+z)
const LAT_SRC_X = 3.2, LAT_DET_X = -2.25;   // LAT: beam travels +x -> -x

const ease = Easing.bezier(0.45, 0, 0.2, 1);
const sec = (f: number, fps: number) => f / fps;
const ramp = (t: number, a: number, b: number) =>
  interpolate(t, [a, b], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });

// ---------- assets ----------
function useObj(url: string) {
  const [geo, setGeo] = useState<THREE.BufferGeometry | null>(null);
  const [handle] = useState(() => delayRender(`load ${url}`));
  useEffect(() => {
    new OBJLoader().load(staticFile(url), (o) => {
      const mesh = o.children[0] as THREE.Mesh;
      setGeo(mesh.geometry);
      continueRender(handle);
    }, undefined, (e) => cancelRender(e));
  }, [url, handle]);
  return geo;
}

function useTex(url: string) {
  const [tex, setTex] = useState<THREE.Texture | null>(null);
  const [handle] = useState(() => delayRender(`tex ${url}`));
  useEffect(() => {
    new THREE.TextureLoader().load(staticFile(url), (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      setTex(t);
      continueRender(handle);
    }, undefined, (e) => cancelRender(e));
  }, [url, handle]);
  return tex;
}

// ---------- camera ----------
type Key = { t: number; pos: [number, number, number]; look: [number, number, number] };
const lerp3 = (a: number[], b: number[], k: number) => a.map((v, i) => v + (b[i] - v) * k) as [number, number, number];

function FrameProbe() {
  useFrame((state) => {
    const v = new THREE.Vector2(); state.gl.getDrawingBufferSize(v);
    console.log("PROBE size", JSON.stringify(state.size), "dpr", state.viewport.dpr, "buffer", v.x, v.y,
      "canvas", state.gl.domElement.width, state.gl.domElement.height);
    state.gl.render(state.scene, state.camera);
  }, 1);
  return null;
}

// Post-processing driven manually: @react-three/postprocessing's <EffectComposer> drew nothing under
// Remotion + ANGLE (verified), while a manual render inside useFrame works.
function PostFX() {
  const { gl, scene, camera, size } = useThree();
  const composer = useMemo(() => {
    const noise = new NoiseEffect({ blendFunction: BlendFunction.SOFT_LIGHT, premultiply: false });
    noise.blendMode.opacity.value = 0.1;
    const c = new PPComposer(gl, { frameBufferType: THREE.HalfFloatType });
    c.addPass(new RenderPass(scene, camera));
    c.addPass(new EffectPass(camera,
      new BloomEffect({ mipmapBlur: true, intensity: 1.1, luminanceThreshold: 0.62, luminanceSmoothing: 0.2, radius: 0.7 }),
      noise,
      new VignetteEffect({ offset: 0.25, darkness: 0.75 })));
    return c;
  }, [gl, scene, camera]);
  useEffect(() => composer.setSize(size.width, size.height), [composer, size]);
  useFrame((_, delta) => composer.render(delta), 1);
  return null;
}

function CameraRig({ t }: { t: number }) {
  const { camera } = useThree();
  // orbit during s1, then cut-free moves between framed views
  const orbit = (a: number): [number, number, number] => [8.4 * Math.sin(a), 0.6, -8.4 * Math.cos(a)];
  const keys: Key[] = [
    { t: 0, pos: orbit(-0.5), look: [0, 0, 0] },
    { t: T.s1[1], pos: orbit(0.9), look: [0, 0, 0] },
    { t: T.s2[0] + 1.5, pos: [8.2, 1.6, -2.4], look: [0, 0, -0.3] },
    { t: T.s3[1], pos: [7.6, 1.2, -1.2], look: [0, 0, -0.2] },
    { t: T.s4[0] + 4, pos: [6.0, 4.6, -7.4], look: [-0.4, -0.1, 0.2] },
    { t: SPIKE_DURATION, pos: [5.0, 4.0, -7.8], look: [-0.5, -0.1, 0.2] },
  ];
  let i = 0;
  while (i < keys.length - 2 && t > keys[i + 1].t) i++;
  const a = keys[i], b = keys[i + 1];
  // s1 is a true orbit (circular path), other segments ease linearly
  let pos: [number, number, number];
  if (i === 0) {
    pos = orbit(interpolate(t, [a.t, b.t], [-0.5, 0.9], { extrapolateRight: "clamp" }));
  } else {
    pos = lerp3(a.pos, b.pos, ramp(t, a.t, b.t));
  }
  camera.position.set(...pos);
  camera.lookAt(new THREE.Vector3(...lerp3(a.look, b.look, ramp(t, a.t, b.t))));
  return null;
}

// ---------- detector with row-by-row reveal ----------
const detectorShader = {
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    uniform sampler2D map; uniform float progress; uniform float opacity; varying vec2 vUv;
    void main(){
      float row = 1.0 - vUv.y;                       // top row first
      vec3 img = texture2D(map, vUv).rgb;
      vec3 panel = vec3(0.03, 0.05, 0.08);
      float shown = step(row, progress);
      float edge = smoothstep(0.018, 0.0, abs(row - progress)) * step(progress, 0.999) * step(0.001, progress);
      vec3 col = mix(panel, img, shown) + edge * vec3(0.55, 0.85, 1.6);
      gl_FragColor = vec4(col, opacity);
    }`,
};

function Detector({ tex, progress, opacity, size, position, rotation }: {
  tex: THREE.Texture; progress: number; opacity: number; size: [number, number];
  position: [number, number, number]; rotation: [number, number, number];
}) {
  const uniforms = useMemo(() => ({ map: { value: tex }, progress: { value: 0 }, opacity: { value: 1 } }), [tex]);
  uniforms.progress.value = progress;
  uniforms.opacity.value = opacity;
  return (
    <group position={position} rotation={rotation}>
      <mesh>
        <planeGeometry args={size} />
        <shaderMaterial args={[detectorShader]} uniforms={uniforms} transparent side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
      {/* thin frame */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(size[0] * 1.02, size[1] * 1.02)]} />
        <lineBasicMaterial color={new THREE.Color(0.45, 0.75, 1.0)} transparent opacity={0.6 * opacity} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

// ---------- parallel X-ray beam: a scan line of rays sweeping top -> bottom ----------
function Beam({ progress, opacity, axis }: { progress: number; opacity: number; axis: "ap" | "lat" }) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  const N = 34;
  const y = H / 2 - progress * H;
  const pos: number[] = [], col: number[] = [];
  for (let k = 0; k < N; k++) {
    const u = (k + 0.5) / N - 0.5;
    const jitter = Math.sin(k * 12.9898) * 0.012;
    if (axis === "ap") {
      const x = u * W;
      // bright before the body, dimmer after it (attenuated)
      pos.push(x, y + jitter, AP_SRC_Z, x, y + jitter, -D / 2, x, y + jitter, -D / 2, x, y + jitter, AP_DET_Z);
    } else {
      const z = u * D;
      pos.push(LAT_SRC_X, y + jitter, z, W / 2, y + jitter, z, W / 2, y + jitter, z, LAT_DET_X, y + jitter, z);
    }
    col.push(1.4, 2.1, 3.0, 1.4, 2.1, 3.0, 0.5, 0.8, 1.2, 0.25, 0.4, 0.6);
  }
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  if (opacity <= 0.001) return null;
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent opacity={opacity} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

function SourcePanel({ opacity, axis }: { opacity: number; axis: "ap" | "lat" }) {
  if (opacity <= 0.001) return null;
  const isAp = axis === "ap";
  const size: [number, number] = isAp ? [W, H] : [D, H];
  return (
    <group position={isAp ? [0, 0, AP_SRC_Z] : [LAT_SRC_X, 0, 0]} rotation={isAp ? [0, 0, 0] : [0, Math.PI / 2, 0]}>
      {/* faint emitter surface + glowing outline: reads as "the source", never as a wall */}
      <mesh>
        <planeGeometry args={size} />
        <meshBasicMaterial color={new THREE.Color(0.3, 0.55, 1.0)} transparent opacity={0.025 * opacity}
          side={THREE.DoubleSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(...size)]} />
        <lineBasicMaterial color={new THREE.Color(0.6, 1.0, 2.2)} transparent opacity={0.9 * opacity} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

// ---------- the 3D scene ----------
type Assets = {
  pelvis: THREE.BufferGeometry; femurs: THREE.BufferGeometry; spine: THREE.BufferGeometry;
  ap: THREE.Texture; lat: THREE.Texture;
};

function Scene({ t, fx, assets }: { t: number; fx: boolean; assets: Assets }) {
  const { pelvis, femurs, spine, ap, lat } = assets;

  const boneMat = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: new THREE.Color("#efe6d2"), roughness: 0.55, clearcoat: 0.35, clearcoatRoughness: 0.4,
    sheen: 0.4, sheenColor: new THREE.Color("#ffd9a8"), transparent: true,
  }), []);
  const otherMat = useMemo(() => boneMat.clone(), [boneMat]);

  // phases
  const apOn = ramp(t, T.s2[0] + 1.0, T.s2[0] + 2.2);
  const apScan = ramp(t, T.s2[0] + 2.5, T.s2[1] - 0.5);
  const apBeamOut = 1 - ramp(t, T.s2[1] - 0.6, T.s2[1] + 0.4);
  const latOn = ramp(t, T.s4[0] + 0.5, T.s4[0] + 1.8);
  const latScan = ramp(t, T.s4[0] + 2.0, T.s4[0] + 7.5);
  const latBeamOut = 1 - ramp(t, T.s4[0] + 7.3, T.s4[0] + 8.2);
  const ghost = ramp(t, T.s2[0] + 1.0, T.s2[0] + 2.5);   // bones turn translucent so rays read through them
  boneMat.opacity = 1 - 0.55 * ghost;
  boneMat.depthWrite = ghost < 0.5;
  otherMat.opacity = (1 - 0.7 * ghost) * 0.9;
  otherMat.depthWrite = ghost < 0.5;

  const spin = interpolate(t, [0, SPIKE_DURATION], [0, 0.0]);
  return (
    <>
      <CameraRig t={t} />
      <color attach="background" args={["#05070b"]} />
      <fog attach="fog" args={["#05070b", 9, 20]} />
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

      <group rotation={[0, spin, 0]}>
        {pelvis && <mesh geometry={pelvis} material={boneMat} />}
        {femurs && <mesh geometry={femurs} material={otherMat} />}
        {spine && <mesh geometry={spine} material={otherMat} />}
      </group>

      {/* AP setup */}
      <SourcePanel opacity={apOn} axis="ap" />
      <Beam axis="ap" progress={apScan} opacity={apOn * apBeamOut * (apScan > 0 ? 1 : 0)} />
      {ap && <Detector tex={ap} progress={apScan} opacity={apOn} size={[W, H]} position={[0, 0, AP_DET_Z]} rotation={[0, Math.PI, 0]} />}

      {/* LAT setup */}
      <SourcePanel opacity={latOn} axis="lat" />
      <Beam axis="lat" progress={latScan} opacity={latOn * latBeamOut * (latScan > 0 ? 1 : 0)} />
      {lat && <Detector tex={lat} progress={latScan} opacity={latOn} size={[D, H]} position={[LAT_DET_X, 0, 0]} rotation={[0, Math.PI / 2, 0]} />}

      {/* floor grid for depth cues */}
      <gridHelper args={[30, 60, "#1f4a6e", "#12283c"]} position={[0, -H / 2 - 0.25, 0]} />

      {fx && <PostFX />}
    </>
  );
}

// ---------- 2D overlays ----------
function Caption({ id, t }: { id: string; t: number }) {
  const [a, b] = T[id];
  const o = ramp(t, a - 0.2, a + 0.3) * (1 - ramp(t, b + 0.1, b + 0.5));
  if (o <= 0) return null;
  return (
    <div style={{
      position: "absolute", bottom: 64, left: 240, right: 240, textAlign: "center", direction: "rtl",
      fontFamily, fontSize: 38, lineHeight: 1.45, color: "#e9f1ff", opacity: o,
      textShadow: "0 2px 18px rgba(0,0,0,0.9)",
    }}>{CAPTIONS[id]}</div>
  );
}

function Title({ t }: { t: number }) {
  const o = ramp(t, 0.3, 1.3) * (1 - ramp(t, T.s1[1] - 1.5, T.s1[1]));
  return (
    <div style={{ position: "absolute", top: 90, width: "100%", textAlign: "center", opacity: o, direction: "rtl" }}>
      <div style={{ fontFamily, fontWeight: 700, fontSize: 84, color: "white", letterSpacing: -1 }}>
        משני צילומי רנטגן לאגן תלת-ממדי
      </div>
      <div style={{ fontFamily, fontSize: 34, color: "#8fb8e8", marginTop: 10 }}>
        סרטון ניסיון · ראייה ממוחשבת · אדיר ונועה
      </div>
    </div>
  );
}

function Equation({ t }: { t: number }) {
  const [a, b] = T.s3;
  const o = ramp(t, a - 0.1, a + 0.7) * (1 - ramp(t, b + 0.3, b + 1.0));
  if (o <= 0) return null;
  // highlight terms in step with the narration
  const d = b - a;
  const hl = (k0: number, k1: number) => ramp(t, a + d * k0, a + d * k0 + 0.3) * (1 - ramp(t, a + d * k1, a + d * k1 + 0.3));
  const c = (k: number) => `rgba(120,200,255,${0.15 + 0.85 * k})`;
  const tex = String.raw`\htmlStyle{color:${hl(0.12, 0.34) > 0.5 ? "#7fd0ff" : "white"}}{I} = \htmlStyle{color:${hl(0.34, 0.55) > 0.5 ? "#7fd0ff" : "white"}}{I_0}\, \htmlStyle{color:${hl(0.55, 1.2) > 0.5 ? "#ffcf7f" : "white"}}{e^{-\int \mu(s)\,ds}}`;
  const html = katex.renderToString(tex, { trust: true, strict: false, displayMode: true });
  return (
    <div style={{
      position: "absolute", left: 110, top: 300, width: 760, padding: "42px 30px", borderRadius: 28, opacity: o,
      background: "rgba(8,14,24,0.72)", border: `1.5px solid ${c(0.6)}`, boxShadow: "0 0 60px rgba(60,140,255,0.18)",
      transform: `translateY(${(1 - o) * 30}px)`,
    }}>
      <div style={{ fontSize: 66, color: "white" }} dangerouslySetInnerHTML={{ __html: html }} />
      <div style={{ fontFamily, direction: "rtl", fontSize: 30, color: "#9fb8d8", textAlign: "center", lineHeight: 1.6 }}>
        μ = מקדם ההנחתה של הרקמה · ∫ = סכום לאורך הקרן
      </div>
    </div>
  );
}

function Question({ t }: { t: number }) {
  const o = ramp(t, T.s4[1] - 3.5, T.s4[1] - 2.5);
  return (
    <div style={{ position: "absolute", top: 80, width: "100%", textAlign: "center", opacity: o, direction: "rtl",
      fontFamily, fontWeight: 700, fontSize: 64, color: "white", textShadow: "0 0 40px rgba(80,160,255,0.6)" }}>
      2 תמונות שטוחות ⟵ צורה תלת-ממדית?
    </div>
  );
}

export const Spike = ({ fx = true }: { fx?: boolean }) => {
  // assets load in the DOM tree (not inside the R3F tree), so the canvas only mounts once they exist
  const pelvis = useObj("data/spike/pelvis.obj");
  const femurs = useObj("data/spike/femurs.obj");
  const spine = useObj("data/spike/spine.obj");
  const ap = useTex("data/spike/drr_ap.png");
  const lat = useTex("data/spike/drr_lat.png");
  const ready = pelvis && femurs && spine && ap && lat;
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = sec(frame, fps);
  return (
    <AbsoluteFill style={{ backgroundColor: "#05070b" }}>
      {ready && <ThreeCanvas width={width} height={height} camera={{ fov: 38, near: 0.1, far: 60 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
        <Scene t={t} fx={fx} assets={{ pelvis, femurs, spine, ap, lat }} />
      </ThreeCanvas>}
      <Title t={t} />
      <Equation t={t} />
      <Question t={t} />
      {["s1", "s2", "s3", "s4"].map((id) => <Caption key={id} id={id} t={t} />)}
      {["s1", "s2", "s3", "s4"].map((id) => (
        <Sequence key={id} from={Math.round(T[id][0] * fps)}>
          <Audio src={staticFile(`audio/spike/${id}.mp3`)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
