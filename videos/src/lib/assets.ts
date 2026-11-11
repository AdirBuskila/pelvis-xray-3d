// Asset loaders. Load in the DOM tree and pass results into the 3D scene as props:
// loading inside the R3F tree finishes after Remotion has already drawn the frame (verified in the spike).
import { useEffect, useState } from "react";
import { cancelRender, continueRender, delayRender, staticFile } from "remotion";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

export function useObj(url: string | null) {
  const [geo, setGeo] = useState<THREE.BufferGeometry | null>(null);
  const [handle] = useState(() => (url ? delayRender(`obj ${url}`) : null));
  useEffect(() => {
    if (!url || handle === null) return;
    new OBJLoader().load(staticFile(url), (o) => {
      setGeo((o.children[0] as THREE.Mesh).geometry);
      continueRender(handle);
    }, undefined, (e) => cancelRender(e));
  }, [url, handle]);
  return geo;
}

export function useTex(url: string | null, srgb = true) {
  const [tex, setTex] = useState<THREE.Texture | null>(null);
  const [handle] = useState(() => (url ? delayRender(`tex ${url}`) : null));
  useEffect(() => {
    if (!url || handle === null) return;
    new THREE.TextureLoader().load(staticFile(url), (t) => {
      if (srgb) t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      setTex(t);
      continueRender(handle);
    }, undefined, (e) => cancelRender(e));
  }, [url, handle, srgb]);
  return tex;
}

export function useJson<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [handle] = useState(() => delayRender(`json ${url}`));
  useEffect(() => {
    fetch(staticFile(url)).then((r) => r.json()).then((d) => { setData(d); continueRender(handle); })
      .catch((e) => cancelRender(e));
  }, [url, handle]);
  return data;
}

/** raw uint8 volume (x fastest) -> THREE.Data3DTexture, for slice/volume scenes */
export function useVolume(url: string | null, dims: [number, number, number]) {
  const [tex, setTex] = useState<THREE.Data3DTexture | null>(null);
  const [handle] = useState(() => (url ? delayRender(`vol ${url}`) : null));
  useEffect(() => {
    if (!url || handle === null) return;
    fetch(staticFile(url)).then((r) => r.arrayBuffer()).then((buf) => {
      const t = new THREE.Data3DTexture(new Uint8Array(buf), ...dims);
      t.format = THREE.RedFormat; t.type = THREE.UnsignedByteType;
      t.minFilter = t.magFilter = THREE.LinearFilter; t.unpackAlignment = 1; t.needsUpdate = true;
      setTex(t); continueRender(handle);
    }).catch((e) => cancelRender(e));
  }, [url, handle, dims[0], dims[1], dims[2]]);
  return tex;
}
