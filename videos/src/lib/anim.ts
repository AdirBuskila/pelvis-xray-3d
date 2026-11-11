import { Easing, interpolate } from "remotion";

export const ease = Easing.bezier(0.45, 0, 0.2, 1);
/** 0 -> 1 between seconds a and b, eased and clamped */
export const ramp = (t: number, a: number, b: number, e = ease) =>
  interpolate(t, [a, b], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: e });
/** fades in over `fin` after a, fades out over `fout` before b */
export const window01 = (t: number, a: number, b: number, fin = 0.4, fout = 0.4) =>
  ramp(t, a, a + fin) * (1 - ramp(t, b - fout, b));
export const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
export const lerp3 = (a: number[], b: number[], k: number) =>
  a.map((v, i) => v + (b[i] - v) * k) as [number, number, number];
