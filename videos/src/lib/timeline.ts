// Narration drives the clock: every segment's start/end comes from its measured mp3 duration.
export type Segment = { id: string; text: string; caption?: string; pauseAfter?: number };
export type Script = { voice: string; rate?: string; lead?: number; tail?: number; segments: Segment[] };
export type Timeline = {
  seg: Record<string, [number, number]>;     // id -> [start, end] seconds
  order: string[];
  total: number;                              // seconds
  script: Script;
};

export function buildTimeline(script: Script, durations: Record<string, number>, gap = 0.45): Timeline {
  const seg: Record<string, [number, number]> = {};
  let t = script.lead ?? 1.2;
  for (const s of script.segments) {
    const d = durations[s.id];
    if (d === undefined) throw new Error(`no narration audio for segment ${s.id} - run prep/tts.py`);
    seg[s.id] = [t, t + d];
    t += d + (s.pauseAfter ?? gap);
  }
  return { seg, order: script.segments.map((s) => s.id), total: t + (script.tail ?? 1.5), script };
}

/** seconds from the start of segment `id` (negative before it starts) */
export const since = (tl: Timeline, id: string, t: number) => t - tl.seg[id][0];
/** progress 0..1 through segment `id` */
export const through = (tl: Timeline, id: string, t: number) => {
  const [a, b] = tl.seg[id];
  return Math.min(1, Math.max(0, (t - a) / (b - a)));
};
/** time span covering a list of consecutive segments */
export const span = (tl: Timeline, from: string, to: string): [number, number] => [tl.seg[from][0], tl.seg[to][1]];
