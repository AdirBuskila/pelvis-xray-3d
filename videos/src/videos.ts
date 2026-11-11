// Registry of the series. Each video folder exports its component and its narration timeline.
import type { ComponentType } from "react";
import { V00, tl as tl00 } from "./v00/Video";

export const VIDEOS: { id: string; component: ComponentType; seconds: number }[] = [
  { id: "V00", component: V00, seconds: tl00.total },
];
