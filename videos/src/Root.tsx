import { Composition } from "remotion";
import { Spike, SPIKE_DURATION } from "./spike/Spike";
import { FPS, W, H } from "./lib/style";
import { VIDEOS } from "./videos";

export const Root = () => (
  <>
    {VIDEOS.map((v) => (
      <Composition key={v.id} id={v.id} component={v.component} durationInFrames={Math.ceil(v.seconds * FPS)}
        fps={FPS} width={W} height={H} />
    ))}
    <Composition id="Spike" component={Spike} durationInFrames={Math.ceil(SPIKE_DURATION * FPS)} fps={FPS} width={W} height={H} />
  </>
);
