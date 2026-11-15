import { Composition } from "remotion";
import { Spike, SPIKE_DURATION } from "./spike/Spike";
import { FPS, W, H } from "./lib/style";
import { VIDEOS } from "./videos";
import { DECK, DeckAmbiguity, DeckHero, DeckHull, DeckRig, DeckVoxels } from "./deck/Deck";

export const Root = () => (
  <>
    {VIDEOS.map((v) => (
      <Composition key={v.id} id={v.id} component={v.component} durationInFrames={Math.ceil(v.seconds * FPS)}
        fps={FPS} width={W} height={H} />
    ))}
    {/* text-free clips for the presentation deck */}
    <Composition id="DeckHero" component={DeckHero} durationInFrames={DECK.hero} fps={FPS} width={W} height={H} />
    <Composition id="DeckRig" component={DeckRig} durationInFrames={DECK.rig} fps={FPS} width={W} height={H} />
    <Composition id="DeckAmbiguity" component={DeckAmbiguity} durationInFrames={DECK.ambiguity} fps={FPS} width={W} height={H} />
    <Composition id="DeckVoxels" component={DeckVoxels} durationInFrames={DECK.voxels} fps={FPS} width={W} height={H} />
    <Composition id="DeckHull" component={DeckHull} durationInFrames={DECK.hull} fps={FPS} width={W} height={H} />
    <Composition id="Spike" component={Spike} durationInFrames={Math.ceil(SPIKE_DURATION * FPS)} fps={FPS} width={W} height={H} />
  </>
);
