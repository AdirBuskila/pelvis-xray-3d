// Generates a Remotion entry that registers only the given videos, so a local render isn't
// broken by other videos that are mid-edit. Usage: node scripts/entry.mjs V09,V10 -> entries/V09_V10.tsx
import fs from "node:fs";
const ids = process.argv[2].split(",");
const name = ids.join("_");
const lines = [
  `import { Composition, registerRoot } from "remotion";`,
  `import { FPS, W, H } from "../src/lib/style";`,
  ...ids.map((id) => `import { ${id}, tl as tl${id} } from "../src/${id.toLowerCase()}/Video";`),
  `const R = () => (<>`,
  ...ids.map((id) => `  <Composition id="${id}" component={${id}} durationInFrames={Math.ceil(tl${id}.total * FPS)} fps={FPS} width={W} height={H} />`),
  `</>);`,
  `registerRoot(R);`,
];
fs.mkdirSync("entries", { recursive: true });
fs.writeFileSync(`entries/${name}.tsx`, lines.join("\n") + "\n");
console.log(`entries/${name}.tsx`);
