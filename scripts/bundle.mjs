/**
 * bundle — esbuild step of `npm run build`.
 *
 * tsc emits the .d.ts files; this bundles the two runtime entry points
 * (library + server) with TypeBox tree-shaken INTO the bundle — consumers
 * never download @sinclair/typebox's 5.2 MB dual-format package for the
 * ~200 kB subset this library actually uses.
 *
 * open-jev and @huggingface/transformers stay EXTERNAL: they are optional
 * peers resolved from the consumer's own tree (dynamic imports).
 */

import { build } from "esbuild";

const result = await build({
  entryPoints: ["src/index.ts", "src/server.ts"],
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "node",
  target: "node20",
  sourcemap: true,
  external: ["open-jev", "@huggingface/transformers"],
  metafile: true,
  logLevel: "info",
});

// Report the tree-shaking win
let typeboxBytes = 0;
for (const [file, { bytes, inputs }] of Object.entries(result.metafile.outputs)) {
  for (const input of Object.keys(inputs)) {
    if (input.includes("typebox")) typeboxBytes += inputs[input].bytesInOutput;
  }
}
console.log(`typebox bytes bundled in: ${(typeboxBytes / 1024).toFixed(0)} kB (vs 5.2 MB as a dependency)`);
