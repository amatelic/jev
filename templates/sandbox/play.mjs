// sandbox/play.mjs — throwaway jev playground.
//
// Edit ./states.txt (one event per line) and run:  npm run sandbox
// This folder is never committed; restore defaults:  npm run sandbox:reset
//
// To ask different questions, edit the question below — see open-jev's
// noul() and score() builders for yes/no and rating questions.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { OpenJev, choice } from "open-jev";

// Prefer the local fine-tuned model, fall back to the cached baseline.
// (local models need an ABSOLUTE path — relative paths go to the HF hub)
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = existsSync(join(root, "models/events-v1/onnx"))
  ? join(root, "models/events-v1")
  : "kev-0.6b";

// The category question (same options/descriptions as eval.mjs).
const OPTIONS = {
  concert: "Live music performance, band, DJ, tour, festival",
  sports: "Sports match, game, race, tournament, athletic competition",
  exhibition: "Art exhibition, museum show, gallery, design fair",
  networking: "Meetup, social gathering, community event, book club",
  dining: "Food event, dinner, brunch, cooking, tasting",
  outdoor: "Hiking, walking tour, outdoor activity, nature",
  wellness: "Yoga, meditation, breathwork, spa, sound healing",
};

const states = readFileSync(new URL("./states.txt", import.meta.url), "utf8")
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith("#"));

console.log(`model: ${MODEL} · ${states.length} states from states.txt\n`);
const rt = await OpenJev.load({ model: MODEL, device: "cpu", dtype: "q4" });

for (const state of states) {
  const [answer] = await rt.decide(state, [
    choice("What type of event is this?", Object.keys(OPTIONS), OPTIONS),
  ]);
  console.log(`${String(answer.choice).padEnd(12)} ${(answer.confidence * 100).toFixed(0).padStart(3)}%  ${state}`);
}

await rt.dispose?.();
