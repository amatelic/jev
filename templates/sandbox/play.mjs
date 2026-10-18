// sandbox/play.mjs — throwaway jev playground.
//
// Edit ./states.txt (one event per line) and run:  npm run sandbox
// This folder is never committed; restore defaults:  npm run sandbox:reset
//
// Optional labels: end a line with " | category" to check the prediction:
//   HC Sparta Praha hockey match | sports        → ✓/✗ mark + tally
// Unlabeled lines just print the prediction.
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

// "text | label" → { text, label? }
const states = readFileSync(new URL("./states.txt", import.meta.url), "utf8")
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith("#"))
  .map((line) => {
    const sep = line.indexOf(" | ");
    return sep === -1 ? { text: line } : { text: line.slice(0, sep), label: line.slice(sep + 3) };
  });

console.log(`model: ${MODEL} · ${states.length} states from states.txt\n`);
const rt = await OpenJev.load({ model: MODEL, device: "cpu", dtype: "q4" });

let labeled = 0, correct = 0;
for (const { text, label } of states) {
  const [answer] = await rt.decide(text, [
    choice("What type of event is this?", Object.keys(OPTIONS), OPTIONS),
  ]);
  const predicted = String(answer.choice);
  const conf = `${(answer.confidence * 100).toFixed(0).padStart(3)}%`;
  if (label === undefined) {
    console.log(`  ? ${predicted.padEnd(12)} ${conf}  ${text}`);
  } else {
    labeled++;
    const ok = predicted === label;
    if (ok) correct++;
    console.log(`${ok ? "✓" : "✗"} ${predicted.padEnd(12)} ${conf}  ${text}${ok ? "" : `  (expected ${label})`}`);
  }
}

await rt.dispose?.();
if (labeled > 0) console.log(`\n${correct}/${labeled} labeled correct`);
