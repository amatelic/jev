// ─────────────────────────────────────────────────────────────────────
// sandbox/play.mjs — throwaway jev playground.
//
// This folder is a gitignored working copy of templates/sandbox/:
// edit anything here freely, nothing is committed. Restore the
// pristine default with:  npm run sandbox:reset
//
// The default setup loads a model and asks all three jev question
// types (choice / noul / score) about a few example event states —
// the same shape used by eval.mjs. Swap in your own states/questions.
// ─────────────────────────────────────────────────────────────────────

import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { OpenJev, choice, noul, score } from "open-jev";

// The runtime loads through `open-jev` because it works straight from
// node_modules — no build step. To exercise THIS repo's createJev()
// instead, run `npm run build` once and swap the import + load call for:
//     import { createJev } from "../dist/index.js";   // needs tsc dist

// ── 1. Pick a model ─────────────────────────────────────────────────
// Prefer a local packaged fine-tune (models/<name> with an onnx/ subdir,
// newest first); fall back to the cached kev-0.6b baseline.
const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function pickModel() {
  const dir = join(root, "models");
  let names = [];
  try {
    names = readdirSync(dir);
  } catch {
    // no models/ directory — hub alias fallback below
  }
  const packaged = names
    .filter((n) => existsSync(join(dir, n, "onnx")))
    .map((n) => ({ n, mtime: statSync(join(dir, n)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);
  if (!packaged.length) return { id: "kev-0.6b", label: "kev-0.6b (cached hub alias)" };
  return { id: join(dir, packaged[0].n), label: `models/${packaged[0].n} (local fine-tune)` };
}

const model = pickModel();

// ── 2. Load the runtime ─────────────────────────────────────────────
let rt;
const t0 = Date.now();
try {
  rt = await OpenJev.load({ model: model.id, device: "cpu", dtype: "q4" });
} catch (err) {
  console.error(`error: failed to load ${model.label} — ${err.message}`);
  console.error("hint: `node scripts/doctor.mjs` diagnoses the environment");
  process.exit(1);
}
const loadMs = Date.now() - t0;
console.log(`model: ${model.label} · loaded in ${loadMs} ms\n`);

// ── 3. Example states + all three question types ────────────────────
const STATES = [
  "SABATON The Legendary Tour power metal concert at O2 arena",
  "Prague International Marathon 2026",
  "Yoga in the Park morning session",
];

const QUESTIONS = [
  // choice: event categorization (categories/descriptions from eval.mjs)
  choice("What type of event is this?", ["concert", "sports", "exhibition", "networking", "dining", "outdoor", "wellness"], {
    concert: "Live music performance, band, DJ, tour, festival",
    sports: "Sports match, game, race, tournament, athletic competition",
    exhibition: "Art exhibition, museum show, gallery, design fair",
    networking: "Meetup, social gathering, community event, book club",
    dining: "Food event, dinner, brunch, cooking, tasting",
    outdoor: "Hiking, walking tour, outdoor activity, nature",
    wellness: "Yoga, meditation, breathwork, spa, sound healing",
  }),
  // noul: yes/no statement about the state
  noul("This event is happening within the next 7 days."),
  // score: ordered rating scale (first level = lowest)
  score("Rate the data quality of this event record.", ["poor", "fair", "good", "excellent"]),
];

let answered = 0;
for (const state of STATES) {
  const t = Date.now();
  const [category, soon, quality] = await rt.decide(state, QUESTIONS);
  const ms = Date.now() - t;
  console.log(`state: ${state}`);
  console.log(`  choice → ${category.choice}  (confidence ${(category.confidence * 100).toFixed(1)}%)`);
  console.log(`  noul   → ${soon.answer ? "yes" : "no"}  (p(yes) ${(soon.probability * 100).toFixed(1)}%)`);
  console.log(`  score  → ${quality.level}  (${quality.score.toFixed(2)}/3, confidence ${(quality.confidence * 100).toFixed(1)}%)`);
  console.log(`  decide: ${ms} ms\n`);
  answered += QUESTIONS.length;
}

console.log(`summary: ${STATES.length} states × 3 question types → ${answered} answers · model ${model.label} · load ${loadMs} ms`);
