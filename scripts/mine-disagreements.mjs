/**
 * mine-disagreements — docs/TRAINING.md §4.3, round-2 training data.
 *
 * Generates a fresh candidate pool (new seed, TRAIN entity banks only — never
 * the holdout banks, so the v1 holdout stays a clean eval for events-v2),
 * runs the model ensemble over it, and keeps the records worth learning from:
 *
 *   - events-v1 is WRONG (gold known from the generator)
 *   - the models DISAGREE on the category
 *   - events-v1 confidence < 0.7 (near-miss)
 *
 * Output (kev trainer format, gold labels, provenance in _meta):
 *   data/kev-mined-v2.jsonl   — the mined records
 *   data/kev-train-v2.jsonl   — kev-train + mined ×2 (errors weighted double)
 *
 * Usage: node scripts/mine-disagreements.mjs [--seed 999] [--pool 400]
 *   [--models kev-0.6b,open-jev,events-v1]   (kev-4b opt-in: slow first load)
 */

import { readFileSync, writeFileSync } from "node:fs";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { OpenJev, choice } from "open-jev";
import { generate } from "./generate-training-data.mjs";
import { CATEGORIES, CATEGORY_DESCRIPTIONS } from "./shared/question.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

// ── CLI ─────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i !== -1 && argv[i + 1] ? argv[i + 1] : fallback;
};
const SEED = Number(opt("seed", 999));
const POOL = Number(opt("pool", 400));
const MODELS = opt("models", "kev-0.6b,open-jev,events-v1").split(",").map((s) => s.trim());

const localPath = (name) => (existsSync(join(ROOT, "models", name, "onnx"))
  ? join(ROOT, "models", name)
  : name); // hub alias

// ── Pool: fresh instances from TRAIN banks (never holdout banks) ────
console.log(`generating pool: seed=${SEED} target=${POOL} (train split only)`);
const { train: pool, stats } = generate({ seed: SEED, total: POOL });
if (stats.overlap || stats.gateLeaks) {
  console.error(`pool safety failed: overlap=${stats.overlap} gateLeaks=${stats.gateLeaks}`);
  process.exit(1);
}
console.log(`pool: ${pool.length} records · gate-clean ✓ split-clean ✓`);

// ── Ensemble voting ─────────────────────────────────────────────────
const QUESTION = choice("What type of event is this?", CATEGORIES, CATEGORY_DESCRIPTIONS);

const runtimes = [];
for (const m of MODELS) {
  const t0 = Date.now();
  const rt = await OpenJev.load({ model: localPath(m), device: "cpu", dtype: "q4" });
  console.log(`loaded ${m} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  runtimes.push({ name: m, rt });
}

const mined = [];
const report = { wrong: 0, disagree: 0, lowConf: 0, perCatErrors: {}, confusions: [] };
const t0 = Date.now();

for (const [i, rec] of pool.entries()) {
  const gold = rec.meta.cat;
  const votes = [];
  for (const { rt } of runtimes) {
    const [a] = await rt.decide(rec.state, [QUESTION]);
    votes.push({ pick: String(a.choice), conf: a.confidence });
  }
  const champ = votes[runtimes.findIndex((r) => r.name === "events-v1")] ?? votes[votes.length - 1];
  const uniquePicks = new Set(votes.map((v) => v.pick));

  const wrong = champ.pick !== gold;
  const disagree = uniquePicks.size > 1;
  const lowConf = champ.conf < 0.7;

  if (wrong) {
    report.wrong++;
    report.perCatErrors[gold] = (report.perCatErrors[gold] ?? 0) + 1;
    report.confusions.push(`${gold}→${champ.pick}  "${rec.state.replace(/^Event: /, "").slice(0, 46)}"`);
  }
  if (disagree) report.disagree++;
  if (lowConf) report.lowConf++;

  if (wrong || disagree || lowConf) {
    mined.push({
      state: rec.state,
      questions: {
        category: {
          type: "choice",
          instructions: "What type of event is this?",
          criteria: Object.fromEntries(CATEGORIES.map((c) => [c, CATEGORY_DESCRIPTIONS[c]])),
          label: gold,
        },
      },
      _meta: { source: "disagreement-mining", split: "train", cat: gold, sub: rec.meta.sub,
               id: `mine-${String(i).padStart(4, "0")}`, lang: rec.meta.lang,
               votes: Object.fromEntries(runtimes.map((r, k) => [r.name, `${votes[k].pick}@${votes[k].conf.toFixed(2)}`])) },
    });
  }

  if ((i + 1) % 50 === 0) {
    const rate = (i + 1) / ((Date.now() - t0) / 1000);
    console.log(`  ${i + 1}/${pool.length} (${rate.toFixed(1)} rec/s) · mined so far: ${mined.length}`);
  }
}

for (const { rt } of runtimes) await rt.dispose?.();

// ── Build the v2 training set: original + mined ×2 ──────────────────
const original = readFileSync(join(ROOT, "data", "kev-train.jsonl"), "utf8").trim().split("\n");
const minedLines = mined.map((r) => JSON.stringify(r));
const v2 = [...original, ...minedLines, ...minedLines]; // errors weighted 2×
writeFileSync(join(ROOT, "data", "kev-mined-v2.jsonl"), minedLines.join("\n") + "\n");
writeFileSync(join(ROOT, "data", "kev-train-v2.jsonl"), v2.join("\n") + "\n");

// ── Report ──────────────────────────────────────────────────────────
console.log(`\n── mining report (pool ${pool.length}, ${((Date.now() - t0) / 1000).toFixed(0)}s) ──`);
console.log(`events-v1 wrong: ${report.wrong} · ensemble disagreements: ${report.disagree} · low-conf (<0.7): ${report.lowConf}`);
console.log(`mined records: ${mined.length} (deduped by condition)`);
console.log(`errors by gold category: ${JSON.stringify(report.perCatErrors)}`);
console.log(`v2 training set: ${v2.length} = ${original.length} original + ${minedLines.length} mined ×2`);
if (report.confusions.length) {
  console.log("\ntop confusions:");
  for (const c of report.confusions.slice(0, 12)) console.log(`  ✗ ${c}`);
}
console.log(`\nnext: node scripts/pipeline.mjs --from train --name events-v2 --data ${join(ROOT, "data", "kev-train-v2.jsonl")} --detach --yes`);
