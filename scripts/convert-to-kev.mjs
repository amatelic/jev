/**
 * Convert data/{train,holdout}.jsonl (jev eval format) into kev's
 * training JSONL (vendor/kev kev/data.py load_records schema):
 *
 *   {"state": "...",
 *    "questions": {"category": {"type": "choice",
 *                               "instructions": "...",
 *                               "criteria": {"opt": "desc", ...},   // key ORDER = option order
 *                               "label": "opt"}}}
 *
 * Usage: node scripts/convert-to-kev.mjs [train|holdout] [> out.jsonl]
 * Writes to data/kev-<split>.jsonl by default.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { CATEGORIES, CATEGORY_DESCRIPTIONS } from "./shared/question.mjs";

const split = process.argv[2] ?? "train";
const out = process.argv[3] ?? `data/kev-${split}.jsonl`;

// criteria preserves eval.mjs's CATEGORIES order exactly (question_keys
// returns list(criteria) — order is the contract, kev/api.py:94).
const criteria = Object.fromEntries(CATEGORIES.map((c) => [c, CATEGORY_DESCRIPTIONS[c]]));

const lines = readFileSync(`data/${split}.jsonl`, "utf8").trim().split("\n");
const outLines = lines.map((line) => {
  const r = JSON.parse(line);
  const expected = r.expected.choice;
  return JSON.stringify({
    state: r.state,
    questions: {
      category: {
        type: "choice",
        instructions: r.question.prompt,
        criteria: { ...criteria },
        label: expected,
      },
    },
    _meta: { source: "jev-events", split: r.meta.split, cat: r.meta.cat, sub: r.meta.sub, id: r.meta.id, lang: r.meta.lang },
  });
});

writeFileSync(out, outLines.join("\n") + "\n");
console.log(`wrote ${outLines.length} records to ${out}`);

// Sanity: labels all valid categories, criteria order matches CATEGORIES
const first = JSON.parse(outLines[0]);
const keys = Object.keys(first.questions.category.criteria);
if (JSON.stringify(keys) !== JSON.stringify(CATEGORIES)) {
  console.error("criteria order mismatch!");
  process.exit(1);
}
console.log("criteria order ✓  sample:", first.questions.category.label, "|", first.state.slice(0, 50));
