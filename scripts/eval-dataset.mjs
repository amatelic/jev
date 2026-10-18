/**
 * Dataset-driven evaluation — scores a model against data/holdout.jsonl
 * (or data/train.jsonl, or the official 50-event gate) using the same
 * metrics as eval.mjs: accuracy, per-category P/R/F1, calibration bins.
 *
 * This is the evaluation half of the docs/TRAINING.md optimization loop:
 *   baseline eval → fine-tune → eval holdout → eval gate → decide
 *
 * Usage:
 *   node scripts/eval-dataset.mjs                       # kev-0.6b on holdout
 *   node scripts/eval-dataset.mjs kev-4b                # any alias/id/path
 *   node scripts/eval-dataset.mjs kev-0.6b --gate       # official 50-event set
 *   node scripts/eval-dataset.mjs /path/to/my-kev       # local fine-tuned model
 *   node scripts/eval-dataset.mjs --list                # available datasets/models
 */

import { readFileSync, existsSync, statSync } from "node:fs";
import { CATEGORIES, CATEGORY_DESCRIPTIONS } from "./shared/question.mjs";
import { reexecIfWrongArch } from "./shared/arm64.mjs";

// Positionals + flags — order-independent: `--gate kev-4b` and `kev-4b --gate`
// both work; unknown flags/positionals error instead of being silently ignored.
const argv = process.argv.slice(2);
const ALLOWED_FLAGS = new Set(["--gate", "--list"]);
const flags = new Set(); // flags actually present on THIS command line
const positionals = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a.startsWith("--")) {
    if (!ALLOWED_FLAGS.has(a)) {
      console.error(`error: unknown flag "${a}" — valid: ${[...ALLOWED_FLAGS].join(" ")} <model> [split]`);
      process.exit(2);
    }
    flags.add(a);
  } else {
    positionals.push(a);
  }
}
const MODEL = positionals[0] ?? "kev-0.6b";
const SPLIT = positionals[1];
const GATE = flags.has("--gate");
if (positionals.length > 2) {
  console.error(`error: unexpected extra argument(s): ${positionals.slice(2).join(" ")}`);
  process.exit(2);
}

// ─── Load records ───────────────────────────────────────────────────

function loadJsonl(path) {
  if (!existsSync(path)) {
    console.error(`dataset not found: ${path} — run node scripts/generate-training-data.mjs first`);
    process.exit(1);
  }
  return readFileSync(path, "utf8").trim().split("\n").map((l) => JSON.parse(l));
}

function loadGate() {
  // Parse the official 50-event set out of eval.mjs (single source of truth).
  const src = readFileSync(new URL("../eval.mjs", import.meta.url), "utf8");
  const texts = [...src.matchAll(/\{ text: "([^"]+)", cat: "(\w+)", sub: "([\w-]+)" \}/g)].map((m) => ({
    text: m[1], cat: m[2], sub: m[3],
  }));
  if (texts.length !== 50) {
    console.error(`gate parse error: extracted ${texts.length}/50 events from eval.mjs`);
    process.exit(1);
  }
  return texts.map((t) => ({
    state: `Event: ${t.text}`,
    expected: { choice: t.cat, probabilities: {} },
    meta: { cat: t.cat, sub: t.sub, source: "official-gate", split: "gate", id: `gate-${t.cat}-${t.sub}` },
  }));
}

// ─── Metrics (mirrors eval.mjs) ─────────────────────────────────────

function report(results, label) {
  const correct = results.filter((r) => r.correct).length;
  const accuracy = ((correct / results.length) * 100).toFixed(1);

  const perCat = {};
  for (const cat of CATEGORIES) {
    const tp = results.filter((r) => r.predicted === cat && r.expected === cat).length;
    const fp = results.filter((r) => r.predicted === cat && r.expected !== cat).length;
    const fn = results.filter((r) => r.predicted !== cat && r.expected === cat).length;
    const p = tp + fp > 0 ? ((tp / (tp + fp)) * 100).toFixed(0) : "—";
    const r_ = tp + fn > 0 ? ((tp / (tp + fn)) * 100).toFixed(0) : "—";
    const f1 = tp + fp + fn > 0 ? ((2 * tp / (2 * tp + fp + fn)) * 100).toFixed(0) : "—";
    perCat[cat] = { tp, fp, fn, p, r_, f1, n: results.filter((x) => x.expected === cat).length };
  }

  const bins = [
    { range: [0, 0.5], label: "<50%" },
    { range: [0.5, 0.7], label: "50-70%" },
    { range: [0.7, 0.85], label: "70-85%" },
    { range: [0.85, 0.95], label: "85-95%" },
    { range: [0.95, 1.01], label: ">95%" },
  ].map((bin) => {
    const inBin = results.filter((r) => r.confidence >= bin.range[0] && r.confidence < bin.range[1]);
    const avgConf = inBin.length ? (inBin.reduce((s, r) => s + r.confidence, 0) / inBin.length * 100).toFixed(0) : "—";
    const actual = inBin.length ? (inBin.filter((r) => r.correct).length / inBin.length * 100).toFixed(0) : "—";
    const gap = inBin.length ? Math.abs(Number(avgConf) - Number(actual)).toFixed(0) : "—";
    return { label: bin.label, count: inBin.length, avgConf, actual, gap };
  });

  console.log(`\n  ${label}: ${correct}/${results.length} = ${accuracy}% accuracy`);
  console.log("  Per-category (P / R / F1):");
  for (const [cat, m] of Object.entries(perCat)) {
    if (m.n === 0) continue;
    const icon = Number(m.f1) >= 70 ? "✓" : Number(m.f1) >= 40 ? "⚠" : "✗";
    console.log(`    ${icon} ${cat.padEnd(14)} P:${String(m.p).padStart(4)}  R:${String(m.r_).padStart(4)}  F1:${String(m.f1).padStart(4)}  (n=${m.n})`);
  }
  console.log("  Calibration (stated vs actual, gap ≤10pp ok):");
  for (const c of bins) {
    if (c.count === 0) continue;
    const well = c.gap === "0" ? "✓" : Number(c.gap) <= 10 ? "≈" : "✗";
    console.log(`    ${c.label.padEnd(10)} n=${String(c.count).padStart(4)}  stated ${String(c.avgConf).padStart(3)}%  actual ${String(c.actual).padStart(3)}%  gap ${String(c.gap).padStart(3)}pp ${well}`);
  }

  const errors = results.filter((r) => !r.correct);
  if (errors.length) {
    console.log(`  Misclassified (${Math.min(errors.length, 8)} of ${errors.length}):`);
    for (const e of errors.slice(0, 8)) {
      console.log(`    ✗ exp=${e.expected.padEnd(12)} got=${e.predicted.padEnd(12)} @${(e.confidence * 100).toFixed(0)}%  "${e.text.slice(0, 44)}"`);
    }
  }
  return { accuracy: Number(accuracy), correct, total: results.length, perCat, bins };
}

// A model arg that is an existing local directory is fully on disk — nothing
// to fetch — so eval can default to offline mode when HF_HUB_OFFLINE is unset
// (HF connectivity is flaky). Remote ids/paths stay untouched so first-time
// downloads still work when online.
function defaultOfflineForLocalModel(model) {
  if (process.env.HF_HUB_OFFLINE !== undefined) return;
  try {
    if (statSync(model).isDirectory()) process.env.HF_HUB_OFFLINE = "1";
  } catch {
    // not a local path — leave the environment alone
  }
}

// ─── Main ───────────────────────────────────────────────────────────

async function main() {
  reexecIfWrongArch();

  if (process.argv.includes("--list")) {
    console.log("models: kev-0.6b, kev-4b, open-jev, <hf-id>, <local-path>");
    console.log("datasets: --gate (official 50), holdout (250), train (1000)");
    return;
  }

  defaultOfflineForLocalModel(MODEL);

  const records = GATE ? loadGate() : loadJsonl(`data/${SPLIT ?? "holdout"}.jsonl`);
  const datasetLabel = GATE ? "official-gate (50 events)" : `${SPLIT ?? "holdout"} (${records.length} records)`;

  // open-jev → @huggingface/transformers → sharp loads arch-specific native
  // bindings at import time, so it must be imported only AFTER the arm64
  // re-exec above (under x64-under-Rosetta the import itself throws).
  const { OpenJev, choice } = await import("open-jev");

  console.log(`Evaluating ${MODEL} on ${datasetLabel}`);
  const tLoad = performance.now();
  const rt = await OpenJev.load({ model: MODEL, device: "cpu", dtype: "q4" });
  console.log(`loaded in ${Math.round(performance.now() - tLoad)}ms`);

  const question = choice(
    "What type of event is this?",
    CATEGORIES,
    CATEGORY_DESCRIPTIONS,
  ); // same builder as eval.mjs — same token stream
  const results = [];
  const t0 = performance.now();

  for (const rec of records) {
    const [ans] = await rt.decide(rec.state, [question]);
    results.push({
      text: rec.state.replace(/^Event: /, "").slice(0, 44),
      expected: rec.meta.cat,
      predicted: ans?.choice ?? "(none)",
      confidence: ans?.confidence ?? 0,
      correct: (ans?.choice ?? "(none)") === rec.meta.cat,
    });
  }
  const totalMs = Math.round(performance.now() - t0);

  const summary = report(results, datasetLabel);
  console.log(`  latency: ${Math.round(totalMs / records.length)}ms avg (${totalMs}ms total)`);

  await rt.dispose?.();

  // Machine-readable line for CI / comparison scripts
  console.log(`\nRESULT\t${MODEL}\t${GATE ? "gate" : (SPLIT ?? "holdout")}\tacc=${summary.accuracy}\tavg_ms=${Math.round(totalMs / records.length)}`);
}

main().catch((e) => {
  console.error("eval failed:", e.message?.slice(0, 300));
  process.exit(1);
});
