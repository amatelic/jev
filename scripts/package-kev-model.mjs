/**
 * Package a fine-tuned kev model directory for the open-jev JS runtime
 * (docs/TRAINING.md §7): config.json with a proper "kev" section, full
 * tokenizer set, and the onnx/ graph files from scripts/export+quantize.
 *
 * Usage: node scripts/package-kev-model.mjs <name> [--from-base kev-0.6b]
 * Reads : models/<name>/model_q4.onnx (+_data, model_q4f16.*) — from quantize step
 * Writes: models/<name>/{config.json, tokenizer files}
 */

import { cpSync, existsSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const name = process.argv[2];
if (!name) {
  console.error("usage: node scripts/package-kev-model.mjs <model-dir-name>");
  process.exit(1);
}

const BASE_SNAP = join(
  homedir(), ".cache/huggingface/hub/models--onnx-community--kev-0.6b-ONNX/snapshots",
);
const baseDir = existsSync(BASE_SNAP)
  ? join(BASE_SNAP, readFileSync(join(BASE_SNAP, "../refs/main"), "utf8").trim())
  : null;
if (!baseDir || !existsSync(baseDir)) {
  console.error("kev-0.6b-ONNX base snapshot not cached — run scripts/fetch-hf-cache.sh first");
  process.exit(1);
}

const outDir = `models/${name}`;
const tidied = existsSync(`${outDir}/onnx/model_q4.onnx`); // already in the loadable layout
if (!tidied && !existsSync(`${outDir}/model_q4.onnx`)) {
  console.error(`missing ${outDir}/model_q4.onnx — run export + quantize first`);
  console.error(`(or ${outDir}/onnx/model_q4.onnx for an already-packaged model)`);
  process.exit(1);
}
// Re-running on an already-tidied model only refreshes config.json + tokenizer.

// ── tokenizer files: copy the full set from the base snapshot ──────
for (const f of [
  "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
  "vocab.json", "merges.txt", "added_tokens.json",
]) {
  cpSync(join(baseDir, f), join(outDir, f));
  console.log(`✓ ${f}`);
}

// ── config.json: base + edited kev section ─────────────────────────
const config = JSON.parse(readFileSync(join(baseDir, "config.json"), "utf8"));

// pass trained lengths via env or CLI: PKG_TRAINED_STATE / PKG_TRAINED_BRANCH
const trainedState = Number(process.env.PKG_TRAINED_STATE ?? 384);
const trainedBranch = Number(process.env.PKG_TRAINED_BRANCH ?? 1024);

// display name: --display-name="..." wins; else preserve an existing one
const displayName =
  process.argv.find((a) => a.startsWith("--display-name="))?.slice("--display-name=".length)
  ?? (existsSync(join(outDir, "config.json"))
      ? readFileSync(join(outDir, "config.json"), "utf8").match(/"display_name"\s*:\s*"([^"]*)"/)?.[1]
      : undefined);
if (displayName) config.display_name = displayName;

config.kev = {
  ...config.kev,
  description: `Fine-tuned kev-0.6b (jev events domain): ${name}. Same contract as the base — pointer head over </opt>×<decide>, block-causal branches in-graph.`,
  trained_max_state_tokens: trainedState,
  trained_max_branch_tokens: trainedBranch,
  lora: { r: 16, alpha: 32, merged: true, init_from: "jaredpalmer/kev-0.6b" },
};

// external data flags reflect what actually shipped
const external = {};
if (existsSync(`${outDir}/model_q4.onnx_data`)) external["model_q4.onnx"] = 1;
if (existsSync(`${outDir}/model_q4f16.onnx_data`)) external["model_q4f16.onnx"] = 1;
config["transformers.js_config"] = {
  dtype: existsSync(`${outDir}/model_q4f16.onnx`) ? "q4f16" : "q4",
  ...(Object.keys(external).length ? { use_external_data_format: external } : {}),
};

writeFileSync(join(outDir, "config.json"), JSON.stringify(config, null, 2) + "\n");
console.log(`✓ config.json (kev section: trained ${trainedState}/${trainedBranch}, dtype ${config["transformers.js_config"].dtype})`);

// size report
let total = 0;
for (const f of ["model.onnx", "model_q4.onnx", "model_q4.onnx_data", "model_q4f16.onnx", "model_q4f16.onnx_data"]) {
  const p = existsSync(join(outDir, f)) ? join(outDir, f) : join(outDir, "onnx", f);
  if (existsSync(p)) {
    const mb = statSync(p).size / 1e6;
    total += mb;
    console.log(`  ${f}: ${mb.toFixed(0)} MB`);
}
}
console.log(`  package total (onnx): ${total.toFixed(0)} MB`);
console.log(`\nnext: node scripts/eval-dataset.mjs $(pwd)/${outDir}`);
