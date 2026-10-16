/**
 * eval:gate wrapper — runs the official 50-event gate offline.
 *
 * Picks models/<name> (default events-v1, override: arg or JEV_MODEL env)
 * when it is packaged (models/<name>/onnx exists), else falls back to the
 * kev-0.6b baseline, then spawns scripts/eval-dataset.mjs <model> --gate
 * with HF_HUB_OFFLINE=1 (HF network is flaky; models are cached).
 *
 * Re-execs through an arm64 node when available (shared resolver in
 * scripts/shared/arm64.mjs) — the default shell node is x64-under-Rosetta
 * and can't load the native ONNX runtime.
 *
 * Usage: node scripts/eval-gate.mjs [model-name]     (npm run eval:gate)
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolveArm64Node } from "./shared/arm64.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const name = process.argv[2] && !process.argv[2].startsWith("--") ? process.argv[2] : (process.env.JEV_MODEL ?? "events-v1");
const local = join(root, "models", name);
const model = existsSync(join(local, "onnx")) ? local : "kev-0.6b";

console.log(`eval:gate → ${model}${model === "kev-0.6b" && name !== "kev-0.6b" ? ` (models/${name} not packaged — baseline fallback)` : ""}`);
const child = spawn(resolveArm64Node() ?? process.execPath, [join(root, "scripts", "eval-dataset.mjs"), model, "--gate"], {
  cwd: root,
  env: { ...process.env, HF_HUB_OFFLINE: "1" },
  stdio: "inherit",
});
child.on("error", (err) => {
  console.error(`error: failed to start eval — ${err.message}`);
  process.exit(1);
});
child.on("close", (code, signal) => process.exit(signal ? 1 : (code ?? 1)));
