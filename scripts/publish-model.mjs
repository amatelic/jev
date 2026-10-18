/**
 * publish-model — deploy a packaged local model to HuggingFace and register it.
 *
 * Uploads config.json + tokenizer + the q4 ONNX artifact from models/<name>/
 * to a HF repo (default <HF_USER|amatelic>/<name>-ONNX), then records the
 * alias → repo id in models.remote.json and regenerates src/models.generated.ts
 * (models:sync merges the manifest with the local scan — the published entry
 * replaces the local one in listModels()/resolution).
 *
 * Consumers then get createJev({ model: "<name>" }) with auto-download,
 * exactly like the shipped kev-* models.
 *
 * Token: HF_TOKEN env var or ~/.cache/huggingface/token.
 * Uploads run through vendor/kev's uv env (huggingface_hub) — resilient,
 * resumable, LFS-aware.
 *
 * Usage:
 *   node scripts/publish-model.mjs <name> --dry-run     # plan only, no token needed
 *   node scripts/publish-model.mjs <name> --repo org/name [--include-fp32]
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const name = args.find((a) => !a.startsWith("--"));
const dryRun = args.includes("--dry-run");
const includeFp32 = args.includes("--include-fp32");
const repoIdx = args.indexOf("--repo");
const repoArg = repoIdx !== -1 ? args[repoIdx + 1] : undefined;
if (repoIdx !== -1 && (!repoArg || repoArg.startsWith("--") || !/^[\w.-]+\/[\w.-]+$/.test(repoArg))) {
  console.error("error: --repo needs a value like org/name");
  process.exit(1);
}

if (!name) {
  console.error("usage: node scripts/publish-model.mjs <name> [--repo org/name] [--include-fp32] [--dry-run]");
  process.exit(1);
}

const dir = join(process.cwd(), "models", name);
const manifestPath = join(process.cwd(), "models.remote.json");

// ── validate payload ────────────────────────────────────────────────

if (!existsSync(join(dir, "onnx"))) {
  console.error(`error: models/${name}/onnx missing — package the model first (npm run pipeline)`);
  process.exit(1);
}
try {
  const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8"));
  if (typeof config.kev !== "object" && typeof config.open_jev !== "object") {
    throw new Error('config.json has neither a "kev" nor "open_jev" section');
  }
} catch (e) {
  console.error(`error: models/${name}/config.json invalid — ${e.message}`);
  process.exit(1);
}

const FILES = [
  "config.json",
  "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
  "vocab.json", "merges.txt", "added_tokens.json",
  "onnx/model_q4.onnx", "onnx/model_q4.onnx_data",
];
if (includeFp32) FILES.push("onnx/model.onnx", "onnx/model.onnx_data");

const files = FILES.filter((f) => existsSync(join(dir, f)));
const totalMb = Math.round(files.reduce((n, f) => n + statSync(join(dir, f)).size, 0) / 1e6);
const repo = repoArg ?? `${process.env.HF_USER ?? "amatelic"}/${name}-ONNX`;

// ── token (not needed for dry-run) ─────────────────────────────────

function readToken() {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN;
  const p = join(homedir(), ".cache/huggingface/token");
  return existsSync(p) ? readFileSync(p, "utf8").trim() : null;
}

console.log(`publish ${name} → ${repo}`);
console.log(`  files (${files.length}, ${totalMb} MB):`);
for (const f of files) console.log(`    ↑ ${f}`);

if (dryRun) {
  console.log("\ndry-run: nothing uploaded. Remove --dry-run to publish (needs an HF token).");
  process.exit(0);
}

const token = readToken();
if (!token) {
  console.error("\nerror: no HF token — set HF_TOKEN or log in once:");
  console.error("  cd vendor/kev && uv run python -c 'from huggingface_hub import login; login()'");
  process.exit(1);
}
if (!existsSync(join(process.cwd(), "vendor", "kev", ".venv"))) {
  console.error("\nerror: vendor/kev/.venv missing — the upload runs through its huggingface_hub.");
  console.error("  node scripts/doctor.mjs --fix --all   # bootstraps it");
  process.exit(1);
}

// ── upload via vendor/kev's huggingface_hub ─────────────────────────

const PY = `
import sys
from huggingface_hub import HfApi
repo, token, files = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=token)
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
for pair in files.split("\\n"):
    if not pair:
        continue
    local, rpath = pair.split("\\t", 1)
    print(f"  uploading {rpath} ...", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=rpath,
                    repo_id=repo, repo_type="model",
                    commit_message=f"add {rpath}")
print("  upload complete", flush=True)
`;

const pairs = files.map((f) => `${join(dir, f)}\t${f}`).join("\n");
const r = spawnSync(
  "uv", ["run", "--project", "vendor/kev", "python", "-c", PY, repo, token, pairs],
  { stdio: "inherit", cwd: process.cwd() },
);
if (r.status !== 0) {
  console.error(`\nerror: upload failed (exit ${r.status}) — nothing was registered locally; fix and re-run`);
  process.exit(1);
}

// ── register: models.remote.json + regenerate the typed registry ────

let manifest = {};
if (existsSync(manifestPath)) {
  try { manifest = JSON.parse(readFileSync(manifestPath, "utf8")); } catch { /* recreate below */ }
}
manifest[name] = repo;
writeFileSync(manifestPath, JSON.stringify({
  $comment: "Alias → HuggingFace repo id for published fine-tunes. Maintained by scripts/publish-model.mjs; models:sync merges it with the local models/ scan (the published entry replaces the local one in listings/resolution).",
  ...manifest,
}, null, 2) + "\n");

console.log(`\nregistered ${name} → ${repo} in models.remote.json`);
spawnSync("node", ["scripts/sync-models.mjs"], { stdio: "inherit", cwd: process.cwd() });
console.log(`\ndone — commit models.remote.json + src/models.generated.ts with this change.`);
console.log(`consumers now get createJev({ model: "${name}" }) with auto-download.`);
