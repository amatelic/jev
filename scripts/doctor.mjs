#!/usr/bin/env node
/**
 * doctor.mjs — one-command environment doctor + guided bootstrap.
 *
 * Answers "is this machine ready for the jev pipeline?" in a single run;
 * --fix repairs what it can.
 *
 * Usage:
 *   node scripts/doctor.mjs              # checks only (⚠ allowed, exit 0)
 *   node scripts/doctor.mjs --fix        # repair ✗ items, re-check, before/after
 *   node scripts/doctor.mjs --fix --all  # also bootstrap ⚠ heavy items
 *                                         # (HF fetches, vendor/kev clone — downloads!)
 *   node scripts/doctor.mjs --json       # machine-readable results
 *
 * Checks (registry below, mirrors pipeline.mjs STAGES): node-arch, npm-deps,
 * sharp-arch, hf-cache, vendor-kev, artifacts, disk.
 *
 * Runs correctly under the default x64-under-Rosetta shell node — that's the
 * whole point: it only diagnoses (sysctl/df/du, file presence, `uv ... import`)
 * and never loads the native ONNX runtime itself.
 */

import { spawn, spawnSync } from "node:child_process";
import { existsSync, readdirSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const VENDOR = join(ROOT, "vendor", "kev");
const HF_CACHE = process.env.HF_HUB_CACHE ?? join(homedir(), ".cache", "huggingface", "hub");
const DISK_WARN_GB = 10; // ONNX artifacts are GB-scale

const ICONS = { ok: "✅", fail: "✗", warn: "⚠", info: "ℹ" };
const pad = (s, n) => String(s).padEnd(n);
const tilde = (p) => (p && p.startsWith(homedir()) ? p.replace(homedir(), "~") : p);

// ─── Resolved node (the ARM64 build native ONNX inference needs) ─────

function newestArm64NvmNode() {
  const base = join(process.env.NVM_DIR ?? join(homedir(), ".nvm"), "versions", "node");
  let entries;
  try {
    entries = readdirSync(base);
  } catch {
    return null;
  }
  const cands = entries.filter((d) => /^v\d+\.\d+\.\d+-arm64$/.test(d)); // e.g. v24.21.0-arm64
  if (!cands.length) return null;
  const key = (d) => d.replace(/^v/, "").split(/[.-]/).map(Number).slice(0, 3);
  cands.sort((a, b) => {
    const va = key(a);
    const vb = key(b);
    for (let i = 0; i < 3; i++) if (va[i] !== vb[i]) return vb[i] - va[i];
    return 0;
  });
  const node = join(base, cands[0], "bin", "node");
  return existsSync(node) ? node : null;
}

function resolveNode() {
  if (process.arch === "arm64") return { bin: process.execPath, arch: "arm64" };
  const arm64 = newestArm64NvmNode();
  return arm64 ? { bin: arm64, arch: "arm64" } : { bin: process.execPath, arch: process.arch };
}

const RESOLVED = resolveNode(); // what pipeline/eval-gate re-exec through

// ─── Small helpers ───────────────────────────────────────────────────

const underRosetta = () => {
  const r = spawnSync("sysctl", ["-n", "sysctl.proc_translated"], { encoding: "utf8" });
  return r.status === 0 && r.stdout.trim() === "1";
};

const dirSize = (dir) => {
  const r = spawnSync("du", ["-sh", dir], { encoding: "utf8" });
  return r.status === 0 ? r.stdout.trim().split(/\s+/)[0] : "?";
};

const onPath = (cmd) => {
  const r = spawnSync(cmd, ["--version"], { encoding: "utf8" });
  return r.error ? null : (r.stdout || r.stderr).trim().split("\n")[0];
};

let liveChild = null;

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(cmd, args, { cwd: opts.cwd ?? ROOT, env: opts.env, stdio: "inherit" });
    } catch (err) {
      console.error(`   fix: cannot start "${cmd}" — ${err.message}`);
      resolve(1);
      return;
    }
    liveChild = child;
    child.on("error", (err) => {
      liveChild = null;
      console.error(`   fix: "${cmd}" failed to start — ${err.message} (is it on PATH?)`);
      resolve(127);
    });
    child.on("close", (code) => {
      liveChild = null;
      resolve(code ?? 1);
    });
  });
}

const npmInstall = () =>
  run("npm", ["install"], {
    // prepend the resolved node's bin dir so the matching-arch npm/node is used
    env: { ...process.env, PATH: `${dirname(RESOLVED.bin)}${process.env.PATH ? `:${process.env.PATH}` : ""}` },
  });

// ─── Check implementations (sync, read-only; status = { state, detail, info?, ... }) ──

function checkNodeArch() {
  const { arch, platform } = process;
  if (platform === "darwin" && arch === "x64") {
    const where = underRosetta() ? "x64 under Rosetta" : "x64 native (Intel)";
    const arm64 = newestArm64NvmNode();
    if (arm64) {
      return {
        state: "warn",
        detail: `${where} — model loading needs the ARM64 build; found ${tilde(arm64)}`,
        info: ["pipeline / eval:gate re-exec through it automatically"],
        hint: "start shells from the arm64 node, or nvm install --arch arm64 24",
      };
    }
    return {
      state: "fail",
      detail: `${where} — model loading needs the ARM64 build; none found under $NVM_DIR/versions/node/*-arm64`,
      hint: "nvm install --arch arm64 24",
    };
  }
  return { state: "ok", detail: `${arch} native on ${platform} — OK for native ONNX inference` };
}

function checkNpmDeps() {
  const nm = join(ROOT, "node_modules");
  if (!existsSync(nm)) return { state: "fail", detail: "node_modules missing" };
  const missing = ["@huggingface/transformers", "open-jev", "onnxruntime-node"]
    .filter((p) => !existsSync(join(nm, ...p.split("/"))));
  if (missing.length) return { state: "fail", detail: `missing: ${missing.join(", ")}` };

  // native binding must match the RESOLVED node arch (napi-v6/<platform>/<arch>)
  let have = "none";
  let wantExists = false;
  try {
    const binDir = join(nm, "onnxruntime-node", "bin");
    const napi = readdirSync(binDir).find((d) => d.startsWith("napi-"));
    const platDir = join(binDir, napi, process.platform);
    const archs = readdirSync(platDir);
    have = archs.join("|") || "none";
    wantExists = archs.includes(RESOLVED.arch);
  } catch {
    /* missing binding dirs entirely */
  }
  if (!wantExists) {
    return {
      state: "fail",
      detail: `onnxruntime-node binding is "${have}" but resolved node is ${RESOLVED.arch} (${tilde(RESOLVED.bin)})`,
    };
  }
  return {
    state: "ok",
    detail: `transformers + open-jev + onnxruntime-node present; ${process.platform}/${RESOLVED.arch} binding matches resolved node`,
  };
}

function checkSharpArch() {
  const sharp = join(ROOT, "node_modules", "sharp");
  if (!existsSync(sharp)) return { state: "ok", detail: "sharp not installed — nothing to check" };
  const imgDir = join(ROOT, "node_modules", "@img");
  let archs = [];
  try {
    const want = `sharp-${process.platform}-${RESOLVED.arch}`; // e.g. sharp-darwin-arm64
    archs = readdirSync(imgDir).filter((d) => /^sharp-\w+-\w+/.test(d) && !d.startsWith("sharp-libvips"));
    if (readdirSync(imgDir).some((d) => d.startsWith(want))) {
      return { state: "ok", detail: `@img/${want} matches resolved node (${RESOLVED.arch})` };
    }
  } catch {
    /* @img missing */
  }
  return {
    state: "fail",
    detail: `sharp installed but @img/sharp-${process.platform}-${RESOLVED.arch} missing (have: ${archs.join(", ") || "none"}) — wrong-arch install`,
  };
}

const HF_REPOS = [
  { repo: "onnx-community/kev-0.6b-ONNX", critical: true, note: "eval baseline" },
  { repo: "onnx-community/kev-4b-ONNX", note: "kev-4b eval" },
  { repo: "onnx-community/open-jev-deberta-v3-large-ONNX", note: "open-jev eval" },
  { repo: "jaredpalmer/kev-0.6b", note: "train stage only" },
  { repo: "Qwen/Qwen3-0.6B-Base", note: "train stage only" },
];

function hfCacheState() {
  const missing = [];
  const lines = [];
  for (const r of HF_REPOS) {
    const dir = join(HF_CACHE, `models--${r.repo.replaceAll("/", "--")}`);
    const present = existsSync(join(dir, "refs", "main")); // fetch-hf-cache.sh writes refs/main
    if (!present) missing.push(r);
    lines.push(`${present ? "✓" : "✗"} ${r.repo} ${present ? `(${dirSize(dir)}, ${r.note})` : `MISSING (${r.note}${r.critical ? ", eval-critical" : ""})`}`);
  }
  const state = missing.some((m) => m.critical) ? "fail" : missing.length ? "warn" : "ok";
  const detail = `${HF_REPOS.length - missing.length}/${HF_REPOS.length} repos cached${missing.length ? ` — missing: ${missing.map((m) => m.repo).join(", ")}` : ""}`;
  return { state, detail, info: lines, missing };
}

function checkVendorKev() {
  const uv = onPath("uv");
  if (!uv) return { state: "warn", detail: "uv not on PATH — needed for train/export/quantize", hint: "install uv (curl -LsSf https://astral.sh/uv/install.sh | sh)" };
  if (!existsSync(VENDOR)) {
    return { state: "warn", detail: "vendor/kev missing — only needed for train/export/quantize (--fix --all bootstraps)", info: [`${uv}`] };
  }
  if (!existsSync(join(VENDOR, ".venv"))) {
    return { state: "fail", detail: "vendor/kev present but .venv missing — needs uv sync", info: [`${uv}`] };
  }
  const imp = spawnSync("uv", ["run", "python", "-c", "import onnx, onnxruntime, onnxconverter_common"], { cwd: VENDOR, encoding: "utf8" });
  if (imp.status !== 0) {
    return {
      state: "fail",
      detail: `export deps not importable in vendor/kev/.venv (onnx, onnxruntime, onnxconverter-common; exit ${imp.status ?? "?"})`,
      info: [`${uv}`],
    };
  }
  return { state: "ok", detail: `vendor/kev/.venv ready — onnx · onnxruntime · onnxconverter-common import clean`, info: [`${uv}`] };
}

function checkArtifacts() {
  const missing = ["train.jsonl", "holdout.jsonl"].filter((f) => !existsSync(join(ROOT, "data", f)));
  const packaged = existsSync(join(ROOT, "models", "events-v1", "onnx"));
  return {
    state: missing.length ? "warn" : "ok",
    detail: missing.length
      ? `missing data/${missing.join(", data/")} — regenerate (~1s, deterministic, seed 42)`
      : "data/train.jsonl + data/holdout.jsonl present",
    info: [`${packaged ? "packaged model models/events-v1/onnx present" : "models/events-v1/onnx not packaged yet (info only — run the pipeline)"}`],
  };
}

function checkDisk() {
  const r = spawnSync("df", ["-h", ROOT], { encoding: "utf8" });
  if (r.status !== 0) return { state: "info", detail: "df unavailable — skipping disk check" };
  const parts = r.stdout.trim().split("\n").pop().trim().split(/\s+/);
  const avail = parts[3] ?? "?";
  const m = /^([\d.]+)([KMGT]i?)/.exec(avail);
  if (m) {
    const units = { K: 1e3, M: 1e6, G: 1e9, T: 1e12, Ki: 1024, Mi: 1024 ** 2, Gi: 1024 ** 3, Ti: 1024 ** 4 };
    const gb = (Number(m[1]) * units[m[2]]) / 1e9;
    if (gb < DISK_WARN_GB) return { state: "warn", detail: `only ${avail} free on ${ROOT} — ONNX artifacts are GB-scale` };
  }
  return { state: "ok", detail: `${avail} free on ${tilde(ROOT)}` };
}

// ─── Fix implementations (--fix; may download — never run by the checks) ──

let OPTS = { fix: false, all: false, json: false };

async function fixNpmDeps() {
  console.log("   $ npm install");
  await npmInstall();
}

async function fixSharpArch() {
  console.log("   $ rm -rf node_modules/sharp node_modules/@img && npm install");
  rmSync(join(ROOT, "node_modules", "sharp"), { recursive: true, force: true });
  rmSync(join(ROOT, "node_modules", "@img"), { recursive: true, force: true });
  await npmInstall();
}

async function fixHfCache() {
  const { missing } = hfCacheState(); // recompute at fix time
  const targets = OPTS.all ? missing : missing.filter((m) => m.critical);
  if (!targets.length) return;
  for (const t of targets) {
    console.log(`   $ bash scripts/fetch-hf-cache.sh ${t.repo}`);
    await run("bash", [join(ROOT, "scripts", "fetch-hf-cache.sh"), t.repo], { cwd: ROOT });
  }
}

async function fixVendorKev() {
  if (!existsSync(VENDOR)) {
    console.log("   $ git clone --depth 1 https://github.com/jaredpalmer/kev vendor/kev");
    const code = await run("git", ["clone", "--depth", "1", "https://github.com/jaredpalmer/kev", VENDOR], { cwd: ROOT });
    if (code !== 0) return;
  }
  console.log("   $ uv sync   (cwd: vendor/kev)");
  await run("uv", ["sync"], { cwd: VENDOR });
  console.log("   $ uv pip install onnx onnxruntime onnxconverter-common onnx-ir   (cwd: vendor/kev)");
  await run("uv", ["pip", "install", "onnx", "onnxruntime", "onnxconverter-common", "onnx-ir"], { cwd: VENDOR });
}

async function fixArtifacts() {
  console.log("   $ node scripts/pipeline.mjs --only data --yes   (deterministic, no downloads)");
  await run(RESOLVED.bin, ["scripts/pipeline.mjs", "--only", "data", "--yes"], { cwd: ROOT });
}

// ─── Check registry (like pipeline.mjs STAGES) ───────────────────────
// Each check: { id, title, check: () => status, fix?: async () => void, hint }
// status: { state: "ok"|"fail"|"warn"|"info", detail, info?: string[], hint? }

const CHECKS = [
  { id: "node-arch", title: "node arch (Rosetta)", check: checkNodeArch, hint: "nvm install --arch arm64 24" }, // no fix — can't install node for you
  { id: "npm-deps", title: "npm dependencies", check: checkNpmDeps, fix: fixNpmDeps, hint: "npm install" },
  { id: "sharp-arch", title: "sharp arch match", check: checkSharpArch, fix: fixSharpArch, hint: "rm -rf node_modules/sharp node_modules/@img && npm install" },
  { id: "hf-cache", title: "huggingface cache", check: hfCacheState, fix: fixHfCache, hint: "bash scripts/fetch-hf-cache.sh <org/repo>" },
  { id: "vendor-kev", title: "vendored kev (train/export)", check: checkVendorKev, fix: fixVendorKev, hint: "git clone --depth 1 https://github.com/jaredpalmer/kev vendor/kev && cd vendor/kev && uv sync && uv pip install onnx onnxruntime onnxconverter-common onnx-ir" },
  { id: "artifacts", title: "pipeline artifacts", check: checkArtifacts, fix: fixArtifacts, hint: "node scripts/pipeline.mjs --only data" },
  { id: "disk", title: "disk space", check: checkDisk, hint: `free up space — keep >${DISK_WARN_GB} GB (ONNX artifacts are GB-scale)` },
];

// ─── CLI ─────────────────────────────────────────────────────────────

const USAGE = `usage: node scripts/doctor.mjs [--fix] [--all] [--json]

checks: ${CHECKS.map((c) => c.id).join(", ")}
  default  checks only — exit 0 if no ✗ (⚠ allowed), exit 1 otherwise
  --fix    run fixes for ✗ items, then re-check and print before/after
  --all    with --fix: also bootstrap ⚠ items (HF fetches, vendor/kev clone — downloads!)
  --json   machine-readable results`;

function parseArgs(argv) {
  const opts = { fix: false, all: false, json: false };
  for (const a of argv) {
    if (a === "--fix") opts.fix = true;
    else if (a === "--all") opts.all = true;
    else if (a === "--json") opts.json = true;
    else if (a === "--help" || a === "-h") {
      console.log(USAGE);
      process.exit(0);
    } else {
      console.error(`error: unknown flag "${a}"\n${USAGE}`);
      process.exit(2);
    }
  }
  if (opts.all) opts.fix = true; // --all only means something together with --fix
  return opts;
}

// ─── Output ──────────────────────────────────────────────────────────

const anyFail = (results) => results.some((r) => r.state === "fail");

function runChecks() {
  return CHECKS.map((c) => {
    const status = c.check();
    return {
      id: c.id,
      title: c.title,
      hint: status.hint ?? c.hint,
      ...status, // state, detail, info?, extras (missing, ...) stripped for --json
      fix: c.fix ? true : undefined,
    };
  });
}

function printHeader() {
  const rosetta = process.platform === "darwin" && process.arch === "x64" && underRosetta() ? ", Rosetta" : "";
  console.log("doctor — is this machine ready for the jev pipeline?");
  console.log(`repo: ${ROOT}`);
  console.log(`shell node:   ${tilde(process.execPath)} (${process.arch}, ${process.platform}${rosetta})`);
  if (RESOLVED.bin !== process.execPath) {
    console.log(`resolved node: ${tilde(RESOLVED.bin)} (${RESOLVED.arch}) — used for native ONNX`);
  }
}

function printResults(results, heading) {
  if (heading) console.log(`\n──────── ${heading} ────────`);
  for (const r of results) {
    console.log(`  ${ICONS[r.state]} ${pad(r.id, 12)} ${r.detail}`);
    for (const line of r.info ?? []) console.log(`       ${line}`);
    if (r.state === "fail" || r.state === "warn") console.log(`       hint: ${r.hint}`);
  }
}

function printTransitions(before, after) {
  const changes = before
    .map((b, i) => ({ id: b.id, from: b.state, to: after[i].state, detail: after[i].detail }))
    .filter((c) => c.from !== c.to);
  console.log("\n──────── before → after ────────");
  if (!changes.length) {
    console.log("  (no state changes)");
    return;
  }
  for (const c of changes) console.log(`  ${c.id}: ${ICONS[c.from]} → ${ICONS[c.to]}  ${c.detail}`);
}

async function applyFixes(results) {
  let ran = 0;
  if (!OPTS.json) console.log("\n──────── fixes ────────");
  for (const [i, c] of CHECKS.entries()) {
    const r = results[i];
    const fixable = c.fix && (r.state === "fail" || (r.state === "warn" && OPTS.all));
    if (!fixable) continue;
    console.log(`\n→ ${r.id}: ${r.detail}`);
    await c.fix();
    ran++;
  }
  if (!ran) console.log(`  nothing fixable${OPTS.all ? "" : " (⚠ items bootstrap with --fix --all)"}`);
  return ran;
}

// ─── Main ────────────────────────────────────────────────────────────

async function main() {
  OPTS = parseArgs(process.argv.slice(2));
  if (!OPTS.json) printHeader();

  let results = runChecks();
  if (!OPTS.json) printResults(results, "checks");

  if (OPTS.fix) {
    const before = results;
    const ran = await applyFixes(results);
    if (ran) {
      results = runChecks();
      if (!OPTS.json) {
        printResults(results, "after");
        printTransitions(before, results);
      }
    }
  }

  if (OPTS.json) {
    console.log(
      JSON.stringify(
        {
          ok: !anyFail(results),
          shell: { node: process.execPath, arch: process.arch, platform: process.platform },
          resolved: RESOLVED,
          checks: results.map(({ id, title, state, detail, hint, info }) => ({ id, title, state, detail, hint, ...(info ? { info } : {}) })),
        },
        null,
        2,
      ),
    );
  } else {
    const fails = results.filter((r) => r.state === "fail").length;
    const warns = results.filter((r) => r.state === "warn").length;
    console.log(
      `\n${fails ? `✗ not ready — ${fails} failing check(s)` : warns ? `⚠ ready with warnings (${warns})` : "✅ ready — all checks green"}`,
    );
  }
  process.exit(anyFail(results) ? 1 : 0);
}

process.on("SIGINT", () => {
  if (liveChild) liveChild.kill("SIGINT");
  else process.exit(130);
});

main().catch((err) => {
  console.error(`error: ${err?.stack ?? err}`);
  process.exit(1);
});
