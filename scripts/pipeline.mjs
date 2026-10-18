#!/usr/bin/env node
/**
 * pipeline.mjs — single-command orchestrator for the events fine-tune
 * pipeline (docs/TRAINING-PIPELINE.md):
 *
 *   (fetch →) data → train → export → quantize → package → eval
 *
 * Usage:
 *   node scripts/pipeline.mjs --dry-run                # print plan, execute nothing
 *   node scripts/pipeline.mjs --only data --yes        # regenerate datasets (~1s)
 *   node scripts/pipeline.mjs --from export --yes      # reuse vendor/kev/runs/<name>
 *   node scripts/pipeline.mjs --name events-v1           # full pipeline (~15 min)
 *   node scripts/pipeline.mjs --name events-v1 --detach # fire-and-forget (implies --yes)
 *   node scripts/pipeline.mjs status                    # table of detached runs
 *
 * Flags:
 *   --name <model>   run/model dir name (default events-v1)
 *   --only a,b,c     run exactly these stages (pipeline order preserved)
 *   --skip a,b       remove stages from the plan
 *   --from <stage>   start at <stage> (e.g. --from export reuses runs/<name>)
 *   --dry-run        print the exact commands, exit 0 (overrides --detach)
 *   --yes            non-interactive (skip the y/N prompt)
 *   --detach         spawn the arm64-resolved node as a detached child running
 *                    this same pipeline (implies --yes — a detached child has
 *                    no TTY to answer the y/N prompt). Output appends to
 *                    runs/<name>/pipeline.log; the parent writes
 *                    runs/<name>/job.json {name,pid,startedAt,args} and exits 0;
 *                    the child writes runs/<name>/job.status {code,finishedAt,
 *                    results} on every exit path (finish/SIGINT/fatal error).
 *                    Re-detaching the same name is allowed: job.json is
 *                    overwritten, the log is appended, and any stale job.status
 *                    is deleted first (else `status` would lie about the new run).
 *   status           subcommand (not a stage; parsed before flag parsing):
 *                    table of runs/* from job.json + job.status — state
 *                    (running/done/failed/crashed), elapsed or exit code, log
 *                    path; always exits 0 (it is a read).
 *
 * Notes:
 * - train/export/quantize run via the vendored kev uv env with cwd=vendor/kev.
 * - eval forces HF_HUB_OFFLINE=1 (models are already cached; network is flaky).
 * - quantize drops model*.onnx* at models/<name>/ top level; package-kev-model.mjs
 *   needs them there, so the mv-into-onnx/ tidy-up runs right after packaging
 *   (final layout matches the verified models/events-v1).
 */

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, openSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";

import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import readline from "node:readline/promises";

import { currentShortSha, finishEntry, startEntry } from "./progress.mjs"; // PROGRESS.md model log (train-stage runs only)
import { resolveArm64Node } from "./shared/arm64.mjs"; // glob-newest arm64 resolver, NVM_DIR-aware

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const RUNS_DIR = join(ROOT, "runs"); // repo-root runs/ (throwaway, gitignored) — NOT vendor/kev/runs

// State for detached children (PIPELINE_DETACHED=1 is set by the --detach parent).
const DETACHED = process.env.PIPELINE_DETACHED === "1";
let detachedName = null; // set in main() so every exit path can record job.status
let interrupted = false; // set by SIGINT so finish() reports 130 in job.status
let progressEntryName = null; // set in main() when startEntry() logged this run; finish() closes it

// Verified reference numbers (docs/TRAINING-PIPELINE.md, EVENTS-V1.md).
const BASELINES = [
  { model: "open-jev", dataset: "gate", acc: 84.0 },
  { model: "kev-0.6b", dataset: "gate", acc: 88.0 },
  { model: "kev-0.6b", dataset: "holdout", acc: 88.0 },
  { model: "kev-4b", dataset: "gate", acc: 94.0 },
];
const PREVIOUS_BEST = { model: "events-v1", gate: 100.0, holdout: 96.4 };

// ─── Stage registry ──────────────────────────────────────────────────
// Each stage: { name, description, optional, run: (ctx) => step[] }
// step: { cmd, args, cwd, env (overrides), label, parseResults }

const STAGES = [
  {
    name: "fetch",
    optional: true, // default-off; enable via --only fetch or --from fetch
    description: "curl kev-0.6b-ONNX base into ~/.cache/huggingface (already cached here)",
    run: (ctx) => [
      { cmd: "bash", args: ["scripts/fetch-hf-cache.sh", "onnx-community/kev-0.6b-ONNX"], cwd: ctx.root },
    ],
  },
  {
    name: "data",
    description: "regenerate deterministic datasets (seed 42) + convert to kev JSONL",
    run: (ctx) => [
      { cmd: ctx.nodeBin, args: ["scripts/generate-training-data.mjs"], cwd: ctx.root, label: "generate" },
      { cmd: ctx.nodeBin, args: ["scripts/convert-to-kev.mjs", "train"], cwd: ctx.root, label: "convert train" },
      { cmd: ctx.nodeBin, args: ["scripts/convert-to-kev.mjs", "holdout"], cwd: ctx.root, label: "convert holdout" },
    ],
  },
  {
    name: "train",
    description: "LoRA fine-tune kev-0.6b in vendor/kev (~13 min on mps) — DO NOT run casually",
    run: (ctx) => [
      {
        cmd: "uv",
        cwd: ctx.vendor,
        args: [
          "run", "python", "-m", "kev.train",
          "--data", "../../data/kev-train.jsonl",
          "--init_from", "jaredpalmer/kev-0.6b",
          "--lr", "2e-5", "--epochs", "3", "--batch", "8", "--accum", "1",
          "--device", "mps",
          "--out", `runs/${ctx.name}`,
        ],
      },
    ],
  },
  {
    name: "export",
    description: "export runs/<name> to fp32 ONNX + parity check (vendor/kev uv env)",
    run: (ctx) => [
      {
        cmd: "uv",
        cwd: ctx.vendor,
        args: ["run", "python", "../../scripts/export-kev-onnx.py", `runs/${ctx.name}`, `../../models/${ctx.name}`],
      },
    ],
  },
  {
    name: "quantize",
    description: "int4-quantize the fp32 graph (pointer head stays fp32) (vendor/kev uv env)",
    run: (ctx) => [
      {
        cmd: "uv",
        cwd: ctx.vendor,
        args: ["run", "python", "../../scripts/quantize-kev-onnx.py", `../../models/${ctx.name}`],
      },
    ],
  },
  {
    name: "package",
    description: "write config.json + tokenizer + tidy graphs into onnx/ (loadable layout)",
    run: (ctx) => [
      { cmd: ctx.nodeBin, args: ["scripts/package-kev-model.mjs", ctx.name], cwd: ctx.root, label: "package" },
    ],
  },
  {
    name: "eval",
    description: "official 50-event gate + 250-record holdout (offline, HF_HUB_OFFLINE=1)",
    run: (ctx) => {
      const modelPath = join(ctx.root, "models", ctx.name);
      const env = { HF_HUB_OFFLINE: "1" };
      return [
        { cmd: ctx.nodeBin, args: ["scripts/eval-dataset.mjs", modelPath, "--gate"], cwd: ctx.root, env, label: "eval gate", parseResults: true },
        { cmd: ctx.nodeBin, args: ["scripts/eval-dataset.mjs", modelPath], cwd: ctx.root, env, label: "eval holdout", parseResults: true },
      ];
    },
  },
];
const STAGE_NAMES = STAGES.map((s) => s.name);

// ─── CLI ─────────────────────────────────────────────────────────────

const USAGE = `usage: node scripts/pipeline.mjs [--name <model>] [--only a,b,c] [--skip a,b]
                 [--from <stage>] [--dry-run] [--yes] [--detach]
       node scripts/pipeline.mjs status

stages: ${STAGE_NAMES.join(", ")}   (fetch is optional and default-off)`;

function parseArgs(argv) {
  const opts = { name: "events-v1", only: null, skip: [], from: null, dryRun: false, yes: false, detach: false };
  const value = (flag, v) => {
    if (v === undefined || v === "") {
      console.error(`error: ${flag} requires a value\n${USAGE}`);
      process.exit(2);
    }
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const [flag, eqVal] = a.split(/=(.*)/, 2);
    const next = () => {
      if (eqVal !== undefined) {
        if (eqVal === "") {
          console.error(`error: ${flag} requires a non-empty value\n${USAGE}`);
          process.exit(2);
        }
        return eqVal;
      }
      return value(flag, argv[++i]);
    };
    if (flag === "--name") opts.name = next();
    else if (flag === "--only") opts.only = next().split(",").map((s) => s.trim()).filter(Boolean);
    else if (flag === "--skip") opts.skip = next().split(",").map((s) => s.trim()).filter(Boolean);
    else if (flag === "--from") opts.from = next();
    else if (flag === "--dry-run") opts.dryRun = true;
    else if (flag === "--detach") opts.detach = true;
    else if (flag === "--yes" || flag === "-y") opts.yes = true;
    else if (flag === "--help" || flag === "-h") { console.log(USAGE); process.exit(0); }
    else {
      console.error(`error: unknown flag "${a}"\n${USAGE}`);
      process.exit(2);
    }
  }
  const assertStages = (names, flag) => {
    for (const n of names) {
      if (!STAGE_NAMES.includes(n)) {
        console.error(`error: unknown stage "${n}" for ${flag} — valid: ${STAGE_NAMES.join(", ")}`);
        process.exit(2);
      }
    }
  };
  if (opts.only) assertStages(opts.only, "--only");
  assertStages(opts.skip, "--skip");
  if (opts.from) assertStages([opts.from], "--from");
  return opts;
}

// ─── status subcommand ───────────────────────────────────────────────

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null; // missing, empty, or corrupt
  }
}

function pidAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err?.code === "EPERM"; // EPERM = exists, owned by someone else
  }
}

function statusCommand() {
  const jobs = [];
  if (existsSync(RUNS_DIR)) {
    for (const entry of readdirSync(RUNS_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const job = readJson(join(RUNS_DIR, entry.name, "job.json"));
      if (!job) continue;
      jobs.push({ name: job.name ?? entry.name, job, dir: join(RUNS_DIR, entry.name) });
    }
  }
  jobs.sort((a, b) => String(a.job.startedAt ?? "").localeCompare(String(b.job.startedAt ?? "")));

  const counts = { running: 0, done: 0, failed: 0, crashed: 0 };
  const rows = jobs.map(({ name, job, dir }) => {
    const status = readJson(join(dir, "job.status"));
    let state, detail;
    if (status && Number.isInteger(status.code)) {
      state = status.code === 0 ? "done" : "failed";
      detail = `exit ${status.code}`;
    } else if (Number.isInteger(job.pid) && pidAlive(job.pid)) {
      state = "running";
      const started = Date.parse(job.startedAt ?? "");
      detail = Number.isNaN(started) ? "elapsed ?" : `elapsed ${fmtDur(Date.now() - started)}`;
    } else {
      state = "crashed";
      detail = Number.isInteger(job.pid) ? `pid ${job.pid} dead, no job.status` : "invalid pid, no job.status";
    }
    counts[state]++;
    return [name, state, detail, `${relative(ROOT, dir)}/pipeline.log`];
  });

  if (!rows.length) {
    console.log(`no detached runs found (${relative(ROOT, RUNS_DIR)}/*/job.json)`);
  } else {
    const widths = [0, 1, 2].map((i) => Math.max(...rows.map((r) => r[i].length)) + 2);
    console.log(`${"name".padEnd(widths[0])}${"state".padEnd(widths[1])}${"detail".padEnd(widths[2])}log`);
    for (const r of rows) console.log(r[0].padEnd(widths[0]) + r[1].padEnd(widths[1]) + r[2].padEnd(widths[2]) + r[3]);
  }
  const crashed = counts.crashed ? `, ${counts.crashed} crashed` : "";
  console.log(`${counts.running} running, ${counts.done} done, ${counts.failed} failed${crashed}`);
  process.exit(0); // status is a read — always exit 0
}

// ─── Plan + preflight ────────────────────────────────────────────────

function buildPlan(opts) {
  let selected;
  if (opts.only) {
    selected = new Set(opts.only);
  } else if (opts.from) {
    selected = new Set(STAGE_NAMES.slice(STAGE_NAMES.indexOf(opts.from)));
  } else {
    selected = new Set(STAGES.filter((s) => !s.optional).map((s) => s.name));
  }
  for (const s of opts.skip) selected.delete(s);
  return STAGES.filter((s) => selected.has(s.name));
}

function preflight(plan, ctx) {
  const issues = [];
  const has = (n) => plan.some((s) => s.name === n);

  if ((has("train") || has("export") || has("quantize")) && !existsSync(ctx.vendor)) {
    issues.push("vendor/kev missing — clone github.com/jaredpalmer/kev (gitignored, has its own .venv)");
  }
  if (has("train") && !has("data") && !existsSync(join(ctx.root, "data", "kev-train.jsonl"))) {
    issues.push("data/kev-train.jsonl missing — run the data stage first (node scripts/pipeline.mjs --only data)");
  }
  if (has("export") && !has("train") && !existsSync(ctx.runDir)) {
    issues.push(`${ctx.runDir.replace(`${ctx.root}/`, "")} missing — --from export reuses an existing run; run the train stage first (or drop --from)`);
  }
  if (has("package") && !has("quantize") && !existsSync(join(ctx.modelDir, "model_q4.onnx"))) {
    issues.push(`models/${ctx.name}/model_q4.onnx missing — run the quantize stage before package`);
  }
  if (has("eval") && !has("package") && !existsSync(join(ctx.modelDir, "onnx"))) {
    issues.push(`models/${ctx.name}/onnx/ missing — eval needs a packaged model; run the package stage first`);
  }
  if (has("eval") && !has("data") && !existsSync(join(ctx.root, "data", "holdout.jsonl"))) {
    issues.push("data/holdout.jsonl missing — run the data stage first");
  }
  return issues;
}

// ─── Execution ───────────────────────────────────────────────────────

let liveChild = null;

// ─── Detached execution (--detach parent / PIPELINE_DETACHED=1 child) ─

function detachSpawn(opts, argv) {
  const nodeBin = resolveArm64Node() ?? process.execPath;
  const runDir = join(RUNS_DIR, opts.name);
  mkdirSync(runDir, { recursive: true });
  // a stale job.status from a previous run would make `status` lie about this one
  const statusPath = join(runDir, "job.status");
  if (existsSync(statusPath)) rmSync(statusPath);
  const logPath = join(runDir, "pipeline.log");
  const out = openSync(logPath, "a");
  const err = openSync(logPath, "a");
  const childArgs = argv.filter((a) => a !== "--detach");
  if (!childArgs.includes("--yes") && !childArgs.includes("-y")) childArgs.push("--yes"); // --detach implies --yes
  const child = spawn(nodeBin, [fileURLToPath(import.meta.url), ...childArgs], {
    cwd: ROOT,
    env: { ...process.env, PIPELINE_DETACHED: "1" },
    detached: true,
    stdio: ["ignore", out, err],
  });
  child.unref();
  try {
    writeFileSync(
      join(runDir, "job.json"),
      JSON.stringify({ name: opts.name, pid: child.pid, startedAt: new Date().toISOString(), args: childArgs }, null, 2) + "\n",
    );
  } catch (err2) {
    console.error(`error: cannot write ${relative(ROOT, runDir)}/job.json — ${err2.message}`);
    try {
      child.kill("SIGKILL"); // don't leave an untracked orphan running
    } catch { /* already gone */ }
    process.exit(1);
  }
  console.log(`detached: ${opts.name} — pid ${child.pid} — log ${relative(ROOT, logPath)} — check: node scripts/pipeline.mjs status`);
  process.exit(0);
}

function writeJobStatus(code, results) {
  if (!DETACHED || !detachedName) return;
  try {
    const runDir = join(RUNS_DIR, detachedName);
    mkdirSync(runDir, { recursive: true });
    writeFileSync(join(runDir, "job.status"), JSON.stringify({ code, finishedAt: new Date().toISOString(), results }, null, 2) + "\n");
  } catch (err) {
    console.error(`warning: could not write job.status — ${err?.message ?? err}`);
  }
}

function cmdLine(step, ctx) {
  const envPrefix = step.env ? Object.entries(step.env).map(([k, v]) => `${k}=${v}`).join(" ") + " " : "";
  return `${envPrefix}${step.cmd} ${step.args.join(" ")}`;
}

function runStep(step) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(step.cmd, step.args, { cwd: step.cwd, env: { ...process.env, ...step.env }, stdio: ["inherit", "pipe", "pipe"] });
    } catch (err) {
      console.error(`error: cannot start "${step.cmd}" — ${err.message}`);
      resolve({ code: 127, signal: null, tail: [], stdout: [] });
      return;
    }
    liveChild = child;
    const tail = [];
    const stdout = [];
    const sink = (isErr) => {
      let buf = "";
      return (chunk) => {
        const text = chunk.toString();
        process[isErr ? "stderr" : "stdout"].write(text);
        buf += text;
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line) continue;
          tail.push(isErr ? `stderr: ${line}` : line);
          if (tail.length > 2000) tail.shift();
          if (!isErr) stdout.push(line);
        }
      };
    };
    child.stdout.on("data", sink(false));
    child.stderr.on("data", sink(true));
    child.on("error", (err) => {
      liveChild = null;
      console.error(`error: "${step.cmd}" failed to start — ${err.message} (is it on PATH?)`);
      resolve({ code: 127, signal: null, tail, stdout });
    });
    child.on("close", (code, signal) => {
      liveChild = null;
      resolve({ code, signal, tail, stdout });
    });
  });
}

// RESULT<TAB>model<TAB>gate|holdout<TAB>acc=NN.N<TAB>avg_ms=N  (eval-dataset.mjs)
function parseResultLine(line, ctx) {
  if (!line.startsWith("RESULT\t")) return null;
  const [, model, dataset, acc, avgMs] = line.split("\t");
  if (!model || !dataset || !acc?.startsWith("acc=") || !avgMs?.startsWith("avg_ms=")) return null;
  const display = model.startsWith(ctx.root) ? model.slice(ctx.root.length + 1) : model;
  return { model: display, dataset, acc: Number(acc.slice(4)), avgMs: Number(avgMs.slice(7)) };
}

function fmtDur(ms) {
  const s = ms / 1000;
  if (s < 120) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m${String(Math.round(s - m * 60)).padStart(2, "0")}s`;
}

const pad = (s, n) => String(s).padEnd(n);
const fmtDelta = (d) => `${d >= 0 ? "+" : ""}${d.toFixed(1)}`;

function printSummary(results, ctx) {
  if (!results.length) return;
  console.log("\n──────── eval summary ────────");
  console.log(`${pad("model", 22)}${pad("dataset", 10)}${pad("acc", 9)}avg_ms`);
  for (const r of results) {
    console.log(`${pad(r.model, 22)}${pad(r.dataset, 10)}${pad(r.acc.toFixed(1), 9)}${r.avgMs}`);
  }
  console.log("");
  for (const r of results) {
    const parts = [];
    for (const b of BASELINES) {
      if (b.dataset === r.dataset) parts.push(`${fmtDelta(r.acc - b.acc)} vs ${b.model} ${b.acc.toFixed(1)}`);
    }
    const prev = PREVIOUS_BEST[r.dataset];
    if (ctx.name === PREVIOUS_BEST.model && prev !== undefined) {
      parts.push(`${fmtDelta(r.acc - prev)} vs previous best ${prev.toFixed(1)}`);
    }
    console.log(`  ${r.model} ${r.dataset} ${r.acc.toFixed(1)}  →  ${parts.join("  ·  ")}`);
  }
  console.log(`\nbaselines: ${BASELINES.map((b) => `${b.model} ${b.dataset} ${b.acc.toFixed(1)}`).join("  ·  ")}`);
  if (ctx.name === PREVIOUS_BEST.model) {
    console.log(`previous best (${PREVIOUS_BEST.model}): gate ${PREVIOUS_BEST.gate.toFixed(1)}  ·  holdout ${PREVIOUS_BEST.holdout.toFixed(1)}`);
  }
}

function finish(timings, results, ctx, code) {
  if (timings.length) {
    console.log("\n──────── stage wall-times ────────");
    for (const [name, ms] of timings) console.log(`  ${pad(name, 10)} ${fmtDur(ms)}`);
  }
  printSummary(results, ctx);
  writeJobStatus(interrupted ? 130 : code, results);
  if (progressEntryName) {
    try {
      finishEntry(
        progressEntryName,
        results.map(({ dataset, acc, avgMs }) => ({ dataset, acc, avgMs })),
        { failed: code !== 0 },
      );
    } catch (err) {
      console.error(`warning: PROGRESS.md update failed — ${err?.message ?? err}`); // progress must never break the pipeline
    }
  }
  process.exit(code);
}

// ─── Progress log hook (PROGRESS.md) ─────────────────────────────────

// "lr 2e-5 · 3 epochs · batch 8" from the train stage's step args.
function trainHyper(plan, ctx) {
  const stage = plan.find((s) => s.name === "train");
  if (!stage) return null;
  const args = stage.run(ctx).find((st) => st.args?.includes("--lr"))?.args ?? [];
  const val = (f) => {
    const i = args.indexOf(f);
    return i >= 0 ? args[i + 1] : null;
  };
  const parts = [];
  if (val("--lr")) parts.push(`lr ${val("--lr")}`);
  if (val("--epochs")) parts.push(`${val("--epochs")} epochs`);
  if (val("--batch")) parts.push(`batch ${val("--batch")}`);
  return parts.join(" · ") || null;
}

// ─── Output ──────────────────────────────────────────────────────────

function printPlan(plan, ctx, opts, issues) {
  const excluded = STAGES.filter((s) => !plan.some((p) => p.name === s.name)).map((s) => s.name);
  const nodeNote = resolveArm64Node() ? "" : "  ⚠ no arm64 node found (ONNX eval may fail under Rosetta)";
  console.log(`pipeline plan — name=${opts.name}`);
  console.log(`repo: ${ctx.root}   node: ${ctx.nodeBin}${nodeNote}`);
  for (const [i, stage] of plan.entries()) {
    console.log(`\n  [${i + 1}/${plan.length}] ${stage.name} — ${stage.description}`);
    for (const step of stage.run(ctx)) {
      const cwdNote = step.cwd && step.cwd !== ctx.root ? `   (cwd: ${step.cwd.replace(`${ctx.root}/`, "")})` : "";
      console.log(`      $ ${cmdLine(step, ctx)}${cwdNote}`);
    }
  }
  if (excluded.length) console.log(`\n  not selected: ${excluded.join(", ")}`);
  if (issues.length) {
    for (const issue of issues) console.error(`\npreflight: ✗ ${issue}`);
  } else {
    console.log("\npreflight: ✓ all preconditions present");
  }
}

// ─── Main ────────────────────────────────────────────────────────────

async function main() {
  const argv = process.argv.slice(2);
  if (argv[0] === "status") statusCommand(); // subcommand, not a stage — parsed before flags; exits 0
  const opts = parseArgs(argv);
  if (DETACHED) {
    detachedName = opts.name; // every child exit path below records job.status
    opts.yes = true; // a detached child has no TTY — non-interactive by definition
  }
  if (opts.detach) {
    if (opts.dryRun) {
      console.log("note: --dry-run overrides --detach — printing the plan instead");
    } else {
      detachSpawn(opts, argv); // spawns arm64-resolved node, writes job.json, exits 0
    }
  }
  const nodeBin = resolveArm64Node() ?? process.execPath;
  const ctx = {
    name: opts.name,
    root: ROOT,
    vendor: join(ROOT, "vendor", "kev"),
    nodeBin,
  };
  ctx.runDir = join(ctx.vendor, "runs", ctx.name);
  ctx.modelDir = join(ctx.root, "models", ctx.name);

  const plan = buildPlan(opts);
  if (!plan.length) {
    console.error("error: empty plan — check --only/--skip/--from");
    process.exit(2);
  }
  const issues = preflight(plan, ctx);
  printPlan(plan, ctx, opts, issues);
  if (opts.dryRun) {
    console.log("\ndry-run: nothing executed");
    return;
  }
  if (issues.length) {
    console.error("\nerror: preflight failed — fix the above, then re-run");
    process.exit(1);
  }

  const stepCount = plan.reduce((n, s) => n + s.run(ctx).length, 0);
  if (!opts.yes) {
    if (!process.stdin.isTTY) {
      console.error("\nerror: non-interactive shell — re-run with --yes");
      process.exit(2);
    }
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const answer = await rl.question(`\nrun ${plan.length} stage(s), ${stepCount} command(s)? [y/N] `);
    rl.close();
    if (!/^y(es)?$/i.test(answer.trim())) {
      console.log("aborted");
      process.exit(130);
    }
  }

  // Progress log: only runs that actually train get a PROGRESS.md entry
  // (--dry-run returned above; --only data/--only eval never reach here with
  // a "train" stage). Progress failure must never break the pipeline.
  if (plan.some((s) => s.name === "train")) {
    try {
      startEntry({
        name: ctx.name,
        gitSha: currentShortSha(),
        base: "jaredpalmer/kev-0.6b",
        hyper: trainHyper(plan, ctx),
        dataNotes: "seed 42 · kev-train.jsonl",
      });
      progressEntryName = ctx.name;
      console.log(`progress: logged training start for ${ctx.name} (PROGRESS.md)`);
    } catch (err) {
      console.error(`warning: PROGRESS.md start failed — ${err?.message ?? err}`);
    }
  }

  const timings = [];
  const results = [];
  for (const [i, stage] of plan.entries()) {
    console.log(`\n── [${i + 1}/${plan.length}] ${stage.name}: ${stage.description}`);
    const t0 = performance.now();
    for (const step of stage.run(ctx)) {
      if (step.label) console.log(`   $ ${cmdLine(step, ctx)}`);
      const r = await runStep(step);
      if (r.code !== 0 || r.signal) {
        console.error(`\n✗ stage "${stage.name}"${step.label ? ` (${step.label})` : ""} failed (${r.signal ? `signal ${r.signal}` : `exit ${r.code}`})`);
        const tail = r.tail.slice(-15);
        if (tail.length) console.error(`last ${tail.length} output line(s):\n  ${tail.join("\n  ")}`);
        finish(timings, results, ctx, r.code ?? 1);
      }
      if (step.parseResults) {
        for (const line of r.stdout) {
          const parsed = parseResultLine(line, ctx);
          if (parsed) results.push(parsed);
        }
      }
    }
    const dt = performance.now() - t0;
    timings.push([stage.name, dt]);
    console.log(`✓ ${stage.name} (${fmtDur(dt)})`);
  }
  finish(timings, results, ctx, 0);
}

process.on("SIGINT", () => {
  interrupted = true;
  if (liveChild) liveChild.kill("SIGINT");
  else {
    writeJobStatus(130, []);
    process.exit(130);
  }
});

main().catch((err) => {
  console.error(`error: ${err?.stack ?? err}`);
  writeJobStatus(1, []);
  process.exit(1);
});
