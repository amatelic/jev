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
 *   node scripts/pipeline.mjs --name events-v1         # full pipeline (~15 min)
 *
 * Flags:
 *   --name <model>   run/model dir name (default events-v1)
 *   --only a,b,c     run exactly these stages (pipeline order preserved)
 *   --skip a,b       remove stages from the plan
 *   --from <stage>   start at <stage> (e.g. --from export reuses runs/<name>)
 *   --dry-run        print the exact commands, exit 0
 *   --yes            non-interactive (skip the y/N prompt)
 *
 * Notes:
 * - train/export/quantize run via the vendored kev uv env with cwd=vendor/kev.
 * - eval forces HF_HUB_OFFLINE=1 (models are already cached; network is flaky).
 * - quantize drops model*.onnx* at models/<name>/ top level; package-kev-model.mjs
 *   needs them there, so the mv-into-onnx/ tidy-up runs right after packaging
 *   (final layout matches the verified models/events-v1).
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import readline from "node:readline/promises";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const ARM64_NVM_NODE = join(homedir(), ".nvm/versions/node/v24.21.0-arm64/bin/node");

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
    description: "write config.json + tokenizer, then tidy model*.onnx* into onnx/ (loadable layout)",
    run: (ctx) => [
      { cmd: ctx.nodeBin, args: ["scripts/package-kev-model.mjs", ctx.name], cwd: ctx.root, label: "package" },
      {
        cmd: "bash",
        args: ["-c", `mkdir -p models/${ctx.name}/onnx && mv models/${ctx.name}/model*.onnx* models/${ctx.name}/onnx/`],
        cwd: ctx.root,
        label: "move graphs into onnx/",
      },
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
                 [--from <stage>] [--dry-run] [--yes]

stages: ${STAGE_NAMES.join(", ")}   (fetch is optional and default-off)`;

function parseArgs(argv) {
  const opts = { name: "events-v1", only: null, skip: [], from: null, dryRun: false, yes: false };
  const value = (flag, v) => {
    if (v === undefined || v === "") {
      console.error(`error: ${flag} requires a value\n${USAGE}`);
      process.exit(2);
    }
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const [flag, eqVal] = a.includes("=") ? a.split(/=(.*)/, 2) : [a, undefined];
    const next = () => (eqVal !== undefined ? eqVal : value(flag, argv[++i]));
    if (flag === "--name") opts.name = next();
    else if (flag === "--only") opts.only = next().split(",").map((s) => s.trim()).filter(Boolean);
    else if (flag === "--skip") opts.skip = next().split(",").map((s) => s.trim()).filter(Boolean);
    else if (flag === "--from") opts.from = next();
    else if (flag === "--dry-run") opts.dryRun = true;
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

function resolveArm64Node() {
  if (process.arch === "arm64") return process.execPath;
  return existsSync(ARM64_NVM_NODE) ? ARM64_NVM_NODE : null;
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
  process.exit(code);
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
  const opts = parseArgs(process.argv.slice(2));
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
  if (liveChild) liveChild.kill("SIGINT");
  else process.exit(130);
});

main().catch((err) => {
  console.error(`error: ${err?.stack ?? err}`);
  process.exit(1);
});
