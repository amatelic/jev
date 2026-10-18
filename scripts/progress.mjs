#!/usr/bin/env node
/**
 * progress.mjs — auto-maintained model log (PROGRESS.md, repo root).
 *
 * PROGRESS.md answers "what is the newest model, what were the results,
 * what is the difference vs the previous one". Entries live in a fenced
 * ```json block inside the file — the single source of truth the markdown
 * tables are rendered from. Every write re-renders the whole file from
 * the entries, so manual edits to entries are overwritten (use --note).
 *
 * API (pure core, thin IO wrappers — importable from pipeline.mjs):
 *   parseEntries(text)                       text -> entries array (throws on corrupt json)
 *   renderMarkdown(entries)                  entries -> full file content
 *   computeDeltas(entries, index)            entry -> deltas vs previous entry with results
 *   readEntries() / startEntry(input) / finishEntry(name, results, {failed})
 *   currentShortSha()                        spawnSync git rev-parse --short HEAD
 *
 * CLI:
 *   node scripts/progress.mjs start --name X [--base B] [--hyper "..."] [--data-notes "..."] [--note "..."]
 *   node scripts/progress.mjs finish --name X --gate 100.0 --gate-ms 240 --holdout 96.4 --holdout-ms 170
 *   node scripts/progress.mjs finish --name X --failed
 *   node scripts/progress.mjs finish --name X --from-job runs/<name>/job.status
 *   node scripts/progress.mjs list
 *
 * Delta rule: each result is compared against the most recent EARLIER entry
 * that has results — preferring an earlier entry with the same name (an
 * eval re-run of the same model compares against its own last result).
 * The first entry (no earlier results) renders "—".
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const PROGRESS_PATH = join(ROOT, "PROGRESS.md");

export const STATUSES = ["training", "evaluated", "failed"];

// ─── Pure core ───────────────────────────────────────────────────────

// Tolerant: any ```json fence anywhere in the file; first one that parses
// to an array wins. Surrounding prose is ignored.
const JSON_FENCE = /```json[^\n]*\r?\n([\s\S]*?)```/g;

export function parseEntries(text) {
  const fences = [...String(text).matchAll(JSON_FENCE)];
  for (const m of fences) {
    let parsed;
    try {
      parsed = JSON.parse(m[1]);
    } catch {
      continue;
    }
    if (Array.isArray(parsed)) return parsed;
  }
  if (fences.length) {
    throw new Error("PROGRESS.md has a ```json block that is not a JSON array — fix or delete it manually");
  }
  return []; // no fence yet (fresh file)
}

const fmtDelta = (d) => `${d >= 0 ? "+" : ""}${d.toFixed(1)}`;
const entryDate = (e) => String(e.startedAt ?? "").slice(0, 10) || "?";

// For entry at `index`: the most recent EARLIER entry with results —
// same-name entries preferred (re-evals compare against their own last run).
function previousWithResults(entries, index) {
  let sameName = null;
  for (let i = index - 1; i >= 0; i--) {
    const e = entries[i];
    if (!e?.results?.length) continue;
    if (!sameName && e.name === entries[index].name) sameName = e;
    if (sameName) break; // found the newest same-name one
  }
  if (sameName) return sameName;
  for (let i = index - 1; i >= 0; i--) {
    if (entries[i]?.results?.length) return entries[i];
  }
  return null;
}

// deltas: { gate: { prev, prevAcc, dAcc, prevMs, dMs }, ... } | null
export function computeDeltas(entries, index) {
  const entry = entries[index];
  if (!entry?.results?.length) return null;
  const prev = previousWithResults(entries, index);
  if (!prev) return null;
  const deltas = {};
  for (const r of entry.results) {
    const p = prev.results.find((x) => x.dataset === r.dataset);
    if (!p) continue;
    deltas[r.dataset] = {
      prev: prev.name,
      prevAcc: p.acc,
      dAcc: Number((r.acc - p.acc).toFixed(1)),
      prevMs: p.avgMs,
      dMs: Number((r.avgMs - p.avgMs).toFixed(1)),
    };
  }
  return Object.keys(deltas).length ? deltas : null;
}

const resultCell = (r, delta) =>
  `${r.acc.toFixed(1)} (${delta ? `prev ${delta.prev} ${delta.prevAcc.toFixed(1)} → ${fmtDelta(delta.dAcc)}` : "—"}) @${r.avgMs}ms`;

function renderEntry(entry) {
  const rows = [];
  const sha = entry.gitSha ? ` · ${entry.gitSha}` : "";
  rows.push(`| ${entry.name} | ${entry.status} · ${entryDate(entry)}${sha} |`, "| --- | --- |");
  if (entry.base) rows.push(`| base | ${entry.base} |`);
  if (entry.hyper) rows.push(`| hyper | ${entry.hyper} |`);
  if (entry.dataNotes) rows.push(`| data | ${entry.dataNotes} |`);
  const deltas = entry.deltas ?? {};
  if (entry.results?.length) {
    for (const r of entry.results) rows.push(`| ${r.dataset} | ${resultCell(r, deltas[r.dataset])} |`);
  } else {
    rows.push(`| results | ${entry.status === "training" ? "pending" : "—"} |`);
  }
  if (entry.note) rows.push(`| note | ${entry.note} |`);
  return rows.join("\n");
}

export function renderMarkdown(entries) {
  const header = [
    "# Model progress log",
    "",
    "Fine-tuned model history: newest model, results, and deltas vs the previous entry with results.",
    "Auto-written by `scripts/progress.mjs` — manual edits to entries will be overwritten; use `--note`.",
    "",
    "- Deltas compare against the most recent earlier entry with results (an earlier entry with the",
    "  same name wins — eval re-runs compare against the model's own last result); first entry renders “—”.",
    "- The pipeline only writes here when its plan includes the `train` stage; `--only data`,",
    "  `--only eval`, `--dry-run`, and `status` never modify this file.",
    "- `progress.mjs finish` updates the name's latest result-less entry, or appends a fresh one",
    "  if none exists (that is how eval-only re-runs get recorded).",
    "",
    "## Entries (newest first)",
    "",
  ].join("\n");
  const body = entries.length
    ? [...entries]
        .reverse()
        .map(renderEntry)
        .join("\n\n")
    : "_no entries yet — `node scripts/progress.mjs start --name <model>`_";
  const fence = ["", "## Source data (single source of truth — the tables above render from this)", "", "```json", JSON.stringify(entries, null, 2), "```", ""].join("\n");
  return `${header}${body}\n${fence}`;
}

// ─── IO wrappers ─────────────────────────────────────────────────────

export function readEntries() {
  if (!existsSync(PROGRESS_PATH)) return [];
  return parseEntries(readFileSync(PROGRESS_PATH, "utf8"));
}

export function writeEntries(entries) {
  writeFileSync(PROGRESS_PATH, renderMarkdown(entries));
}

export function currentShortSha() {
  try {
    const r = spawnSync("git", ["rev-parse", "--short", "HEAD"], { cwd: ROOT, encoding: "utf8" });
    return r.status === 0 ? r.stdout.trim() : null;
  } catch {
    return null;
  }
}

export function startEntry(input = {}) {
  const { name, startedAt, gitSha, base, hyper, dataNotes, note } = input;
  if (!name) throw new Error("startEntry: name is required");
  const entries = readEntries();
  entries.push({
    name,
    startedAt: startedAt ?? new Date().toISOString(),
    status: "training",
    gitSha: gitSha ?? currentShortSha(),
    base: base ?? null,
    dataNotes: dataNotes ?? null,
    hyper: hyper ?? null,
    results: [],
    deltas: null,
    note: note ?? null,
  });
  writeEntries(entries);
  return entries[entries.length - 1];
}

// Fills the LAST entry with `name` lacking results (or appends a fresh one —
// how eval-only re-runs get recorded), computes deltas, rewrites the file.
export function finishEntry(name, results = [], { failed = false } = {}) {
  if (!name) throw new Error("finishEntry: name is required");
  const entries = readEntries();
  let index = -1;
  for (let i = entries.length - 1; i >= 0; i--) {
    if (entries[i].name === name && !entries[i].results?.length) {
      index = i;
      break;
    }
  }
  if (index === -1) {
    entries.push({
      name,
      startedAt: new Date().toISOString(),
      status: "training",
      gitSha: currentShortSha(),
      base: null,
      dataNotes: null,
      hyper: null,
      results: [],
      deltas: null,
      note: null,
    });
    index = entries.length - 1;
  }
  entries[index].results = results;
  entries[index].deltas = computeDeltas(entries, index);
  entries[index].status = failed ? "failed" : "evaluated";
  writeEntries(entries);
  return entries[index];
}

// ─── CLI ─────────────────────────────────────────────────────────────

const USAGE = `usage: node scripts/progress.mjs start --name X [--base B] [--hyper "..."] [--data-notes "..."] [--note "..."]
       node scripts/progress.mjs finish --name X [--gate A --gate-ms M] [--holdout A --holdout-ms M]
                                         [--from-job runs/<name>/job.status] [--failed]
       node scripts/progress.mjs list`;

function parseFlags(argv) {
  const opts = { _: [] };
  const numeric = (flag, raw) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) {
      console.error(`error: ${flag} needs a finite number`);
      process.exit(2);
    }
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const [flag, eqVal] = a.includes("=") ? a.split(/=(.*)/, 2) : [a, undefined];
    const value = () => {
      if (eqVal !== undefined) return eqVal;
      const v = argv[++i];
      if (v === undefined || v === "") {
        console.error(`error: ${flag} requires a value\n${USAGE}`);
        process.exit(2);
      }
      return v;
    };
    if (flag === "--name") opts.name = value();
    else if (flag === "--base") opts.base = value();
    else if (flag === "--hyper") opts.hyper = value();
    else if (flag === "--data-notes") opts.dataNotes = value();
    else if (flag === "--note") opts.note = value();
    else if (flag === "--gate") opts.gate = numeric(flag, value());
    else if (flag === "--gate-ms") opts.gateMs = numeric(flag, value());
    else if (flag === "--holdout") opts.holdout = numeric(flag, value());
    else if (flag === "--holdout-ms") opts.holdoutMs = numeric(flag, value());
    else if (flag === "--from-job") opts.fromJob = value();
    else if (flag === "--failed") opts.failed = true;
    else if (flag === "--help" || flag === "-h") {
      console.log(USAGE);
      process.exit(0);
    } else {
      console.error(`error: unknown flag "${a}"\n${USAGE}`);
      process.exit(2);
    }
  }
  return opts;
}

const num = (v) => (Number.isFinite(v) ? v : null);

function cliFinish(opts) {
  if (!opts.name) {
    console.error(`error: finish requires --name\n${USAGE}`);
    process.exit(2);
  }
  let results = [];
  let failed = opts.failed === true;
  if (opts.fromJob) {
    let job;
    try {
      job = JSON.parse(readFileSync(opts.fromJob, "utf8"));
    } catch (err) {
      console.error(`error: cannot read job status ${opts.fromJob} — ${err.message}`);
      process.exit(1);
    }
    failed = failed || (Number.isInteger(job.code) && job.code !== 0);
    results = (job.results ?? []).map(({ dataset, acc, avgMs }) => ({ dataset, acc, avgMs }));
  }
  for (const [acc, ms, dataset] of [
    [num(opts.gate), num(opts.gateMs), "gate"],
    [num(opts.holdout), num(opts.holdoutMs), "holdout"],
  ]) {
    if (acc !== null) results.push({ dataset, acc, avgMs: ms ?? 0 });
  }
  if (!results.length && !failed) {
    console.error(`error: finish needs results (--gate/--holdout), --failed, or --from-job\n${USAGE}`);
    process.exit(2);
  }
  const entry = finishEntry(opts.name, results, { failed });
  console.log(`progress: ${entry.name} → ${entry.status} (${entry.results.length} result(s)) — PROGRESS.md updated`);
}

function cliList() {
  const entries = readEntries();
  if (!entries.length) {
    console.log(`no entries (${PROGRESS_PATH.replace(`${ROOT}/`, "")})`);
    return;
  }
  const rows = [...entries].reverse().map((e) => {
    const res = e.results?.length
      ? e.results
          .map((r) => {
            const d = e.deltas?.[r.dataset];
            return `${r.dataset} ${r.acc.toFixed(1)}${d ? ` (${fmtDelta(d.dAcc)} vs ${d.prev})` : ""} @${r.avgMs}ms`;
          })
          .join(" · ")
      : e.status === "training"
        ? "pending"
        : "—";
    return [e.name, e.status, entryDate(e), res];
  });
  const widths = [0, 1, 2].map((i) => Math.max(...rows.map((r) => r[i].length)) + 2);
  console.log(`${"name".padEnd(widths[0])}${"status".padEnd(widths[1])}${"date".padEnd(widths[2])}results`);
  for (const r of rows) console.log(r[0].padEnd(widths[0]) + r[1].padEnd(widths[1]) + r[2].padEnd(widths[2]) + r[3]);
  console.log(`${entries.length} entr${entries.length === 1 ? "y" : "ies"} — ${PROGRESS_PATH.replace(`${ROOT}/`, "")}`);
}

function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === "start") {
    const opts = parseFlags(rest);
    if (!opts.name) {
      console.error(`error: start requires --name\n${USAGE}`);
      process.exit(2);
    }
    const entry = startEntry(opts);
    console.log(`progress: started ${entry.name} (${entry.gitSha ?? "no sha"}) — PROGRESS.md updated`);
  } else if (cmd === "finish") {
    cliFinish(parseFlags(rest));
  } else if (cmd === "list") {
    parseFlags(rest); // flag validation only
    cliList();
  } else {
    console.error(cmd ? `error: unknown command "${cmd}"\n${USAGE}` : USAGE);
    process.exit(cmd ? 2 : 0);
  }
}

// CLI when invoked directly; importable otherwise (pipeline.mjs imports the API).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
