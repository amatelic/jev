/**
 * sandbox runner — one-command throwaway jev playground.
 *
 * templates/sandbox/ is the committed pristine default; sandbox/ is a
 * gitignored working copy (like .env.example → .env). First run copies
 * the template into sandbox/ and executes sandbox/play.mjs under an
 * arm64 node (shared/arm64.mjs resolver) with HF_HUB_OFFLINE=1 unless
 * already set. Edits to sandbox/ are never committed; --reset restores
 * the pristine copy.
 *
 * Usage:
 *   npm run sandbox            # create-if-missing, then run
 *   npm run sandbox -- --flag  # extra args forwarded to sandbox/play.mjs
 *   npm run sandbox:reset      # rm -rf sandbox/ + fresh template copy
 */

import { spawn } from "node:child_process";
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolveArm64Node } from "./shared/arm64.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const template = join(root, "templates", "sandbox");
const work = join(root, "sandbox");

if (!existsSync(template)) {
  console.error("error: templates/sandbox/ is missing — the checkout is incomplete");
  process.exit(1);
}

if (process.argv[2] === "--reset") {
  rmSync(work, { recursive: true, force: true });
  cpSync(template, work, { recursive: true });
  console.log("sandbox: created from templates/sandbox (edits are not saved — reset with npm run sandbox:reset)");
  process.exit(0);
}

if (!existsSync(work)) {
  cpSync(template, work, { recursive: true });
  console.log("sandbox: created from templates/sandbox (edits are not saved — reset with npm run sandbox:reset)");
}

const env = { ...process.env };
if (env.HF_HUB_OFFLINE === undefined) env.HF_HUB_OFFLINE = "1";

const child = spawn(resolveArm64Node() ?? process.execPath, [join(work, "play.mjs"), ...process.argv.slice(2)], {
  cwd: root,
  env,
  stdio: "inherit",
});
child.on("error", (err) => {
  console.error(`error: failed to start sandbox — ${err.message}`);
  process.exit(1);
});
child.on("close", (code, signal) => process.exit(signal ? 1 : (code ?? 1)));
