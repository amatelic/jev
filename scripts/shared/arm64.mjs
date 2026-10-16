/**
 * arm64 re-exec helpers.
 *
 * The default shell node on this machine is x64-under-Rosetta and cannot
 * load the ARM64 onnxruntime-node native binaries. Model-loading scripts
 * call reexecIfWrongArch() as their first statement so they self-correct
 * instead of requiring a manual PATH export.
 *
 * Node >=20 built-ins only. No dependencies.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

function versionNumbers(name) {
  const m = /^v(\d+)\.(\d+)\.(\d+)/.exec(name);
  return m ? [+m[1], +m[2], +m[3]] : [0, 0, 0];
}

/**
 * Resolve a native arm64 node executable:
 *   - process.execPath when already running arm64,
 *   - else the newest ~/.nvm/versions/node/*-arm64/bin/node,
 *   - else null (no arm64 toolchain installed).
 * Pure lookup — no spawning, no env changes.
 */
export function resolveArm64Node() {
  if (process.arch === "arm64") return process.execPath;
  const versionsDir = join(homedir(), ".nvm", "versions", "node");
  let dirs;
  try {
    dirs = readdirSync(versionsDir);
  } catch {
    return null;
  }
  const newestFirst = dirs
    .filter((d) => d.endsWith("-arm64"))
    .sort((a, b) => {
      const [ma, mb] = [versionNumbers(a), versionNumbers(b)];
      for (let i = 0; i < 3; i += 1) {
        if (ma[i] !== mb[i]) return mb[i] - ma[i];
      }
      return 0;
    });
  for (const dir of newestFirst) {
    const bin = join(versionsDir, dir, "bin", "node");
    if (existsSync(bin)) return bin;
  }
  return null;
}

/**
 * Re-exec the current script under an arm64 node when running x64 (or
 * other arch) on darwin and an arm64 toolchain is available. Prints one
 * notice line, runs the child synchronously with inherited stdio and the
 * same argv, then exits with the child's code. When no arm64 binary is
 * found, returns quietly — the caller proceeds and the eventual model
 * load error explains what is missing.
 */
export function reexecIfWrongArch() {
  if (process.platform !== "darwin" || process.arch === "arm64") return;
  if (!process.argv[1]) return;
  const bin = resolveArm64Node();
  if (!bin) return;
  console.log(`re-exec under arm64 node: ${bin}`);
  const child = spawnSync(bin, [resolve(process.argv[1]), ...process.argv.slice(2)], {
    stdio: "inherit",
  });
  if (child.error) {
    console.error(`re-exec failed: ${child.error.message}`);
    process.exit(1);
  }
  process.exit(child.status ?? 1);
}
