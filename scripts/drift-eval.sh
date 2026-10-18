#!/usr/bin/env bash
# drift-eval.sh — periodic local regression check for gate accuracy drift.
#
# Runs the official 50-event gate for the kev-0.6b baseline and — when
# present — the fine-tuned models/events-v1, appends each RESULT line to
# data/eval-history.log with an ISO 8601 UTC timestamp, and exits non-zero
# if accuracy drops below the thresholds below.
#
# Usage: bash scripts/drift-eval.sh

set -euo pipefail

# No PATH/arm64 handling needed here: scripts/eval-dataset.mjs self-re-execs
# through the shared ARM64 resolver (scripts/shared/arm64.mjs) when started
# under the x64-under-Rosetta shell node.

# Models are pre-cached in ~/.cache/huggingface; forbid all network access.
export HF_HUB_OFFLINE=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HISTORY="data/eval-history.log"
mkdir -p data

# Thresholds — baselines measured on this machine (docs/TRAINING-PIPELINE.md):
#   kev-0.6b gate 88.0%  → fail below 84.0 (~4pt slack for eval variance)
#   events-v1 gate 100%  → fail below 95.0 (tighter: this is the shipped model)
KEV_MIN="${KEV_MIN:-84.0}"
EVENTS_MIN="${EVENTS_MIN:-95.0}"

# run_gate <model-id-or-path> <min-acc>
# Prints one PASS/FAIL line, appends "TS<TAB>RESULT..." to the history log.
run_gate() {
  local model="$1" min="$2" out result acc verdict
  echo "── gate: $model ──"
  out="$(mktemp)"
  if ! node scripts/eval-dataset.mjs "$model" --gate 2>&1 | tee "$out"; then
    rm -f "$out"
    echo "FAIL $model (eval exited non-zero)"
    return 1
  fi
  result="$(grep -m1 '^RESULT' "$out" || true)"
  rm -f "$out"
  if [ -z "$result" ]; then
    echo "FAIL $model (no RESULT line in eval output)"
    return 1
  fi
  acc="$(printf '%s\n' "$result" | cut -f4 | cut -d= -f2)"
  verdict="$(awk -v a="$acc" -v m="$min" 'BEGIN { print (a + 0 >= m + 0) ? "PASS" : "FAIL" }')"
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$result" >> "$HISTORY"
  echo "$verdict $model gate acc=$acc (min $min)"
  [ "$verdict" = "PASS" ]
}

failures=0
run_gate "kev-0.6b" "$KEV_MIN" || failures=$((failures + 1))

if [ -d "models/events-v1/onnx" ]; then
  run_gate "$(pwd)/models/events-v1" "$EVENTS_MIN" || failures=$((failures + 1))
else
  echo "SKIP models/events-v1 (models/events-v1/onnx not present)"
fi

if [ "$failures" -gt 0 ]; then
  echo "drift-eval: $failures model(s) below threshold"
  exit 1
fi
echo "drift-eval: all gates above threshold"
