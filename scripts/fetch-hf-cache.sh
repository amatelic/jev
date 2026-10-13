#!/usr/bin/env bash
# Deterministic HF cache builder — downloads a model repo into the exact
# ~/.cache/huggingface/hub layout transformers.js expects, using curl
# (resumable, with timeouts). Workaround for hanging fetches in
# transformers.js on flaky networks.
#
# Usage: bash scripts/fetch-hf-cache.sh <org/repo> [...more repos]
set -uo pipefail

CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
REPO_BASE="https://huggingface.co"

fetch_repo() {
  local repo="$1"
  local safe="${repo//\//--}"                    # onnx-community/kev-0.6b-ONNX -> models--onnx-community--kev-0.6b-ONNX
  safe="models--${safe}"
  local dir="$CACHE/$safe"

  echo "==> $repo"

  # Commit sha (pin the snapshot so refs/snapshots stay consistent)
  local sha
  sha=$(curl -sS --max-time 30 "$REPO_BASE/api/models/$repo" | sed -n 's/.*"sha":"\([0-9a-f]\{40\}\)".*/\1/p' | head -1)
  if [[ -z "$sha" ]]; then
    echo "    FAILED to resolve sha for $repo" >&2
    return 1
  fi
  echo "    sha: $sha"

  mkdir -p "$dir/refs" "$dir/snapshots/$sha"
  echo -n "$sha" > "$dir/refs/main"

  # File list from the API
  local files
  files=$(curl -sS --max-time 30 "$REPO_BASE/api/models/$repo" \
    | tr ',' '\n' | sed -n 's/.*"rfilename":"\([^"]*\)".*/\1/p')

  local snap="$dir/snapshots/$sha"
  local failures=0
  for f in $files; do
    [[ "$f" == .gitattributes || "$f" == README.md ]] && continue
    local dest="$snap/$f"
    mkdir -p "$(dirname "$dest")"

    # Complete-file marker: dest exists only after a fully successful download
    if [[ -f "$dest" ]]; then
      echo "    ✓ $f (cached)"
      continue
    fi

    echo "    ↓ $f"
    if curl -sS --fail --retry 8 --retry-delay 15 --retry-all-errors \
         --max-time 7200 -C - -L \
         -o "$dest.part" "$REPO_BASE/$repo/resolve/main/$f"; then
      mv "$dest.part" "$dest"
    else
      echo "    FAILED: $f" >&2
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "    $repo: $failures file(s) failed" >&2
    return 1
  fi
  echo "    done: $repo → $snap"
}

rc=0
for repo in "$@"; do
  fetch_repo "$repo" || rc=1
done
exit $rc
