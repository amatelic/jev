# Training Pipeline README

End-to-end pipeline for fine-tuning, exporting, packaging, and evaluating a
custom JEV decision model — the practical realization of
[TRAINING.md](../TRAINING.md), verified on an M1 Max with no cloud GPUs.

- Run report with measured numbers: [EVENTS-V1.md](EVENTS-V1.md)
- Upstream trainer (vendored, gitignored): [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)

## What this produces

A directory (`models/<name>/`) that the standard `open-jev` runtime loads
directly — same contract as the shipped `kev-0.6b` / `kev-4b` models:

```
models/events-v1/
├── config.json              # "kev" section + transformers.js_config
├── tokenizer.json           # full Qwen tokenizer set (copied from base)
├── vocab.json  merges.txt  added_tokens.json  special_tokens_map.json  tokenizer_config.json
└── onnx/
    ├── model.onnx           # fp32 export (parity-checked, ~2.3 GB external data)
    ├── model.onnx_data
    └── model_q4.onnx        # int4-quantized — what Node CPU inference loads
```

Verified outcome of the first run (this machine, CPU q4):

| Model | Official gate (50) | Generated holdout (250) | Avg latency |
|---|---|---|---|
| open-jev | 84.0% | — | 127 ms |
| kev-0.6b | 88.0% | 88.0% | 218–241 ms |
| kev-4b | 94.0% | — | 927 ms |
| **events-v1 (fine-tuned)** | **100%** | **96.4%** | 170–240 ms |

## How the model works (30-second version)

kev is **not** a text generator. One forward pass over a packed sequence →
one score per option → softmax within each question:

```
<|fim_prefix|> Event: KoRn heavy metal live show          ← state (segment 0)
<|fim_middle|> What type of event is this?                ← question branch (segment k)
<|box_start|> concert: Live music... <|box_end|>          ← option 1
<|box_start|> sports: Sports match... <|box_end|>         ← option 2
<|fim_suffix|>                                            ← the "decide" token
```

- A **block-causal mask** lets each branch attend to the state + itself only.
- The **pointer head** scores options: `logit(opt) = k(h[</opt>]) · q(h[<decide>]) × 0.0625`.
- Training = **cross-entropy over the option softmax** against the gold label.
- Trainable parts: **LoRA adapters** (r=16, ~10.6M params) on a frozen
  Qwen3-0.6B backbone + the **pointer head** (2 MB) — warm-started from
  `jaredpalmer/kev-0.6b`, so 1,000 records and ~13 minutes on a laptop suffice.

## Prerequisites

```bash
# ARM64 Node (native ONNX inference; x64-under-Rosetta needs the ARM64 build)
nvm install --arch arm64 24   # or use any installed <version>-arm64

# Python env for the trainer (uv — never global pip)
git clone --depth 1 https://github.com/jaredpalmer/kev vendor/kev
cd vendor/kev && uv sync && uv pip install onnx onnxruntime onnxconverter-common onnx-ir
```

## Pipeline, stage by stage

### Stage 0 — cache models (network-resilient)

`transformers.js` fetches can hang or 429 on flaky networks. This fetches via
curl — resumable, exact HF cache layout:

```bash
bash scripts/fetch-hf-cache.sh onnx-community/kev-0.6b-ONNX \
                               onnx-community/open-jev-deberta-v3-large-ONNX \
                               onnx-community/kev-4b-ONNX
```

Then run everything with `HF_HUB_OFFLINE=1` to force cache-only reads.

### Stage 1 — generate training data

```bash
node scripts/generate-training-data.mjs            # 1,250 records (seeded, reproducible)
node scripts/convert-to-kev.mjs train              # → data/kev-train.jsonl
node scripts/convert-to-kev.mjs holdout            # → data/kev-holdout.jsonl
```

Design rules baked into the generator:

| Rule | Why |
|---|---|
| Question built from `scripts/shared/question.mjs` | token-for-token identity with `eval.mjs` prompts — training must match serving exactly |
| Holdout uses **different template phrasings AND different entity banks** | no template/entity leakage between splits |
| Official 50-event gate texts parsed from `eval.mjs`, excluded from both splits | never train on (or hold out on) the acceptance gate — 19 exact collisions were caught and dropped in the first run |
| ~30% Czech / 70% English states | the baselines' known weakness (multilingual titles) |
| Stratified 7 categories, ≥5/category per split | per-category metrics stay meaningful |

### Stage 2 — fine-tune (LoRA delta, ~13 min on M1 Max)

```bash
cd vendor/kev
uv run python -m kev.train \
  --data ../../data/kev-train.jsonl \
  --init_from jaredpalmer/kev-0.6b \    # warm start: released LoRA + pointer head
  --lr 2e-5 \                           # small: this is a delta tune
  --epochs 3 --batch 8 --accum 1 \
  --device mps \                        # mps | cuda | cpu
  --out runs/events-v1
```

Output: `runs/events-v1/` (`adapter_model.safetensors` + `head.pt` + provenance).
`--init_from` matters: the head already knows the task shape from 12.5k upstream
records, so loss starts ~0.24 instead of ~2.3 and 1,000 domain records are
enough (the guide's floor: <400 records = noise).

### Stage 3 — merge + export to ONNX

```bash
cd vendor/kev
uv run python ../../scripts/export-kev-onnx.py runs/events-v1 ../../models/events-v1
```

This fills TRAINING.md §5's "gap you must fill" (no published tool does it):

1. Loads the run (`Checkpoint.load`) and **merges the LoRA** into the backbone
2. Wraps it as `(input_ids, attention_mask) → logits [B,S]`, deriving
   **seg / block-causal mask / positions in-graph from the delimiter ids** —
   the exact contract the JS runtime expects
3. Applies the pointer head; values are read at each `<|box_end|>` position
4. `torch.onnx.export` (opset 17, dynamic batch/sequence axes)
5. **Parity check** vs the torch model on real records (first run: max |Δlogit| = 1e-5)
6. Consolidates torch's ~200 loose external tensor files into
   `model.onnx` + `model.onnx_data` (the HF layout)

Known ONNX-op traps handled: `cummin` (→ argmax formulation for
"first `<|fim_suffix|>` at index ≥ i") and `EyeLike` (→ dropped; `tril` already
contains the diagonal and the JS runtime never pads single sequences).

### Stage 4 — quantize to q4

```bash
cd vendor/kev
uv run python ../../scripts/quantize-kev-onnx.py ../../models/events-v1
```

- int4 MatMul (block 32, symmetric, accuracy_level 4) — what Node CPU loads
- **Pointer head MatMuls excluded, stay fp32** (quantizing the head collapses
  calibration — TRAINING.md §5/§9)
- q4f16 (WebGPU) attempted with int4-first ordering; auto-skipped if the mixed
  int4/fp16 graph fails ORT verification (q4 is unaffected)

### Stage 5 — package

```bash
mkdir -p models/events-v1/onnx && mv models/events-v1/model*.onnx* models/events-v1/onnx/
node scripts/package-kev-model.mjs events-v1
```

Copies the full tokenizer set from the cached base snapshot and writes
`config.json` with a proper `kev` section (delimiters, `trained_max_*`,
`transformers.js_config.dtype`). Local-path loading
(`OpenJev.load({ model: "/abs/path" })`) is verified working.

### Stage 6 — evaluate through the real runtime

```bash
export HF_HUB_OFFLINE=1
node scripts/eval-dataset.mjs "$(pwd)/models/events-v1"          # generated holdout (250)
node scripts/eval-dataset.mjs "$(pwd)/models/events-v1" --gate   # official 50-event gate
node scripts/eval-dataset.mjs kev-0.6b --gate                    # any baseline for comparison
```

Reports accuracy, per-category P/R/F1, confidence calibration bins, latency,
and a machine-readable `RESULT` line. Reads the gate directly from `eval.mjs`
so it can never drift from the shipped acceptance set.

## Script index

| Script | Purpose |
|---|---|
| `scripts/fetch-hf-cache.sh` | curl-based resumable HF cache builder (network-resilient) |
| `scripts/shared/question.mjs` | single source of truth for the category question |
| `scripts/generate-training-data.mjs` | seeded synthetic data + leakage/gate guards |
| `scripts/convert-to-kev.mjs` | jev JSONL → kev trainer JSONL (criteria order = option order!) |
| `scripts/download-model.mjs` | pre-download via open-jev with retry/backoff |
| `scripts/export-kev-onnx.py` | merge → wrap → export → parity → consolidate |
| `scripts/quantize-kev-onnx.py` | q4 (+ best-effort q4f16) with fp32 head |
| `scripts/package-kev-model.mjs` | config.json + tokenizer into a loadable dir |
| `scripts/eval-dataset.mjs` | dataset-driven eval (holdout/gate, any model id/path) |

## One-shot recap (copy-paste)

```bash
export PATH="$HOME/.nvm/versions/node/<ver>-arm64/bin:$PATH"   # ARM64 node
bash scripts/fetch-hf-cache.sh onnx-community/kev-0.6b-ONNX

node scripts/generate-training-data.mjs
node scripts/convert-to-kev.mjs train && node scripts/convert-to-kev.mjs holdout

cd vendor/kev
uv run python -m kev.train --data ../../data/kev-train.jsonl \
  --init_from jaredpalmer/kev-0.6b --lr 2e-5 --epochs 3 --batch 8 --accum 1 \
  --device mps --out runs/events-v1
uv run python ../../scripts/export-kev-onnx.py runs/events-v1 ../../models/events-v1
uv run python ../../scripts/quantize-kev-onnx.py ../../models/events-v1
cd ..
mkdir -p models/events-v1/onnx && mv models/events-v1/model*.onnx* models/events-v1/onnx/
node scripts/package-kev-model.mjs events-v1

HF_HUB_OFFLINE=1 node scripts/eval-dataset.mjs "$(pwd)/models/events-v1" --gate
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Node fetch hangs/429s on model download | use `scripts/fetch-hf-cache.sh` + `HF_HUB_OFFLINE=1` |
| `sharp` module wrong-arch error | reinstall under ARM64 node: `rm -rf node_modules/sharp node_modules/@img && npm install` |
| Export fails on `cummin` / `EyeLike` | already handled in `export-kev-onnx.py`; keep replacements if re-editing |
| q4f16 ORT type error on `Cast` | known onnxconverter-common issue with mixed int4/fp16 graphs; q4 is the CPU path — safe to skip |
| Accuracy fine but confidences wrong | refit temperature on validation, re-export with `--temperature T` |
| Bench (`bench.mjs`) score looks low | bench uses a different taxonomy (workshop/general/market) with a 9/10 ceiling by construction — judge on `eval-dataset.mjs` instead |

## Repo hygiene notes

- `vendor/`, `models/`, `*.onnx*` are gitignored; `data/*.jsonl` (the seeded,
  reproducible datasets) and all `scripts/` are committed.
- The run's provenance (args, base, init checkpoint) is recorded by the trainer
  in `vendor/kev/runs/<name>/` — keep it with any shipped model.
