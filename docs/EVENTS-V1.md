# events-v1 — fine-tuned kev-0.6b for event classification

First full pass of the [TRAINING.md](TRAINING.md) optimization loop, run locally
on an M1 Max (32 GB) — no cloud, no Modal.

## Results (50-event official gate + 250-record holdout, CPU q4, this machine)

| Model | Gate (50) | Holdout (250) | Avg latency | Notes |
|---|---|---|---|---|
| open-jev (baseline) | 84.0% | — | 127 ms | verified locally |
| kev-0.6b (baseline) | 88.0% | 88.0% | 218–241 ms | verified locally |
| kev-4b (baseline) | 94.0% | — | 927 ms | verified locally (491 s one-time session init) |
| **events-v1 (this run)** | **100%** | **96.4%** | **170–240 ms** | kev-0.6b + LoRA delta |

Beats the shipped family's accuracy ceiling (kev-4b's 94%) at kev-0.6b's
latency — TRAINING.md §8's ">94% @ <1051 ms" prize.

- Outdoor F1 on holdout: 57 → **96** (the baseline's worst category — the
  fine-tune lifts the weak categories, not just the mean, per §8's checklist).
- Gate calibration: all 50 predictions in the >95% bin, actual 100%, gap **0pp**.
- Latency parity: same generation cost class as kev-0.6b.
- bench.mjs note: 7/10 — bench uses a different taxonomy (workshop/general/market
  options the fine-tune never saw) plus its known 9/10 ceiling (§8); not a regression.

## Recipe (all artifacts in this repo)

```bash
# 1. training data — 1,000 train / 250 holdout, template+entity split-separated,
#    zero overlap with train, holdout, or the official 50-event gate (checked)
node scripts/generate-training-data.mjs
node scripts/convert-to-kev.mjs train && node scripts/convert-to-kev.mjs holdout

# 2. delta LoRA fine-tune on MPS (warm start from jaredpalmer/kev-0.6b), ~13 min
cd vendor/kev   # github.com/jaredpalmer/kev, uv sync first
uv run python -m kev.train --data ../../data/kev-train.jsonl \
  --init_from jaredpalmer/kev-0.6b --lr 2e-5 --epochs 3 --batch 8 --accum 1 \
  --device mps --out runs/events-v1

# 3. merge + ONNX export (opset 17, in-graph block-causal mask + pointer head,
#    parity vs torch ≤ 1e-5)
uv run python ../../scripts/export-kev-onnx.py runs/events-v1 ../../models/events-v1

# 4. quantize q4 (head fp32!) — q4f16 auto-skipped on Cast type-inference
uv run python ../../scripts/quantize-kev-onnx.py ../../models/events-v1

# 5. package config.json + tokenizer (expects onnx/ subdir)
node scripts/package-kev-model.mjs events-v1

# 6. evaluate through the real open-jev runtime (local path load)
HF_HUB_OFFLINE=1 node scripts/eval-dataset.mjs "$(pwd)/models/events-v1"          # holdout
HF_HUB_OFFLINE=1 node scripts/eval-dataset.mjs "$(pwd)/models/events-v1" --gate   # official gate
```

## Training data design (§4)

- Template-generated per (category, subtype) with slot banks
  (artists/teams/venues/days/months/instructors), 30% Czech / 70% English.
- **Leakage controls**: holdout uses different template phrasings AND different
  entity banks (artists, teams, festivals, museums) than train; the official
  50-event gate texts are parsed from `eval.mjs` at generation time and
  excluded from both splits (19 exact-collision candidates were dropped).
- Question construction is shared with `eval.mjs` via
  `scripts/shared/question.mjs` — token-for-token identical prompts,
  options order, and descriptions.

## Export contract notes (fills the §5/§10 "UNVERIFIED" gaps)

- The wrapper `scripts/export-kev-onnx.py` reproduces the JS runtime contract:
  `(input_ids, attention_mask) → logits[B,S]`, seg/mask/positions derived
  in-graph from delimiter ids; readout = pointer(</opt>, <decide>)·scale read at
  each `<|box_end|>` position.
- `cummin`/`EyeLike` are not ONNX-runtime-implementable — replaced with an
  argmax formulation and dropped (tril already contains the diagonal).
- torch.onnx.export writes ~200 loose external tensor files; the script
  consolidates to `model.onnx` + `model.onnx_data` (the HF layout).
- Local-path loading (`OpenJev.load({ model: "/abs/path" })`) **verified
  working** (was UNVERIFIED in TRAINING.md §10).
- q4 is 901 MB (embedding stays fp32; the shipped base is 383 MB — an fp16
  embedding pass would close most of the gap; not needed for correctness).

## Fixed along the way

- `src/index.ts` native transport passed JEV-schema questions (`kind`/`prompt`)
  to open-jev unchanged → `Question #0 needs non-empty instructions`. The HTTP
  path (server.ts) already mapped them via `choice()/noul()/score()`; the native
  path now maps identically. Full suite: 38/38 passing.

## Next steps

- Ship `events-v1` to a HF repo + add as a first-class alias in `src/index.ts`
  (§7: MODEL_REPOS, MODEL_SIZES, ModelAliasSchema).
- q4f16 (WebGPU): blocked on a Cast type-inference failure in
  onnxconverter-common against the mixed int4/fp16 graph — q4 unaffected.
- Embedding → fp16 to shrink the q4 artifact toward the base's 383 MB.
- Round 2 data: disagreement mining (§4.3) using the three baselines +
  events-v1 ensemble on fresh crawl text.
