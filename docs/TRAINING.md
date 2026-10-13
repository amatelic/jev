# Training your own JEV models

Practical guide to fine-tuning and packaging a custom decision model that loads in the `@amatelic/jev` / `open-jev` runtime. Baselines to beat (50-event eval set, ARM64 Mac CPU, `dtype: "q4"`):

| Model | Accuracy | Latency | Notes |
|---|---|---|---|
| `kev-0.6b` | 88% | 164 ms | fast default |
| `kev-4b` | 94% | 1051 ms | accuracy ceiling of the shipped family |
| `open-jev` | 84% | 136 ms | lightest, 256-token state cap |

TL;DR path: generate ~1,000 labeled records (§4) → LoRA fine-tune kev via the upstream `kev-finetune` skill (§5) → merge + ONNX export + q4 quantize (§5) → craft `config.json` (§7) → eval against the baselines (§8).

## 1. When to train your own

| Situation | Do this |
|---|---|
| Baseline meets target accuracy | Ship it; tune prompts/`descriptions` first |
| Off by a few points on one or two categories | Prompt engineering — sharpen `descriptions`, reword state text; re-run `eval.mjs` |
| Domain vocabulary the baselines misread (e.g. multilingual event titles) | Fine-tune `kev-0.6b` on ≥1,000 records |
| Need to beat kev-4b's 94% | Fine-tune kev-4b (`--init-from jaredpalmer/kev-4b`) on ≥1,000 domain records |
| Need a different question/answer contract | Don't — you'd be forking the runtime, not training a model |
| Have <400 labeled records | Label more first (§4). Upstream measurement: 400 records = inside noise; ~1,000 records = +5pt detectable |

The 50-event eval set is a **gate**, not a training set.

## 2. How the runtime loads models — the `config.json` contract

`OpenJev.load({ model, device, dtype })` → `AutoConfig.from_pretrained(modelId)` → **`detectFamily`**:

- `config.kev` is an object → kev adapter (checked **first**, wins if both exist)
- else `config.open_jev` is an object → open-jev adapter
- else it throws: `Unsupported model: config.json has neither an "open_jev" nor a "kev" section. Use one of open-jev, kev-0.6b, kev-4b or a compatible ONNX conversion.`

This is exactly why `autotrust/JEV-9B` was rejected: no ONNX export and no `open_jev`/`kev` section — and even patching the config wouldn't fix the graph shape (§9).

Keys the runtime actually reads (defaults in parentheses):

| Key | `kev` section | `open_jev` section |
|---|---|---|
| `temperature` | not read (hardcoded `1`) | `1.05` |
| `max_state_tokens` | `8192` | `256` |
| length cap | `max_branch_tokens` → `8192` (state + one branch) | `max_len` → `512` (whole sequence) |
| `max_options` | `255` (choice min 1, score min 2) | not read (hardcoded: choice 2–255, score 2–10) |
| delimiters | `delimiter_ids` (numbers, takes precedence), then `delimiters` (strings); defaults `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|box_start\|>`, `<\|box_end\|>`, `<\|fim_suffix\|>` | n/a |

The sections also carry graph-contract metadata read by humans, not the runtime (`head_dim`, `scale`, `lora`, `trained_max_state_tokens`, `trained_max_branch_tokens`, `inputs`, `output`, `pool`, `markers`, `noul_options`). Keep it accurate — it documents what your ONNX graph must implement.

**ModelId resolution**: `resolveModelId` is `MODELS[alias] ?? alias`. Only `open-jev`, `kev-0.6b`, `kev-4b` are aliases; any other string is passed **verbatim** to transformers.js, so any HF repo id **or local path** works: `OpenJev.load({ model: "/abs/path/my-kev" })`. Caveat: `dtype: "auto"` resolves to `q4` off-WebGPU — a custom fp32/fp16-only repo must pass `dtype` explicitly or the download 404s.

**Required repo/file layout** (verified against the live HF trees):

```
my-kev/
  config.json                 # MUST contain a "kev" (or "open_jev") object section
  tokenizer.json  tokenizer_config.json  special_tokens_map.json  added_tokens.json
  vocab.json  merges.txt      # Qwen (kev family); DeBERTa adds spm.model + open_jev_config.json
  onnx/
    model_q4.onnx             # mandatory — Node device:"cpu" auto-selects q4
    model_q4f16.onnx          # for WebGPU
    # model.onnx / model_fp16.onnx optional
```

`config.json → transformers.js_config` declares the default `dtype` (kev ships `q4f16`, deberta ships `fp16`) and `use_external_data_format` per file — mandatory for >2 GB models (kev-4b q4 is 2.4–2.5 GB; weights live in a sibling `.onnx_data` file).

## 3. What the model must learn — encoding & decoding

All text is tokenized with `add_special_tokens: false`. `choice` with `descriptions` renders each option as `name: description`. `noul` options are fixed `["no","yes"]` — hard-coded pairing; don't retrain with different noul tokens without forking the runtime.

**kev (decoder-only, pointer head)** — `encodeKevSequence`:

```
<|fim_prefix|> state
<|fim_middle|> instr <|box_start|> opt₁ <|box_end|> <|box_start|> opt₂ <|box_end|> …
<|fim_suffix|>
```

Graph inputs: `input_ids`, `attention_mask` only — the ONNX graph derives segments/positions/**block-causal mask in-graph from the delimiter ids** (each question branch attends only to state + itself). Output: `logits (batch, seq)` — one scalar per token; the runtime reads the value at each option's `</opt>` position and softmaxes within the question. The head is a pointer head: `</opt>` hidden state ⋅ its question's `<decide>` hidden state (`head_dim` 256, `scale` 0.0625). Caller text is escaped before tokenizing — `<|name|>` → `<¦name¦>` — so delimiters are unforgeable.

**open-jev (encoder, span pooling)** — `encodeSequence`:

```
[CLS] [STATE] state [Q] instr₁ [OPT] opt₁ [OPT] opt₂ … [Q] instr₂ [OPT] … [SEP]
```

Graph inputs add int64 tensors `seg` (span slot per token), `pair_q`, `pair_opt` (slot ids per question/option pair). Output: `logits (batch, pairs)`. Head: 3-layer MLP over `[mean(q tokens); mean(opt tokens); product]`. Markers `[STATE]`/`[Q]`/`[OPT]` are real added tokens (ids 128001–128003 in `added_tokens.json`) — keep those ids. No escaping.

**Decoding** (`decodeAnswer`; numerically stable temperature-scaled softmax, `exp((x−max)/T)`):

- `choice` → argmax = `choice`, `confidence = p[argmax]`, full `probabilities` record
- `score` → expected value `Σ pᵢ·i`, `normalized = score/(n−1)`, `level = levels[round(score)]`, `confidence = max p`
- `noul` → `probability = p["yes"=index 1]`, `answer = probability ≥ 0.5`, `confidence = max(p, 1−p)`

So what training must teach: **one logit per (question, option) pair whose within-question softmax matches the gold distribution.** Cross-entropy on the option distribution is exactly the upstream loss for both families.

Token budgets (throw on overflow): open-jev state budget = `max_len − (3 + Σ(1+instrLen) + Σ(1+optLen))` → `Questions need N tokens which exceeds the M token context`; kev = `max_branch_tokens − 1 − longestBranch` → `A question needs N tokens…`.

## 4. Preparing training data

One JSONL record per labeled item, reusing the **exact** question construction from `eval.mjs` — training and inference prompts must match token-for-token, `descriptions` included:

```json
{"state":"Event: Deep Purple concert at Budapest Arena",
 "question":{"id":"category","kind":"choice","prompt":"What type of event is this?",
   "options":["concert","sports","exhibition","networking","dining","outdoor","wellness"],
   "descriptions":{"concert":"Live music performance, band, DJ, tour, festival"}},
 "expected":{"choice":"concert","probabilities":{"concert":1.0}},
 "meta":{"cat":"concert","sub":"rock","source":"knowledge-crawler","split":"train","id":"evt-017"}}
```

- `state` ≤ 8000 chars (TypeBox cap in `DecideRequestSchema`) and ≤ 256 tokens if you'll eval on `open-jev`.
- `options` 2–50 items (runtime caps: kev choice 1–255, open-jev choice 2–255, open-jev score 2–10 levels).
- For `noul`/`score` targets, `expected` carries `answer` / `level` instead.
- Never put delimiter text (`<|…|>`) in state/options: the kev runtime escapes it at inference (`<¦…¦>`), so generate training text through the same escape.

**Split rules**: stratified holdout ~20%, ≥5 per category, tagged `split:"holdout"` in `meta`, never trained on. Keep `meta.sub` so a subtype question can be added later.

**Sourcing** (cheapest → highest value per label):

1. **knowledge-crawler corpus** (`packages/knowledge-crawler`): extracted events already carry titles/categories — highest volume; map crawler categories → your taxonomy (with a `general` fallback as in `bench.mjs`), then human-spot-check a stratified sample. The production `events` service is a second corpus.
2. **Synthetic augmentation**: the 50-item eval set is itself template-generated (city/entity swapping Prague↔Valencia↔Budapest, Czech/English mixing, genre substitution within a category's subtype vocabulary, venue-suffix patterns like "at O2 arena"). Parameterize `{city, venue, artist/team, sub}` slots; label = `cat`. The kev repo's `generate_data.py` does LLM labeling at ~$0.25/1k records.
3. **Disagreement mining**: label items where kev-0.6b / kev-4b / open-jev disagree or the 3-model ensemble is wrong, plus confidence < 0.7 and calibration-gap outliers.

## 5. Path A — fine-tune the kev family (LoRA on Qwen3-Base)

Upstream repo: **github.com/jaredpalmer/kev** (Apache-2.0; `kev/train.py`, full provenance per checkpoint in `training_config.json` / `train.log` / `result.json`). Released checkpoints ship as unmerged adapter + `head.pt` and are servable via the repo itself (`python -m kev.serve --run jaredpalmer/kev-0.6b --port 8008`) — a separate serving path from this ONNX runtime.

Verified upstream recipe (kev-0.6b): LoRA r=16, α=32, dropout 0.05, targets all 7 projections (`q,k,v,o,gate,up,down_proj`), peft `task_type: FEATURE_EXTRACTION`, plus the pointer head trained from scratch. 12,576 records (10 public classification sets ×1000, policy minimal-pairs, synthetic rule-structure records), 2 epochs, lr 1e-4, batch 8, bf16, ~12 min on 1×H100.

Recommended: the kev-finetune skill — Modal-based, continues from a released checkpoint:

```bash
npx skills add jaredpalmer/kev@kev-finetune
# repo scripts: extract_workload.py  convert_data.py  generate_data.py
#               plan_size.py  split_data.py  kev_modal.py
modal run scripts/kev_modal.py::train --init-from jaredpalmer/kev-4b --data my-events.jsonl
```

Measured upstream: ~1,000 records → ~15 min, ~$1 on H100 (4B).

### Merge + ONNX export (the gap you must fill)

`optimum-cli export onnx --task text-generation` **cannot** produce this graph — the pointer head and in-graph block-causal mask are custom modules unknown to optimum. Merge the adapter onto the headless Qwen3-Base backbone, load `head.pt`, wrap in a plain `nn.Module` with signature `(input_ids, attention_mask) → logits [B,S]`, export manually:

```python
import torch
model = load_merged("runs/my-v1").eval().float()   # UNVERIFIED entrypoint name — read kev/*.py
dummy_ids  = torch.ones(1, 128, dtype=torch.int64)
dummy_mask = torch.ones(1, 128, dtype=torch.int64)
torch.onnx.export(model, (dummy_ids, dummy_mask), "onnx/model.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids":     {0: "batch", 1: "sequence"},
                  "attention_mask": {0: "batch", 1: "sequence"},
                  "logits":         {0: "batch", 1: "sequence"}},
    opset_version=17, do_constant_folding=True)
```

### Quantize to q4 + q4f16

The wrapper's dtype union is `q4 | q4f16 | fp16 | fp32 | auto` — **q8 / `model_quantized.onnx` (dynamic int8) is NOT accepted**; skip `quantize_dynamic`. q4 is mandatory (Node `device:"cpu"` auto-selects q4); ship q4f16 for WebGPU. Int4 quantizes MatMul only — keep the pointer head fp32 or calibration collapses:

```bash
pip install onnx onnxruntime onnxconverter-common
python - <<'EOF'
import onnx
from onnxruntime.quantization import matmul_4bits_quantizer
m = onnx.load("onnx/model.onnx")
q = matmul_4bits_quantizer.MatMul4BitsQuantizer(m, block_size=32, is_symmetric=True, accuracy_level=4)
q.process()
onnx.save(q.model.model, "onnx/model_q4.onnx")     # >2GB also emits model_q4.onnx_data
from onnxconverter_common import float16
onnx.save(float16.convert_float_to_float16(onnx.load("onnx/model_q4.onnx")), "onnx/model_q4f16.onnx")
EOF
```

Order matters for q4f16: int4 **first**, then fp16-convert the rest — the reverse breaks WebGPU.

## 6. Path B — fine-tune the open-jev / DeBERTa family

Upstream repo: **github.com/kotoba-lang/typed-decisions** (`pip install git+https://github.com/kotoba-lang/typed-decisions`; module `typed_decisions.open_jev`, plus `modal_app.py` and train/val/test JSONL).

Verified upstream recipe (`com-kotobalabs/open-jev-deberta-v3-large`, Apache-2.0): `microsoft/deberta-v3-large` backbone + 3-layer scoring head (`head.safetensors`), markers `[STATE]`/`[Q]`/`[OPT]` added to the tokenizer — **keep the same added-token ids**. 18,000 states / 42,000 questions (banking77 + sst5 + boolq), loss **CE + Brier**, gold-preserving question augmentation at p=0.7 (option shuffle, paraphrase, distractor drop, level relabel, noul negation with flipped gold). 1 epoch, 229 s on 1×H100 (~$0.25). Temperature fitted post-hoc on validation (shipped 1.05).

Train with the repo's transformers `Trainer` setup (it defines the 5-input collate and pair head). Export: same story as Path A — optimum's `text-classification` task cannot emit the 5-input span-pooling graph. Wrap the trained module as `(input_ids, attention_mask, seg, pair_q, pair_opt) → logits [B, pairs]` and `torch.onnx.export` with those five `input_names`, `output_names=["logits"]`, dynamic `batch` axis, opset 17. Quantize identically to §5 (head stays fp32).

## 7. Making open-jev load your model

Copy `onnx-community/kev-0.6b-ONNX/config.json` (or the deberta one) and edit:

- **Base section intact**: `architectures` must stay `["Qwen3Model"]` (headless backbone) for kev, `["DebertaV2Model"]` for open-jev — a `ForCausalLM` class makes transformers.js map to a text-generation class and demand a KV-cache graph. Keep `model_type`, `vocab_size`, `use_cache: false`, `pad_token_id`.
- **`kev` section**: `delimiters`/`delimiter_ids` (must match the tokenizer; originals `<|fim_prefix|>`=151659, `<|fim_middle|>`=151660, `<|box_start|>`=151648, `<|box_end|>`=151649, `<|fim_suffix|>`=151661), `max_state_tokens`/`max_branch_tokens` (8192), `trained_max_state_tokens`/`trained_max_branch_tokens` (set to your training lengths; originals 384/1024), `max_options`, `noul_options: ["no","yes"]`, `head_dim`, `scale`, `lora` metadata.
- **`open_jev` section**: `markers: ["[STATE]","[Q]","[OPT]"]`, `max_state_tokens: 256`, `max_len: 512`, `temperature` (**refit on your validation split** — a stale temperature reports dishonest confidences), `pool: "span"`.
- **`transformers.js_config`**: `{ "dtype": "q4f16", "use_external_data_format": { … } }`.
- Ship the **full tokenizer set**: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`, `added_tokens.json` (Qwen); `spm.model` for DeBERTa.

Test with a local path before pushing to HF:

```bash
node -e '
import("open-jev").then(async ({ OpenJev, choice }) => {
  const rt = await OpenJev.load({ model: "/abs/path/my-kev", device: "cpu", dtype: "q4" });
  const [a] = await rt.decide("Event: Deep Purple concert at Budapest Arena",
    [choice("What type of event is this?",
            ["concert","sports","exhibition","networking","dining","outdoor","wellness"])]);
  console.log(a.choice, a.confidence);
})'
```

To make it a first-class alias in `@amatelic/jev` itself (so `createJev({ model: "my-kev" })` validates), extend `src/index.ts`: `MODEL_REPOS` (alias→repo), `MODEL_SIZES` (bytes), the `ModelAliasSchema` TypeBox literals, and the `family: alias.startsWith("kev") ? "kev" : "open-jev"` heuristic in `models()`.

## 8. Evaluating

`eval.mjs` hardcodes `["kev-0.6b","kev-4b","open-jev"]` as the default of its `models` ternary (single model via `node eval.mjs <model>`), loading each with `OpenJev.load({ model, device: "cpu", dtype: "q4" })`. Any non-alias string passes through as an HF repo id / local path, so **append your model id to that default array — nothing else changes**:

```js
// eval.mjs, in main() — edit the default branch:
const models = process.argv[2] && process.argv[2] !== "--calibration"
  ? [process.argv[2]]
  : ["kev-0.6b", "kev-4b", "open-jev", "yourname/my-kev-events"];
```

`bench.mjs` is a pure env switch: `JEV_MODEL=yourname/my-kev-events node bench.mjs`.

Beat-the-baseline bar (50 events, ARM64 Mac CPU, q4):

| Model | Accuracy | Latency | Bar for your model |
|---|---|---|---|
| `kev-0.6b` | 88% | 164 ms | > 88% @ ≤ ~164 ms |
| `kev-4b` | 94% | 1051 ms | > 94% @ < 1051 ms is the real prize |
| `open-jev` | 84% | 136 ms | > 84% @ ≤ ~136 ms |

Checks beyond headline accuracy:

- **Per-category P/R/F1** (eval.mjs reports it): a fine-tune should lift your worst ✗/⚠ categories, not just the mean.
- **Calibration**: eval.mjs bins confidence into `[0,0.5) [0.5,0.7) [0.7,0.85) [0.85,0.95) [0.95,1.01)` and reports `avgConfidence` vs `actualAccuracy`; gap ≤10pp ≈ acceptable. Top-bin overconfidence → refit `temperature` on validation (§7) and re-ship config; don't tune it on the eval set.
- **Abstains**: eval.mjs never checks `kind:"abstain"` — `cat?.choice ?? "general"` silently converts an abstain into a wrong "general" prediction. Inspect raw answers if accuracy looks oddly low.
- Known eval quirks: `bench.mjs`'s `expected` includes `"market"`, which is **not in its options** — its accuracy ceiling is 9/10 by construction. And if you train on generated data, make sure no holdout template variant leaks into train.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Unsupported model: config.json has neither an "open_jev" nor a "kev" section…` | `detectFamily` found no section (the JEV-9B case), or the section isn't an object | Add a proper `kev` or `open_jev` **object**; `config.kev` is checked first and wins if both exist |
| `Tokenizer does not know the marker token X` | marker/delimiter doesn't tokenize to exactly 1 id with `add_special_tokens:false` | Ship the full tokenizer files incl. `added_tokens.json`; `delimiter_ids` must match the tokenizer |
| Garbage answers or ORT input-name errors after export | exported `ForCausalLM`/`text-generation` → `[B,S,V]` logits + KV-cache inputs; runtime indexes flat per-token logits | Export the wrapped headless module (§5/§6); keep `architectures: ["Qwen3Model"]` |
| Silent accuracy collapse after quantization | pointer/span head got quantized, or wrong q4f16 order | Keep the head fp32; int4 first, then fp16-convert; re-run eval |
| 404 / missing file on load | `dtype:"auto"` → q4 off-WebGPU but you only shipped fp16/fp32; or >2 GB without `use_external_data_format` / missing `.onnx_data` sibling | Pass `dtype` explicitly; declare external data in `transformers.js_config` |
| Accuracy fine, confidences wrong | stale `temperature` (shipped values are calibration-specific per checkpoint) | refit on validation, update `config.json`, re-eval calibration bins |
| Delimiter-like text in state/options | kev escapes `<\|name\|>` → `<¦name¦>` at runtime only | generate training data through the same escape; never emit raw delimiters |

## 10. Hardware & time expectations (M1 Max 64 GB)

M1 Max numbers are derived estimates (MPS), not measurements; the Modal numbers are measured upstream:

| Job | Dataset | M1 Max (local) | Modal H100 |
|---|---|---|---|
| kev-0.6b LoRA (bf16 autocast, grad ckpt; frozen 1.2 GB backbone, ~10M trainable) | 12.5k records | ~1–2 h | — |
| kev-0.6b domain fine-tune (`--init-from`) | 1–2k records | ~10–20 min | ~15 min, ~$1 |
| kev-4b LoRA (fp16/bf16, ~10 GB resident; no bitsandbytes on MPS → no QLoRA) | 12.5k / 1–2k | ~6–10 h / ~1 h | ~15 min, ~$1 |
| deberta-v3-large full FT (fp32 Adam ~7 GB, head from scratch) | 18k states / 2k | ~1.5–2.5 h / ~10–15 min | 229 s, ~$0.25 |

Practical loop: generate ~1,000 records (§4) → Modal fine-tune (`--init-from kev-0.6b` or `kev-4b`) → merge + export + quantize locally (CPU is fine for export) → eval (§8). The 50-event set alone is far too small to train on.

## Caveats — flagged UNVERIFIED by the research

- The exact merge + export entrypoint APIs in `jaredpalmer/kev` and `kotoba-lang/typed-decisions` — `load_merged()` in §5 is illustrative; expect ~100 LOC reading `kev/train.py` / `typed_decisions` to build the wrapper. The onnx-community conversion scripts are unpublished; alternatively ask Xenova for the ones used for `kev-*-ONNX`.
- Local-dir loading (`OpenJev.load({ model: "/abs/path/…" })`) follows from `resolveModelId` passthrough but was not exercised end-to-end through `@amatelic/jev`.
- M1 Max wall-times in §10 are derived estimates, not measurements.
- Whether `head_lr: 0.0` in kev's `training_config.json` means the pointer head shares the main lr.
- kev's full-precision ONNX export is deliberately unpublished (the onnx-community repos ship q4/q4f16 only).
