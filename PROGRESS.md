# Model progress log

Fine-tuned model history: newest model, results, and deltas vs the previous entry with results.
Auto-written by `scripts/progress.mjs` — manual edits to entries will be overwritten; use `--note`.

- Deltas compare against the most recent earlier entry with results (an earlier entry with the
  same name wins — eval re-runs compare against the model's own last result); first entry renders “—”.
- The pipeline only writes here when its plan includes the `train` stage; `--only data`,
  `--only eval`, `--dry-run`, and `status` never modify this file.
- `progress.mjs finish` updates the name's latest result-less entry, or appends a fresh one
  if none exists (that is how eval-only re-runs get recorded).

## Entries (newest first)
| events-v2 | evaluated · 2026-10-18 · bd052bb |
| --- | --- |
| base | jaredpalmer/kev-0.6b |
| hyper | lr 2e-5 · 3 epochs · batch 8 |
| data | seed 42 · kev-train.jsonl |
| gate | 100.0 (prev events-v1 100.0 → +0.0) @176ms |
| gate | 100.0 (prev events-v1 100.0 → +0.0) @166ms |

| events-v1 | evaluated · 2026-10-13 · 31bb5ed |
| --- | --- |
| base | jaredpalmer/kev-0.6b (LoRA delta) |
| hyper | lr 2e-5 · 3 epochs · batch 8 |
| data | seed 42 · 1000 train / 250 holdout · template+entity split-separated |
| gate | 100.0 (—) @240ms |
| holdout | 96.4 (—) @170ms |
| note | first fine-tune; beats kev-4b ceiling at 0.6b latency |

## Source data (single source of truth — the tables above render from this)

```json
[
  {
    "name": "events-v1",
    "startedAt": "2026-10-13T00:00:00.000Z",
    "status": "evaluated",
    "gitSha": "31bb5ed",
    "base": "jaredpalmer/kev-0.6b (LoRA delta)",
    "dataNotes": "seed 42 · 1000 train / 250 holdout · template+entity split-separated",
    "hyper": "lr 2e-5 · 3 epochs · batch 8",
    "results": [
      {
        "dataset": "gate",
        "acc": 100,
        "avgMs": 240
      },
      {
        "dataset": "holdout",
        "acc": 96.4,
        "avgMs": 170
      }
    ],
    "deltas": null,
    "note": "first fine-tune; beats kev-4b ceiling at 0.6b latency"
  },
  {
    "name": "events-v2",
    "startedAt": "2026-10-18T20:37:44.356Z",
    "status": "evaluated",
    "gitSha": "bd052bb",
    "base": "jaredpalmer/kev-0.6b",
    "dataNotes": "1174 = 1000 original + 87 disagreement-mined ×2 (pool seed 999, train banks only)",
    "hyper": "lr 2e-5 · 3 epochs · batch 8",
    "results": [
      {
        "dataset": "gate",
        "acc": 100,
        "avgMs": 176
      },
      {
        "dataset": "holdout",
        "acc": 97.2,
        "avgMs": 163
      }
    ],
    "deltas": {
      "gate": {
        "prev": "events-v1",
        "prevAcc": 100,
        "dAcc": 0,
        "prevMs": 240,
        "dMs": -64
      },
      "holdout": {
        "prev": "events-v1",
        "prevAcc": 96.4,
        "dAcc": 0.8,
        "prevMs": 170,
        "dMs": -7
      }
    },
    "note": null
  }
]
```
