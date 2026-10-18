# sandbox — jev playground

`npm run sandbox` copies this folder into a gitignored `sandbox/` and runs it.

- **`states.txt`** — the events to classify, one per line (`#` = comment).
  Optional label: end a line with ` | category` to check the prediction
  (`✓`/`✗` + a `n/m labeled correct` tally). Unlabeled lines just predict.
- **`play.mjs`** — the code: load model → classify each state → print.

Edits in `sandbox/` are throwaway (invisible to git). Restore defaults: `npm run sandbox:reset`.
