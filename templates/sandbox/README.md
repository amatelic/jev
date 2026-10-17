# sandbox — jev playground

`npm run sandbox` copies this folder into a gitignored `sandbox/` and runs it.

- **`states.txt`** — the events to classify, one per line (`#` = comment). This is the file to play with.
- **`play.mjs`** — the code: load model → classify each state → print `category  confidence  state`.

Edits in `sandbox/` are throwaway (invisible to git). Restore defaults: `npm run sandbox:reset`.
