# @amatelic/jev

**One-line local AI decisions.** Calibrated choice/yes-no/score answers from open models — no API keys, no cloud, no Docker required on ARM64 Mac or Linux.

```typescript
import { createJev } from "@amatelic/jev";

const jev = await createJev(); // kev-0.6b, auto-downloads if needed

const [answer] = await jev.decide(
  "Event: HC Sparta Praha hockey at O2 arena",
  [{ kind: "choice", prompt: "What type of event?", options: ["concert", "sports", "exhibition", "general"] }]
);

console.log(answer.choice);     // "sports"
console.log(answer.confidence); // 0.998
```

## Models

| Alias | Base | Size (q4) | Best for |
|---|---|---|---|
| `kev-0.6b` | Qwen3-0.6B | 383 MB | Fast decisions, prototyping |
| `kev-4b` | Qwen3-4B | 2.4 GB | Better accuracy, nuanced classification |
| `open-jev` | DeBERTa-v3 | 800 MB | Lighter, different tradeoffs |

Models are **auto-downloaded on first use** (HuggingFace CDN → local cache) with progress callbacks. Switch anytime:

```typescript
const jev = await createJev({ model: "kev-4b" });
```

Need a domain-specific decision model? See **[Training your own JEV models](docs/TRAINING.md)** — fine-tune kev or open-jev, export to ONNX, and load it right here. For the worked, verified end-to-end pipeline (data → LoRA → ONNX q4 → eval, 88% → 100% on the event-classification gate), see **[docs/TRAINING-PIPELINE.md](docs/TRAINING-PIPELINE.md)**.

## Configuration

```typescript
const jev = await createJev({
  model: "kev-0.6b",           // default
  dtype: "q4",                  // q4 | q4f16 | fp16 | fp32 | auto
  device: "auto",               // auto | cpu | webgpu | wasm
  mode: "auto",                 // auto: try native, fallback to Docker
  // mode: "native",            // force embedded (fails on x64 Mac)
  // mode: "http",              // force Docker/HTTP (always works)
  // url: "http://localhost:9401", // Docker server URL (mode: "http")
  onProgress: ({ loaded, total }) => {
    console.log(`Downloading: ${(loaded / total * 100).toFixed(0)}%`);
  },
});
```

## Model management

```typescript
// List all models with cache status and display names
const models = await listModels();
// → [
//   { alias: "kev-0.6b",  displayName: "Kev 0.6B — fast default",           repo: "onnx-community/kev-0.6b-ONNX", family: "kev",      cached: true,  downloadSizeBytes: 383111829 },
//   { alias: "kev-4b",    displayName: "Kev 4B — accuracy ceiling",         repo: "onnx-community/kev-4b-ONNX",   family: "kev",      cached: false, downloadSizeBytes: 2528556269 },
//   { alias: "open-jev",  displayName: "Open-Jev (DeBERTa-v3) — lightest",  repo: "onnx-community/open-jev-deberta-v3-large-ONNX", family: "open-jev", cached: false, downloadSizeBytes: 838860800 }
// ] — plus any fine-tuned models registered by `npm run models:sync`

// Pre-download without using (metadata-only cache check is listModels() above)
await download("kev-4b", { onProgress: (p) => ... });
```

## Question types (TypeBox validated)

```typescript
// Choice: pick one from options
{ kind: "choice", prompt: "...", options: ["a", "b", "c"], descriptions?: { a: "...", b: "..." } }

// Noul: yes/no with probability
{ kind: "noul", prompt: "Is this statement true?" }

// Score: rate on ordered levels
{ kind: "score", prompt: "Rate quality", levels: ["bad", "ok", "good", "great"] }
```

All questions and answers are validated with TypeBox schemas at the API boundary. Bad input gets a `400` with field-level error messages, not silent corruption.

## ARM64 Mac / Linux (no Docker)

```bash
npm install @amatelic/jev
# Works immediately — model downloads to ~/.cache/transformers/
```

### Installing from GitHub (not yet on npm)

Install from a **release tarball** — it ships the compiled `dist/`, runs no
build scripts, and downloads no devDependencies:

```jsonc
// package.json — pin the exact release
"dependencies": {
  "@amatelic/jev": "https://github.com/amatelic/jev/releases/download/v0.1.1/amatelic-jev-0.1.1.tgz"
}
```

Tarballs are built by CI (`.github/workflows/release.yml`) on every `v*` tag.
`open-jev` (the inference runtime) is an **optional peer** — keep it in your
own dependencies for native in-process inference; without it `createJev`
falls back to HTTP mode. (The repo itself never carries build output.)

## x64 Mac (Docker fallback)

```bash
# Start the model server once
docker run -d -p 9401:9401 -v jev-cache:/root/.cache \
  node:22-slim npx -y @amatelic/jev serve

# Your code stays the same — mode: "auto" detects and uses HTTP
```

The server answers **both** `POST /decide` and `POST /ai/jev/decide` (the
route prefix Nomadia's crawler uses) — same request/response body.

## Architecture

```
┌─────────────────────────────────────────────┐
│  @amatelic/jev                              │
│                                              │
│  createJev() ──► mode: "auto"               │
│    │                                         │
│    ├── native works? ──► EmbeddedTransport  │
│    │   (ARM64/Linux)     open-jev in-process │
│    │                                         │
│    └── native fails? ──► HttpTransport      │
│        (x64 Mac)         Docker :9401        │
│                                              │
│  Both paths return the same typed answers    │
│  validated by the same TypeBox schemas       │
└─────────────────────────────────────────────┘
```

## API Reference

### `createJev(options?)`
Returns a `JevClient` — loads the model, validates config, auto-downloads if needed.

### `jev.decide(state, questions)`
One forward pass: state + questions → calibrated answers. The core method.

### `jev.models()`
List available models with cache status.

### `jev.switch(model)`
Switch to a different model at runtime.

### `jev.dispose()`
Free memory (model weights).

## License

MIT
