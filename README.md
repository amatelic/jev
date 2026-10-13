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
// List all models with cache status
const models = await listModels();
// → [
//   { alias: "kev-0.6b", sizeBytes: 383111829, cached: true,  family: "kev" },
//   { alias: "kev-4b",   sizeBytes: 2528556269, cached: false, family: "kev" },
//   { alias: "open-jev", sizeBytes: 838860800,  cached: false, family: "open-jev" }
// ]

// Pre-download without using
await download("kev-4b", { onProgress: (p) => ... });

// Check cache without downloading
const info = await modelInfo("kev-4b");
// → { isCached: false, downloadSize: 2528556269, files: [...] }
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

## x64 Mac (Docker fallback)

```bash
# Start the model server once
docker run -d -p 9401:9401 -v jev-cache:/root/.cache \
  node:22-slim npx -y @amatelic/jev serve

# Your code stays the same — mode: "auto" detects and uses HTTP
```

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
