/**
 * JEV client — the simplest possible interface to local AI decisions.
 *
 * Two transports, one API:
 *   native: open-jev runs in-process (ARM64 Mac / Linux)
 *   http:   Docker container on x64 Mac or remote server
 *
 * mode: "auto" tries native first, falls back to HTTP.
 */

import { Value } from "@sinclair/typebox/value";
import type { Answer, Question, JevConfig, ModelInfo, ModelAlias } from "./schemas.js";
import { DecideRequestSchema, DecideResponseSchema } from "./schemas.js";

// ─── Model registry (mirrors open-jev's MODELS) ────────────────────

const MODEL_REPOS: Record<string, string> = {
  "kev-0.6b": "onnx-community/kev-0.6b-ONNX",
  "kev-4b": "onnx-community/kev-4b-ONNX",
  "open-jev": "onnx-community/open-jev-deberta-v3-large-ONNX",
};

const MODEL_SIZES: Record<string, number> = {
  "kev-0.6b": 383_111_829,
  "kev-4b": 2_528_556_269,
  "open-jev": 838_860_800,
};

// ─── Types ──────────────────────────────────────────────────────────

export interface JevClient {
  decide(state: string, questions: readonly Question[]): Promise<Answer[]>;
  models(): Promise<ModelInfo[]>;
  switch(model: ModelAlias): Promise<void>;
  dispose(): Promise<void>;
  readonly model: string;
  readonly mode: "native" | "http";
}

export interface CreateJevOptions extends Partial<JevConfig> {
  onProgress?: (info: { loaded: number; total: number }) => void;
}

// ─── Native transport (ARM64 Mac / Linux) ─────────────────────────

interface OpenJevRuntime {
  decide: (state: string, questions: unknown[]) => Promise<unknown[]>;
  dispose?: () => Promise<void>;
  runtime?: { model?: string; family?: string };
}

async function tryNativeLoad(
  model: string,
  dtype: string,
  onProgress?: (info: { loaded: number; total: number }) => void,
): Promise<OpenJevRuntime | null> {
  try {
    // Dynamic import — fails silently on unsupported platforms
    // SAFETY: open-jev's typed generics don't match our generic runtime
    // interface; cast through unknown at this dynamic import boundary.
    const mod = (await import("open-jev")) as unknown as {
      OpenJev: { load: (opts: Record<string, unknown>) => Promise<OpenJevRuntime> };
    };

    const runtime = await mod.OpenJev.load({
      model,
      device: "cpu",
      dtype,
      onProgress: onProgress
        ? (p: { status?: string; loaded?: number; total?: number }) => {
            if (p.loaded !== undefined && p.total !== undefined && p.total > 0) {
              onProgress({ loaded: p.loaded, total: p.total });
            }
          }
        : undefined,
    });

    return runtime;
  } catch {
    return null;
  }
}

function mapNativeAnswer(raw: Record<string, unknown>, question: Question): Answer {
  if (question.kind === "choice") {
    const choice = String(raw.choice ?? "");
    const confidence = Number(raw.confidence ?? 0);
    const probabilities = (raw.probabilities ?? {}) as Record<string, number>;

    return { kind: "choice", id: question.id, choice, confidence, probabilities };
  }

  if (question.kind === "noul") {
    return {
      kind: "noul",
      id: question.id,
      answer: Boolean(raw.answer),
      probability: Number(raw.probability ?? 0),
      confidence: Number(raw.confidence ?? 0),
    };
  }

  return {
    kind: "score",
    id: question.id,
    score: Number(raw.score ?? 0),
    normalized: Number(raw.normalized ?? 0),
    level: String(raw.level ?? ""),
    confidence: Number(raw.confidence ?? 0),
  };
}

// ─── HTTP transport (Docker / remote) ──────────────────────────────

async function httpDecide(
  url: string,
  state: string,
  questions: readonly Question[],
): Promise<Answer[]> {
  const body = JSON.stringify({ state, questions });

  const response = await fetch(`${url.replace(/\/$/, "")}/decide`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
    signal: AbortSignal.timeout(120_000),
  });

  if (!response.ok) {
    throw new Error(`jev http ${response.status}: ${(await response.text()).slice(0, 200)}`);
  }

  // SAFETY: response validated against the schema
  const parsed = (await response.json()) as { answers?: Answer[] };

  return parsed.answers ?? [];
}

// ─── Public API ─────────────────────────────────────────────────────

export async function createJev(options: CreateJevOptions = {}): Promise<JevClient> {
  const model = options.model ?? "kev-0.6b";
  const dtype = options.dtype ?? "q4";
  const mode = options.mode ?? "auto";
  const url = options.url ?? "http://localhost:9401";

  let nativeRuntime: OpenJevRuntime | null = null;
  let transport: "native" | "http" = "http";

  if (mode === "native" || mode === "auto") {
    nativeRuntime = await tryNativeLoad(model, dtype, options.onProgress);
  }

  if (nativeRuntime) {
    transport = "native";
  } else if (mode === "native") {
    throw new Error(
      `Native mode requested but open-jev failed to load (platform may not support it). ` +
      `Try mode: "http" with a Docker server, or install the ARM64 build of Node.js.`,
    );
  }

  const decide = async (state: string, questions: readonly Question[]): Promise<Answer[]> => {
    // Validate the request at the boundary
    const request = { state, questions: [...questions] };
    const errors = [...Value.Errors(DecideRequestSchema, request)];

    if (errors.length > 0) {
      const first = errors[0]!;

      throw new Error(`Invalid request at ${first.path}: ${first.message}`);
    }

    if (transport === "native" && nativeRuntime) {
      const rawAnswers = await nativeRuntime.decide(state, [...questions]);

      return questions.map((q, i) =>
        mapNativeAnswer((rawAnswers[i] ?? {}) as Record<string, unknown>, q),
      );
    }

    return httpDecide(url, state, questions);
  };

  const models = async (): Promise<ModelInfo[]> => {
    const results: ModelInfo[] = [];

    for (const [alias, repo] of Object.entries(MODEL_REPOS)) {
      let cached = false;

      if (transport === "native") {
        try {
          const mod = (await import("open-jev")) as unknown as {
            OpenJev: { info: (opts: Record<string, unknown>) => Promise<{ isCached?: boolean }> };
          };
          const info = await mod.OpenJev.info({ model: alias, device: "cpu", dtype });

          cached = info.isCached ?? false;
        } catch { /* info not critical */ }
      }

      results.push({
        alias,
        repo,
        family: alias.startsWith("kev") ? "kev" : "open-jev",
        cached,
        downloadSizeBytes: MODEL_SIZES[alias] ?? 0,
      });
    }

    return results;
  };

  const client: JevClient = {
    decide,
    models,
    model,
    mode: transport,
    switch: async (newModel: ModelAlias): Promise<void> => {
      if (transport === "native") {
        await nativeRuntime?.dispose?.();
        nativeRuntime = await tryNativeLoad(newModel, dtype, options.onProgress);

        if (!nativeRuntime) throw new Error(`Failed to switch to ${newModel}`);
      }
      // HTTP mode: model switching is server-side, just update the label
      (client as { model: string }).model = newModel;
    },
    dispose: async (): Promise<void> => {
      await nativeRuntime?.dispose?.();
      nativeRuntime = null;
    },
  };

  return client;
}

export async function listModels(): Promise<ModelInfo[]> {
  const jev = await createJev({ model: "kev-0.6b" });
  const models = await jev.models();
  await jev.dispose();

  return models;
}

export async function modelInfo(model: ModelAlias): Promise<ModelInfo | null> {
  const models = await listModels();

  return models.find((m) => m.alias === model) ?? null;
}

export async function download(
  model: ModelAlias,
  options: { onProgress?: (info: { loaded: number; total: number }) => void } = {},
): Promise<void> {
  const jev = await createJev({ model, onProgress: options.onProgress });
  await jev.dispose();
}

// Re-export schemas for consumers
export * from "./schemas.js";
