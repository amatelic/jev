/**
 * Standalone HTTP server — the Docker target for x64 Mac / remote deployments.
 *
 * Usage:
 *   npx @amatelic/jev serve [--model kev-4b] [--port 9401] [--dtype q4]
 *   docker run -p 9401:9401 -v jev-cache:/root/.cache node:22-slim npx -y @amatelic/jev serve
 */

import { createServer } from "node:http";
import { Value } from "@sinclair/typebox/value";
import { DecideRequestSchema, DecideResponseSchema } from "./schemas.js";
import type { Question, Answer } from "./schemas.js";
import { buildNativeQuestion, mapNativeAnswer } from "./mapping.js";

const PORT = Number.parseInt(process.env.JEV_PORT ?? process.argv.find((a) => a.startsWith("--port="))?.split("=")[1] ?? "9401", 10);
const MODEL = process.env.JEV_MODEL ?? process.argv.find((a) => a.startsWith("--model="))?.split("=")[1] ?? "kev-0.6b";
const DTYPE = process.env.JEV_DTYPE ?? process.argv.find((a) => a.startsWith("--dtype="))?.split("=")[1] ?? "q4";
const VALID_DTYPES = new Set(["q4", "q4f16", "fp16", "fp32", "auto"]);
if (!VALID_DTYPES.has(DTYPE)) {
  console.error(`error: invalid dtype "${DTYPE}" — valid: ${[...VALID_DTYPES].join(", ")}`);
  process.exit(1);
}

async function main(): Promise<void> {
  const { OpenJev } = await import("open-jev");

  console.log(`Loading ${MODEL} (${DTYPE})...`);
  // DTYPE was validated against VALID_DTYPES above; narrow for open-jev's union.
  const runtime = await OpenJev.load({ model: MODEL, device: "cpu", dtype: DTYPE as "q4" | "q4f16" | "fp16" | "fp32" | "auto" });
  console.log(`✅ ${MODEL} loaded`);

  const server = createServer((req, res) => {
    // Both routes, same handler: /decide (documented default) and
    // /ai/jev/decide (the prefix Nomadia's crawler calls).
    if (req.method !== "POST" || (req.url !== "/decide" && req.url !== "/ai/jev/decide")) {
      res.writeHead(404).end();
      return;
    }

    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => { chunks.push(chunk); });
    req.on("end", async () => {
      try {
        // Validate at the boundary
        const parsed = JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
        const errors = [...Value.Errors(DecideRequestSchema, parsed)];

        if (errors.length > 0) {
          const first = errors[0]!;
          res.writeHead(400, { "content-type": "application/json" });
          res.end(JSON.stringify({ ok: false, error: `Invalid at ${first.path}: ${first.message}` }));
          return;
        }

        const { state, questions } = parsed as { state: string; questions: Question[] };
        const t0 = Date.now();
        // SAFETY: shared with the native transport (src/mapping.ts) — the two
        // paths can never drift apart.
        const built = questions.map(buildNativeQuestion);
        const rawAnswers = (await runtime.decide(state, built as never)) as unknown[];

        const answers: Answer[] = questions.map((q, i) =>
          mapNativeAnswer((rawAnswers[i] ?? {}) as Record<string, unknown>, q),
        );

        const response = { ok: true, answers, model: MODEL, durationMs: Date.now() - t0 };

        // Validate the response too (self-check)
        if (!Value.Check(DecideResponseSchema, response)) {
          console.warn("Response failed self-validation");
        }

        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(response));
      } catch (error) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: String((error as Error).message ?? error) }));
      }
    });
  });

  server.listen(PORT, () => {
    console.log(`@amatelic/jev server → http://localhost:${PORT}`);
    console.log(`POST /decide  { state, questions[] }`);
  });
}

void main();
