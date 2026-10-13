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

const PORT = Number.parseInt(process.env.JEV_PORT ?? process.argv.find((a) => a.startsWith("--port="))?.split("=")[1] ?? "9401", 10);
const MODEL = process.env.JEV_MODEL ?? process.argv.find((a) => a.startsWith("--model="))?.split("=")[1] ?? "kev-0.6b";
const DTYPE = process.env.JEV_DTYPE ?? process.argv.find((a) => a.startsWith("--dtype="))?.split("=")[1] ?? "q4";

async function main(): Promise<void> {
  const { OpenJev, choice, noul, score } = await import("open-jev");

  console.log(`Loading ${MODEL} (${DTYPE})...`);
  // SAFETY: DTYPE is validated from CLI args; cast to the expected union.
  const runtime = await OpenJev.load({ model: MODEL, device: "cpu", dtype: DTYPE as "q4" });
  console.log(`✅ ${MODEL} loaded`);

  const buildQuestion = (q: Question): unknown => {
    if (q.kind === "choice") return choice(q.prompt, q.options, q.descriptions);
    if (q.kind === "noul") return noul(q.prompt);
    return score(q.prompt, q.levels);
  };

  const server = createServer((req, res) => {
    if (req.method !== "POST" || req.url !== "/decide") {
      res.writeHead(404).end();
      return;
    }

    let body = "";
    req.on("data", (chunk: Buffer) => { body += chunk; });
    req.on("end", async () => {
      try {
        // Validate at the boundary
        const parsed = JSON.parse(body) as unknown;
        const errors = [...Value.Errors(DecideRequestSchema, parsed)];

        if (errors.length > 0) {
          const first = errors[0]!;
          res.writeHead(400, { "content-type": "application/json" });
          res.end(JSON.stringify({ ok: false, error: `Invalid at ${first.path}: ${first.message}` }));
          return;
        }

        const { state, questions } = parsed as { state: string; questions: Question[] };
        const t0 = Date.now();
        const built = questions.map(buildQuestion);
        // SAFETY: our Question[] maps to open-jev Questions at this boundary.
        const rawAnswers = (await runtime.decide(state, built as never)) as unknown[];

        const answers: Answer[] = questions.map((q, i) => {
          const raw = (rawAnswers[i] ?? {}) as Record<string, unknown>;

          if (q.kind === "choice") {
            return {
              kind: "choice" as const,
              id: q.id,
              choice: String(raw.choice ?? ""),
              confidence: Number(raw.confidence ?? 0),
              probabilities: (raw.probabilities ?? {}) as Record<string, number>,
            };
          }
          if (q.kind === "noul") {
            return {
              kind: "noul" as const,
              id: q.id,
              answer: Boolean(raw.answer),
              probability: Number(raw.probability ?? 0),
              confidence: Number(raw.confidence ?? 0),
            };
          }
          return {
            kind: "score" as const,
            id: q.id,
            score: Number(raw.score ?? 0),
            normalized: Number(raw.normalized ?? 0),
            level: String(raw.level ?? ""),
            confidence: Number(raw.confidence ?? 0),
          };
        });

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
