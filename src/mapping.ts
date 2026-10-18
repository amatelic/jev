/**
 * Question/answer mapping between the public JEV schema (kind/prompt) and
 * open-jev's runtime shape (type/instructions) — shared by the native
 * transport (index.ts) and the HTTP server (server.ts) so the two paths can
 * never drift apart.
 */

import type { Answer, Question } from "./schemas.js";

/** JEV question → open-jev runtime question (what choice()/noul()/score() produce). */
export function buildNativeQuestion(q: Question): Record<string, unknown> {
  if (q.kind === "choice") {
    return { type: "choice", instructions: q.prompt, options: q.options, descriptions: q.descriptions };
  }
  if (q.kind === "noul") {
    return { type: "noul", instructions: q.prompt };
  }
  return { type: "score", instructions: q.prompt, options: q.levels };
}

/** open-jev raw answer → validated-shape JEV answer for the given question. */
export function mapNativeAnswer(raw: Record<string, unknown>, question: Question): Answer {
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
