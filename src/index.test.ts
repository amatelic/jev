/**
 * Unit tests for @amatelic/jev — schemas, validation, model management,
 * and end-to-end decide() with the real model (ARM64 native or HTTP).
 *
 * Usage:
 *   npx vitest run                              # all tests (uses real model)
 *   npx vitest run src/index.test.ts           # unit tests only
 *   JEV_MODE=http npx vitest run src/index.test.ts  # force HTTP transport
 */

import { describe, expect, it, beforeAll, afterAll } from "vitest";
import { Value } from "@sinclair/typebox/value";
import {
  createJev,
  listModels,
  modelInfo,
  // Schemas
  ChoiceQuestionSchema,
  NoulQuestionSchema,
  ScoreQuestionSchema,
  QuestionSchema,
  AnswerSchema,
  DecideRequestSchema,
  DecideResponseSchema,
  JevConfigSchema,
  ModelAliasSchema,
  // Types
  type Question,
  type JevClient,
  type ModelInfo,
} from "./index.js";
import { LOCAL_MODELS, REMOTE_MODELS } from "./models.generated.js";

// ─── Schema validation tests (no model needed) ─────────────────────

describe("schemas", () => {
  describe("ChoiceQuestionSchema", () => {
    const valid = {
      id: "q1",
      kind: "choice",
      prompt: "What type?",
      options: ["a", "b", "c"],
    };

    it("accepts valid choice questions", () => {
      expect(Value.Check(ChoiceQuestionSchema, valid)).toBe(true);
    });

    it("accepts with optional descriptions", () => {
      const withDesc = { ...valid, descriptions: { a: "Option A", b: "Option B" } };
      expect(Value.Check(ChoiceQuestionSchema, withDesc)).toBe(true);
    });

    it("rejects: fewer than 2 options", () => {
      expect(Value.Check(ChoiceQuestionSchema, { ...valid, options: ["only"] })).toBe(false);
    });

    it("rejects: missing prompt", () => {
      expect(Value.Check(ChoiceQuestionSchema, { id: "q", kind: "choice", options: ["a", "b"] })).toBe(false);
    });

    it("rejects: empty id", () => {
      expect(Value.Check(ChoiceQuestionSchema, { ...valid, id: "" })).toBe(false);
    });

    it("rejects: more than 50 options", () => {
      expect(Value.Check(ChoiceQuestionSchema, {
        ...valid,
        options: Array.from({ length: 51 }, (_, i) => `opt${i}`),
      })).toBe(false);
    });

    it("rejects: wrong kind", () => {
      expect(Value.Check(ChoiceQuestionSchema, { ...valid, kind: "noul" })).toBe(false);
    });
  });

  describe("NoulQuestionSchema", () => {
    it("accepts valid noul questions", () => {
      expect(Value.Check(NoulQuestionSchema, {
        id: "q1",
        kind: "noul",
        prompt: "Is this true?",
      })).toBe(true);
    });

    it("rejects: extra properties (options not allowed)", () => {
      expect(Value.Check(NoulQuestionSchema, {
        id: "q1",
        kind: "noul",
        prompt: "Is this true?",
        options: ["yes", "no"],
      })).toBe(false);
    });
  });

  describe("ScoreQuestionSchema", () => {
    it("accepts valid score questions", () => {
      expect(Value.Check(ScoreQuestionSchema, {
        id: "q1",
        kind: "score",
        prompt: "Rate quality",
        levels: ["low", "medium", "high"],
      })).toBe(true);
    });

    it("rejects: fewer than 2 levels", () => {
      expect(Value.Check(ScoreQuestionSchema, {
        id: "q1",
        kind: "score",
        prompt: "Rate",
        levels: ["only"],
      })).toBe(false);
    });
  });

  describe("QuestionSchema (union)", () => {
    it("accepts all three kinds", () => {
      const choice: Question = { id: "c", kind: "choice", prompt: "?", options: ["a", "b"] };
      const noul: Question = { id: "n", kind: "noul", prompt: "?" };
      const score: Question = { id: "s", kind: "score", prompt: "?", levels: ["x", "y"] };

      expect(Value.Check(QuestionSchema, choice)).toBe(true);
      expect(Value.Check(QuestionSchema, noul)).toBe(true);
      expect(Value.Check(QuestionSchema, score)).toBe(true);
    });

    it("rejects unknown kinds", () => {
      expect(Value.Check(QuestionSchema, { id: "x", kind: "unknown", prompt: "?" })).toBe(false);
    });
  });

  describe("DecideRequestSchema", () => {
    it("accepts valid request", () => {
      const request = {
        state: "Some context text",
        questions: [
          { id: "q1", kind: "choice", prompt: "?", options: ["a", "b"] },
          { id: "q2", kind: "noul", prompt: "True?" },
        ],
      };
      expect(Value.Check(DecideRequestSchema, request)).toBe(true);
    });

    it("rejects: empty state", () => {
      expect(Value.Check(DecideRequestSchema, {
        state: "",
        questions: [{ id: "q", kind: "noul", prompt: "?" }],
      })).toBe(false);
    });

    it("rejects: no questions", () => {
      expect(Value.Check(DecideRequestSchema, { state: "text", questions: [] })).toBe(false);
    });

    it("rejects: more than 32 questions", () => {
      const questions = Array.from({ length: 33 }, (_, i) => ({
        id: `q${i}`,
        kind: "noul",
        prompt: "?",
      }));
      expect(Value.Check(DecideRequestSchema, { state: "text", questions })).toBe(false);
    });

    it("rejects: state longer than 8000 chars", () => {
      expect(Value.Check(DecideRequestSchema, {
        state: "x".repeat(8001),
        questions: [{ id: "q", kind: "noul", prompt: "?" }],
      })).toBe(false);
    });
  });

  describe("AnswerSchema", () => {
    it("accepts valid choice answer", () => {
      expect(Value.Check(AnswerSchema, {
        kind: "choice",
        id: "q1",
        choice: "option_a",
        confidence: 0.85,
        probabilities: { option_a: 0.85, option_b: 0.15 },
      })).toBe(true);
    });

    it("accepts valid noul answer", () => {
      expect(Value.Check(AnswerSchema, {
        kind: "noul",
        id: "q1",
        answer: true,
        probability: 0.92,
        confidence: 0.92,
      })).toBe(true);
    });

    it("accepts valid abstain answer", () => {
      expect(Value.Check(AnswerSchema, {
        kind: "abstain",
        id: "q1",
        reason: "below confidence gate",
      })).toBe(true);
    });

    it("rejects: confidence > 1", () => {
      expect(Value.Check(AnswerSchema, {
        kind: "noul",
        id: "q",
        answer: true,
        probability: 1.5,
        confidence: 1.5,
      })).toBe(false);
    });
  });

  describe("JevConfigSchema", () => {
    it("accepts empty config (all defaults)", () => {
      expect(Value.Check(JevConfigSchema, {})).toBe(true);
    });

    it("accepts valid model alias", () => {
      expect(Value.Check(JevConfigSchema, { model: "kev-4b" })).toBe(true);
    });

    it("rejects unknown model alias", () => {
      expect(Value.Check(JevConfigSchema, { model: "gpt-4" })).toBe(false);
    });

    it("accepts all valid modes", () => {
      for (const mode of ["auto", "native", "http"]) {
        expect(Value.Check(JevConfigSchema, { mode })).toBe(true);
      }
    });
  });
});

// ─── Model management tests ─────────────────────────────────────────

describe("model management", () => {
  const HUB_ALIASES = ["kev-0.6b", "kev-4b", "open-jev"];

  it("listModels returns the hub models (plus any local fine-tunes present)", async () => {
    const models = await listModels();
    const aliases = models.map((m: ModelInfo) => m.alias);

    for (const hub of HUB_ALIASES) expect(aliases).toContain(hub);
    expect(models.length).toBeGreaterThanOrEqual(HUB_ALIASES.length);

    for (const m of models) {
      const isHub = HUB_ALIASES.includes(m.alias);

      if (isHub) expect(m.repo).toMatch(/^onnx-community\//);
      else {
        // local fine-tunes come from the generated registry, are always
        // cached (their directory is present), and carry the payload size
        expect(LOCAL_MODELS.map((x) => x.alias)).toContain(m.alias);
        expect(m.cached).toBe(true);
        expect(m.downloadSizeBytes).toBeGreaterThan(0);
      }
      expect(typeof m.cached).toBe("boolean");
      expect(["kev", "open-jev", "unknown"]).toContain(m.family);
    }
  });

  it("generated local model aliases pass ModelAliasSchema (autocomplete is real)", () => {
    for (const m of [...LOCAL_MODELS, ...REMOTE_MODELS]) {
      expect(Value.Check(ModelAliasSchema, m.alias)).toBe(true);
    }
  });

  it("every listing carries a non-empty displayName", async () => {
    const models = await listModels();

    for (const m of models) {
      expect(typeof m.displayName).toBe("string");
      expect(m.displayName.length).toBeGreaterThan(0);
    }
  });

  it("modelInfo returns specific model", async () => {
    const info = await modelInfo("kev-0.6b");

    expect(info).not.toBeNull();
    expect(info?.alias).toBe("kev-0.6b");
    expect(info?.repo).toBe("onnx-community/kev-0.6b-ONNX");
  });

  it("modelInfo returns null for unknown model", async () => {
    const info = await modelInfo("nonexistent" as never);
    expect(info).toBeNull();
  });
});

// ─── Client tests (uses real model — slower) ───────────────────────

describe("createJev + decide", { timeout: 120_000 }, () => {
  let jev: JevClient | null = null;

  beforeAll(async () => {
    try {
      jev = await createJev({
        model: process.env.JEV_MODEL ?? "kev-0.6b",
        mode: process.env.JEV_MODE === "http" ? "http" : "native",
      });
    } catch {
      // Native mode not available on this platform (e.g., x64 Mac without Docker)
      // Tests in this suite will be skipped
      jev = null;
    }
  });

  afterAll(async () => {
    await jev?.dispose?.();
  });

  it("creates a client with the expected mode", () => {
    if (!jev) return; // skip: native unavailable
    expect(["native", "http"]).toContain(jev.mode);
    expect(jev.model).toBe(process.env.JEV_MODEL ?? "kev-0.6b");
  });

  it("answers a choice question with valid structure", async () => {
    if (!jev) return;
    const [answer] = await jev.decide(
      "Event: HC Sparta Praha hockey match at O2 arena",
      [{ id: "type", kind: "choice", prompt: "What type of event?", options: ["concert", "sports", "exhibition", "general"] }],
    );

    expect(answer).toBeDefined();
    expect(answer.kind).toBe("choice");
    expect(["concert", "sports", "exhibition", "general"]).toContain(answer.choice);
    expect(answer.confidence).toBeGreaterThanOrEqual(0);
    expect(answer.confidence).toBeLessThanOrEqual(1);
  });

  it("answers a noul question with boolean + probability", async () => {
    if (!jev) return;
    const [answer] = await jev.decide(
      "The number 2 is greater than the number 1.",
      [{ id: "math", kind: "noul", prompt: "The number 2 is greater than the number 1." }],
    );

    expect(answer.kind).toBe("noul");
    expect(typeof answer.answer).toBe("boolean");
    expect(answer.probability).toBeGreaterThanOrEqual(0);
    expect(answer.probability).toBeLessThanOrEqual(1);
  });

  it("answers a score question with level from the provided levels", async () => {
    if (!jev) return;
    const [answer] = await jev.decide(
      "A comprehensive 3-day music festival with international headliners.",
      [{ id: "scale", kind: "score", prompt: "Rate the scale of this event.", levels: ["small", "medium", "large", "massive"] }],
    );

    expect(answer.kind).toBe("score");
    expect(["small", "medium", "large", "massive"]).toContain(answer.level);
    expect(answer.confidence).toBeGreaterThanOrEqual(0);
  });

  it("answers multiple questions in one decide() call", async () => {
    if (!jev) return;
    const answers = await jev.decide(
      "Event: Jazz concert at the Blue Note club in New York, $50 tickets, Friday night",
      [
        { id: "type", kind: "choice", prompt: "What type?", options: ["concert", "sports", "general"] },
        { id: "is_music", kind: "noul", prompt: "This is a music event." },
        { id: "price", kind: "score", prompt: "How expensive?", levels: ["cheap", "moderate", "expensive", "premium"] },
      ],
    );

    expect(answers).toHaveLength(3);
    expect(answers.map((a) => a.id)).toEqual(["type", "is_music", "price"]);
  });

  it("throws on invalid state (empty)", async () => {
    if (!jev) return;
    await expect(
      jev.decide("", [{ id: "q", kind: "noul", prompt: "?" }]),
    ).rejects.toThrow(/Invalid request/);
  });

  it("throws on invalid question (no options)", async () => {
    if (!jev) return;
    await expect(
      jev.decide("test state", [
        { id: "bad", kind: "choice", prompt: "?", options: [] } as never,
      ]),
    ).rejects.toThrow(/Invalid request/);
  });

  it("produces consistent answers for the same input (deterministic)", async () => {
    if (!jev) return;
    const question: Question = {
      id: "type",
      kind: "choice",
      prompt: "What type of event?",
      options: ["concert", "sports", "general"],
    };

    const [a1] = await jev.decide("Event: Football match between Sparta and Slavia", [question]);
    const [a2] = await jev.decide("Event: Football match between Sparta and Slavia", [question]);

    expect(a1.kind).toBe("choice");
    expect(a2.kind).toBe("choice");

    if (a1.kind === "choice" && a2.kind === "choice") {
      expect(a1.choice).toBe(a2.choice);
    }
  });
});

// ─── HTTP transport tests (skip if no server running) ──────────────

describe("HTTP transport", { timeout: 30_000 }, () => {
  it("connects to a running server", async () => {
    // Skip if no server on :9401
    try {
      const probe = await fetch("http://localhost:9401/decide", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ state: "test", questions: [{ id: "t", kind: "noul", prompt: "test" }] }),
        signal: AbortSignal.timeout(2000),
      });

      if (!probe.ok) return; // Skip silently
    } catch {
      return; // No server running — skip
    }

    const jev = await createJev({ mode: "http", url: "http://localhost:9401" });
    const [answer] = await jev.decide(
      "Event: test concert",
      [{ id: "t", kind: "noul", prompt: "This is about music." }],
    );

    expect(answer.kind).toBe("noul");
    expect(typeof answer.answer).toBe("boolean");
    await jev.dispose();
  });
});
