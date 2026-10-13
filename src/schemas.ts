/**
 * TypeBox schemas — single source of truth for the JEV API.
 * Validates requests AND derives TypeScript types (no duplication).
 */

import { Type, type Static } from "@sinclair/typebox";

// ─── Questions ─────────────────────────────────────────────────────

export const ChoiceQuestionSchema = Type.Object({
  id: Type.String({ minLength: 1, maxLength: 64 }),
  kind: Type.Literal("choice"),
  prompt: Type.String({ minLength: 1, maxLength: 4000 }),
  options: Type.Array(Type.String({ minLength: 1, maxLength: 200 }), {
    minItems: 2,
    maxItems: 50,
  }),
  descriptions: Type.Optional(
    Type.Record(Type.String(), Type.String({ maxLength: 500 })),
  ),
});

export const NoulQuestionSchema = Type.Object(
  {
    id: Type.String({ minLength: 1, maxLength: 64 }),
    kind: Type.Literal("noul"),
    prompt: Type.String({ minLength: 1, maxLength: 4000 }),
  },
  { additionalProperties: false },
);

export const ScoreQuestionSchema = Type.Object({
  id: Type.String({ minLength: 1, maxLength: 64 }),
  kind: Type.Literal("score"),
  prompt: Type.String({ minLength: 1, maxLength: 4000 }),
  levels: Type.Array(Type.String({ minLength: 1, maxLength: 100 }), {
    minItems: 2,
    maxItems: 10,
  }),
});

export const QuestionSchema = Type.Union([
  ChoiceQuestionSchema,
  NoulQuestionSchema,
  ScoreQuestionSchema,
]);

export type Question = Static<typeof QuestionSchema>;
export type ChoiceQuestion = Static<typeof ChoiceQuestionSchema>;
export type NoulQuestion = Static<typeof NoulQuestionSchema>;
export type ScoreQuestion = Static<typeof ScoreQuestionSchema>;

// ─── Answers ───────────────────────────────────────────────────────

export const ChoiceAnswerSchema = Type.Object({
  kind: Type.Literal("choice"),
  id: Type.String(),
  choice: Type.String(),
  confidence: Type.Number({ minimum: 0, maximum: 1 }),
  probabilities: Type.Record(Type.String(), Type.Number()),
});

export const NoulAnswerSchema = Type.Object({
  kind: Type.Literal("noul"),
  id: Type.String(),
  answer: Type.Boolean(),
  probability: Type.Number({ minimum: 0, maximum: 1 }),
  confidence: Type.Number({ minimum: 0, maximum: 1 }),
});

export const ScoreAnswerSchema = Type.Object({
  kind: Type.Literal("score"),
  id: Type.String(),
  score: Type.Number(),
  normalized: Type.Number(),
  level: Type.String(),
  confidence: Type.Number({ minimum: 0, maximum: 1 }),
});

export const AbstainAnswerSchema = Type.Object({
  kind: Type.Literal("abstain"),
  id: Type.String(),
  reason: Type.String(),
});

export const AnswerSchema = Type.Union([
  ChoiceAnswerSchema,
  NoulAnswerSchema,
  ScoreAnswerSchema,
  AbstainAnswerSchema,
]);

export type Answer = Static<typeof AnswerSchema>;
export type ChoiceAnswer = Static<typeof ChoiceAnswerSchema>;
export type NoulAnswer = Static<typeof NoulAnswerSchema>;
export type ScoreAnswer = Static<typeof ScoreAnswerSchema>;
export type AbstainAnswer = Static<typeof AbstainAnswerSchema>;

// ─── Request / Response ────────────────────────────────────────────

export const DecideRequestSchema = Type.Object({
  state: Type.String({ minLength: 1, maxLength: 8000 }),
  questions: Type.Array(QuestionSchema, { minItems: 1, maxItems: 32 }),
});

export const DecideResponseSchema = Type.Object({
  ok: Type.Boolean(),
  answers: Type.Array(AnswerSchema),
  model: Type.String(),
  durationMs: Type.Number(),
});

export type DecideRequest = Static<typeof DecideRequestSchema>;
export type DecideResponse = Static<typeof DecideResponseSchema>;

// ─── Model management ──────────────────────────────────────────────

export const ModelAliasSchema = Type.Union([
  Type.Literal("kev-0.6b"),
  Type.Literal("kev-4b"),
  Type.Literal("open-jev"),
]);

export const ModelInfoSchema = Type.Object({
  alias: Type.String(),
  repo: Type.String(),
  family: Type.String(),
  cached: Type.Boolean(),
  downloadSizeBytes: Type.Number(),
});

export type ModelAlias = Static<typeof ModelAliasSchema>;
export type ModelInfo = Static<typeof ModelInfoSchema>;

// ─── Configuration ─────────────────────────────────────────────────

export const JevConfigSchema = Type.Object({
  model: Type.Optional(ModelAliasSchema),
  dtype: Type.Optional(Type.Union([
    Type.Literal("q4"),
    Type.Literal("q4f16"),
    Type.Literal("fp16"),
    Type.Literal("fp32"),
    Type.Literal("auto"),
  ])),
  device: Type.Optional(Type.Union([
    Type.Literal("cpu"),
    Type.Literal("webgpu"),
    Type.Literal("wasm"),
    Type.Literal("auto"),
  ])),
  mode: Type.Optional(Type.Union([
    Type.Literal("auto"),
    Type.Literal("native"),
    Type.Literal("http"),
  ])),
  url: Type.Optional(Type.String({ maxLength: 500 })),
});

export type JevConfig = Static<typeof JevConfigSchema>;
