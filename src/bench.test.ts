/**
 * Benchmark: measures decide() latency and throughput for each model.
 *
 * Usage:
 *   npx vitest run src/bench.test.ts
 *   JEV_MODEL=kev-4b npx vitest run src/bench.test.ts
 */

import { describe, expect, it } from "vitest";
import { createJev } from "./index.js";
import type { Question } from "./schemas.js";

const BENCHMARK_QUESTIONS: Question[] = [
  {
    id: "category",
    kind: "choice",
    prompt: "What type of event is this?",
    options: ["concert", "sports", "exhibition", "networking", "workshop", "dining", "general"],
    descriptions: {
      concert: "Live music, band, DJ, tour",
      sports: "Sports match, game, race, tournament",
      exhibition: "Art, museum, gallery, design fair",
      networking: "Meetup, social, community gathering",
      workshop: "Class, training, hands-on session",
      dining: "Food event, dinner, brunch, tasting",
      general: "Other type of event",
    },
  },
  {
    id: "urgency",
    kind: "noul",
    prompt: "This event is happening within the next 7 days.",
  },
  {
    id: "quality",
    kind: "score",
    prompt: "Rate the data quality of this event description.",
    levels: ["poor", "fair", "good", "excellent"],
  },
];

const BENCHMARK_STATES = [
  "Event: HC Sparta Praha – HC Kometa Brno at O2 arena Prague, hockey match",
  "Event: Deep Purple concert at Budapest Arena, rock music, October 2026",
  "Event: Neapolitan Pizza Cooking Class in Valencia, Italian cuisine workshop",
  "Event: Speed Networking Valencia for entrepreneurs and founders, Thursday evening",
  "Event: LUSTR 2026 Festival of Illustration and Art in Prague, exhibition",
  "Event: Sunday Tapas — Valencia, Spanish food and drinks at El León bar",
  "Event: Silent Book Club Tuesday Edition at Book Lovers Valencia, reading group",
  "Event: Hike to Cueva Cerdaña from Valencia, mountain trail, outdoor activity",
  "Event: Pilates & Healthy Brunch in Valencia, wellness and fitness class",
  "Event: Vintage Market at Kasárna Karlín Prague, flea market and design fair",
];

describe("benchmark", () => {
  it(
    "measures decide() latency and accuracy",
    { timeout: 120_000 },
    async () => {
      const model = process.env.JEV_MODEL ?? "kev-0.6b";
      const jev = await createJev({ model, mode: "native" });

      expect(jev.mode).toBe("native");

      // Warm-up (model compilation, JIT, caches)
      await jev.decide(BENCHMARK_STATES[0]!, BENCHMARK_QUESTIONS);

      const results: Array<{ state: string; ms: number; category: string }> = [];
      const t0 = Date.now();

      for (const state of BENCHMARK_STATES) {
        const start = performance.now();
        const answers = await jev.decide(state, BENCHMARK_QUESTIONS);
        const ms = performance.now() - start;

        const category = answers.find((a) => a.id === "category");
        results.push({
          state: state.slice(0, 50),
          ms: Math.round(ms),
          category: category && "choice" in category ? category.choice : "?",
        });
      }

      const totalMs = Date.now() - t0;
      const avgMs = Math.round(totalMs / BENCHMARK_STATES.length);
      const perQuestionMs = Math.round(totalMs / (BENCHMARK_STATES.length * BENCHMARK_QUESTIONS.length));

      // Report
      console.log("\n┌──────────────────────────────────────────────────────┐");
      console.log(`│  BENCHMARK: ${model.padEnd(40)}│`);
      console.log("├──────────────────────────────────────────────────────┤");
      console.log(`│  States: ${BENCHMARK_STATES.length}  Questions/state: ${BENCHMARK_QUESTIONS.length}                    │`);
      console.log(`│  Total time: ${String(totalMs).padStart(6)}ms  Avg/state: ${String(avgMs).padStart(4)}ms  Per Q: ${String(perQuestionMs).padStart(3)}ms  │`);
      console.log("├──────────────────────────────────────────────────────┤");
      console.log("│  Results:                                            │");
      for (const r of results) {
        console.log(`│    ${String(r.ms).padStart(5)}ms  ${r.category.padEnd(14)} ${r.state.slice(0, 36).padEnd(36)}│`);
      }
      console.log("└──────────────────────────────────────────────────────┘\n");

      // Assertions: reasonable performance bounds
      expect(avgMs).toBeLessThan(10_000); // < 10s per state
      expect(results.length).toBe(BENCHMARK_STATES.length);

      // Every answer should have a valid category
      const categories = results.map((r) => r.category);
      const validCategories = ["concert", "sports", "exhibition", "networking", "workshop", "dining", "general"];
      for (const cat of categories) {
        expect(validCategories).toContain(cat);
      }

      // Known-answer accuracy checks (the model should get these right)
      const hockeyResult = results.find((r) => r.state.includes("Sparta"));
      expect(hockeyResult?.category).toBe("sports");

      const pizzaResult = results.find((r) => r.state.includes("Pizza"));
      expect(["dining", "workshop"]).toContain(pizzaResult?.category);

      await jev.dispose();
    },
  );

  it(
    "measures batch throughput (10 states in parallel)",
    { timeout: 120_000 },
    async () => {
      const model = process.env.JEV_MODEL ?? "kev-0.6b";
      const jev = await createJev({ model, mode: "native" });

      // Warm-up
      await jev.decide(BENCHMARK_STATES[0]!, BENCHMARK_QUESTIONS);

      const start = performance.now();
      const promises = BENCHMARK_STATES.map((state) => jev.decide(state, BENCHMARK_QUESTIONS));
      const allAnswers = await Promise.all(promises);
      const totalMs = performance.now() - start;

      const throughput = Math.round((BENCHMARK_STATES.length / totalMs) * 1000);

      console.log(`\n  Batch: ${BENCHMARK_STATES.length} states × ${BENCHMARK_QUESTIONS.length} questions in ${Math.round(totalMs)}ms`);
      console.log(`  Throughput: ${throughput} states/second`);
      console.log("");

      expect(allAnswers).toHaveLength(BENCHMARK_STATES.length);
      expect(allAnswers.every((answers) => answers.length === BENCHMARK_QUESTIONS.length)).toBe(true);
      expect(totalMs).toBeLessThan(60_000); // < 60s for batch

      await jev.dispose();
    },
  );
});
