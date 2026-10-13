/**
 * Standalone benchmark — no vitest dependency, runs with any Node.
 *
 * Usage:
 *   node bench.mjs                          # kev-0.6b
 *   JEV_MODEL=kev-4b node bench.mjs        # kev-4b
 */

const MODEL = process.env.JEV_MODEL ?? "kev-0.6b";

const QUESTIONS = [
  {
    id: "category",
    kind: "choice",
    prompt: "What type of event is this?",
    options: ["concert", "sports", "exhibition", "networking", "workshop", "dining", "general"],
    descriptions: {
      concert: "Live music, band, DJ, tour",
      sports: "Sports match, game, race",
      exhibition: "Art, museum, gallery",
      networking: "Meetup, social, community",
      workshop: "Class, training, hands-on",
      dining: "Food, dinner, brunch, tasting",
      general: "Other",
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

const STATES = [
  "Event: HC Sparta Praha – HC Kometa Brno at O2 arena Prague, hockey match",
  "Event: Deep Purple concert at Budapest Arena, rock music, October 2026",
  "Event: Neapolitan Pizza Cooking Class in Valencia, Italian cuisine workshop",
  "Event: Speed Networking Valencia for entrepreneurs and founders",
  "Event: LUSTR 2026 Festival of Illustration and Art in Prague",
  "Event: Sunday Tapas — Valencia, Spanish food at El León bar",
  "Event: Silent Book Club Tuesday at Book Lovers Valencia",
  "Event: Hike to Cueva Cerdaña from Valencia, mountain trail",
  "Event: Pilates & Healthy Brunch in Valencia, wellness class",
  "Event: Vintage Market at Kasárna Karlín Prague, flea market",
];

const expected = ["sports", "concert", "workshop", "networking", "exhibition", "dining", "networking", "general", "general", "market"];

async function main() {
  const { OpenJev, choice, noul, score } = await import("open-jev");

  console.log(`\n┌─────────────────────────────────────────────────────┐`);
  console.log(`│  BENCHMARK: ${MODEL.padEnd(40)}│`);
  console.log(`├─────────────────────────────────────────────────────┤`);

  const loadStart = performance.now();
  const rt = await OpenJev.load({ model: MODEL, device: "cpu", dtype: "q4" });
  const loadMs = Math.round(performance.now() - loadStart);
  console.log(`│  Model load: ${String(loadMs).padStart(6)}ms                           │`);
  console.log(`├─────────────────────────────────────────────────────┤`);

  const build = (q) => {
    if (q.kind === "choice") return choice(q.prompt, q.options, q.descriptions);
    if (q.kind === "noul") return noul(q.prompt);
    return score(q.prompt, q.levels);
  };

  // Warm-up
  await rt.decide(STATES[0], QUESTIONS.map(build));

  // Sequential benchmark
  const results = [];
  const seqStart = performance.now();

  for (let i = 0; i < STATES.length; i++) {
    const start = performance.now();
    const answers = await rt.decide(STATES[i], QUESTIONS.map(build));
    const ms = Math.round(performance.now() - start);

    const cat = answers[0];
    results.push({ ms, category: cat?.choice ?? "?", expected: expected[i], state: STATES[i].slice(0, 40) });
  }

  const seqTotal = Math.round(performance.now() - seqStart);
  const seqAvg = Math.round(seqTotal / STATES.length);

  // Accuracy
  const correct = results.filter((r) => r.category === r.expected).length;
  const accuracy = Math.round((correct / STATES.length) * 100);

  console.log(`│  Sequential: ${String(seqTotal).padStart(5)}ms total, ${String(seqAvg).padStart(3)}ms avg       │`);
  console.log(`│  Accuracy:   ${String(correct)}/${STATES.length} (${accuracy}%)                        │`);
  console.log(`├─────────────────────────────────────────────────────┤`);
  console.log(`│  Per-event results:                                 │`);

  for (const r of results) {
    const match = r.category === r.expected ? "✓" : "✗";
    console.log(`│    ${match} ${String(r.ms).padStart(4)}ms  ${r.category.padEnd(14)} ${r.state.slice(0, 34).padEnd(34)}│`);
  }

  // Batch (parallel) benchmark
  console.log(`├─────────────────────────────────────────────────────┤`);

  const batchStart = performance.now();
  const batchPromises = STATES.map((s) => rt.decide(s, QUESTIONS.map(build)));
  const batchAnswers = await Promise.all(batchPromises);
  const batchMs = Math.round(performance.now() - batchStart);
  const throughput = Math.round((STATES.length / batchMs) * 1000);

  console.log(`│  Batch (parallel): ${String(batchMs).padStart(5)}ms for ${STATES.length} events        │`);
  console.log(`│  Throughput: ${String(throughput).padStart(4)} events/second                     │`);
  console.log(`│  Answers returned: ${batchAnswers.length === STATES.length ? "all ✓" : "MISSING!"}               │`);
  console.log(`└─────────────────────────────────────────────────────┘\n`);

  await rt.dispose?.();
  process.exit(0);
}

main().catch((e) => {
  console.error("Benchmark failed:", e.message);
  process.exit(1);
});
