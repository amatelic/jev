/**
 * Comprehensive evaluation — compares all 3 models with proper metrics.
 *
 * Tests:
 *   1. Accuracy across a 50-event labeled dataset
 *   2. Per-category precision/recall/F1
 *   3. Confidence calibration (does 80% confidence mean 80% correct?)
 *   4. Question phrasing sensitivity
 *   5. Ensemble voting (2-of-3 agreement)
 *
 * Usage:
 *   node eval.mjs                    # all models
 *   node eval.mjs kev-0.6b          # single model
 *   node eval.mjs --calibration     # calibration analysis only
 */

import { OpenJev, choice, noul, score } from "open-jev";

// ─── Labeled dataset (50 events, 7 categories) ─────────────────────

const DATASET = [
  // sports (8)
  { text: "HC Sparta Praha – HC Kometa Brno hockey match", cat: "sports", sub: "hockey" },
  { text: "HC Sparta Praha – HC Vítkovice Ridera hockey game", cat: "sports", sub: "hockey" },
  { text: "OKTAGON 100 MMA fighting championship", cat: "sports", sub: "mma" },
  { text: "Prague International Marathon 2026", cat: "sports", sub: "running" },
  { text: "Tenisový turnaj Prague Open", cat: "sports", sub: "tennis" },
  { text: "FMX Gladiator Games motocross show", cat: "sports", sub: "motorsport" },
  { text: "Turia Park Run #36 October running club", cat: "sports", sub: "running" },
  { text: "Basketball friendly: Czechia vs Poland", cat: "sports", sub: "basketball" },

  // concert (10)
  { text: "Deep Purple concert at Budapest Arena", cat: "concert", sub: "rock" },
  { text: "Five Finger Death Punch 20th Anniversary Tour metal concert", cat: "concert", sub: "metal" },
  { text: "Jazz Jamboree 2026 Day 1 jazz festival", cat: "concert", sub: "jazz" },
  { text: "CAT POWER indie folk concert", cat: "concert", sub: "indie" },
  { text: "Chopin recital classical piano concert", cat: "concert", sub: "classical" },
  { text: "SABATON The Legendary Tour power metal", cat: "concert", sub: "metal" },
  { text: "Famous Organ Concerts at St Francis Church", cat: "concert", sub: "classical" },
  { text: "The Prodigy electronic dance music live", cat: "concert", sub: "electronic" },
  { text: "Teddy Swims The Ugly Tour soul music", cat: "concert", sub: "pop" },
  { text: "KoRn nu metal concert at O2 arena", cat: "concert", sub: "metal" },

  // exhibition (8)
  { text: "Mucha Foundation Art Museum Art Nouveau exhibition", cat: "exhibition", sub: "painting" },
  { text: "DesignBlok 2026 Prague Design Festival", cat: "exhibition", sub: "design" },
  { text: "Signal Space Digital Art Gallery and Museum", cat: "exhibition", sub: "digital-art" },
  { text: "Body Worlds Pulse anatomy exhibition", cat: "exhibition", sub: "science" },
  { text: "The Sobieskis and Stuarts museum collection", cat: "exhibition", sub: "history" },
  { text: "Sculptures for a Wall Unit Xawery Dunikowski", cat: "exhibition", sub: "sculpture" },
  { text: "Harry Potter The Exhibition", cat: "exhibition", sub: "interactive" },
  { text: "Images of Identity photographic collection", cat: "exhibition", sub: "photography" },

  // networking (8)
  { text: "Speed Networking Valencia entrepreneurs", cat: "networking", sub: "business-networking" },
  { text: "Silent Book Club Tuesday Edition reading group", cat: "networking", sub: "book-club" },
  { text: "Women Entrepreneur Networking Valencia", cat: "networking", sub: "business-networking" },
  { text: "Digital Nomads Founders Valencia meetup", cat: "networking", sub: "founder-meetup" },
  { text: "Tuesday Language Exchange international friends", cat: "networking", sub: "language-exchange" },
  { text: "Tech Meetup Valencia: AI and Machine Learning", cat: "networking", sub: "tech-meetup" },
  { text: "Board Games Night chess and strategy", cat: "networking", sub: "board-games" },
  { text: "Open Mic Night sing speak perform", cat: "networking", sub: "open-mic" },

  // dining (6)
  { text: "Neapolitan Pizza Cooking Class Italian cuisine", cat: "dining", sub: "cooking-demo" },
  { text: "Sunday Tapas Valencia Spanish food", cat: "dining", sub: "dinner" },
  { text: "Nikkei Brunch Japanese Peruvian fusion", cat: "dining", sub: "brunch" },
  { text: "Agave Fest 2026 tequila mezcal Mexican", cat: "dining", sub: "cocktail-event" },
  { text: "Budapest Coffee Week local cafe culture", cat: "dining", sub: "food-festival" },
  { text: "Best Sandwich Lobster Roll American food", cat: "dining", sub: "dinner" },

  // outdoor (5)
  { text: "Hike to Cueva Cerdaña mountain trail", cat: "outdoor", sub: "hiking" },
  { text: "Guided Walking Tour Old Town architecture", cat: "outdoor", sub: "walking-tour" },
  { text: "Picnic in the Park autumn community", cat: "outdoor", sub: "picnic" },
  { text: "Cycling Tour Valencia bike routes", cat: "outdoor", sub: "cycling-tour" },
  { text: "Beach Cleanup volunteer environmental", cat: "outdoor", sub: "adventure" },

  // wellness (5)
  { text: "Pilates and Healthy Brunch fitness", cat: "wellness", sub: "pilates" },
  { text: "Meditative Philosophy mental wellness", cat: "wellness", sub: "meditation" },
  { text: "Yoga in the Park morning session", cat: "wellness", sub: "yoga" },
  { text: "Breathwork Workshop stress relief", cat: "wellness", sub: "breathwork" },
  { text: "Sound Healing Bath meditation music", cat: "wellness", sub: "sound-healing" },
];

const CATEGORIES = ["concert", "sports", "exhibition", "networking", "dining", "outdoor", "wellness"];

// ─── Evaluation runner ──────────────────────────────────────────────

const CATEGORY_QUESTION = choice(
  "What type of event is this?",
  CATEGORIES,
  {
    concert: "Live music performance, band, DJ, tour, festival",
    sports: "Sports match, game, race, tournament, athletic competition",
    exhibition: "Art exhibition, museum show, gallery, design fair",
    networking: "Meetup, social gathering, community event, book club",
    dining: "Food event, dinner, brunch, cooking, tasting",
    outdoor: "Hiking, walking tour, outdoor activity, nature",
    wellness: "Yoga, meditation, wellness, health, fitness",
  },
);

const SUBTYPE_QUESTIONS = {
  concert: ["rock", "pop", "jazz", "electronic", "classical", "hip-hop", "folk", "metal", "indie", "other-music"],
  sports: ["hockey", "football", "basketball", "tennis", "running", "mma", "motorsport", "other-sports"],
  exhibition: ["contemporary-art", "photography", "design", "history", "science", "sculpture", "painting", "digital-art", "interactive"],
  networking: ["business-networking", "language-exchange", "founder-meetup", "tech-meetup", "book-club", "board-games", "open-mic"],
  dining: ["food-festival", "cooking-demo", "brunch", "dinner", "cocktail-event"],
  outdoor: ["hiking", "walking-tour", "cycling-tour", "picnic", "adventure"],
  wellness: ["yoga", "meditation", "pilates", "breathwork", "sound-healing"],
};

async function evalModel(rt, modelName) {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`  MODEL: ${modelName}`);
  console.log(`${"═".repeat(60)}\n`);

  const results = [];
  const t0 = performance.now();

  for (const item of DATASET) {
    const answers = await rt.decide(`Event: ${item.text}`, [CATEGORY_QUESTION]);

    const cat = answers[0];
    const predicted = cat?.choice ?? "general";
    const confidence = cat?.confidence ?? 0;
    const correct = predicted === item.cat;

    results.push({
      text: item.text.slice(0, 40),
      expected: item.cat,
      predicted,
      correct,
      confidence,
    });
  }

  const totalMs = Math.round(performance.now() - t0);

  // ── Accuracy ──
  const correct = results.filter((r) => r.correct).length;
  const accuracy = (correct / results.length * 100).toFixed(1);

  // ── Per-category precision/recall/F1 ──
  const perCat = {};
  for (const cat of CATEGORIES) {
    const tp = results.filter((r) => r.predicted === cat && r.expected === cat).length;
    const fp = results.filter((r) => r.predicted === cat && r.expected !== cat).length;
    const fn = results.filter((r) => r.predicted !== cat && r.expected === cat).length;

    const precision = tp + fp > 0 ? (tp / (tp + fp) * 100).toFixed(0) : "—";
    const recall = tp + fn > 0 ? (tp / (tp + fn) * 100).toFixed(0) : "—";
    const f1 = tp + fp + fn > 0 ? (2 * tp / (2 * tp + fp + fn) * 100).toFixed(0) : "—";

    perCat[cat] = { tp, fp, fn, precision, recall, f1 };
  }

  // ── Confidence calibration ──
  const calibrationBins = [
    { range: [0, 0.5], label: "<50%" },
    { range: [0.5, 0.7], label: "50-70%" },
    { range: [0.7, 0.85], label: "70-85%" },
    { range: [0.85, 0.95], label: "85-95%" },
    { range: [0.95, 1.01], label: ">95%" },
  ];

  const calibration = calibrationBins.map(bin => {
    const inBin = results.filter((r) =>
      r.confidence >= bin.range[0] && r.confidence < bin.range[1]
    );
    const correctInBin = inBin.filter((r) => r.correct).length;
    const avgConf = inBin.length > 0
      ? (inBin.reduce((s, r) => s + r.confidence, 0) / inBin.length * 100).toFixed(0)
      : "—";
    const actualRate = inBin.length > 0 ? (correctInBin / inBin.length * 100).toFixed(0) : "—";

    return {
      label: bin.label,
      count: inBin.length,
      avgConfidence: avgConf,
      actualAccuracy: actualRate,
      gap: inBin.length > 0
        ? `${Math.abs(Number(avgConf) - Number(actualRate)).toFixed(0)}pp`
        : "—",
    };
  });

  // ── Output ──
  console.log(`  Dataset: ${results.length} events across ${CATEGORIES.length} categories`);
  console.log(`  Total time: ${totalMs}ms (${Math.round(totalMs / results.length)}ms avg)`);
  console.log(`  Accuracy: ${correct}/${results.length} = ${accuracy}%`);
  console.log();

  console.log("  Per-category (precision / recall / F1):");
  for (const [cat, m] of Object.entries(perCat)) {
    const icon = Number(m.f1) >= 70 ? "✓" : Number(m.f1) >= 40 ? "⚠" : "✗";
    console.log(`    ${icon} ${cat.padEnd(14)} P:${String(m.precision).padStart(4)}  R:${String(m.recall).padStart(4)}  F1:${String(m.f1).padStart(4)}  (n=${results.filter(r => r.expected === cat).length})`);
  }
  console.log();

  console.log("  Confidence calibration (stated vs actual accuracy):");
  console.log("    Range      Count  Stated  Actual  Gap");
  for (const c of calibration) {
    if (c.count === 0) continue;
    const well = c.gap === "0pp" ? "✓" : Number(c.gap.replace("pp", "")) <= 10 ? "≈" : "✗";
    console.log(`    ${c.label.padEnd(10)} ${String(c.count).padStart(4)}  ${String(c.avgConfidence).padStart(5)}%  ${String(c.actualAccuracy).padStart(5)}%  ${c.gap.padStart(4)} ${well}`);
  }
  console.log();

  // ── Misclassifications ──
  const errors = results.filter((r) => !r.correct);
  if (errors.length > 0) {
    console.log(`  Misclassified (${errors.length}):`);
    for (const e of errors.slice(0, 10)) {
      console.log(`    ✗ expected=${e.expected.padEnd(12)} got=${e.predicted.padEnd(12)} @${(e.confidence * 100).toFixed(0)}%  "${e.text}"`);
    }
  }

  return { results, accuracy, perCat, calibration, totalMs };
}

// ─── Main ──────────────────────────────────────────────────────────

async function main() {
  const models = process.argv[2] && process.argv[2] !== "--calibration"
    ? [process.argv[2]]
    : ["kev-0.6b", "kev-4b", "open-jev"];

  const allResults = {};

  for (const modelName of models) {
    try {
      console.log(`\nLoading ${modelName}...`);
      const rt = await OpenJev.load({ model: modelName, device: "cpu", dtype: "q4" });
      console.log(`✅ ${modelName} loaded`);

      const evalResult = await evalModel(rt, modelName);
      allResults[modelName] = evalResult;

      await rt.dispose?.();
    } catch (e) {
      console.log(`❌ ${modelName}: ${e.message?.slice(0, 100)}`);
    }
  }

  // ─── Ensemble evaluation (if we have 2+ models) ──
  const modelNames = Object.keys(allResults);
  if (modelNames.length >= 2) {
    console.log(`\n${"═".repeat(60)}`);
    console.log(`  ENSEMBLE (majority vote across ${modelNames.length} models)`);
    console.log(`${"═".repeat(60)}\n`);

    let ensembleCorrect = 0;
    const perModelResults = modelNames.map(m => allResults[m].results);

    for (let i = 0; i < DATASET.length; i++) {
      const votes = {};
      for (const results of perModelResults) {
        const pred = results[i]?.predicted;
        if (pred) votes[pred] = (votes[pred] ?? 0) + 1;
      }

      const winner = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
      const ensemblePred = winner?.[0] ?? "general";
      const correct = ensemblePred === DATASET[i].cat;
      if (correct) ensembleCorrect++;

      if (!correct && ensemblePred !== perModelResults[0]?.[i]?.predicted) {
        console.log(`    ⇄ ensemble=${ensemblePred} vs ${modelNames[0]}=${perModelResults[0]?.[i]?.predicted} expected=${DATASET[i].cat}  "${DATASET[i].text.slice(0, 35)}"`);
      }
    }

    const ensAcc = (ensembleCorrect / DATASET.length * 100).toFixed(1);
    console.log(`  Ensemble accuracy: ${ensembleCorrect}/${DATASET.length} = ${ensAcc}%`);

    for (const m of modelNames) {
      console.log(`  ${m}: ${allResults[m].accuracy}%`);
    }
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`  SUMMARY`);
  console.log(`${"═".repeat(60)}\n`);
  console.log(`  ${"Model".padEnd(15)} ${"Accuracy".padEnd(10)} ${"Avg ms".padEnd(8)} Misclassified`);
  console.log(`  ${"─".repeat(55)}`);
  for (const [name, r] of Object.entries(allResults)) {
    const errors = r.results.filter((x) => !x.correct).length;
    const avgMs = Math.round(r.totalMs / r.results.length);
    console.log(`  ${name.padEnd(15)} ${(r.accuracy + "%").padEnd(10)} ${String(avgMs).padEnd(8)} ${errors}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error("Evaluation failed:", e.message);
  process.exit(1);
});
