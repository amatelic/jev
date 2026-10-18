/**
 * Shared question definitions — the single source of truth for the
 * category question used in eval.mjs (via scripts/eval-dataset.mjs), the data generator, and training data — eval.mjs keeps a deliberately frozen inline copy so the gate cannot drift; both must stay token-identical.
 *
 * CRITICAL (docs/TRAINING.md §4): training and inference prompts must
 * match token-for-token, descriptions included. Never edit these strings
 * without re-running `node eval.mjs` baselines.
 */

export const CATEGORIES = [
  "concert",
  "sports",
  "exhibition",
  "networking",
  "dining",
  "outdoor",
  "wellness",
];

export const CATEGORY_DESCRIPTIONS = {
  concert: "Live music performance, band, DJ, tour, festival",
  sports: "Sports match, game, race, tournament, athletic competition",
  exhibition: "Art exhibition, museum show, gallery, design fair",
  networking: "Meetup, social gathering, community event, book club",
  dining: "Food event, dinner, brunch, cooking, tasting",
  outdoor: "Hiking, walking tour, outdoor activity, nature",
  wellness: "Yoga, meditation, wellness, health, fitness",
};

/** The exact category question object used by eval.mjs (plain JSON form). */
export function categoryQuestion() {
  return {
    id: "category",
    kind: "choice",
    prompt: "What type of event is this?",
    options: [...CATEGORIES],
    descriptions: { ...CATEGORY_DESCRIPTIONS },
  };
}

/** eval.mjs state format: `Event: ${text}`. */
export function stateFor(text) {
  return `Event: ${text}`;
}
