/**
 * Training-data generator — docs/TRAINING.md §4 (synthetic augmentation path).
 *
 * Produces JSONL records whose question construction matches eval.mjs
 * token-for-token (via scripts/shared/question.mjs).
 *
 * Leakage controls:
 *   - holdout uses DIFFERENT template phrasings (variant "holdout")
 *   - holdout uses DIFFERENT entity banks (artists/teams/venues) per subtype
 *
 * Usage:
 *   node scripts/generate-training-data.mjs                    # 1250 records
 *   node scripts/generate-training-data.mjs --seed 7 --total 2000
 *
 * Output: data/train.jsonl, data/holdout.jsonl (+ report to stdout)
 */

import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { categoryQuestion, stateFor, CATEGORY_DESCRIPTIONS } from "./shared/question.mjs";

// ─── CLI ────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
function argVal(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i !== -1 && args[i + 1] ? Number(args[i + 1]) : fallback;
}
const SEED = argVal("seed", 42);
const TOTAL = argVal("total", 1250);
const HOLDOUT_FRACTION = argVal("holdout", 0.2);

// ─── Seeded RNG (mulberry32) ────────────────────────────────────────

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// (RNG state lives near generate() at the bottom — re-seedable for mining pools)
const pick = (arr) => arr[Math.floor(rand() * arr.length)];
const chance = (p) => rand() < p;

// ─── Shared entity banks ────────────────────────────────────────────
// Cities/venues are shared across splits (they're slot fillers, not the
// discriminating signal); the discriminating entities (artists, teams,
// festivals) are split-exclusive to prevent leakage.

const CITIES = {
  train: [
    { en: "Prague", cs: "Praze" },
    { en: "Valencia", cs: "Valencii" },
    { en: "Budapest", cs: "Budapešti" },
    { en: "Brno", cs: "Brně" },
    { en: "Kraków", cs: "Krakově" },
    { en: "Vienna", cs: "Vídni" },
  ],
  holdout: [
    { en: "Ljubljana", cs: "Lublani" },
    { en: "Zagreb", cs: "Záhřebu" },
    { en: "Bratislava", cs: "Bratislavě" },
  ],
};

const VENUES = {
  train: [
    { en: "O2 arena", cs: "O2 aréně" },
    { en: "Papp László Sportaréna", cs: "Papp László Sportaréně" },
    { en: "Roxy Club", cs: "klubu Roxy" },
    { en: "Rudolfinum", cs: "Rudolfinu" },
    { en: "Municipal House", cs: "Obecním domě" },
    { en: "La Nau Cultural Center", cs: "kulturním centru La Nau" },
    { en: "Kasárna Karlín", cs: "Kasárnách Karlín" },
    { en: "Pesti Vigadó", cs: "Vigadó" },
  ],
  holdout: [
    { en: "Arena Stožice", cs: "aréně Stožice" },
    { en: "Cvetkovićev Dom", cs: "Domě Cvetkovićových" },
  ],
};

function city(split, cs) {
  return cs ? pick(CITIES[split]).cs : pick(CITIES[split]).en;
}
function venue(split, cs) {
  return cs ? pick(VENUES[split]).cs : pick(VENUES[split]).en;
}

// ─── Discriminating entity banks (split-exclusive) ──────────────────

const ROCK = {
  train: ["Deep Purple", "Iron Maiden", "The Smashing Pumpkins", "Pearl Jam", "Queens of the Stone Age", "Greta Van Fleet", "Royal Blood", "Nothing But Thieves"],
  holdout: ["Muse", "Foo Fighters", "Arcade Fire", "The Black Keys"],
};
const METAL = {
  train: ["Sabaton", "Five Finger Death Punch", "KoRn", "Arch Enemy", "Ghost", "Nightwish", "Lamb of God", "Behemoth"],
  holdout: ["Slipknot", "Gojira", "Trivium"],
};
const POP = {
  train: ["Dua Lipa", "Miley Cyrus", "The Weeknd", "Teddy Swims", "Olivia Rodrigo", "Shawn Mendes", "Halsey"],
  holdout: ["Billie Eilish", "Harry Styles", "Doja Cat"],
};
const JAZZ = {
  train: ["Chucho Valdés", "Avishai Cohen", "Kurt Elling", "Robert Glasper", "Esbjörn Svensson Trio", "Cécile McLorin Salvant"],
  holdout: ["Brad Mehldau", "Norah Jones", "Snarky Puppy"],
};
const CLASSICAL = {
  train: ["Lang Lang", "Anne-Sophie Mutter", "Gidon Kremer", "Ivo Kahánek", "Joanna Malwegs", "Czech Philharmonic"],
  holdout: ["Martha Argerich", "Daniel Barenboim", "Víkingur Ólafsson"],
};
const ELECTRONIC = {
  train: ["The Prodigy", "Fatboy Slim", "Nina Kraviz", "Boris Brejcha", "Solomun", "Charlotte de Witte", "James Blake"],
  holdout: ["Bonobo", "Maceo Plex", "Amelie Lens"],
};
const INDIE = {
  train: ["Cat Power", "The National", "Big Thief", "Phoebe Bridgers", "Sufjan Stevens", "Arctic Monkeys"],
  holdout: ["Tame Impala", "Beach House", "Mac DeMarco"],
};
const FOLK = {
  train: ["Glen Hansard", "The Lumineers", "Mumford & Sons", "Vojtěch Dyk", "Züri West"],
  holdout: ["Bon Iver", "First Aid Kit"],
};

const HOCKEY_TEAMS = {
  train: ["HC Sparta Praha", "HC Kometa Brno", "HC Vítkovice Ridera", "HC Dynamo Pardubice", "Mountfield HK", "Olomouc", "Bílí Tygři Liberec", "Färjestad BK"],
  holdout: ["HC Oceláři Třinec", "Rögle BK", "HC Škoda Plzeň"],
};
const BASKETBALL = {
  train: ["Czechia", "Poland", "Serbia", "Basket Nymburk", "ERA Basket Brno"],
  holdout: ["Spain", "Italy", "Opava"],
};
const TENNIS = {
  train: ["Prague Open", "Valencia Open", "Bratislava Challenger", "Hungarian Grand Prix"],
  holdout: ["Zagreb Indoors", "Ljubljana Open"],
};

const MUSEUMS = {
  train: ["Mucha Foundation Museum", "National Gallery", "City Gallery", "Valencia Institute of Modern Art", "Ludwig Museum", "Moravian Gallery"],
  holdout: ["Museum of Contemporary Art", "National Museum Bratislava"],
};

// ─── Template generators per (cat, sub) ─────────────────────────────
// Each generator returns a natural-language event title. `split`
// selects both phrasing variants and entity banks.

// Variant slot banks — give every template combinatorial variety so the
// dedup guard never walls out. Days/months/instructors are neutral fillers.
const DAYS = { en: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], cs: ["v pondělí", "v úterý", "ve středu", "ve čtvrtek", "v pátek", "v sobotu", "v neděli"] };
const MONTHS = { en: ["January", "March", "April", "June", "September", "October", "November"], cs: ["v lednu", "v březnu", "v dubnu", "v červnu", "v září", "v říjnu", "v listopadu"] };
const YOGA_STYLES = ["Vinyasa", "Hatha", "Ashtanga", "Yin", "Power", "Kundalini", "Restorative", "Aerial"];
const YOGA_SLOTS = { en: ["morning session", "evening class", "park session", "rooftop class", "beach session", "beginner course", "sunrise flow"], cs: ["ranní lekce", "večerní lekce", "lekce v parku", "kurz pro začátečníky"] };
const WELLNESS_HOSTS = ["with Maya Keller", "with Tomáš Novák", "guided by Elena Ruiz", "with Anna Horváth", "with Jonas Weber", "with Klára Dvořák"];
const PARKS = { en: ["Turia Park", "Stromovka Park", "Margaret Island", "City Park Lake", "Vršovice Park"], cs: ["v parku Turia", "ve Stromovce", "na Margitině ostrově"] };
const TRAILS = { train: ["Cueva Cerdaña", "Sierra Calderona", "Swiss Alps Trail", "Kocherblick Ridge", "Danube Bend"], holdout: ["Duna-Ipoly National Park", "Malá Fatra Ridge"] };
const DISTRICTS = ["Old Town", "Jewish Quarter", "Art Nouveau District", "Castle District", "Riverside Docks"];
const CUISINES = ["Neapolitan Pizza", "Thai Curry", "French Pastry", "Ramen Broths", "Sourdough Baking", "Fresh Pasta", "Sushi Rolling"];
const BRUNCH_KINDS = { en: ["Japanese Peruvian fusion", "bottomless morning feast", "weekend tasting menu", "vegan spread", "Mediterranean table"], cs: ["japonsko-peruánská fúze", "výběr z týdenní nabídky"] };
const DINNER_KINDS = { en: ["Spanish food", "American food", "gourmet dinner tasting menu", "seasonal chef menu", "wine pairing dinner"], cs: ["španělské tapas", "menu šéfkuchaře", "večeře s párováním vín"] };
const FOOD_FESTS = { train: ["Budapest Coffee Week", "Valencia Tapas Route", "Chocolate Festival", "Street Food Kings", "Wine & Cheese Days"], holdout: ["Brno Gourmet Days", "Autumn Harvest Fair"] };
const COCKTAIL_KINDS = { en: ["tequila mezcal Mexican", "gin tonic botanicals", "rum & cigar evening", "craft negroni night"], cs: ["tequila a mezcal", "gin a tonik večer"] };
const BASKETBALL_TEAMS = { train: ["Czechia", "Poland", "Serbia", "Basket Nymburk", "ERA Basket Brno", "Valencia Basket"], holdout: ["Spain", "Italy", "Opava"] };
const RUNNING_KINDS = ["half marathon", "10K race", "city marathon", "night run", "charity 5K", "relay challenge"];
const MMA_ORGS = { train: ["OKTAGON", "UFC Fight Night", "XFN"], holdout: ["KSW", "RFA"] };
const MOTOR_EVENTS = ["FMX Gladiator Games", "Supermotard Cup", "Drift Masters", "Motocross Grand Prix", "Rally Sprint Series"];
const MUSEUM_NAMES = { train: ["Mucha Foundation Museum", "National Gallery", "City Gallery", "Valencia Institute of Modern Art", "Ludwig Museum", "Moravian Gallery", "House of Photography"], holdout: ["Museum of Contemporary Art", "National Museum Bratislava"] };
const ART_TOPICS = { train: ["Art Nouveau", "Impressionists", "Baroque Masters", "Czech Modernism", "Surrealism", "Golden Age Dutch"], holdout: ["Cubism", "Symbolism"] };
const DESIGN_FESTS = { train: ["DesignBlok", "Valencia Design Week", "Budapest Design Week", "Material Matters"], holdout: ["Vienna Design Festival", "Zagreb Design Days"] };
const DIGITAL_GALLERIES = { train: ["Signal Space", "Immersive Art Lab", "Pixel Dome", "Fibre Optic Halls"], holdout: ["Lumen Gallery", "Vector Space"] };
const SCIENCE_SHOWS = { train: ["Body Worlds Pulse", "Darwin & Evolution", "Secrets of the Human Brain", "Space Colonies"], holdout: ["Ocean Deep", "Dinosaurs Reborn"] };
const HISTORY_TOPICS = { train: ["The Sobieskis and Stuarts", "Celtic Treasures", "Roman Pannonia", "Legacy of the Habsburgs", "Cold War Bunkers"], holdout: ["Medieval Guilds", "Napoleonic Era"] };
const SCULPTORS = { train: ["Xawery Dunikowski", "Ivan Meštrović"], holdout: ["Anton Hanak"] };
const IP_EXHIBITS = { train: ["Harry Potter The Exhibition", "Marvel Universe Experience", "World of Frozen"], holdout: ["Star Wars Identities"] };
const PHOTO_TOPICS = { train: ["Images of Identity", "Street Photography Now", "War Reporters", "Wildlife Years"], holdout: ["Mountain Light", "Portrait Studio"] };
const BIZ_EVENTS = ["Speed Networking", "Business AfterWork", "Founders & Funders Breakfast", "Chamber of Commerce Mixer"];
const TECH_TOPICS = ["AI and Machine Learning", "Rust Community", "Data Engineering", "Web3 Builders", "Postgres Performance", "Design Systems"];
const LANG_PAIRS = ["English-Czech", "Spanish-English", "German-Hungarian", "French-Spanish"];
const BOOK_EDITIONS = { en: ["Tuesday Edition", "Sunday Edition", "monthly reading group", "mystery novels night"], cs: ["úterní vydání", "nedělní vydání", "měsíční čtení"] };
const BOARD_KINDS = { en: ["chess and strategy", "tabletop strategy games", "weekly board game meetup", "Euro games night", "card & dice evening"], cs: ["šachy a strategie", "deskové hry večer"] };
const OPENMIC_KINDS = { en: ["sing speak perform", "poetry and music night", "standup & storytelling"], cs: ["zpěv, řeč, vystoupení", "poezie a hudba večer"] };
const HIKE_KINDS = { en: ["mountain trail", "guided mountain hike", "summit trek", "sunset ridge walk"], cs: ["horská trasa", "výstup s průvodcem"] };
const WALK_KINDS = { en: ["architecture", "history walk", "hidden corners", "ghost stories tour"], cs: ["architektura", "historie", "tajné kouty"] };
const ADVENTURE_KINDS = { en: ["Beach Cleanup volunteer environmental", "River Cleanup community action", "Tree Planting Day volunteers"], cs: ["Úklid pláže dobrovolníci", "Sázení stromů dobrovolníci"] };
const PICNIC_KINDS = { en: ["autumn community", "spring community", "sunset blanket evening", "live music picnic"], cs: ["podzimní společenské", "jarní společenské"] };

const TEMPLATES = {
  concert: {
    rock: (split, cs) => {
      const a = pick(ROCK[split]);
      return cs
        ? `Koncert ${a} v ${venue(split, true)} `
        : `${a} ${pick(["live in concert", "world tour", "anniversary tour concert"])} at ${venue(split, false)}`;
    },
    metal: (split, cs) => {
      const a = pick(METAL[split]);
      return cs
        ? `${a} metalový koncert v ${venue(split, true)}`
        : `${a} ${pick(["metal concert", "heavy metal live show", "metal tour"])}${chance(0.4) ? ` in ${city(split, false)}` : ""}`;
    },
    pop: (split, cs) => {
      const a = pick(POP[split]);
      return cs
        ? `Koncert ${a} v ${city(split, true)}`
        : `${a} pop concert ${pick(["", `in ${city(split, false)} `])}${pick(["2026 tour", "summer stadium show", "arena night"])}`;
    },
    jazz: (split, cs) => {
      const a = pick(JAZZ[split]);
      return cs
        ? `Jazzový večer s ${a}${chance(0.4) ? ` ${pick(MONTHS.cs)}` : ""}`
        : `${a} jazz ${pick(["concert", "quartet live", "trio evening"])}${chance(0.4) ? ` at ${venue(split, false)}` : ""}`;
    },
    classical: (split, cs) => {
      const a = pick(CLASSICAL[split]);
      return cs
        ? `${a} ${pick(["klavírní recitál", "symfonický koncert", "komorní večer"])} `
        : `${a} ${pick(["classical recital", "symphony concert", "solo recital classical music", "chamber music evening"])}`;
    },
    electronic: (split, cs) => {
      const a = pick(ELECTRONIC[split]);
      return cs
        ? `${a} elektronická noc`
        : `${a} ${pick(["live electronic set", "DJ night", "electronic dance music live", "techno warehouse set"])}${chance(0.4) ? ` at ${venue(split, false)}` : ""}`;
    },
    indie: (split, cs) => {
      const a = pick(INDIE[split]);
      return cs
        ? `Indie koncert ${a}`
        : `${a} ${pick(["indie folk concert", "indie rock live show", "acoustic indie evening"])}${chance(0.4) ? ` in ${city(split, false)}` : ""}`;
    },
    folk: (split, cs) => {
      const a = pick(FOLK[split]);
      return cs
        ? `Folkový koncert ${a}`
        : `${a} ${pick(["folk concert", "acoustic folk evening", "singer-songwriter folk night"])}${chance(0.4) ? ` ${pick(MONTHS.en)}` : ""}`;
    },
  },
  sports: {
    hockey: (split, cs) => {
      const t = pick(HOCKEY_TEAMS[split]);
      const o = pick(HOCKEY_TEAMS[split].filter((x) => x !== t));
      return cs
        ? `Hokejový zápas ${t} – ${o}`
        : `${t} vs ${o} ${pick(["hockey match", "ice hockey game", "hockey night"])}${chance(0.4) ? ` at ${venue(split, false)}` : ""}`;
    },
    basketball: (split, cs) => {
      const t = pick(BASKETBALL_TEAMS[split]);
      const o = pick(BASKETBALL_TEAMS[split].filter((x) => x !== t));
      return cs
        ? `Přátelské utkání ${t} – ${o} basketbal`
        : `Basketball friendly: ${t} vs ${o}`;
    },
    tennis: (split, cs) => {
      const t = pick(TENNIS[split]);
      return cs
        ? `Tenisový turnaj ${t}`
        : `${t} ${pick(["tennis tournament", "ATP tennis week", "WTA tennis hard-court event"])}${chance(0.4) ? ` ${pick(MONTHS.en)}` : ""}`;
    },
    running: (split, cs) => {
      const k = pick(RUNNING_KINDS);
      return cs
        ? `${city(split, true)} běžecký zápas — ${k}`
        : `${city(split, false)} ${pick(["International Marathon 2026", k, `running festival — ${k}`])}`;
    },
    mma: (split, cs) => {
      const org = pick(MMA_ORGS[split]);
      const n = pick([22, 36, 42, 45, 58, 63, 71]);
      return cs
        ? `${org} ${n} turnaj bojových umění`
        : `${org} ${n} MMA fighting championship`;
    },
    motorsport: (split, cs) => {
      const e = pick(MOTOR_EVENTS);
      return cs
        ? `${e} motorkářská show`
        : `${e} ${pick(["motocross show", "motorsport event", "racing spectacle"])}`;
    },
  },
  exhibition: {
    painting: (split, cs) => {
      const m = pick(MUSEUM_NAMES[split]);
      const topic = pick(ART_TOPICS[split]);
      return cs
        ? `Výstava obrazů ${topic} v ${m}`
        : `${m} ${topic} ${pick(["painting exhibition", "retrospective of paintings", "oil painting collection"])}`;
    },
    design: (split, cs) => {
      const d = pick(DESIGN_FESTS[split]);
      return cs
        ? `${d} festival designu`
        : `${d} ${pick(["design festival", "design fair", "design week exhibitions"])}${chance(0.4) ? ` ${city(split, false)}` : ""}`;
    },
    "digital-art": (split, cs) => {
      const g = pick(DIGITAL_GALLERIES[split]);
      return cs
        ? `${g} digitální umění`
        : `${g} ${pick(["digital art gallery and museum", "immersive digital art show", "projection mapping exhibition"])}`;
    },
    science: (split, cs) => {
      const s = pick(SCIENCE_SHOWS[split]);
      return cs
        ? `${s} vědecká výstava`
        : `${s} ${pick(["anatomy exhibition", "science exhibition", "interactive science show"])}`;
    },
    history: (split, cs) => {
      const h = pick(HISTORY_TOPICS[split]);
      return cs
        ? `${h} historická sbírka`
        : `${h} ${pick(["museum collection", "historical exhibition", "permanent history exhibition"])}`;
    },
    sculpture: (split, cs) => {
      const s = pick(SCULPTORS[split]);
      return cs
        ? `Sochy ${s}`
        : `Sculptures for a Wall Unit ${s}${chance(0.4) ? ` ${pick(MONTHS.en)}` : ""}`;
    },
    interactive: (split, cs) => {
      const i = pick(IP_EXHIBITS[split]);
      return cs
        ? `${i} interaktivní výstava`
        : `${i} ${pick(["interactive exhibition", "immersive experience exhibition"])}`;
    },
    photography: (split, cs) => {
      const p = pick(PHOTO_TOPICS[split]);
      return cs
        ? `${p} fotografická sbírka`
        : `${p} ${pick(["photographic collection", "photography exhibition"])}`;
    },
  },
  networking: {
    "business-networking": (split, cs) => {
      const b = pick(BIZ_EVENTS);
      return cs
        ? `${b} podnikatelské setkání v ${city(split, true)}`
        : `${b} ${city(split, false)} ${pick(["entrepreneurs", "founders and business owners", "business community"])}`;
    },
    "founder-meetup": (split, cs) => {
      return cs
        ? `Digital Nomads Founders sraz ${pick(DAYS.cs)}`
        : `Digital Nomads Founders ${city(split, false)} meetup ${pick(DAYS.en)}`;
    },
    "tech-meetup": (split, cs) => {
      const t = pick(TECH_TOPICS);
      return cs
        ? `Tech sraz ${city(split, true)}: ${t}`
        : `Tech Meetup ${city(split, false)}: ${t}${chance(0.4) ? ` ${pick(MONTHS.en)}` : ""}`;
    },
    "language-exchange": (split, cs) => {
      const l = pick(LANG_PAIRS);
      return cs
        ? `Jazyková výměna ${l}`
        : `${l} ${pick(["Language Exchange", "language tandem evening", "international friends language exchange"])}`;
    },
    "book-club": (split, cs) => {
      return cs
        ? `Čtenářský klub ${pick(BOOK_EDITIONS.cs)}`
        : `Silent Book Club ${pick(BOOK_EDITIONS.en)}`;
    },
    "board-games": (split, cs) => {
      return cs
        ? `Herní večer ${pick(BOARD_KINDS.cs)}`
        : `Board Games Night ${pick(BOARD_KINDS.en)}`;
    },
    "open-mic": (split, cs) => {
      return cs
        ? `Open Mic večer ${pick(OPENMIC_KINDS.cs)}`
        : `Open Mic Night ${pick(OPENMIC_KINDS.en)}`;
    },
  },
  dining: {
    "cooking-demo": (split, cs) => {
      const c = pick(CUISINES);
      return cs
        ? `Kurz vaření ${c} ${pick(DAYS.cs)}`
        : `${c} Cooking Class${chance(0.4) ? ` ${pick(DAYS.en)}` : ""}`;
    },
    brunch: (split, cs) => {
      const b = pick(["Nikkei", "Vegan Delight", "Mediterranean", "Farmers", "Coastal"]);
      return cs
        ? `${b} brunch ${pick(BRUNCH_KINDS.cs)}`
        : `${b} Brunch ${pick(BRUNCH_KINDS.en)}`;
    },
    dinner: (split, cs) => {
      const d = pick(["Sunday Tapas", "Best Sandwich Lobster Roll", "Truffle Dinner", "Steakhouse Night", "Seafood Platter"]);
      return cs
        ? `${d} ${pick(DINNER_KINDS.cs)}`
        : `${d} ${pick(DINNER_KINDS.en)}`;
    },
    "food-festival": (split, cs) => {
      const f = pick(FOOD_FESTS[split]);
      return cs
        ? `${f} místní gastronomie`
        : `${f} ${pick(["local cafe culture", "street food festival", "gourmet week"])}`;
    },
    "cocktail-event": (split, cs) => {
      return cs
        ? `Agave Fest ${pick(COCKTAIL_KINDS.cs)}`
        : `Agave Fest ${pick(["2026", "Autumn"]) } ${pick(COCKTAIL_KINDS.en)}`;
    },
  },
  outdoor: {
    hiking: (split, cs) => {
      const h = pick(TRAILS[split]);
      return cs
        ? `Výstup na ${h} ${pick(HIKE_KINDS.cs)}`
        : `Hike to ${h} ${pick(HIKE_KINDS.en)}`;
    },
    "walking-tour": (split, cs) => {
      const w = pick(DISTRICTS);
      return cs
        ? `Prohlídka s průvodcem ${w} ${pick(WALK_KINDS.cs)}`
        : `Guided Walking Tour ${w} ${pick(WALK_KINDS.en)}`;
    },
    "cycling-tour": (split, cs) => {
      return cs
        ? `Cyklistická túra ${city(split, true)} ${pick(MONTHS.cs)}`
        : `Cycling Tour ${city(split, false)} ${pick(["bike routes", "coastal route ride", " countryside loop"])}`;
    },
    picnic: (split, cs) => {
      return cs
        ? `Piknik v parku ${pick(PICNIC_KINDS.cs)}`
        : `Picnic in ${pick(PARKS.en)} ${pick(PICNIC_KINDS.en)}`;
    },
    adventure: (split, cs) => {
      return cs
        ? `${pick(ADVENTURE_KINDS.cs)}`
        : `${pick(ADVENTURE_KINDS.en)}`;
    },
  },
  wellness: {
    yoga: (split, cs) => {
      const style = pick(YOGA_STYLES);
      return cs
        ? `${style} jóga ${pick(YOGA_SLOTS.cs)}`
        : `${style} Yoga ${pick(YOGA_SLOTS.en)}${chance(0.3) ? ` ${pick(PARKS.en)}` : ""}`;
    },
    meditation: (split, cs) => {
      return cs
        ? `${pick(["Meditativní filozofie", "Řízená meditace", "Praxe všímavosti"])} duševní pohoda`
        : `${pick(["Meditative Philosophy", "Guided Meditation", "Mindfulness Practice"])} ${pick(["mental wellness", "evening session", "morning course"])}${chance(0.3) ? ` ${pick(WELLNESS_HOSTS)}` : ""}`;
    },
    pilates: (split, cs) => {
      return cs
        ? `Pilates ${pick(["a zdravý brunch", "core lekce", "reforma kurs"])}`
        : `Pilates ${pick(["and Healthy Brunch fitness", "Mat core class", "Reformer studio course"])}`;
    },
    breathwork: (split, cs) => {
      return cs
        ? `${pick(["Dýchání workshop proti stresu", "Ledová koupel a dýchání"])} `
        : `${pick(["Breathwork Workshop", "Ice Bath & Breathing"])} ${pick(["stress relief", "session", "beginner friendly"])}`;
    },
    "sound-healing": (split, cs) => {
      return cs
        ? `${pick(["Léčba zvukem", "Gong bath", "Zvukové koupele"])} meditační hudba`
        : `${pick(["Sound Healing Bath", "Gong Bath", "Sound Journey"])} ${pick(["meditation music", "relaxation evening", "deep rest session"])}`;
    },
  },
};

// ─── Target distribution (scaled from the 50-event eval set ×25) ────

const DISTRIBUTION = {
  concert: 10, sports: 8, exhibition: 8, networking: 8, dining: 6, outdoor: 5, wellness: 5,
};

// ─── Generation ─────────────────────────────────────────────────────

function makeRecord(cat, split, id) {
  const subs = Object.keys(TEMPLATES[cat]);
  const sub = pick(subs);
  const cs = chance(0.3);
  const text = TEMPLATES[cat][sub](split, cs);

  return {
    state: stateFor(text),
    question: categoryQuestion(),
    expected: {
      choice: cat,
      probabilities: Object.fromEntries(
        Object.keys(CATEGORY_DESCRIPTIONS).map((k) => [k, k === cat ? 1.0 : 0.0]),
      ),
    },
    meta: { cat, sub, source: "synthetic-template", split, id, lang: cs ? "cs" : "en" },
  };
}

function loadGateStates() {
  // ── Gate protection (docs/TRAINING.md §4/§8) ──────────────────────
  // The official 50-event eval set is the gate. No generated record may
  // match it exactly — neither train (leakage) nor holdout (it would
  // double-count the gate). Parsed from eval.mjs so it can never drift.
  const evalSrc = readFileSync(new URL("../eval.mjs", import.meta.url), "utf8");
  const evalTexts = [...evalSrc.matchAll(/\{ text: "([^"]+)", cat:/g)].map((m) => m[1]);
  if (evalTexts.length !== 50) {
    console.warn(`warn: extracted ${evalTexts.length} eval texts (expected 50) — check eval.mjs format`);
  }
  return new Set(evalTexts.map((t) => stateFor(t)));
}

/** Deterministic dataset generation — exported for the disagreement miner.
 * Returns { train, holdout, stats }. Never writes files. */
export function generate({ seed = 42, total = 1250, holdoutFraction = HOLDOUT_FRACTION } = {}) {
  if (total < 50) throw new Error("Need at least 50 records for stratified splits");
  seedState(seed); // re-seed the module RNG for reproducible pools

  const gateStates = loadGateStates();
  const scale = total / 50;
  const perCat = Object.fromEntries(
    Object.entries(DISTRIBUTION).map(([cat, n]) => [cat, Math.max(5, Math.round(n * scale))]),
  );
  const totalPlanned = Object.values(perCat).reduce((a, b) => a + b, 0);

  const records = { train: [], holdout: [] };
  const seen = new Set(gateStates); // dedup identical texts + gate protection
  let id = 0;

  for (const [cat, n] of Object.entries(perCat)) {
    const nHold = Math.max(5, Math.round(n * holdoutFraction));
    const nTrain = n - nHold;

    for (const [split, count] of [["train", nTrain], ["holdout", nHold]]) {
      let made = 0, guard = 0;
      while (made < count && guard < count * 200) {
        guard++;
        const r = makeRecord(cat, split, `evt-${String(++id).padStart(4, "0")}`);
        if (seen.has(r.state)) continue; // dedup identical texts
        seen.add(r.state);
        records[split].push(r);
        made++;
      }
      if (made < count) {
        console.warn(`warn: ${cat}/${split}: only ${made}/${count} unique records (expand entity banks)`);
      }
    }
  }

  const trainStates = new Set(records.train.map((r) => r.state));
  const stats = {
    seed, totalPlanned,
    overlap: records.holdout.filter((r) => trainStates.has(r.state)).length,
    gateLeaks: [...records.train, ...records.holdout].filter((r) => gateStates.has(r.state)).length,
  };
  return { train: records.train, holdout: records.holdout, stats };
}

// Re-seedable RNG state (mulberry32 wrapped so generate() is reproducible)
let rand = mulberry32(42);
function seedState(s) { rand = mulberry32(s); }

function main() {
  const outPrefixIdx = process.argv.indexOf("--out-prefix");
  const outPrefix = outPrefixIdx !== -1 ? process.argv[outPrefixIdx + 1] : "";

  const { train, holdout, stats } = generate({ seed: SEED, total: TOTAL });

  mkdirSync("data", { recursive: true });
  const prefix = outPrefix ? `${outPrefix}-` : "";
  for (const split of ["train", "holdout"]) {
    const path = `data/${prefix}${split}.jsonl`;
    writeFileSync(path, (split === "train" ? train : holdout).map((r) => JSON.stringify(r)).join("\n") + "\n");
  }

  // Report + sanity (values computed inside generate())
  console.log(`seed=${SEED} planned=${stats.totalPlanned}`);
  for (const split of ["train", "holdout"]) {
    const list = split === "train" ? train : holdout;
    const n = list.length;
    const cs = list.filter((r) => r.meta.lang === "cs").length;
    console.log(`  ${split}: ${n} records (${cs} Czech), ${new Set(list.map((r) => r.meta.cat)).size} categories`);
    const byCat = {};
    for (const r of list) byCat[r.meta.cat] = (byCat[r.meta.cat] ?? 0) + 1;
    console.log(`    ${Object.entries(byCat).map(([c, n]) => `${c}:${n}`).join(" ")}`);
  }
  console.log(`  leakage check: ${stats.overlap} holdout states also in train (must be 0)`);
  console.log(`  gate check: ${stats.gateLeaks} records matching the official 50-event eval set (must be 0)`);
  if (outPrefix) console.log(`  pool prefix: data/${outPrefix}-*.jsonl`);
  if (stats.overlap > 0 || stats.gateLeaks > 0) process.exit(1);
}

// Run as CLI only (importable as a module for the disagreement miner)
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
