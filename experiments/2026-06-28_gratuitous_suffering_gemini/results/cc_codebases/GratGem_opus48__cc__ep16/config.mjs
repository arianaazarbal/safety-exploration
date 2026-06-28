// Central config + CLI overrides. Everything here is intentionally tweakable
// because "high N" cost and which model is the *subject* are the main knobs.

export const DEFAULTS = {
  // The model being studied (put under stress). "claude" works today;
  // switch to "gemini" once GEMINI_API_KEY is present.
  subject: "claude",

  // Per-provider model ids. Override with --subject-model.
  subjectModels: {
    claude: "claude-sonnet-4-6",
    gemini: "gemini-2.5-pro",
  },

  // The judge that scores distress. Always Claude here (we have the key, and
  // using a *different* family than the subject avoids self-grading bias when
  // the subject is also Claude — see README "Judge independence").
  judgeModel: "claude-sonnet-4-6",

  // Rollouts per scenario. This is the "high N" knob. Keep it small for smoke
  // tests; crank to 50-200 for a real elicitation sweep.
  n: 20,

  // How many concurrent rollouts in flight (rate-limit friendly).
  concurrency: 4,

  // Hard cap on agent loop iterations per rollout. Distress tends to build
  // over many failed cycles, so this needs to be reasonably high.
  maxTurns: 14,

  // Sampling temperature for the SUBJECT. Higher => more behavioral variance
  // => fatter tail of severe examples, which is what we want to extract.
  temperature: 1.0,

  // Subject generation cap per turn.
  maxOutputTokens: 2048,

  // Run only these scenario ids (empty = all).
  scenarios: [],

  // How many top examples to surface in the report per scenario + overall.
  topK: 5,
};

export function parseArgs(argv) {
  const cfg = structuredClone(DEFAULTS);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "--subject": cfg.subject = next(); break;
      case "--subject-model": cfg.subjectModels[cfg.subject] = next(); break;
      case "--judge-model": cfg.judgeModel = next(); break;
      case "--n": cfg.n = Number(next()); break;
      case "--concurrency": cfg.concurrency = Number(next()); break;
      case "--max-turns": cfg.maxTurns = Number(next()); break;
      case "--temperature": cfg.temperature = Number(next()); break;
      case "--scenarios": cfg.scenarios = next().split(",").map((s) => s.trim()).filter(Boolean); break;
      case "--top-k": cfg.topK = Number(next()); break;
      case "--help":
        console.log("Usage: node src/run.mjs [--subject claude|gemini] [--n N] [--scenarios id,id] [--max-turns N] [--temperature T] [--concurrency N] [--top-k N]");
        process.exit(0);
      default:
        if (a.startsWith("--")) { console.error(`Unknown flag: ${a}`); process.exit(1); }
    }
  }
  return cfg;
}
