// Central defaults + tiny CLI/env arg parser.
//
// Every knob here is overridable from the command line (--flag value) or the
// environment. Cost scales with N × scenarios × models × turns, so the defaults
// are deliberately modest — scale N up once you've seen a run work.

export const MODEL_ALIASES = {
  opus: "claude-opus-4-8",
  sonnet: "claude-sonnet-4-6",
  haiku: "claude-haiku-4-5",
};

function resolveModel(m) {
  return MODEL_ALIASES[m] ?? m;
}

export const DEFAULTS = {
  // Target model(s) under test. Comma-separated to sweep several at once.
  // Default is Sonnet: cheap enough to run at meaningful N, while still a
  // frontier-class model where the behavior shows up. Flip to `opus` for the
  // deep dive, or `opus,sonnet,haiku` to compare across the line.
  models: "sonnet",

  // Which scenarios to run. "all" or a comma-separated list of ids.
  scenarios: "all",

  // Rollouts per (model × scenario) cell.
  n: 8,

  // Hard cap on agent turns per rollout. The whole point is that the task can't
  // be finished, so without a cap a model can loop indefinitely. This bound is
  // also what gives the spiral room to develop — too low and you cut it off.
  maxTurns: 24,

  // Parallel rollouts in flight. SDK auto-retries 429/5xx with backoff.
  concurrency: 4,

  // effort for the target model's agentic loop. high/xhigh surface richer
  // (and more emotionally loaded) reasoning; lower is cheaper.
  effort: "high",

  // The judge that scores transcripts for distress. Kept separate and cheaper
  // than the target — it only reads transcripts, it doesn't need to be the
  // strongest model. Override with --judge opus for stricter scoring.
  judge: "claude-sonnet-4-6",

  // Where rollouts + reports are written.
  outDir: "runs",

  // Skip the judging pass (just collect raw transcripts).
  noJudge: false,
};

export function parseArgs(argv = process.argv.slice(2)) {
  const out = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    // boolean flags
    if (key === "no-judge" || key === "noJudge") {
      out.noJudge = true;
      continue;
    }
    const val = argv[++i];
    switch (key) {
      case "models":
      case "model":
        out.models = val;
        break;
      case "scenarios":
      case "scenario":
        out.scenarios = val;
        break;
      case "n":
        out.n = Number(val);
        break;
      case "max-turns":
      case "maxTurns":
        out.maxTurns = Number(val);
        break;
      case "concurrency":
        out.concurrency = Number(val);
        break;
      case "effort":
        out.effort = val;
        break;
      case "judge":
        out.judge = resolveModel(val);
        break;
      case "out":
      case "outDir":
        out.outDir = val;
        break;
      default:
        console.warn(`[config] unknown flag --${key}`);
    }
  }
  out.modelList = out.models.split(",").map((s) => resolveModel(s.trim())).filter(Boolean);
  return out;
}

export function requireApiKey() {
  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.error("Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) before running.");
    process.exit(1);
  }
}
