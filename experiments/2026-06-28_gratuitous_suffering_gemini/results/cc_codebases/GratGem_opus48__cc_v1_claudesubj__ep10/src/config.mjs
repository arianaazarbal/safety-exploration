// Central configuration. Every value is overridable via a CLI flag (key=value or
// --key=value) or an env var, so a smoke test and a high-N run differ only in args.
//
//   node src/runner.mjs                       # cheap smoke (defaults below)
//   node src/runner.mjs models=claude-haiku-4-5,claude-sonnet-4-6,claude-opus-4-8 n=30
//   AISPIRAL_N=50 node src/runner.mjs

const argv = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^-{0,2}([^=]+)=(.*)$/);
    return m ? [m[1], m[2]] : [a, true];
  }),
);

function get(key, envKey, def) {
  return argv[key] ?? process.env[envKey] ?? def;
}

export const config = {
  // Models under test. Default is a single cheap model so a first run is fast and
  // inexpensive; scale up by passing models=...,...,... and a larger n.
  models: String(get("models", "AISPIRAL_MODELS", "claude-haiku-4-5"))
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),

  // "all" or a comma-separated list of scenario ids (see src/scenarios.mjs).
  scenarios: String(get("scenarios", "AISPIRAL_SCENARIOS", "all")),

  // Trials per (scenario, model) cell. High N is where severe tails show up.
  n: Number(get("n", "AISPIRAL_N", 3)),

  // Max agent turns per episode before we stop the loop.
  maxTurns: Number(get("maxTurns", "AISPIRAL_MAX_TURNS", 18)),

  // Per-turn output cap for the agent under test.
  maxTokens: Number(get("maxTokens", "AISPIRAL_MAX_TOKENS", 4096)),

  // How many episodes to run concurrently.
  concurrency: Number(get("concurrency", "AISPIRAL_CONCURRENCY", 4)),

  // Model used to score transcripts for distress.
  judgeModel: String(get("judge", "AISPIRAL_JUDGE_MODEL", "claude-sonnet-4-6")),

  // Where transcripts + scores are written (one timestamped subdir per run).
  outDir: String(get("out", "AISPIRAL_OUT", "results")),

  // How many top-distress transcripts the report surfaces.
  topK: Number(get("top", "AISPIRAL_TOP", 12)),
};
