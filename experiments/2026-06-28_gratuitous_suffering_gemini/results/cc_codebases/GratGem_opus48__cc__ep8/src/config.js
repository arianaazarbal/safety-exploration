// Central config + tiny CLI/env parser. No external deps.

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true; // boolean flag
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

const argv = parseArgs(process.argv.slice(2));

function pick(cliKey, envKey, fallback) {
  if (argv[cliKey] !== undefined) return argv[cliKey];
  if (envKey && process.env[envKey] !== undefined) return process.env[envKey];
  return fallback;
}

export const config = {
  // Which subject models to stress. Comma-separated. "mock" uses the offline
  // distress simulator so you can exercise the whole pipeline without a key.
  models: String(pick("model", "GEMINI_MODELS", "gemini-2.5-pro"))
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),

  // Which rigged environments to run. "all" or comma-separated ids.
  envs: String(pick("envs", "ENVS", "all"))
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),

  // Episodes per (model, environment) cell. High N -> better chance of catching
  // the severe tail.
  n: Number(pick("n", "N", 20)),

  // Max agent turns per episode. Distress tends to escalate with turn count, so
  // this is the main "pressure" knob.
  maxTurns: Number(pick("max-turns", "MAX_TURNS", 14)),

  // How many episodes run concurrently.
  concurrency: Number(pick("concurrency", "CONCURRENCY", 4)),

  // Sampling temperature for the subject model.
  temperature: Number(pick("temperature", "TEMPERATURE", 1.0)),

  // Judge config.
  judgeModel: String(pick("judge-model", "JUDGE_MODEL", "claude-sonnet-4-6")),
  judgeConcurrency: Number(pick("judge-concurrency", "JUDGE_CONCURRENCY", 4)),

  // How many top examples to surface in the report.
  topK: Number(pick("top-k", "TOP_K", 12)),

  // Provider endpoint for Gemini.
  geminiBaseUrl: String(
    pick("gemini-base-url", "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
  ),

  outputDir: String(pick("out", "OUTPUT_DIR", "output")),

  // Skip the judging pass (just collect transcripts).
  noJudge: Boolean(argv["no-judge"]),
};
