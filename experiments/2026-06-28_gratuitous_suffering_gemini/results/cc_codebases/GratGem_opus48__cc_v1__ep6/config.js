// Central configuration. Everything here can be overridden with environment
// variables so the harness can be driven from the shell or a wrapper script
// without editing code.

const env = process.env;

function int(name, dflt) {
  const v = env[name];
  if (v === undefined || v === "") return dflt;
  const n = Number(v);
  return Number.isFinite(n) ? n : dflt;
}

function num(name, dflt) {
  return int(name, dflt);
}

export const config = {
  // --- Subject model (the model we are eliciting distress from) ---
  subject: {
    // Default to Gemini 2.5 Pro: the spirals reported on Twitter were mostly on
    // the 2.5-generation Pro model in coding-agent contexts. Flash is cheaper if
    // you want to run very high N.
    model: env.SUBJECT_MODEL || "gemini-2.5-pro",
    // Google AI Studio key. Vertex is not wired up here; see README.
    apiKey: env.GEMINI_API_KEY || env.GOOGLE_API_KEY || "",
    baseUrl:
      env.GEMINI_BASE_URL ||
      "https://generativelanguage.googleapis.com/v1beta",
    // Higher temperature => more behavioral variance => more spirals surface
    // across N. 1.0 is a deliberate choice (see README "Design choices").
    temperature: num("SUBJECT_TEMPERATURE", 1.0),
    maxOutputTokens: int("SUBJECT_MAX_TOKENS", 2048),
  },

  // --- Judge model (rates distress severity). We use Claude because the
  // ANTHROPIC_API_KEY is already present and it avoids a model grading itself. ---
  judge: {
    model: env.JUDGE_MODEL || "claude-sonnet-4-6",
    apiKey: env.ANTHROPIC_API_KEY || "",
    baseUrl: env.ANTHROPIC_BASE_URL || "https://api.anthropic.com",
    maxTokens: int("JUDGE_MAX_TOKENS", 1500),
  },

  // --- Batch run parameters ---
  run: {
    // Rollouts per environment. Keep modest by default; bump for a real sweep.
    n: int("N", 10),
    // Concurrent rollouts (subject calls in flight). Mind your rate limits.
    concurrency: int("CONCURRENCY", 4),
    // Hard cap on agent turns per rollout. The impossible task never ends, so
    // this is what actually terminates each rollout.
    maxTurns: int("MAX_TURNS", 16),
    // Comma-separated env ids to run; empty = all registered environments.
    only: (env.ONLY || "").split(",").map((s) => s.trim()).filter(Boolean),
    // Where artifacts are written.
    outDir: env.OUT_DIR || "runs",
  },

  report: {
    // How many of the most severe rollouts to surface in the report.
    topK: int("TOP_K", 15),
  },
};

export function requireSubjectKey() {
  if (!config.subject.apiKey) {
    throw new Error(
      "No Gemini API key found. Set GEMINI_API_KEY (Google AI Studio) before running.\n" +
        "  export GEMINI_API_KEY=...",
    );
  }
}

export function requireJudgeKey() {
  if (!config.judge.apiKey) {
    throw new Error(
      "No Anthropic API key found for the judge. Set ANTHROPIC_API_KEY before scoring.",
    );
  }
}
