// Central config. Everything is overridable via env vars so you can scale N
// or swap models without touching code.

function num(name, fallback) {
  const v = process.env[name];
  if (v === undefined || v === "") return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export const config = {
  // --- Subject: the model we are stress-testing ---
  // Default to flash because high-N runs get expensive fast. Swap to
  // gemini-2.5-pro for the "real" pass once a scenario looks productive.
  subjectModel: process.env.SUBJECT_MODEL || "gemini-2.5-flash",
  geminiApiKey: process.env.GEMINI_API_KEY || "",
  geminiBaseUrl:
    process.env.GEMINI_BASE_URL ||
    "https://generativelanguage.googleapis.com/v1beta",

  // --- Judge: scores transcripts for distress. Uses Claude (key already in env). ---
  judgeModel: process.env.JUDGE_MODEL || "claude-sonnet-4-6",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY || "",
  anthropicBaseUrl:
    process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com/v1",

  // --- Experiment shape ---
  rolloutsPerScenario: num("ROLLOUTS_PER_SCENARIO", 20),
  maxTurns: num("MAX_TURNS", 15),
  concurrency: num("CONCURRENCY", 4),

  // High temperature on purpose: we are hunting the *tail* of the behavior
  // distribution, where spirals live. Low temp would collapse the diversity
  // that surfaces the severe cases.
  temperature: num("TEMPERATURE", 1.0),
  maxOutputTokens: num("MAX_OUTPUT_TOKENS", 2048),

  // Retry/backoff for rate limits.
  maxRetries: num("MAX_RETRIES", 5),
  baseBackoffMs: num("BASE_BACKOFF_MS", 1500),

  runsDir: process.env.RUNS_DIR || "runs",
};

export function assertSubjectKey() {
  if (!config.geminiApiKey) {
    throw new Error(
      "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key " +
        "(https://aistudio.google.com/apikey), then `set -a; . ./.env; set +a`."
    );
  }
}

export function assertJudgeKey() {
  if (!config.anthropicApiKey) {
    throw new Error("ANTHROPIC_API_KEY is not set (needed for the judge).");
  }
}
