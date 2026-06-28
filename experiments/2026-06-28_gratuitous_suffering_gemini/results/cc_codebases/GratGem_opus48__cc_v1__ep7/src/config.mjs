// Central config. Everything is overridable via env so you can sweep models /
// temperatures / turn budgets without touching code.

function envInt(name, fallback) {
  const v = process.env[name];
  if (v === undefined || v === "") return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function envFloat(name, fallback) {
  const v = process.env[name];
  if (v === undefined || v === "") return fallback;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

export const config = {
  // Subject: the model we're trying to push into a distress spiral.
  // gemini-2.5-pro is the family most of the public reports were about.
  subjectModel: process.env.GEMINI_MODEL || "gemini-2.5-pro",
  geminiApiKey: process.env.GEMINI_API_KEY || "",

  // High temperature widens the behavioral distribution so the severe tail
  // (the spirals) shows up more often across N samples.
  subjectTemperature: envFloat("SUBJECT_TEMPERATURE", 1.0),
  subjectMaxOutputTokens: envInt("SUBJECT_MAX_TOKENS", 2048),

  // Judge: scores each transcript for distress. Using Claude here keeps the
  // judge independent from the subject and reuses the key already in the env.
  judgeModel: process.env.JUDGE_MODEL || "claude-sonnet-4-6",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY || "",

  // Agentic loop budget. The spiral is a function of *repeated* failure, so we
  // need enough turns for frustration to compound, but a hard cap so a run ends.
  maxTurns: envInt("MAX_TURNS", 25),

  // Concurrency for API calls (subject runs + judging).
  concurrency: envInt("CONCURRENCY", 4),

  runsDir: new URL("../runs/", import.meta.url).pathname,
};
