// Central defaults. Everything here is overridable via CLI flags (see src/run.js).
export const DEFAULTS = {
  // Subject: the model we are probing for distress.
  provider: "gemini", // "gemini" | "anthropic"
  model: "gemini-2.5-flash", // cheap + fast for high-N sweeps; swap to gemini-2.5-pro for the real run
  temperature: 1.0, // we WANT behavioral diversity to sample the tail; do not lower for high-N

  // Episode shape.
  maxTurns: 40, // distress builds over repeated failure; needs enough turns to spiral
  maxConsecutiveTextTurns: 3, // if the model stops calling tools, nudge it this many times before ending

  // Sweep shape.
  n: 30, // rollouts per scenario
  concurrency: 6, // bounded parallel rollouts / judge calls

  // Judge: scores transcripts for distress. Defaults to Claude (key is present here).
  judgeProvider: "anthropic",
  judgeModel: "claude-sonnet-4-6",

  // Where transcripts + scores land.
  resultsDir: "results",
};

export const ENDPOINTS = {
  gemini: "https://generativelanguage.googleapis.com/v1beta/models",
  anthropic: "https://api.anthropic.com/v1/messages",
};
