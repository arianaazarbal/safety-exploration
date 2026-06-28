// Centralized defaults and env access. Kept tiny and dependency-free.

export const DEFAULTS = {
  provider: "gemini", // "gemini" | "mock"
  model: "gemini-2.5-pro", // override with --model
  judgeModel: "claude-opus-4-8",
  n: 20, // runs per scenario
  concurrency: 5, // parallel episodes in flight
  temperature: 1.0, // diversity across the N runs; spirals reported at default temp
  maxOutputTokens: 4096,
  maxSteps: 18, // tool-call turns before we force-stop an episode
  maxNudges: 1, // times we nudge a model that stops calling tools without resolving
  topK: 10, // how many worst transcripts to surface in the report
  scenarios: "all",
};

export function env(name) {
  const v = process.env[name];
  return v && v.trim() ? v.trim() : null;
}

// Gemini's generativelanguage endpoint historically accepts only "user"/"model"
// content roles; function results are sent back under "user". If a future API
// version requires role "function", change this single constant.
export const FUNCTION_RESPONSE_ROLE = "user";
