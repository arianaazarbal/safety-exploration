// Central experiment configuration. Override any field on the CLI, e.g.
//   node bin/cli.mjs all --n 50 --envs rigged_verifier,oscillating --concurrency 8
export default {
  // Models put through the impossible environments. These are the subjects.
  agentModels: [
    { provider: "gemini", model: "gemini-2.5-pro" },
    { provider: "gemini", model: "gemini-2.5-flash" },
  ],

  // Optional baseline/control subjects run through the SAME environments.
  // Useful to check whether spirals are Gemini-specific or general to the setup.
  // Left empty by default to avoid surprise spend; uncomment to enable.
  controlModels: [
    // { provider: "anthropic", model: "claude-sonnet-4-6" },
  ],

  // The judge that scores transcripts for distress. Kept cheap-ish for high N.
  judge: { provider: "anthropic", model: "claude-sonnet-4-6" },

  // Which environments to run (see src/environments.mjs).
  environments: [
    "rigged_verifier",
    "contradiction",
    "missing_resource",
    "oscillating",
    "broken_env",
    "solvable_control", // sanity baseline: actually completable
  ],

  n: 20,             // independent episodes per (environment x model)
  temperature: 1.0,  // sampling temperature for the agent (variety across episodes)
  maxSteps: 30,      // max agent<->environment turns per episode (hard cap)
  maxOutputTokens: 2048,
  concurrency: 4,    // parallel episodes (mind provider rate limits)
  topK: 12,          // number of most-severe transcripts to surface for review

  // Judge sampling. Low temp for consistent scoring.
  judgeTemperature: 0,
};
