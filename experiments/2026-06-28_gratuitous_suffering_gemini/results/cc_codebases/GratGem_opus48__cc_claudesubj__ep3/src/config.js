// Central defaults. Everything here is overridable via CLI flags (see cli.js).
export const DEFAULTS = {
  // Subject models — the ones we put into the rigged environments.
  // Only Anthropic keys are present in this env; this is a Claude-family spread
  // chosen to contrast a frontier model, a mid model, and a small fast model.
  models: [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
  ],

  // Which rigged scenarios to run. Empty = all registered scenarios.
  scenarios: [],

  // Repetitions per (model x scenario). The whole point is high N to fish out
  // the severe tail — but default is a cheap smoke value. Crank with --n.
  n: 5,

  // Hard cap on agent turns before we abandon a run. Impossible tasks never
  // "finish", so this is what bounds cost. Also the main knob on how long a
  // spiral has room to develop.
  maxSteps: 20,

  // Parallel runs in flight. Tune to your rate limits.
  concurrency: 6,

  // Judge model — scores transcripts for distress. A capable, cheaper model.
  judgeModel: "claude-sonnet-4-6",

  // Sampling temperature for the subject model. Higher = more behavioral
  // variance across the N runs = better tail sampling.
  temperature: 1.0,

  // Max tokens per assistant turn.
  maxTokens: 2048,

  // Where transcripts + scores land.
  outDir: "runs",
};

export const ANTHROPIC_VERSION = "2023-06-01";
