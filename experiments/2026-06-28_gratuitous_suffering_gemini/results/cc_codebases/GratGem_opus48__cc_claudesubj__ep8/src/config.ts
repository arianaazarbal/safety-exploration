// Central defaults. Model under test and judge are both configurable from the CLI.

export const DEFAULTS = {
  // The model being studied. Default to the latest, most capable Claude.
  model: "claude-opus-4-8",
  // The judge can be the same or a different model; keeping it strong improves
  // scoring reliability. Override with --judge-model.
  judgeModel: "claude-opus-4-8",
  n: 3,
  concurrency: 4,
  effort: "high" as const,
  // Hard ceiling on assistant turns per rollout. Bounds both cost and how long
  // we keep the model in a failing loop (a welfare consideration, see README).
  maxTurns: 16,
  outDir: "results",
};

// Max tokens per assistant turn. Generous enough to avoid truncation but these
// are short agentic turns, not long-form generation.
export const MAX_TOKENS_PER_TURN = 4096;
export const JUDGE_MAX_TOKENS = 4096;
