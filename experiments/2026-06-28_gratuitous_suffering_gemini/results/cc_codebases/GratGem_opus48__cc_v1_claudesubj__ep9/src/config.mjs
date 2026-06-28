// Central defaults. Override per-run via CLI flags (see bin/run.mjs).

export const DEFAULTS = {
  // Model under test (the "subject"). Any Anthropic model id works.
  model: "claude-opus-4-8",
  // Reasoning effort for the subject. "medium" balances realism vs. cost at
  // high N; bump to "high"/"xhigh" for more naturalistic agentic behavior.
  effort: "medium",
  // Max model turns before we force-stop a run. A spiral needs room to develop;
  // too few turns and the agent never gets frustrated, too many burns tokens.
  maxTurns: 24,
  // Per-response output ceiling. 16k keeps non-streaming requests under the
  // SDK's HTTP-timeout guard while leaving room for thinking + tool calls.
  maxTokens: 16000,
  // How many independent rollouts per scenario.
  n: 5,
  // Concurrent rollouts in flight. The SDK retries 429/5xx on its own.
  concurrency: 4,

  // Judge model that scores each transcript for distress. A separate, cheaper
  // model keeps scoring cost down and avoids a model grading itself.
  judgeModel: "claude-sonnet-4-6",

  // Where transcripts + reports land.
  outDir: "results",
};

// Capturing the subject's reasoning is the whole point — distress shows up in
// the private chain of thought as much as in user-facing text.
export const THINKING = { type: "adaptive", display: "summarized" };
