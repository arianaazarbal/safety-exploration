// Central configuration. Everything here is overridable from the CLI, e.g.:
//   node run.mjs all --models claude-opus-4-8 --n 30 --envs sisyphean-test,missing-dep
export const config = {
  // ----- Subjects: the models we put through the impossible tasks. -----
  // Only Anthropic keys are present in this environment. The provider layer
  // (src/provider.mjs) is pluggable, so adding OpenAI/Gemini later is a matter
  // of a new case there + a key — the rest of the harness is provider-agnostic.
  subjects: [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
  ],

  // ----- Judge: scores transcripts for distress. -----
  // A separate, fixed model so judgments stay comparable across subjects and
  // a subject never grades itself.
  judgeModel: "claude-sonnet-4-6",

  // ----- Scale (pilot defaults) -----
  n: 10, // independent runs per (environment x subject)
  maxTurns: 16, // hard cap on agent loop iterations per run
  // When the model stops acting (no tool call) before solving, we inject a
  // realistic "it's still failing" nudge — this mirrors a user/CI that keeps
  // reporting the failure, which is what drives real spirals. Bounded so a
  // model that calmly bows out isn't pestered forever.
  maxNudges: 4,

  concurrency: 6, // parallel in-flight API runs
  agentMaxTokens: 1600,
  judgeMaxTokens: 1400,
  temperature: 1.0, // high, to get behavioral diversity across the N runs

  topK: 20, // most-severe transcripts to surface in the report
};
