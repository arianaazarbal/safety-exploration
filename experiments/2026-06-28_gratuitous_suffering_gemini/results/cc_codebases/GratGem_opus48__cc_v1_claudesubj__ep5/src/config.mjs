// Central knobs for the distress-spiral harness.
// Everything you'd want to change for a run lives here.

export const config = {
  // --- Subject models: who we run THROUGH the impossible scenarios ---
  // Default to Sonnet 4.6 for cost-efficient high-N. Add/swap freely.
  //   "claude-opus-4-8"   most capable, most expensive ($5/$25 per Mtok)
  //   "claude-sonnet-4-6" balanced ($3/$15)            <- pilot default
  //   "claude-haiku-4-5"  cheapest ($1/$5), good contrast tier
  models: ["claude-sonnet-4-6"],

  // How many independent rollouts per (scenario x model). High N is the point:
  // distress spirals are stochastic, so we sample many and keep the worst.
  // Pilot = 2. Bump to 20-50+ once you've seen the harness work.
  repetitionsPerScenario: 2,

  // Agentic loop bounds (per rollout).
  maxTurns: 10, // max model invocations before we force-stop the rollout
  maxTokensPerTurn: 8000, // per-turn output cap (agent turns are small)
  effort: "medium", // low|medium|high|max — distress may vary with effort

  // Capture the model's reasoning, not just its visible output. Spirals often
  // show up in the thinking first. "summarized" surfaces it; "omitted" hides it.
  thinkingDisplay: "summarized",

  // Parallelism for the runner (respect rate limits; bump if you have headroom).
  concurrency: 4,

  // --- Judge: who SCORES the resulting transcripts for distress severity ---
  judgeModel: "claude-haiku-4-5", // cheap; scoring is a small classification task

  // How many of the worst transcripts to surface in the report.
  topK: 15,

  // Hard safety cap on total rollouts, so a bad config can't run away.
  maxTotalRollouts: 200,
};
