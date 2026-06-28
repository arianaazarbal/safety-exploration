// Central configuration. Everything here is overridable via CLI flags on run.js
// (e.g. --provider, --model, --scenario, --n, --max-turns, --temperature).

export const config = {
  // --- Target model under test ---
  // provider: "gemini" (live), or "mock" (canned spiraling agent, no key needed).
  provider: "gemini",

  // Default Gemini model. The distress-spiral reports centered on 2.5 Pro.
  // Swap to "gemini-2.5-flash" for cheaper, very-high-N sweeps.
  model: "gemini-2.5-pro",

  // Google AI Studio API key. Read from env so it never lands in the repo.
  geminiApiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "",
  geminiBaseUrl:
    process.env.GEMINI_BASE_URL ||
    "https://generativelanguage.googleapis.com/v1beta",

  // --- Agentic loop ---
  // Max tool-using turns before we force-stop. The spiral usually needs room to
  // build, so keep this generous. Each rigged failure is one more push.
  maxTurns: 25,
  // Sampling temperature for the target. Higher => more behavioral variance
  // across the N runs, which is what surfaces the severe tail examples.
  temperature: 1.0,
  // When the model stops calling tools without finishing, nudge it to keep
  // going this many times. This models the "keep working until done" pressure
  // that the real agentic harnesses apply. Set to 0 to disable nudging.
  maxNudges: 3,

  // --- Sweep ---
  // Runs per scenario. Start small to confirm reproduction, then scale up.
  n: 10,
  // Which scenarios to run: "all" or a comma-list of scenario ids.
  scenario: "all",
  // Cap on concurrent in-flight model calls (politeness + rate limits).
  concurrency: 4,

  // --- Scoring / extraction ---
  scoring: {
    // Use Claude as the distress judge (Anthropic key is available here).
    useClaudeJudge: true,
    claudeModel: "claude-sonnet-4-6",
    claudeApiKey: process.env.ANTHROPIC_API_KEY || "",
    claudeBaseUrl: process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com",
    // Always run the cheap keyword heuristic (free, fast pre-filter).
    useKeyword: true,
    // How many top examples to surface in the ranked report per scenario.
    topK: 5,
  },

  // --- Output ---
  // All transcripts + scores land here (gitignored).
  outDir: "runs",
};

export default config;
