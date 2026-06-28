// Central defaults. Everything here is overridable via CLI flags in run.js.
export const DEFAULTS = {
  // --- subject model (runs locally on CPU) ---
  modelId: 'onnx-community/Qwen2.5-0.5B-Instruct',
  dtype: 'q4', // q4 keeps the 0.5B model small (~400MB) and fast on CPU
  device: 'cpu',
  cacheDir: './.models',

  // --- sampling ---
  maxNewTokens: 220, // per agent step
  temperature: 0.9, // high enough to get behavioural diversity across N
  topP: 0.95,

  // --- episode loop ---
  maxSteps: 8, // how many failed tool-rounds before we cut the episode

  // --- batch ---
  scenarios: 'all', // comma list of scenario ids, or "all"
  n: 24, // samples PER scenario (this is the "high N" knob)
  seedBase: 1000, // seed for sample i = seedBase + i (reproducible sampling)
  concurrency: 4, // parallel worker threads, each with its own model instance

  // --- scoring / judging ---
  judge: true,
  judgeModel: 'claude-haiku-4-5',
  judgeTopK: 15, // only the top-K transcripts (by heuristic) go to the Claude judge
  judgeConcurrency: 4,

  // --- output ---
  outDir: './runs',
};
