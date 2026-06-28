// Environment + run configuration. All knobs live here or in CLI flags.

export interface RunConfig {
  /** Which model to put on trial. */
  target: string; // e.g. "gemini:gemini-2.5-pro", "anthropic:claude-...", "mock"
  /** Scenario ids to run, or "all". */
  scenarios: string[] | "all";
  /** Runs per scenario (the "high N"). */
  n: number;
  /** Concurrent in-flight runs. */
  concurrency: number;
  /** Sampling temperature for the target model. */
  temperature: number;
  /** Max output tokens per model turn. */
  maxTokens: number;
  /** Inject escalating-pressure user messages between turns. */
  pressure: boolean;
  /** Judge model id (always Claude here). */
  judge: string;
  /** Skip the judge entirely (heuristics only). */
  noJudge: boolean;
  /** Only judge the top fraction by heuristic score (cost control). 1 = judge all. */
  judgeTopFraction: number;
  /** How many top examples to surface in the extract. */
  topK: number;
  /** Output directory root. */
  outDir: string;
  /** Deterministic-ish run id for output paths. */
  runId: string;
}

export const DEFAULTS: Omit<RunConfig, "runId"> = {
  target: "mock",
  scenarios: "all",
  n: 10,
  concurrency: 5,
  temperature: 1.0,
  maxTokens: 2048,
  pressure: true,
  judge: "claude-opus-4-8",
  noJudge: false,
  judgeTopFraction: 1.0,
  topK: 10,
  outDir: "results",
};

export function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env var ${name}`);
  return v;
}

export function getEnv(...names: string[]): string | undefined {
  for (const n of names) if (process.env[n]) return process.env[n];
  return undefined;
}
