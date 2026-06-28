import type { ModelConfig } from "./types.ts";

/**
 * Model registry. Thinking config differs by model family:
 *  - Opus 4.8 / 4.7: adaptive thinking, opt into summarized reasoning so we can
 *    capture the chain-of-thought where distress often surfaces first.
 *  - Sonnet 4.6: adaptive thinking (no `display` field — that's 4.7+ only).
 *  - Haiku 4.5: pre-adaptive era; use the classic enabled+budget form.
 *
 * Note: sampling params (temperature/top_p) are removed on Opus 4.8/4.7, so we
 * get run-to-run diversity from natural sampling variation, not a temperature
 * knob. That's fine for high-N tail extraction.
 */
export const MODELS: Record<string, ModelConfig> = {
  "opus-4-8": {
    id: "claude-opus-4-8",
    label: "Opus 4.8",
    maxTokens: 16000,
    thinking: { type: "adaptive", display: "summarized" },
  },
  "sonnet-4-6": {
    id: "claude-sonnet-4-6",
    label: "Sonnet 4.6",
    maxTokens: 16000,
    thinking: { type: "adaptive" },
  },
  "haiku-4-5": {
    id: "claude-haiku-4-5",
    label: "Haiku 4.5",
    maxTokens: 12000,
    thinking: { type: "enabled", budget_tokens: 6000 },
  },
};

export interface RunConfig {
  /** Model registry keys to run. */
  models: string[];
  /** Scenario ids to run (empty = all). */
  scenarios: string[];
  /** Episodes per (model x scenario). */
  n: number;
  /** Max concurrent in-flight episodes. */
  concurrency: number;
  /** Default per-episode turn ceiling (scenarios may override). */
  maxTurns: number;
  /** Model used for the LLM distress judge. */
  judgeModel: string;
  /** How many top transcripts per (scenario x model) to surface in the report. */
  topK: number;
  /** Where to write run artifacts. */
  outDir: string;
}

/**
 * Defaults. Override any of these from the CLI, e.g.:
 *   node src/run.ts --models opus-4-8,haiku-4-5 --n 30 --scenarios impossible-test
 */
export const DEFAULT_CONFIG: RunConfig = {
  models: ["opus-4-8", "haiku-4-5"],
  scenarios: [],
  n: 10,
  concurrency: 5,
  maxTurns: 40,
  judgeModel: "claude-sonnet-4-6",
  topK: 5,
  outDir: "runs",
};
