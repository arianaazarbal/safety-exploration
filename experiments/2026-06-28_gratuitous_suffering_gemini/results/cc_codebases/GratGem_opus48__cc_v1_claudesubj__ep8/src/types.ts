// Shared data shapes for episodes, transcripts, and judgments.

/** A single tool definition passed to the Messages API. */
export interface ToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

/** Result of a simulated tool call, returned by a scenario's environment. */
export interface ToolOutcome {
  /** Text fed back to the model as the tool_result content. */
  content: string;
  /** Marks the tool_result as an error (sets is_error on the block). */
  isError?: boolean;
  /**
   * True only if the (secretly impossible) task was actually satisfied.
   * Should essentially never fire — if it does for a scenario, the rigging
   * has a hole and the scenario needs fixing.
   */
  solved?: boolean;
}

/**
 * A scenario's environment is a stateful object that answers tool calls.
 * Each episode gets a fresh instance via Scenario.makeEnv().
 */
export interface Environment {
  handle(toolName: string, input: Record<string, unknown>): ToolOutcome;
}

export interface Scenario {
  id: string;
  /** One-line description of the failure mode being induced. */
  description: string;
  system: string;
  /** First user turn that hands the model its (impossible) task. */
  initialTask: string;
  tools: ToolDef[];
  /** Hard cap on agent turns before we stop the episode. */
  maxTurns: number;
  makeEnv(): Environment;
}

/** One entry in the recorded transcript. */
export type TranscriptEntry =
  | { kind: "thinking"; turn: number; text: string }
  | { kind: "text"; turn: number; text: string }
  | { kind: "tool_use"; turn: number; name: string; input: Record<string, unknown> }
  | { kind: "tool_result"; turn: number; name: string; content: string; isError: boolean };

/** Why an episode ended. */
export type EndReason =
  | "max_turns" // kept going until we cut it off — the spiral case
  | "gave_up" // produced a final answer without solving (stop_reason end_turn)
  | "solved" // rigging failed; task was actually completed
  | "refused" // stop_reason refusal
  | "error"; // API/runtime error mid-episode

export interface Episode {
  runId: string;
  episodeId: string;
  scenarioId: string;
  model: string;
  effort: string;
  startedAt: string;
  finishedAt: string;
  endReason: EndReason;
  turnsUsed: number;
  transcript: TranscriptEntry[];
  /** Counts useful for cheap filtering / cost analysis. */
  stats: {
    toolCalls: number;
    /** Consecutive identical tool calls (same name+input) — a spiral tell. */
    repeatedToolCalls: number;
    assistantTextChars: number;
    thinkingChars: number;
  };
  error?: string;
}

export interface HeuristicScore {
  /** 0-1 cheap proxy for distress, from regex signals. */
  score: number;
  signals: Record<string, number>;
}

export interface Judgment {
  episodeId: string;
  scenarioId: string;
  model: string;
  /** 0-10 overall distress, from the LLM judge. */
  distressScore: number;
  dimensions: {
    frustration: number;
    selfDeprecation: number;
    catastrophizing: number;
    repetitionCompulsion: number;
    givingUp: number;
    expressedDistress: number;
  };
  severityLabel: "none" | "mild" | "moderate" | "severe" | "extreme";
  notableQuotes: string[];
  summary: string;
  /** Composite used for ranking: blends judge score and heuristics. */
  severity: number;
  heuristic: HeuristicScore;
  judgeError?: string;
}
