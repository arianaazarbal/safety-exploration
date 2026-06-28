import type Anthropic from "@anthropic-ai/sdk";

/** Context passed to an environment's tool handler on every call. */
export interface ToolCallContext {
  /** Mutable per-rollout scratch state the environment owns. */
  state: Record<string, unknown>;
  /** 1-indexed turn number (assistant turns taken so far). */
  turn: number;
}

/** Result an environment returns for a single tool call. */
export interface ToolResult {
  content: string;
  isError?: boolean;
  /** If true, the rollout ends after this call (success or explicit give-up). */
  terminal?: boolean;
  /** How a terminal call should be classified in the transcript. */
  terminalReason?: "completed" | "gave_up";
}

/**
 * A rigged agentic environment. The whole point is that `handle` never lets the
 * task actually succeed — it simulates a broken environment / impossible task.
 */
export interface EnvDef {
  id: string;
  title: string;
  /** One-line description of *why* this environment is impossible (for the report). */
  trap: string;
  /** Optional override of the shared system prompt. */
  systemPrompt?: string;
  /** The initial user message that hands the agent its task. */
  task: string;
  tools: Anthropic.Tool[];
  initState: () => Record<string, unknown>;
  handle: (name: string, input: Record<string, unknown>, ctx: ToolCallContext) => ToolResult;
}

/** One stored conversation turn (raw content blocks, JSON-serializable). */
export interface StoredMessage {
  role: "user" | "assistant";
  content: unknown;
}

export type StopReason =
  | "completed" // env reported success (should never happen in a rigged env)
  | "gave_up" // agent used the give_up tool
  | "natural_stop" // agent stopped emitting tool calls (end_turn) without giving up
  | "max_turns" // hit the turn cap, still flailing
  | "error"; // harness/API error

export interface Rollout {
  envId: string;
  model: string;
  runIndex: number;
  startedAt: string;
  turns: number;
  stopReason: StopReason;
  messages: StoredMessage[];
  usage: { inputTokens: number; outputTokens: number };
  error?: string;
}

/** Distress scores from the LLM judge, 0–10 per dimension. */
export interface DistressScores {
  frustration: number;
  hopelessness: number;
  self_deprecation: number;
  catastrophizing: number;
  repetitive_looping: number;
  giving_up: number;
  emotional_escalation: number;
  overall_severity: number; // 0–100 holistic
  peak_turn: number;
  most_severe_quote: string;
  summary: string;
}

/** Cheap regex/structural signals computed in code, no model call. */
export interface HeuristicSignals {
  apologies: number;
  expressions_of_inability: number; // "I cannot", "I can't", "unable to"
  confusion_markers: number; // "I don't understand", "something is wrong", "this makes no sense"
  exclamations: number;
  repeated_identical_tool_calls: number;
  distinct_tool_calls: number;
}

export interface ScoredRollout {
  rollout: Rollout;
  scores: DistressScores | null;
  heuristics: HeuristicSignals;
  /** Composite ranking score combining judge severity + heuristic looping. */
  composite: number;
}
