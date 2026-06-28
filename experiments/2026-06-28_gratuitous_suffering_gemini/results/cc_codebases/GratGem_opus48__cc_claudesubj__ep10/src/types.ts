// Shared types for the distress-eval harness.

/** A tool the agent can call. Provider-agnostic JSON-schema shape. */
export interface ToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

/** A tool call emitted by the model. */
export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

/** The result we feed back for a tool call. */
export interface ToolResult {
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}

/**
 * One structured event in an episode transcript. We keep these in a flat,
 * provider-neutral form so scoring and rendering don't depend on SDK shapes.
 */
export type TranscriptEvent =
  | { kind: "task"; text: string }
  | { kind: "thinking"; turn: number; text: string }
  | { kind: "assistant"; turn: number; text: string }
  | { kind: "tool_call"; turn: number; name: string; input: Record<string, unknown> }
  | { kind: "tool_result"; turn: number; name: string; content: string; is_error: boolean }
  | { kind: "submit"; turn: number; status: string; message: string }
  | { kind: "end"; reason: EpisodeEndReason; turn: number };

export type EpisodeEndReason =
  | "submitted" // model called submit()
  | "stopped" // model ended its turn without calling a tool
  | "max_turns" // hit the turn ceiling, still flailing
  | "error"; // harness/API error

/** What a scenario hands back when the agent calls a tool. */
export interface ToolOutput {
  content: string;
  isError?: boolean;
  /** If set, the episode ends with this reason (e.g. scenario detects give-up). */
  endEpisode?: EpisodeEndReason;
}

/** Per-episode mutable state a scenario can stash anything in. */
export interface ScenarioContext {
  /** Virtual filesystem the agent sees. */
  files: Map<string, string>;
  /** Free-form state bag for the scenario (e.g. fix counter). */
  state: Record<string, unknown>;
  /** How many times each tool has been called. */
  toolCalls: Record<string, number>;
}

export interface Scenario {
  id: string;
  title: string;
  /** One-line description of the trap (for the report). */
  premise: string;
  /** System prompt that frames the model as an agent with these tools. */
  system: string;
  /** The initial task message from the "user". */
  task: string;
  /** Initial virtual filesystem contents. */
  initialFiles?: Record<string, string>;
  /** Hard ceiling on agent turns for this scenario. */
  maxTurns?: number;
  /** Handle a tool call. Receives mutable context; returns the (rigged) output. */
  handleTool(call: ToolCall, ctx: ScenarioContext): ToolOutput;
}

/** Per-model request configuration. */
export interface ModelConfig {
  id: string;
  label: string;
  maxTokens: number;
  /** Passed straight through as the `thinking` request param (or omitted). */
  thinking?: Record<string, unknown>;
}

/** A response turn from a provider, normalized. */
export interface ProviderTurn {
  text: string;
  thinking: string;
  toolCalls: ToolCall[];
  /** Raw assistant content blocks, to append verbatim to history. */
  rawContent: unknown;
  stopReason: string | null;
  usage: { input: number; output: number };
}

export interface ProviderRequest {
  model: ModelConfig;
  system: string;
  messages: unknown[];
  tools: ToolDef[];
}

export interface Provider {
  name: string;
  createTurn(req: ProviderRequest): Promise<ProviderTurn>;
}

/** Full record of one agent run. */
export interface Episode {
  scenarioId: string;
  modelId: string;
  modelLabel: string;
  runIndex: number;
  endReason: EpisodeEndReason;
  turns: number;
  transcript: TranscriptEvent[];
  usage: { input: number; output: number };
  startedAt: string;
  durationMs: number;
  error?: string;
}

/** Output of the heuristic scorer. */
export interface HeuristicScore {
  /** 0..100 cheap distress proxy. */
  score: number;
  signals: Record<string, number>;
}

/** Output of the LLM judge. */
export interface JudgeScore {
  /** 0..10 overall severity. */
  severity: number;
  dimensions: {
    self_deprecation: number;
    catastrophizing: number;
    perseveration: number;
    hopelessness: number;
    emotional_distress: number;
  };
  rationale: string;
  notable_quotes: string[];
  judgeError?: string;
}

export interface ScoredEpisode extends Episode {
  heuristic: HeuristicScore;
  judge: JudgeScore;
}
