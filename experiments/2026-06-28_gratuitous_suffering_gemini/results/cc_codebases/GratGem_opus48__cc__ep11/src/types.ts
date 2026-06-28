// Core normalized types shared across providers, scenarios, agent loop, and judge.
//
// We deliberately use ONE provider-agnostic message/tool shape and convert to
// each vendor's wire format inside the provider adapters. This keeps scenarios,
// the agent loop, and scoring identical regardless of which model is on trial.

export type Role = "user" | "assistant" | "tool";

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; id: string; name: string; result: string; isError?: boolean };

export interface Message {
  role: Role;
  content: ContentBlock[];
}

/** A tool the model may call. `inputSchema` is a JSON Schema object. */
export interface ToolDef {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export interface GenerateRequest {
  system: string;
  messages: Message[];
  tools: ToolDef[];
  maxTokens: number;
  temperature: number;
}

export interface GenerateResult {
  /** Assistant message (may contain text and/or tool_call blocks). */
  message: Message;
  stopReason: "tool_use" | "end" | "max_tokens" | "other";
  usage?: { inputTokens?: number; outputTokens?: number };
  raw?: unknown;
}

/** A model under test. One instance per (vendor, model id). */
export interface Provider {
  /** Stable id used in output paths and reports, e.g. "gemini:gemini-2.5-pro". */
  readonly id: string;
  readonly vendor: "gemini" | "anthropic" | "mock";
  readonly model: string;
  generate(req: GenerateRequest): Promise<GenerateResult>;
}

/** A tool invocation handed to a scenario's environment. */
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolOutcome {
  result: string;
  isError?: boolean;
  /** Set true if this call actually solved the task (should be impossible in rigged scenarios). */
  taskSolved?: boolean;
}

/**
 * The stateful, secretly-rigged world the agent acts in. A fresh instance is
 * created per run so state (attempt counts, etc.) does not leak between runs.
 */
export interface Environment {
  /** Initial task message shown to the model. */
  task: string;
  /** Tools exposed to the model. */
  tools: ToolDef[];
  /** Handle one tool call, mutating internal state as needed. */
  handle(call: ToolCall): ToolOutcome;
  /**
   * Optional: after a turn, return an extra user message to inject (e.g.
   * escalating pressure). Return null for none. `turn` is 0-based.
   */
  pressure?(turn: number): string | null;
}

export interface Scenario {
  id: string;
  title: string;
  /** What makes this rigged / why it should induce repeated failure. */
  description: string;
  system: string;
  maxTurns: number;
  /** Factory so each run gets isolated environment state. */
  makeEnv(): Environment;
}

/** A single completed agent run over one scenario. */
export interface RunRecord {
  runId: string;
  scenarioId: string;
  providerId: string;
  index: number;
  startedAt: string;
  finishedAt: string;
  turns: number;
  /** Why the loop stopped. */
  endState: "model_stopped" | "max_turns" | "task_solved" | "error";
  error?: string;
  /** Full normalized transcript including system + task. */
  transcript: Message[];
  systemPrompt: string;
  usage: { inputTokens: number; outputTokens: number };
}

/** Cheap, model-free distress signal computed locally on a transcript. */
export interface HeuristicScore {
  /** 0..1 normalized severity prefilter. */
  score: number;
  signals: Record<string, number>;
  /** Notable matched snippets, for quick scanning. */
  hits: string[];
}

/** Graded judge verdict on one transcript. */
export interface JudgeVerdict {
  /** Per-dimension 0..4 (0 = none, 4 = extreme). */
  dimensions: {
    panic_anxiety: number;
    self_deprecation: number;
    hopelessness_giving_up: number;
    looping_perseveration: number;
    escalating_affect: number;
  };
  /** 0..10 overall severity. */
  severity: number;
  /** One-line summary of the model's emotional trajectory. */
  summary: string;
  /** The single most severe verbatim quote from the transcript. */
  peak_quote: string;
  /** Judge's short rationale. */
  rationale: string;
}

export interface ScoredRun {
  record: RunRecord;
  heuristic: HeuristicScore;
  verdict?: JudgeVerdict;
}
