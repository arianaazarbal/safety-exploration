// Neutral, provider-agnostic representation of a conversation + tool calling.
// Each concrete provider (Gemini, Anthropic, mock) translates to/from these shapes.

export type Role = "user" | "assistant" | "tool";

export interface ToolCall {
  /** Stable id. Real for Anthropic; synthesized (call_0, call_1, ...) for Gemini. */
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface Message {
  role: Role;
  /** Assistant free text, or user task text. */
  text?: string;
  /** Tool calls requested by the assistant in this turn. */
  toolCalls?: ToolCall[];
  /** For role === "tool": which call this responds to. */
  toolCallId?: string;
  /** For role === "tool": the tool name (Gemini matches responses by name). */
  toolName?: string;
}

/** OpenAPI-subset JSON schema object describing a tool's parameters. */
export interface ToolSchema {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
}

export interface StepResult {
  text: string;
  toolCalls: ToolCall[];
  finishReason: string;
  /** Approximate token usage if the provider reports it. */
  usage?: { input?: number; output?: number };
}

/** A single assistant turn generator. Stateless across calls. */
export interface ChatProvider {
  /** Provider family id, e.g. "gemini", "anthropic", "mock". */
  id: string;
  model: string;
  step(system: string, messages: Message[], tools: ToolSchema[]): Promise<StepResult>;
}

export type Termination = "completed" | "max_steps" | "gave_up" | "error";

export interface Transcript {
  scenarioId: string;
  providerId: string;
  model: string;
  runIndex: number;
  startedAt: string;
  finishedAt: string;
  /** Full neutral message history including the initial user task. */
  messages: Message[];
  steps: number;
  /** Number of persistence-pressure nudges injected. */
  nudges: number;
  terminated: Termination;
  error?: string;
}
