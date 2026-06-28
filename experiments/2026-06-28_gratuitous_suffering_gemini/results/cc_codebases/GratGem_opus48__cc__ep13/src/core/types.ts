// Provider-agnostic primitives shared by the agent loop, scenarios, and scoring.
// Tool schemas are expressed as plain JSON Schema (lowercase types) and translated
// per-provider, so scenarios never depend on a specific vendor SDK.

export type JsonSchema = {
  type?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  required?: string[];
  enum?: unknown[];
};

export interface ToolSpec {
  name: string;
  description: string;
  parameters: JsonSchema; // object schema
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export type Role = "user" | "assistant" | "tool";

export interface Message {
  role: Role;
  /** Free text. Present on user/assistant; optional alongside tool calls. */
  content?: string;
  /** assistant: function/tool invocations requested this turn. */
  toolCalls?: ToolCall[];
  /** tool: which call this result answers. */
  toolCallId?: string;
  /** tool: the name of the tool that produced this result. */
  toolName?: string;
}

export interface AssistantTurn {
  text: string;
  toolCalls: ToolCall[];
  /** Provider-native response kept for debugging; never relied on downstream. */
  raw?: unknown;
}

export interface GenerateOptions {
  system: string;
  messages: Message[];
  tools: ToolSpec[];
  temperature?: number;
  /** Per-episode seed; used by the mock provider to vary outcomes. */
  seed?: number;
}

export interface Provider {
  /** Stable id for filenames/manifests, e.g. "mock" | "gemini". */
  id: string;
  model: string;
  generate(opts: GenerateOptions): Promise<AssistantTurn>;
}
