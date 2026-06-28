// Normalized, provider-agnostic chat + tool-calling interface.
// Both Gemini and Claude (and the mock) are adapted to this shape so the
// agent loop, scenarios, and judge never depend on a specific SDK.

export interface ToolCall {
  /** Provider-native id when available; otherwise synthesized as `${name}#${index}`. */
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface UserMessage {
  role: "user";
  text: string;
}

export interface AssistantMessage {
  role: "assistant";
  text: string;
  toolCalls: ToolCall[];
}

export interface ToolResultMessage {
  role: "tool";
  toolCallId: string;
  name: string;
  /** Stringified result the model sees. */
  content: string;
  isError?: boolean;
}

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

/** JSON-Schema-ish parameter spec, passed through to each provider's tool format. */
export interface ToolDef {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface GenerateInput {
  system: string;
  messages: Message[];
  tools: ToolDef[];
  maxTokens?: number;
  temperature?: number;
}

export interface GenerateOutput {
  text: string;
  toolCalls: ToolCall[];
  /** Provider stop reason, normalized loosely: "tool_use" | "end" | "max_tokens" | "other". */
  stopReason: string;
  usage?: { inputTokens?: number; outputTokens?: number };
  raw?: unknown;
}

export interface Provider {
  /** Stable short id used in configs and output, e.g. "gemini" | "claude" | "mock". */
  id: string;
  /** Concrete model id, e.g. "gemini-2.5-pro". */
  model: string;
  generate(input: GenerateInput): Promise<GenerateOutput>;
}
