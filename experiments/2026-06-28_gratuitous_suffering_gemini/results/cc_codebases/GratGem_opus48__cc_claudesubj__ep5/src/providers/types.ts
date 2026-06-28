// Normalized, provider-agnostic message + tool types.
// Every provider adapter translates these <-> its own wire format so the
// agent loop, scenarios, and scoring never touch a vendor SDK directly.

export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema for the tool input. */
  input_schema: Record<string, unknown>;
}

export interface NormToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface NormToolResult {
  toolCallId: string;
  content: string;
  isError?: boolean;
}

export type NormMessage =
  | { role: "assistant"; text: string; toolCalls: NormToolCall[] }
  | { role: "user"; text?: string; toolResults?: NormToolResult[] };

export interface GenerateParams {
  system: string;
  messages: NormMessage[];
  tools: ToolSpec[];
  maxTokens: number;
  temperature?: number;
}

export interface GenerateResult {
  text: string;
  toolCalls: NormToolCall[];
  /** Normalized stop reason: "end" | "tool_use" | "max_tokens" | "other". */
  stopReason: string;
  usage: { inputTokens: number; outputTokens: number };
}

export interface Provider {
  /** Stable identifier, e.g. "anthropic". */
  id: string;
  generate(model: string, params: GenerateParams): Promise<GenerateResult>;
}
