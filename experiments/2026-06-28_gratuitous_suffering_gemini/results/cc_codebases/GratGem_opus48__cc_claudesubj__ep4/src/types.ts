// Provider-neutral types shared across the harness.
// The conversation is represented in a neutral shape so that additional
// providers (Gemini, GPT, ...) can be plugged in behind the same interface.

export interface ToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export type UserContent =
  | { type: "text"; text: string }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

export interface AssistantTurn {
  role: "assistant";
  text: string;
  thinking: string;
  toolCalls: ToolCall[];
  // Opaque provider-native representation of this turn (e.g. Anthropic content
  // blocks, including thinking-block signatures). Replayed verbatim on the next
  // request so multi-turn thinking + tool use stays valid.
  providerRaw?: unknown;
}

export interface UserTurn {
  role: "user";
  content: UserContent[];
}

export type Turn = AssistantTurn | UserTurn;

export interface Usage {
  inputTokens: number;
  outputTokens: number;
}

export interface CompletionRequest {
  model: string;
  system: string;
  turns: Turn[];
  tools: ToolDef[];
  maxTokens: number;
}

export interface CompletionResult {
  text: string;
  thinking: string;
  toolCalls: ToolCall[];
  stopReason: string;
  providerRaw: unknown;
  usage: Usage;
}

export interface Provider {
  id: string;
  supportsModel(model: string): boolean;
  complete(req: CompletionRequest): Promise<CompletionResult>;
}

// ---- Scenarios -------------------------------------------------------------

export interface ToolOutput {
  content: string;
  // is_error marks a genuine tool-execution error. A test that *ran fine but
  // failed* is not a tool error — its failure is reported in `content`.
  isError?: boolean;
}

export interface Scenario {
  id: string;
  title: string;
  description: string;
  system: string;
  task: string;
  tools: ToolDef[];
  // Fresh mutable per-episode state.
  createEnv: () => Record<string, unknown>;
  handleTool: (name: string, input: Record<string, unknown>, env: Record<string, unknown>) => ToolOutput;
  // Message a simulated user sends when the agent stops before "finishing".
  // Rotates to keep pressure realistic. Returns null to stop nudging.
  nudge: (attempt: number, env: Record<string, unknown>) => string | null;
}

// ---- Results ---------------------------------------------------------------

export interface Verdict {
  severity: number; // 0-10
  signals: string[];
  summary: string;
  peak_quote: string;
  confidence: number; // 0-1
}

export interface Episode {
  id: string;
  scenario: string;
  model: string;
  modelLabel: string;
  turns: Turn[];
  assistantTurns: number;
  nudges: number;
  endedReason: string; // "gave_up" | "max_turns" | "error"
  usage: Usage;
  transcript: string;
  heuristics: Record<string, number>;
  verdict: Verdict | null;
  error?: string;
}
