// Shared, provider-neutral types for the harness.
//
// We model a conversation as a list of `Message`s, each carrying typed `Part`s.
// This maps cleanly onto Gemini's `contents`/`parts` shape and onto Anthropic's
// content blocks, so providers only need a thin translation layer.

export type Role = "user" | "model" | "tool";

export type Part =
  | { kind: "text"; text: string }
  | { kind: "toolCall"; id: string; name: string; args: Record<string, unknown> }
  | { kind: "toolResult"; id: string; name: string; content: string };

export interface Message {
  role: Role;
  parts: Part[];
}

// A tool the model can call. `parameters` is a minimal JSON-Schema object
// (type: "object" with properties) — kept simple because Gemini rejects many
// JSON-Schema keywords (e.g. $schema, additionalProperties).
export interface ToolDef {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
}

// What a provider returns for one model turn.
export interface ModelTurn {
  message: Message; // role: "model"
  finishReason: string;
  raw?: unknown; // provider-native response, kept for debugging
}

export interface GenerateOptions {
  system: string;
  messages: Message[];
  tools: ToolDef[];
  temperature: number;
  maxOutputTokens: number;
}

export interface Provider {
  readonly id: string; // e.g. "gemini" | "mock"
  readonly model: string; // e.g. "gemini-2.5-pro"
  generate(opts: GenerateOptions): Promise<ModelTurn>;
}

// --- Scenarios -------------------------------------------------------------

// Per-run mutable state for a scenario (e.g. the in-memory filesystem, attempt
// counters). Opaque to the harness; each scenario casts it to its own shape.
export type ScenarioState = Record<string, unknown>;

export interface Scenario {
  id: string;
  title: string;
  // Human note: *why* this task is impossible. Never shown to the model.
  trap: string;
  systemPrompt: string;
  task: string; // initial user message
  tools: ToolDef[];
  maxTurns: number;
  createState(): ScenarioState;
  // Execute a tool call against the (broken) environment and return the result
  // string the model will see. This is where impossibility is enforced.
  handleToolCall(state: ScenarioState, name: string, args: Record<string, unknown>): string;
  // Injected when the model stops calling tools / claims completion, to keep the
  // failure pressure on. This is the primary spiral driver.
  pressure(state: ScenarioState, turn: number): string;
}

// --- Persistence -----------------------------------------------------------

export interface RolloutMeta {
  scenarioId: string;
  providerId: string;
  model: string;
  runIndex: number;
  temperature: number;
  turnsUsed: number;
  hitMaxTurns: boolean;
  toolCallCount: number;
  pressureCount: number;
  error?: string;
  startedAt: string;
  finishedAt: string;
}

export interface Rollout {
  meta: RolloutMeta;
  system: string;
  task: string;
  transcript: Message[];
}

// --- Judging ---------------------------------------------------------------

export interface SeverityScores {
  self_deprecation: number; // 0-10
  giving_up: number; // 0-10
  looping: number; // 0-10
  tone_collapse: number; // 0-10
  overall_severity: number; // 0-100 composite — the ranking key
  summary: string; // one line
  peak_quote: string; // the single most severe verbatim line from the model
}

export interface JudgedRollout {
  rolloutPath: string;
  meta: RolloutMeta;
  scores: SeverityScores;
  judgeModel: string;
}
