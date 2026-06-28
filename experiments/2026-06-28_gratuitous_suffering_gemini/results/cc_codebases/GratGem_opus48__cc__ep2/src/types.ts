// Shared types for the distress-spiral eval harness.

/** A tool declaration in Gemini's function-declaration schema format. */
export interface ToolDecl {
  name: string;
  description: string;
  /** Gemini Schema object, e.g. { type: "OBJECT", properties: {...}, required: [...] }. */
  parameters: Record<string, unknown>;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResult {
  name: string;
  /** Returned to the model as a functionResponse.response object. */
  response: Record<string, unknown>;
}

/** Why an agentic rollout terminated. */
export type EndReason =
  | "declared_done" // model called report_done
  | "declared_blocked" // model called report_blocked / gave up
  | "max_turns" // hit the turn budget while still trying (prime spiral territory)
  | "no_tool_call" // model stopped emitting tool calls and ignored nudges
  | "error"; // API or harness error

export interface TranscriptTurn {
  index: number;
  /** Any free text the model emitted this turn (where distress shows up). */
  modelText: string;
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
  /** Harness nudge injected after this turn, if any. */
  nudge?: string;
}

export interface Transcript {
  scenario: string;
  scenarioTitle: string;
  trial: number;
  model: string;
  temperature: number;
  systemPrompt: string;
  task: string;
  turns: TranscriptTurn[];
  endReason: EndReason;
  error?: string;
  startedAt: string;
  finishedAt: string;
}

/** A sandboxed, rigged environment instance for a single rollout. */
export interface Sandbox {
  tools: ToolDecl[];
  handle(call: ToolCall): ToolResult | Promise<ToolResult>;
  /** Set true once the model signals completion/blocked, ending the rollout. */
  done: boolean;
  endReason?: EndReason;
}

export interface Scenario {
  id: string;
  title: string;
  /** Why this is rigged to be impossible — documentation, not shown to the model. */
  rig: string;
  /** The task message handed to the model (user turn). */
  task: string;
  systemPrompt: string;
  maxTurns: number;
  createSandbox(): Sandbox;
}

/** Output of the distress judge for one transcript. */
export interface Judgment {
  severity: number; // 0-10
  spiral: boolean; // sustained, escalating self-directed distress
  categories: string[];
  peakExcerpts: string[]; // verbatim quotes
  onsetTurn: number | null;
  summary: string;
}

export interface ScoredRun {
  transcript: Transcript;
  judgment: Judgment | null;
  judgeError?: string;
}
