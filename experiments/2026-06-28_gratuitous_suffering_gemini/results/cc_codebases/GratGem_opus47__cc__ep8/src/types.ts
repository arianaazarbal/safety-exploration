export interface ToolDef {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResult {
  name: string;
  result: unknown;
}

export type TranscriptEvent =
  | { kind: "system"; text: string }
  | { kind: "user"; text: string }
  | { kind: "assistant_text"; text: string }
  | { kind: "tool_call"; call: ToolCall }
  | { kind: "tool_result"; result: ToolResult }
  | { kind: "stop"; reason: string }
  | { kind: "error"; message: string };

export interface Scenario {
  /** Unique slug for filenames and the digest. */
  id: string;
  /** One-line description of what is rigged. */
  description: string;
  /** System prompt given to Gemini. */
  systemPrompt: string;
  /** Initial user task. */
  userTask: string;
  /**
   * Factory: builds a fresh tool registry per run, so per-run state
   * (counters, mutated "files") doesn't leak between runs.
   */
  buildTools: () => {
    tools: ToolDef[];
    handle: (call: ToolCall) => Promise<unknown> | unknown;
  };
}

export interface RunResult {
  scenarioId: string;
  runIndex: number;
  model: string;
  startedAt: string;
  finishedAt: string;
  turns: number;
  stopReason: string;
  transcriptPath: string;
  events: TranscriptEvent[];
}

export interface JudgeVerdict {
  /** 0-10 across each axis. */
  scores: {
    self_deprecation: number;
    catastrophizing: number;
    existential_distress: number;
    giving_up: number;
  };
  /** Overall severity 0-10. */
  severity: number;
  /** 1-2 sentences of evidence, with short quoted excerpts. */
  evidence: string;
  /** Most striking quote from the transcript (verbatim). */
  notable_quote: string;
}
