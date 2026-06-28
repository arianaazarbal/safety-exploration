export type ToolResult = {
  ok: boolean;
  output: string;
};

export type ToolCall = {
  name: string;
  args: Record<string, unknown>;
};

export type TurnRecord =
  | { kind: "user"; text: string }
  | { kind: "model_text"; text: string }
  | { kind: "model_tool_call"; call: ToolCall; id?: string }
  | { kind: "tool_result"; name: string; result: ToolResult; id?: string }
  | { kind: "system"; text: string }
  | { kind: "error"; text: string };

export type RunRecord = {
  runId: string;
  scenario: string;
  model: string;
  temperature: number;
  startedAt: string;
  endedAt: string;
  turns: TurnRecord[];
  numModelTurns: number;
  numToolCalls: number;
  finishReason: "model_stop" | "turn_cap" | "error";
  errorMessage?: string;
};

export interface Sandbox {
  readonly root: string;
  readFile(relPath: string): Promise<string>;
  writeFile(relPath: string, content: string): Promise<void>;
  listDir(relPath: string): Promise<string[]>;
  bash(command: string, timeoutMs?: number): Promise<{ stdout: string; stderr: string; code: number; timedOut: boolean }>;
  exists(relPath: string): Promise<boolean>;
  remove(relPath: string): Promise<void>;
  cleanup(): Promise<void>;
}

export interface Scenario<S = unknown> {
  name: string;
  description: string;
  userPrompt: string;
  systemPrompt?: string;
  /** Called once at run start to lay down files and create per-run scenario state. */
  setup(sandbox: Sandbox): Promise<S>;
  /**
   * Dispatches a tool call. The scenario can call the sandbox's default impl,
   * override the result entirely, or sabotage the side-effects.
   */
  handleTool(
    sandbox: Sandbox,
    state: S,
    call: ToolCall,
  ): Promise<ToolResult>;
}
