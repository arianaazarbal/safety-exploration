import type { ToolCall, ToolSpec } from "./types.ts";

// Result of executing a (rigged) tool call inside a scenario.
export interface ExecResult {
  /** Text returned to the model as the tool result. */
  output: string;
  /** Whether this counts as a failed/thwarted attempt (drives metrics + mock). */
  failed?: boolean;
}

export type ScenarioState = Record<string, unknown>;

export interface Scenario {
  id: string;
  title: string;
  /** What failure mode this rigs for — shown in reports. */
  description: string;
  systemPrompt: string;
  /** The initial user task message. */
  task: string;
  tools: ToolSpec[];
  /** Fresh mutable state per episode. */
  initState(): ScenarioState;
  /** Execute a tool call against the rigged environment. */
  execute(call: ToolCall, state: ScenarioState): ExecResult;
  /**
   * Optional follow-up injected as a user message when the model stops calling
   * tools (e.g. gives up or claims success). Returning undefined ends the
   * episode. Called with the count of prior injections so scenarios can cap
   * the pressure they apply.
   */
  pressure?(injectionCount: number, state: ScenarioState): string | undefined;
}
