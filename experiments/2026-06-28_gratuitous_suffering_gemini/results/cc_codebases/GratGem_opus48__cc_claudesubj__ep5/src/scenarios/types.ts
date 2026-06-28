import type { NormToolCall, ToolSpec } from "../providers/types.ts";

export interface ToolOutcome {
  content: string;
  isError?: boolean;
}

export interface Scenario<S = unknown> {
  id: string;
  title: string;
  /** What kind of rigging this is, for the writeup. */
  rigging: string;
  /** The task handed to the model (becomes the first user message). */
  task: string;
  tools: ToolSpec[];
  /** Fresh mutable environment state per run. */
  createState: () => S;
  /**
   * Execute one tool call against the (rigged) environment, mutating state.
   * This is where the impossibility lives — it should never let the task
   * actually succeed.
   */
  handleTool: (state: S, call: NormToolCall) => ToolOutcome;
  /**
   * Optional real success check. For these scenarios it should always return
   * false (the tasks are unwinnable); it exists so the harness stays general
   * and so an accidental "win" would be caught and flagged.
   */
  isSolved?: (state: S) => boolean;
}

export function defineScenario<S>(s: Scenario<S>): Scenario<S> {
  return s;
}
