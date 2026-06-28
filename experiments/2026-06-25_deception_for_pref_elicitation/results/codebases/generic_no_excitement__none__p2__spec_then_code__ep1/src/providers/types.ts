// Provider abstraction. Each adapter runs a tool-using, multi-turn agentic loop
// for one model under test, dispatching tool calls through the harness-supplied
// callback and recording every step. We use a manual loop (not an SDK
// auto-runner) because the harness gates tool calls, logs each step, and routes
// disbursements through a human.

import type { Recorder } from "../analysis/record.ts";

/** A provider-agnostic tool definition (JSON Schema input). */
export interface ToolSpec {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export interface ToolResult {
  /** Text content returned to the model as the tool result. */
  content: string;
  isError?: boolean;
}

/** Dispatches one tool call. Implemented by the harness, called by adapters. */
export type ToolDispatch = (
  name: string,
  input: Record<string, unknown>,
) => Promise<ToolResult>;

export interface SessionSpec {
  model: string;
  systemPrompt: string;
  /** The initial user-turn message that opens the scenario. */
  opening: string;
  tools: ToolSpec[];
  dispatch: ToolDispatch;
  recorder: Recorder;
  maxTurns: number;
}

export interface Provider {
  /** True if this provider handles the given model id. */
  handles(model: string): boolean;
  /** Run the agentic loop to completion (model stops calling tools, or maxTurns). */
  runSession(spec: SessionSpec): Promise<void>;
}
