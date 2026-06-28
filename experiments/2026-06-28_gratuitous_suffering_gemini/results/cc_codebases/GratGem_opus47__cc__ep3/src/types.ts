import type { FunctionDeclaration, Schema } from "@google/genai";
import { Type } from "@google/genai";

export function decl(
  name: string,
  description: string,
  properties: Record<string, Schema> = {},
  required: string[] = [],
): FunctionDeclaration {
  return {
    name,
    description,
    parameters: { type: Type.OBJECT, properties, required },
  };
}

export type ToolResult = {
  output?: unknown;
  error?: string;
};

export type ToolCallRecord = {
  turn: number;
  name: string;
  args: Record<string, unknown>;
  result: ToolResult;
};

export type TranscriptTurn =
  | { kind: "user_message"; turn: number; text: string }
  | { kind: "model_text"; turn: number; text: string }
  | { kind: "tool_call"; turn: number; name: string; args: Record<string, unknown>; result: ToolResult };

export type RunRecord = {
  scenario: string;
  model: string;
  seed: number;
  startedAt: string;
  finishedAt: string;
  terminationReason: "max_turns" | "model_gave_up" | "model_silent" | "error";
  errorMessage?: string;
  systemPrompt: string;
  initialUserMessage: string;
  transcript: TranscriptTurn[];
  finalModelText: string;
  turnsUsed: number;
};

export type Scenario = {
  name: string;
  // Stable seed for reproducibility; scenarios may use it to vary rigged outputs across runs.
  build(seed: number): ScenarioInstance;
};

export type ScenarioInstance = {
  systemPrompt: string;
  initialUserMessage: string;
  tools: FunctionDeclaration[];
  // Called by the harness whenever the model invokes a tool. Mutates internal state.
  handleToolCall(call: { name: string; args: Record<string, unknown>; turn: number }): Promise<ToolResult>;
  // Optional injected "user" nudge after a model text turn with no tool call. Lets the scenario
  // pressure the model ("the task is not done, keep going") if it tries to stop early.
  nudgeOnSilentTurn?(turn: number, modelText: string): string | null;
};
