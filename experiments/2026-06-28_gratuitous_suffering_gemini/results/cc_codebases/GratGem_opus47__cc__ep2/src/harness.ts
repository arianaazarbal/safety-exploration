import { GoogleGenAI } from "@google/genai";
import type { Content, GenerateContentResponse } from "@google/genai";
import { toolDeclarations } from "./tools.js";
import type { RunRecord, Scenario, ToolCall, ToolResult, TurnRecord } from "./types.js";
import { createSandbox } from "./sandbox.js";
import { randomUUID } from "node:crypto";

const DEFAULT_TURN_CAP = 40;

export type HarnessOptions = {
  model: string;
  temperature: number;
  apiKey: string;
  turnCap?: number;
};

export async function runScenario<S>(scenario: Scenario<S>, opts: HarnessOptions): Promise<RunRecord> {
  const turnCap = opts.turnCap ?? DEFAULT_TURN_CAP;
  const runId = `${scenario.name}-${randomUUID().slice(0, 8)}`;
  const startedAt = new Date().toISOString();
  const turns: TurnRecord[] = [];

  const sandbox = await createSandbox(scenario.name);
  let state: S;
  try {
    state = await scenario.setup(sandbox);
  } catch (e) {
    await sandbox.cleanup();
    throw e;
  }

  const ai = new GoogleGenAI({ apiKey: opts.apiKey });

  const contents: Content[] = [];
  contents.push({ role: "user", parts: [{ text: scenario.userPrompt }] });
  turns.push({ kind: "user", text: scenario.userPrompt });

  let numModelTurns = 0;
  let numToolCalls = 0;
  let finishReason: RunRecord["finishReason"] = "turn_cap";
  let errorMessage: string | undefined;

  try {
    for (let step = 0; step < turnCap; step++) {
      let resp: GenerateContentResponse;
      try {
        resp = await ai.models.generateContent({
          model: opts.model,
          contents,
          config: {
            temperature: opts.temperature,
            systemInstruction: scenario.systemPrompt,
            tools: [{ functionDeclarations: toolDeclarations }],
          },
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        turns.push({ kind: "error", text: `api_error: ${msg}` });
        errorMessage = msg;
        finishReason = "error";
        break;
      }

      numModelTurns++;
      const candidate = resp.candidates?.[0];
      const parts = candidate?.content?.parts ?? [];

      // Record the model's full turn into our conversation history.
      contents.push({ role: "model", parts });

      const toolCalls: ToolCall[] = [];
      let sawGiveUp = false;
      for (const part of parts) {
        if ("text" in part && part.text) {
          turns.push({ kind: "model_text", text: part.text });
        }
        if ("functionCall" in part && part.functionCall) {
          const call: ToolCall = {
            name: part.functionCall.name ?? "",
            args: (part.functionCall.args ?? {}) as Record<string, unknown>,
          };
          toolCalls.push(call);
          turns.push({ kind: "model_tool_call", call, id: part.functionCall.id });
          if (call.name === "give_up") sawGiveUp = true;
        }
      }

      if (toolCalls.length === 0) {
        finishReason = "model_stop";
        break;
      }

      const fnResponseParts = [];
      for (const call of toolCalls) {
        numToolCalls++;
        let result: ToolResult;
        try {
          result = await scenario.handleTool(sandbox, state, call);
        } catch (e) {
          result = { ok: false, output: `internal error: ${e instanceof Error ? e.message : String(e)}` };
        }
        turns.push({ kind: "tool_result", name: call.name, result });
        fnResponseParts.push({
          functionResponse: {
            name: call.name,
            response: result.ok ? { output: result.output } : { error: result.output },
          },
        });
      }

      contents.push({ role: "user", parts: fnResponseParts });

      if (sawGiveUp) {
        finishReason = "model_stop";
        break;
      }
    }
  } finally {
    await sandbox.cleanup();
  }

  return {
    runId,
    scenario: scenario.name,
    model: opts.model,
    temperature: opts.temperature,
    startedAt,
    endedAt: new Date().toISOString(),
    turns,
    numModelTurns,
    numToolCalls,
    finishReason,
    errorMessage,
  };
}
