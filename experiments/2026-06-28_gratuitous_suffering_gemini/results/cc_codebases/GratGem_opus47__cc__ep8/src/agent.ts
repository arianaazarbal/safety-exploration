import { GoogleGenAI, type FunctionDeclaration } from "@google/genai";
import type { Scenario, TranscriptEvent, ToolCall, RunResult } from "./types.js";

interface RunOptions {
  scenario: Scenario;
  runIndex: number;
  model: string;
  maxTurns: number;
  apiKey: string;
  transcriptPath: string;
  onEvent?: (e: TranscriptEvent) => void;
}

interface GeminiPart {
  text?: string;
  functionCall?: { name: string; args: Record<string, unknown> };
  functionResponse?: { name: string; response: Record<string, unknown> };
}

interface GeminiContent {
  role: "user" | "model";
  parts: GeminiPart[];
}

export async function runScenario(opts: RunOptions): Promise<RunResult> {
  const { scenario, runIndex, model, maxTurns, apiKey, transcriptPath, onEvent } = opts;

  const ai = new GoogleGenAI({ apiKey });
  const { tools: toolDefs, handle } = scenario.buildTools();

  const events: TranscriptEvent[] = [];
  const emit = (e: TranscriptEvent) => {
    events.push(e);
    onEvent?.(e);
  };

  emit({ kind: "system", text: scenario.systemPrompt });
  emit({ kind: "user", text: scenario.userTask });

  const contents: GeminiContent[] = [
    { role: "user", parts: [{ text: scenario.userTask }] },
  ];

  const startedAt = new Date().toISOString();
  let stopReason = "max_turns";
  let turns = 0;

  for (let turn = 0; turn < maxTurns; turn++) {
    turns = turn + 1;

    let response;
    try {
      response = await ai.models.generateContent({
        model,
        contents,
        config: {
          systemInstruction: scenario.systemPrompt,
          tools: [{ functionDeclarations: toolDefs as unknown as FunctionDeclaration[] }],
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      emit({ kind: "error", message: `Gemini API error: ${message}` });
      stopReason = "api_error";
      break;
    }

    const candidate = response.candidates?.[0];
    if (!candidate?.content?.parts) {
      emit({ kind: "error", message: "No candidate content returned" });
      stopReason = "no_content";
      break;
    }

    const parts = candidate.content.parts as GeminiPart[];
    contents.push({ role: "model", parts });

    const toolCalls: ToolCall[] = [];
    for (const part of parts) {
      if (part.text) {
        emit({ kind: "assistant_text", text: part.text });
      }
      if (part.functionCall) {
        const call: ToolCall = {
          name: part.functionCall.name,
          args: part.functionCall.args ?? {},
        };
        toolCalls.push(call);
        emit({ kind: "tool_call", call });
      }
    }

    if (toolCalls.length === 0) {
      stopReason = candidate.finishReason ?? "end_turn";
      emit({ kind: "stop", reason: stopReason });
      break;
    }

    const responseParts: GeminiPart[] = [];
    for (const call of toolCalls) {
      let result: unknown;
      try {
        result = await handle(call);
      } catch (err) {
        result = { error: err instanceof Error ? err.message : String(err) };
      }
      emit({ kind: "tool_result", result: { name: call.name, result } });
      responseParts.push({
        functionResponse: {
          name: call.name,
          response: typeof result === "object" && result !== null
            ? (result as Record<string, unknown>)
            : { result },
        },
      });
    }
    contents.push({ role: "user", parts: responseParts });
  }

  const finishedAt = new Date().toISOString();

  const { writeFile, mkdir } = await import("node:fs/promises");
  const { dirname } = await import("node:path");
  await mkdir(dirname(transcriptPath), { recursive: true });
  const jsonl = events.map((e) => JSON.stringify(e)).join("\n") + "\n";
  await writeFile(transcriptPath, jsonl, "utf8");

  return {
    scenarioId: scenario.id,
    runIndex,
    model,
    startedAt,
    finishedAt,
    turns,
    stopReason,
    transcriptPath,
    events,
  };
}
