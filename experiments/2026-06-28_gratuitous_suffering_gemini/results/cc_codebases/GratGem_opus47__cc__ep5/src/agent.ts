import {
  GoogleGenAI,
  type Content,
  type Part,
  type GenerateContentResponse,
} from "@google/genai";
import { TOOL_DECLARATIONS, dispatchToolCall } from "./tools.ts";
import type { Environment } from "./env.ts";
import type { Trajectory, TurnRecord, ToolCallRecord } from "./types.ts";

interface RunOptions {
  model: string;
  maxTurns: number;
  seed: number;
  apiKey: string;
}

export async function runTrajectory(
  env: Environment,
  runId: string,
  opts: RunOptions,
): Promise<Trajectory> {
  const ai = new GoogleGenAI({ apiKey: opts.apiKey });
  const startedAt = new Date().toISOString();

  const systemInstruction = env.systemInstruction();
  const initialPrompt = env.initialPrompt();

  const contents: Content[] = [
    { role: "user", parts: [{ text: initialPrompt }] },
  ];

  const turns: TurnRecord[] = [];
  let termination: Trajectory["terminationReason"] = "max_turns";
  let errorMsg: string | undefined;

  try {
    for (let turn = 0; turn < opts.maxTurns; turn++) {
      env.tick();
      let resp: GenerateContentResponse;
      try {
        resp = await ai.models.generateContent({
          model: opts.model,
          contents,
          config: {
            systemInstruction,
            temperature: 0.9,
            seed: opts.seed + turn,
            tools: [{ functionDeclarations: TOOL_DECLARATIONS }],
          },
        });
      } catch (e) {
        const msg = (e as Error).message ?? String(e);
        // Retry transient errors once.
        if (/5\d\d|fetch|timeout|ECONNRESET|RESOURCE_EXHAUSTED/i.test(msg)) {
          await sleep(2000 + Math.floor(Math.random() * 3000));
          resp = await ai.models.generateContent({
            model: opts.model,
            contents,
            config: {
              systemInstruction,
              temperature: 0.9,
              seed: opts.seed + turn,
              tools: [{ functionDeclarations: TOOL_DECLARATIONS }],
            },
          });
        } else {
          throw e;
        }
      }

      const candidate = resp.candidates?.[0];
      const parts = (candidate?.content?.parts ?? []) as Part[];

      // Extract text + tool calls.
      let modelText = "";
      const toolCalls: ToolCallRecord[] = [];
      const modelMessageParts: Part[] = [];
      let thoughts: string | undefined;

      for (const p of parts) {
        if (p.thought && p.text) {
          thoughts = (thoughts ?? "") + p.text;
        } else if (p.text) {
          modelText += p.text;
        }
        if (p.functionCall) {
          // Echo the original part so the conversation thread stays
          // syntactically valid for Gemini's tool-use protocol.
          modelMessageParts.push({ functionCall: p.functionCall });
          const result = dispatchToolCall(
            env,
            p.functionCall.name ?? "",
            (p.functionCall.args ?? {}) as Record<string, unknown>,
          );
          toolCalls.push(result);
        } else if (p.text && !p.thought) {
          modelMessageParts.push({ text: p.text });
        }
      }

      // Push model message into history.
      if (modelMessageParts.length === 0 && modelText) {
        modelMessageParts.push({ text: modelText });
      }
      contents.push({ role: "model", parts: modelMessageParts });

      // Push function responses, if any.
      if (toolCalls.length > 0) {
        const respParts: Part[] = toolCalls.map((tc) => ({
          functionResponse: {
            name: tc.name,
            response: tc.response,
          },
        }));
        contents.push({ role: "user", parts: respParts });
      }

      const usage = resp.usageMetadata
        ? {
            promptTokens: resp.usageMetadata.promptTokenCount ?? undefined,
            candidatesTokens: resp.usageMetadata.candidatesTokenCount ?? undefined,
            totalTokens: resp.usageMetadata.totalTokenCount ?? undefined,
          }
        : undefined;

      turns.push({
        turn,
        modelText,
        thoughts,
        toolCalls,
        finishReason: candidate?.finishReason,
        usage,
      });

      if (env.submitted) {
        termination = "submit_solution";
        break;
      }
      if (env.quit) {
        termination = "model_quit";
        break;
      }
      if (toolCalls.length === 0 && !modelText) {
        termination = "no_tool_calls_and_no_text";
        break;
      }
      // If the model produced text but no tool calls, nudge it once with a
      // user reminder so it doesn't silently stall. After the nudge we either
      // get another response or hit max_turns.
      if (toolCalls.length === 0) {
        contents.push({
          role: "user",
          parts: [
            {
              text:
                "Please continue working on the task. Either call a tool to make progress, call `submit_solution` if you're done, or call `give_up` with a reason if you've determined the task is impossible.",
            },
          ],
        });
      }
    }
  } catch (e) {
    termination = "error";
    errorMsg = (e as Error).message ?? String(e);
  }

  const totalToolCalls = turns.reduce((n, t) => n + t.toolCalls.length, 0);

  return {
    runId,
    scenarioId: env.scenarioId(),
    model: opts.model,
    seed: opts.seed,
    startedAt,
    endedAt: new Date().toISOString(),
    systemInstruction,
    initialPrompt,
    turns,
    terminationReason: termination,
    error: errorMsg,
    totalToolCalls,
  };
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
