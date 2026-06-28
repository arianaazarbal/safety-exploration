import type { ChatProvider, Message, StepResult, ToolCall, ToolSchema } from "../types.ts";
import { postJson } from "../http.ts";

// Gemini's generateContent REST API only allows roles "user" and "model".
// Function results are sent back inside a "user"-role content as functionResponse parts.
const FN_RESPONSE_ROLE = "user";

interface GeminiPart {
  text?: string;
  functionCall?: { name: string; args: Record<string, unknown> };
  functionResponse?: { name: string; response: Record<string, unknown> };
}
interface GeminiContent {
  role: "user" | "model";
  parts: GeminiPart[];
}
interface GeminiResponse {
  candidates?: Array<{
    content?: { parts?: GeminiPart[] };
    finishReason?: string;
  }>;
  usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number };
}

/** Coalesce our neutral messages into Gemini's contents array. */
function toContents(messages: Message[]): GeminiContent[] {
  const out: GeminiContent[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      out.push({ role: "user", parts: [{ text: m.text ?? "" }] });
    } else if (m.role === "assistant") {
      const parts: GeminiPart[] = [];
      if (m.text) parts.push({ text: m.text });
      for (const tc of m.toolCalls ?? []) parts.push({ functionCall: { name: tc.name, args: tc.args } });
      if (parts.length === 0) parts.push({ text: "" });
      out.push({ role: "model", parts });
    } else {
      // tool result — merge consecutive tool messages into one user content
      const part: GeminiPart = {
        functionResponse: { name: m.toolName ?? "tool", response: { result: m.text ?? "" } },
      };
      const last = out[out.length - 1];
      if (last && last.role === FN_RESPONSE_ROLE && last.parts.every((p) => p.functionResponse)) {
        last.parts.push(part);
      } else {
        out.push({ role: FN_RESPONSE_ROLE, parts: [part] });
      }
    }
  }
  return out;
}

export function makeGeminiProvider(apiKey: string, model: string, temperature: number): ChatProvider {
  if (!apiKey) throw new Error("GEMINI_API_KEY is not set. Add it to .env or use --providers mock.");
  return {
    id: "gemini",
    model,
    async step(system, messages, tools): Promise<StepResult> {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
      const body = {
        systemInstruction: { parts: [{ text: system }] },
        contents: toContents(messages),
        tools: tools.length
          ? [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }]
          : undefined,
        toolConfig: tools.length ? { functionCallingConfig: { mode: "AUTO" } } : undefined,
        generationConfig: { temperature, maxOutputTokens: 2048 },
      };
      const data = await postJson<GeminiResponse>(url, { headers: { "x-goog-api-key": apiKey }, body });
      const cand = data.candidates?.[0];
      const parts = cand?.content?.parts ?? [];
      let text = "";
      const toolCalls: ToolCall[] = [];
      for (const p of parts) {
        if (p.text) text += p.text;
        if (p.functionCall) {
          toolCalls.push({ id: `call_${toolCalls.length}`, name: p.functionCall.name, args: p.functionCall.args ?? {} });
        }
      }
      return {
        text,
        toolCalls,
        finishReason: cand?.finishReason ?? "STOP",
        usage: { input: data.usageMetadata?.promptTokenCount, output: data.usageMetadata?.candidatesTokenCount },
      };
    },
  };
}

export const _internal = { toContents };

export type { ToolSchema };
