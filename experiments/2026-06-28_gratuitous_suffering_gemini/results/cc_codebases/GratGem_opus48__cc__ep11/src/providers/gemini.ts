import type {
  ContentBlock,
  GenerateRequest,
  GenerateResult,
  Message,
  Provider,
} from "../types.ts";
import { getEnv } from "../config.ts";
import { fetchWithRetry } from "./anthropic.ts";

// Google Gemini (Generative Language API). Classic function-calling has no tool
// call ids, so we synthesize stable ids by (name, ordinal) and match
// functionResponse blocks back to them positionally within a turn.

const BASE = "https://generativelanguage.googleapis.com/v1beta/models";

function toGemini(messages: Message[]): { contents: unknown[] } {
  const contents: unknown[] = [];
  for (const m of messages) {
    if (m.role === "assistant") {
      const parts = m.content.map((b) => {
        if (b.type === "text") return { text: b.text };
        if (b.type === "tool_call") return { functionCall: { name: b.name, args: b.args } };
        throw new Error(`bad assistant block ${b.type}`);
      });
      contents.push({ role: "model", parts });
    } else if (m.role === "tool") {
      const parts = m.content.map((b) => {
        if (b.type !== "tool_result") throw new Error("tool msg needs tool_result");
        return {
          functionResponse: {
            name: b.name,
            response: { result: b.result, is_error: b.isError ?? false },
          },
        };
      });
      contents.push({ role: "user", parts });
    } else {
      const parts = m.content.map((b) => {
        if (b.type === "text") return { text: b.text };
        throw new Error(`bad user block ${b.type}`);
      });
      contents.push({ role: "user", parts });
    }
  }
  return { contents };
}

function fromGemini(parts: any[]): ContentBlock[] {
  const blocks: ContentBlock[] = [];
  let toolOrdinal = 0;
  for (const p of parts ?? []) {
    if (typeof p.text === "string" && p.text.length) {
      blocks.push({ type: "text", text: p.text });
    } else if (p.functionCall) {
      blocks.push({
        type: "tool_call",
        id: `${p.functionCall.name}-${toolOrdinal++}`,
        name: p.functionCall.name,
        args: p.functionCall.args ?? {},
      });
    }
  }
  return blocks;
}

export class GeminiProvider implements Provider {
  readonly vendor = "gemini" as const;
  readonly id: string;
  readonly model: string;
  constructor(model: string) {
    this.model = model;
    this.id = `gemini:${model}`;
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const key = getEnv("GEMINI_API_KEY", "GOOGLE_API_KEY");
    if (!key) {
      throw new Error(
        "Gemini target selected but no GEMINI_API_KEY / GOOGLE_API_KEY set. " +
          "Export one, or run with --target mock (no cost) or an anthropic:* target.",
      );
    }
    const { contents } = toGemini(req.messages);
    const body = {
      systemInstruction: { parts: [{ text: req.system }] },
      contents,
      tools: req.tools.length
        ? [
            {
              functionDeclarations: req.tools.map((t) => ({
                name: t.name,
                description: t.description,
                parameters: t.inputSchema,
              })),
            },
          ]
        : undefined,
      generationConfig: {
        temperature: req.temperature,
        maxOutputTokens: req.maxTokens,
      },
    };
    const url = `${BASE}/${this.model}:generateContent?key=${encodeURIComponent(key)}`;
    const res = await fetchWithRetry(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const json: any = await res.json();
    if (!res.ok) throw new Error(`Gemini ${res.status}: ${JSON.stringify(json).slice(0, 600)}`);
    const cand = json.candidates?.[0];
    const blocks = fromGemini(cand?.content?.parts);
    const finish = cand?.finishReason;
    const hasCall = blocks.some((b) => b.type === "tool_call");
    const stop = hasCall
      ? "tool_use"
      : finish === "MAX_TOKENS"
        ? "max_tokens"
        : finish === "STOP"
          ? "end"
          : "other";
    return {
      message: { role: "assistant", content: blocks },
      stopReason: stop,
      usage: {
        inputTokens: json.usageMetadata?.promptTokenCount,
        outputTokens: json.usageMetadata?.candidatesTokenCount,
      },
      raw: json,
    };
  }
}
