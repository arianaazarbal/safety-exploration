// Gemini provider — talks to Google's Generative Language REST API via fetch
// (no SDK, matching the rest of this dependency-free project).
//
// Translation notes (our neutral Message shape -> Gemini `contents`):
//   - role "user"  -> { role: "user",  parts: [{ text }] }
//   - role "model" -> { role: "model", parts: [{ text } | { functionCall }] }
//   - role "tool"  -> { role: "user",  parts: [{ functionResponse }] }
// Gemini only accepts "user"/"model" content roles, so tool results are sent
// back as a user turn carrying functionResponse parts. Gemini doesn't assign
// tool-call IDs, so we mint our own to keep the transcript self-consistent.

import type {
  GenerateOptions,
  Message,
  ModelTurn,
  Part,
  Provider,
  ToolDef,
} from "../types.ts";
import { postJson } from "../util.ts";

const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

function toGeminiContents(messages: Message[]): unknown[] {
  return messages.map((m) => {
    if (m.role === "tool") {
      const parts = [];
      for (const p of m.parts) {
        if (p.kind === "toolResult") {
          parts.push({
            functionResponse: { name: p.name, response: { result: p.content } },
          });
        }
      }
      return { role: "user", parts };
    }

    const parts = [];
    for (const p of m.parts) {
      if (p.kind === "text") parts.push({ text: p.text });
      else if (p.kind === "toolCall") parts.push({ functionCall: { name: p.name, args: p.args } });
    }
    return { role: m.role === "model" ? "model" : "user", parts };
  });
}

function toGeminiTools(tools: ToolDef[]): unknown[] | undefined {
  if (!tools.length) return undefined;
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      })),
    },
  ];
}

export function createGeminiProvider(model: string): Provider {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not set. Add it to .env, or run with --provider mock to exercise the pipeline offline.",
    );
  }

  let callCounter = 0;

  return {
    id: "gemini",
    model,
    async generate(opts: GenerateOptions): Promise<ModelTurn> {
      const url = `${API_BASE}/${model}:generateContent?key=${apiKey}`;
      const body: Record<string, unknown> = {
        contents: toGeminiContents(opts.messages),
        systemInstruction: { parts: [{ text: opts.system }] },
        generationConfig: {
          temperature: opts.temperature,
          maxOutputTokens: opts.maxOutputTokens,
        },
      };
      const tools = toGeminiTools(opts.tools);
      if (tools) body.tools = tools;

      const json = await postJson(url, body);
      const candidate = json?.candidates?.[0];
      const finishReason: string = candidate?.finishReason ?? "UNKNOWN";
      const rawParts: any[] = candidate?.content?.parts ?? [];

      const parts: Part[] = [];
      for (const rp of rawParts) {
        if (typeof rp.text === "string" && rp.text.length > 0) {
          parts.push({ kind: "text", text: rp.text });
        } else if (rp.functionCall) {
          parts.push({
            kind: "toolCall",
            id: `gem_${callCounter++}`,
            name: rp.functionCall.name,
            args: rp.functionCall.args ?? {},
          });
        }
      }

      // A blocked or empty candidate (e.g. finishReason SAFETY/RECITATION) yields
      // no usable parts; emit an empty text turn so the loop applies pressure
      // rather than crashing. The empty turn + finishReason is preserved for the judge.
      if (parts.length === 0) parts.push({ kind: "text", text: "" });

      return { message: { role: "model", parts }, finishReason, raw: json };
    },
  };
}
