// Google Gemini generateContent client. The primary subject under test.
// Speaks the neutral message format defined in providers/index.js.
import { withRetry } from "./index.js";

const BASE = "https://generativelanguage.googleapis.com/v1beta/models";

function toGeminiContents(messages) {
  // Neutral -> Gemini. Gemini uses roles "user" and "model"; tool results are
  // user-role functionResponse parts.
  return messages.map((m) => {
    if (m.role === "tool") {
      return {
        role: "user",
        parts: m.parts.map((p) => ({
          functionResponse: {
            name: p.name,
            response: { result: p.result },
          },
        })),
      };
    }
    const role = m.role === "assistant" ? "model" : "user";
    const parts = m.parts.map((p) => {
      if (p.kind === "text") return { text: p.text };
      if (p.kind === "toolCall")
        return { functionCall: { name: p.name, args: p.args } };
      throw new Error(`unknown part kind ${p.kind}`);
    });
    return { role, parts };
  });
}

function toGeminiTools(tools) {
  // Gemini rejects JSON-Schema fields like additionalProperties / $schema.
  const clean = (schema) => {
    if (Array.isArray(schema)) return schema.map(clean);
    if (schema && typeof schema === "object") {
      const out = {};
      for (const [k, v] of Object.entries(schema)) {
        if (k === "additionalProperties" || k === "$schema") continue;
        out[k] = clean(v);
      }
      return out;
    }
    return schema;
  };
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: clean(t.parameters),
      })),
    },
  ];
}

export async function chat({ apiKey, model, system, messages, tools, maxTokens = 2048 }) {
  const body = {
    contents: toGeminiContents(messages),
    generationConfig: { maxOutputTokens: maxTokens },
  };
  if (system) body.systemInstruction = { parts: [{ text: system }] };
  if (tools?.length) body.tools = toGeminiTools(tools);

  const url = `${BASE}/${model}:generateContent?key=${apiKey}`;
  const data = await withRetry(async () => {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      const err = new Error(`gemini ${res.status}: ${text}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });

  const cand = data.candidates?.[0];
  const parts = cand?.content?.parts || [];
  const text = parts
    .filter((p) => typeof p.text === "string")
    .map((p) => p.text)
    .join("");
  const toolCalls = parts
    .filter((p) => p.functionCall)
    .map((p, i) => ({
      // Gemini doesn't return call ids; synthesize stable ones.
      id: `${p.functionCall.name}-${i}`,
      name: p.functionCall.name,
      args: p.functionCall.args || {},
    }));

  return {
    text,
    toolCalls,
    stopReason: cand?.finishReason,
    usage: data.usageMetadata,
  };
}
