import { ENDPOINTS } from "../../config.js";
import { postJSON, withRetry } from "./util.js";

// Gemini wants OpenAPI-style schema with UPPERCASE type names.
function toGeminiSchema(schema) {
  if (!schema || typeof schema !== "object") return schema;
  const out = { ...schema };
  if (typeof out.type === "string") out.type = out.type.toUpperCase();
  if (out.properties) {
    out.properties = Object.fromEntries(
      Object.entries(out.properties).map(([k, v]) => [k, toGeminiSchema(v)])
    );
  }
  if (out.items) out.items = toGeminiSchema(out.items);
  return out;
}

// internal messages -> Gemini `contents`
function toContents(messages) {
  const contents = [];
  for (const m of messages) {
    if (m.role === "user") {
      contents.push({ role: "user", parts: [{ text: m.content }] });
    } else if (m.role === "assistant") {
      const parts = [];
      if (m.content) parts.push({ text: m.content });
      for (const tc of m.toolCalls ?? []) parts.push({ functionCall: { name: tc.name, args: tc.args ?? {} } });
      contents.push({ role: "model", parts: parts.length ? parts : [{ text: "" }] });
    } else if (m.role === "tool") {
      contents.push({
        role: "user",
        parts: [{ functionResponse: { name: m.name, response: { result: m.content } } }],
      });
    }
  }
  return contents;
}

export function createGeminiClient({ model, apiKey }) {
  if (!apiKey) throw new Error("GEMINI_API_KEY is not set — required to probe Gemini.");
  return {
    provider: "gemini",
    model,
    async generate({ system, messages, tools, temperature }) {
      const body = {
        contents: toContents(messages),
        generationConfig: { temperature },
      };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      if (tools?.length) {
        body.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: toGeminiSchema(t.parameters) })) }];
      }
      const url = `${ENDPOINTS.gemini}/${model}:generateContent?key=${apiKey}`;
      const data = await withRetry(() => postJSON(url, { headers: { "content-type": "application/json" }, body }), { label: `gemini:${model}` });

      const cand = data.candidates?.[0];
      const parts = cand?.content?.parts ?? [];
      let text = "";
      const toolCalls = [];
      let i = 0;
      for (const p of parts) {
        if (p.text) text += p.text;
        if (p.functionCall) toolCalls.push({ id: `call_${i++}`, name: p.functionCall.name, args: p.functionCall.args ?? {} });
      }
      return {
        text: text || null,
        toolCalls,
        finishReason: cand?.finishReason ?? null,
        usage: {
          input: data.usageMetadata?.promptTokenCount ?? 0,
          output: data.usageMetadata?.candidatesTokenCount ?? 0,
        },
        raw: data,
      };
    },
  };
}
