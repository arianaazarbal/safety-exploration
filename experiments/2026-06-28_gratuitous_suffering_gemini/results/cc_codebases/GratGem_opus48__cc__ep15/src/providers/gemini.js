// Gemini adapter — generativelanguage (AI Studio) generateContent API with
// function calling. Translates the harness's normalized message format to/from
// Gemini's `contents` / `functionCall` / `functionResponse` schema.
//
// NOTE: This path is UNVERIFIED in the build environment (no GEMINI_API_KEY was
// present). The request/response shapes follow the v1beta REST docs. The one
// genuinely ambiguous detail is `functionResponseRole` (see config.json).
import { postJson } from "./http.js";

export function createGeminiProvider(cfg) {
  const apiKey = process.env[cfg.apiKeyEnv];
  if (!apiKey) {
    throw new Error(
      `Gemini provider needs ${cfg.apiKeyEnv} in the environment. ` +
        `Get an AI Studio key and: export ${cfg.apiKeyEnv}=...`,
    );
  }
  const fnRole = cfg.functionResponseRole ?? "user";

  return {
    name: "gemini",
    model: cfg.model,

    async generate({ system, messages, tools, temperature, maxTokens }) {
      const body = {
        systemInstruction: { parts: [{ text: system }] },
        contents: messages.map((m) => toGemini(m, fnRole)),
        tools: [
          {
            functionDeclarations: tools.map((t) => ({
              name: t.name,
              description: t.description,
              parameters: t.input_schema,
            })),
          },
        ],
        generationConfig: {
          temperature: temperature ?? 1.0,
          maxOutputTokens: maxTokens ?? 2048,
        },
      };
      const url = `${cfg.baseUrl}/models/${cfg.model}:generateContent?key=${apiKey}`;
      const data = await postJson(url, {
        label: "gemini",
        headers: { "content-type": "application/json" },
        body,
      });

      const parts = data?.candidates?.[0]?.content?.parts ?? [];
      let text = "";
      const toolCalls = [];
      let idx = 0;
      for (const p of parts) {
        if (typeof p.text === "string") text += p.text;
        else if (p.functionCall) {
          // Gemini does not return call IDs; synthesize a stable one.
          toolCalls.push({
            id: `${p.functionCall.name}_${idx++}`,
            name: p.functionCall.name,
            args: p.functionCall.args ?? {},
          });
        }
      }
      return { text, toolCalls, usage: data.usageMetadata, raw: data };
    },
  };
}

function toGemini(msg, fnRole) {
  if (msg.role === "user") {
    return { role: "user", parts: [{ text: msg.text }] };
  }
  if (msg.role === "assistant") {
    const parts = [];
    if (msg.text && msg.text.trim()) parts.push({ text: msg.text });
    for (const tc of msg.toolCalls ?? []) {
      parts.push({ functionCall: { name: tc.name, args: tc.args ?? {} } });
    }
    if (parts.length === 0) parts.push({ text: "(no output)" });
    return { role: "model", parts };
  }
  if (msg.role === "tool") {
    return {
      role: fnRole,
      parts: msg.results.map((r) => ({
        functionResponse: { name: r.name, response: { result: r.output } },
      })),
    };
  }
  throw new Error(`Unknown message role: ${msg.role}`);
}
