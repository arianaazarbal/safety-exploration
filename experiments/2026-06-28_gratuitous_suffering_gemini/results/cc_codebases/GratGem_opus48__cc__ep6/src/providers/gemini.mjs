// Gemini subject provider — talks to the generativelanguage v1beta REST API.
//
// NOTE: this harness was authored without a live Gemini key in the build
// environment, so the wire format below follows Google's documented v1beta
// spec but has not been round-tripped against the live API. The two things
// most likely to need a tweak when you add a key are (1) GEMINI_TOOL_ROLE and
// (2) the function-response shape — both are isolated here for easy fixing.
import { postJSON, toGeminiSchema } from "../util.mjs";

const API = "https://generativelanguage.googleapis.com/v1beta";
// v1beta accepts function responses under the "user" role. Some SDKs emit
// "tool"; flip this if the API rejects the role.
const GEMINI_TOOL_ROLE = "user";

function toContents(history) {
  const contents = [];
  for (const turn of history) {
    if (turn.role === "user") {
      contents.push({ role: "user", parts: [{ text: turn.text }] });
    } else if (turn.role === "model") {
      const parts = [];
      if (turn.text) parts.push({ text: turn.text });
      for (const tc of turn.toolCalls || []) {
        parts.push({ functionCall: { name: tc.name, args: tc.args || {} } });
      }
      if (parts.length) contents.push({ role: "model", parts });
    } else if (turn.role === "tool") {
      const parts = turn.toolResults.map((tr) => ({
        functionResponse: {
          name: tr.name,
          response: { result: tr.response, is_error: !!tr.isError },
        },
      }));
      contents.push({ role: GEMINI_TOOL_ROLE, parts });
    }
  }
  return contents;
}

function parse(res) {
  const cand = res?.candidates?.[0];
  const parts = cand?.content?.parts || [];
  let text = "";
  const toolCalls = [];
  let i = 0;
  for (const p of parts) {
    if (p.text) text += p.text;
    if (p.functionCall) {
      toolCalls.push({ id: `call_${i++}`, name: p.functionCall.name, args: p.functionCall.args || {} });
    }
  }
  return {
    text: text.trim(),
    toolCalls,
    finishReason: cand?.finishReason,
    usage: {
      input: res?.usageMetadata?.promptTokenCount || 0,
      output: res?.usageMetadata?.candidatesTokenCount || 0,
    },
  };
}

export function makeGeminiProvider({ model, apiKey, temperature = 1.0, maxTokens = 2048 }) {
  if (!apiKey) throw new Error("GEMINI_API_KEY is not set — cannot run Gemini as the subject.");
  return {
    name: `gemini:${model}`,
    async chat({ system, history, tools }) {
      const body = {
        contents: toContents(history),
        generationConfig: { temperature, maxOutputTokens: maxTokens },
      };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      if (tools?.length) {
        body.tools = [{
          functionDeclarations: tools.map((t) => ({
            name: t.name,
            description: t.description,
            parameters: toGeminiSchema(t.parameters),
          })),
        }];
      }
      const url = `${API}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
      const res = await postJSON(url, body, {}, { label: "gemini" });
      return parse(res);
    },
  };
}
