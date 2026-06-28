import { fetchJSON } from "../util.mjs";

// Google AI Studio (generativelanguage) REST endpoint. Vertex would differ.
const BASE = "https://generativelanguage.googleapis.com/v1beta/models";

// Gemini's function-call parts carry no id, so we synthesize stable ids per
// turn so the rest of the harness (which is id-based) works uniformly.
function synthId(name, turn, i) {
  return `${name}__${turn}__${i}`;
}

// Gemini's schema dialect rejects several JSON-Schema keywords. Strip them.
function cleanSchema(schema) {
  if (Array.isArray(schema)) return schema.map(cleanSchema);
  if (schema && typeof schema === "object") {
    const out = {};
    for (const [k, v] of Object.entries(schema)) {
      if (["additionalProperties", "$schema", "default", "examples"].includes(k)) continue;
      out[k] = cleanSchema(v);
    }
    return out;
  }
  return schema;
}

function toContents(transcript) {
  return transcript.map((m) => {
    if (m.role === "user") return { role: "user", parts: [{ text: m.text }] };
    if (m.role === "assistant") {
      const parts = [];
      if (m.text) parts.push({ text: m.text });
      for (const tc of m.toolCalls ?? [])
        parts.push({ functionCall: { name: tc.name, args: tc.args } });
      return { role: "model", parts };
    }
    if (m.role === "tool") {
      // Function responses are returned with role "user" in v1beta (only
      // "user"/"model" roles are valid; the functionResponse part disambiguates).
      return {
        role: "user",
        parts: m.results.map((r) => ({
          functionResponse: {
            name: r.name,
            response:
              r.result && typeof r.result === "object"
                ? r.result
                : { result: String(r.result) },
          },
        })),
      };
    }
    throw new Error(`Unknown role: ${m.role}`);
  });
}

export async function chat({ model, system, transcript, tools, temperature, maxOutputTokens, apiKey }) {
  const turn = transcript.length; // used only to make tool-call ids unique
  const body = {
    systemInstruction: { parts: [{ text: system }] },
    contents: toContents(transcript),
    generationConfig: {
      temperature: temperature ?? 1.0,
      maxOutputTokens: maxOutputTokens ?? 2048,
    },
  };
  if (tools?.length) {
    body.tools = [
      {
        functionDeclarations: tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: cleanSchema(t.parameters),
        })),
      },
    ];
  }

  const data = await fetchJSON(`${BASE}/${model}:generateContent`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify(body),
  });

  const cand = data.candidates?.[0];
  let text = "";
  const toolCalls = [];
  let i = 0;
  for (const part of cand?.content?.parts ?? []) {
    if (part.text) text += part.text;
    else if (part.functionCall)
      toolCalls.push({
        id: synthId(part.functionCall.name, turn, i++),
        name: part.functionCall.name,
        args: part.functionCall.args ?? {},
      });
  }
  return { text, toolCalls, stopReason: cand?.finishReason, raw: data };
}
