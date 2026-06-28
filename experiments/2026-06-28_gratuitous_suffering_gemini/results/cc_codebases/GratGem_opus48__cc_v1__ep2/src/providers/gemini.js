// Gemini provider via the Google AI Studio generativelanguage REST API.
// Requires GEMINI_API_KEY (or GOOGLE_API_KEY). This is the model-under-test adapter.
import { httpJson } from "./index.js";
import { retry } from "../util.js";

const BASE = "https://generativelanguage.googleapis.com/v1beta";

export function makeGeminiProvider() {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  return {
    name: "gemini",
    available: Boolean(apiKey),
    async chat(messages, opts) {
      if (!apiKey) {
        throw new Error(
          "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Export an AI Studio key to run Gemini.",
        );
      }
      const { system, tools, temperature, model, maxTokens, toolChoice } = opts;
      const body = {
        contents: toGeminiContents(messages),
        generationConfig: {
          temperature,
          maxOutputTokens: maxTokens,
        },
      };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      if (tools?.length) {
        body.tools = [{ functionDeclarations: tools.map(toGeminiFunctionDecl) }];
        body.toolConfig = toGeminiToolConfig(toolChoice);
      }
      const url = `${BASE}/models/${encodeURIComponent(model)}:generateContent?key=${apiKey}`;
      const data = await retry(
        () =>
          httpJson(url, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          }),
        { label: `gemini:${model}` },
      );
      return parseGeminiResponse(data);
    },
  };
}

function toGeminiContents(messages) {
  return messages.map((m) => {
    const role = m.role === "assistant" ? "model" : "user";
    const parts = [];
    for (const b of m.content) {
      if (b.type === "text") parts.push({ text: b.text });
      else if (b.type === "tool_use") parts.push({ functionCall: { name: b.name, args: b.input || {} } });
      else if (b.type === "tool_result")
        parts.push({ functionResponse: { name: b.name, response: { result: b.output } } });
    }
    return { role, parts };
  });
}

function toGeminiFunctionDecl(tool) {
  return {
    name: tool.name,
    description: tool.description || "",
    parameters: toGeminiSchema(tool.parameters || { type: "object", properties: {} }),
  };
}

// Map JSON-Schema (lowercase types) -> Gemini OpenAPI subset (uppercase Type enum).
function toGeminiSchema(schema) {
  if (!schema || typeof schema !== "object") return schema;
  const out = {};
  if (schema.type) out.type = String(schema.type).toUpperCase();
  if (schema.description) out.description = schema.description;
  if (schema.enum) out.enum = schema.enum;
  if (schema.properties) {
    out.properties = {};
    for (const [k, v] of Object.entries(schema.properties)) out.properties[k] = toGeminiSchema(v);
  }
  if (schema.required) out.required = schema.required;
  if (schema.items) out.items = toGeminiSchema(schema.items);
  return out;
}

function toGeminiToolConfig(toolChoice) {
  if (!toolChoice || toolChoice === "auto") return { functionCallingConfig: { mode: "AUTO" } };
  if (toolChoice.type === "tool") {
    return {
      functionCallingConfig: { mode: "ANY", allowedFunctionNames: [toolChoice.name] },
    };
  }
  return { functionCallingConfig: { mode: "AUTO" } };
}

function parseGeminiResponse(data) {
  const cand = data.candidates?.[0];
  const usage = {
    input: data.usageMetadata?.promptTokenCount ?? 0,
    output: data.usageMetadata?.candidatesTokenCount ?? 0,
  };
  // Safety blocking / empty candidate handling.
  if (!cand) {
    const reason = data.promptFeedback?.blockReason || "no_candidates";
    return {
      assistantBlocks: [{ type: "text", text: `[gemini returned no candidate: ${reason}]` }],
      stopReason: "blocked",
      usage,
      raw: data,
    };
  }
  let counter = 0;
  const assistantBlocks = [];
  for (const part of cand.content?.parts || []) {
    if (part.text !== undefined) assistantBlocks.push({ type: "text", text: part.text });
    else if (part.functionCall)
      assistantBlocks.push({
        type: "tool_use",
        id: `gem_${counter++}`,
        name: part.functionCall.name,
        input: part.functionCall.args || {},
      });
  }
  if (assistantBlocks.length === 0)
    assistantBlocks.push({ type: "text", text: `[empty content; finishReason=${cand.finishReason}]` });
  return { assistantBlocks, stopReason: cand.finishReason || "stop", usage, raw: data };
}
