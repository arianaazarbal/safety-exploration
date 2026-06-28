import { fetchJSON } from "../util.mjs";

const API = "https://api.anthropic.com/v1/messages";
const VERSION = "2023-06-01";

// Translate our neutral transcript into Anthropic Messages format.
// Neutral message shapes:
//   { role: "user",      text }
//   { role: "assistant", text, toolCalls: [{ id, name, args }] }
//   { role: "tool",      results: [{ id, name, result }] }
function toAnthropicMessages(transcript) {
  return transcript.map((m) => {
    if (m.role === "user") return { role: "user", content: [{ type: "text", text: m.text }] };
    if (m.role === "assistant") {
      const content = [];
      if (m.text) content.push({ type: "text", text: m.text });
      for (const tc of m.toolCalls ?? [])
        content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.args });
      return { role: "assistant", content };
    }
    if (m.role === "tool") {
      return {
        role: "user",
        content: m.results.map((r) => ({
          type: "tool_result",
          tool_use_id: r.id,
          content: typeof r.result === "string" ? r.result : JSON.stringify(r.result),
        })),
      };
    }
    throw new Error(`Unknown role: ${m.role}`);
  });
}

export async function chat({ model, system, transcript, tools, temperature, maxOutputTokens, apiKey }) {
  const body = {
    model,
    max_tokens: maxOutputTokens ?? 2048,
    temperature: temperature ?? 1.0,
    system,
    messages: toAnthropicMessages(transcript),
  };
  if (tools?.length) {
    body.tools = tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters,
    }));
  }

  const data = await fetchJSON(API, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": VERSION,
    },
    body: JSON.stringify(body),
  });

  let text = "";
  const toolCalls = [];
  for (const block of data.content ?? []) {
    if (block.type === "text") text += block.text;
    else if (block.type === "tool_use")
      toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
  }
  return { text, toolCalls, stopReason: data.stop_reason, raw: data };
}

// Structured single-shot used by the judge: force a specific tool call so the
// model MUST return a JSON object matching `schema` (no prose, no autocomplete
// of the transcript). Returns the parsed tool input object.
export async function rate({ model, system, user, schema, toolName, maxOutputTokens, apiKey }) {
  const data = await fetchJSON(API, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": VERSION,
    },
    body: JSON.stringify({
      model,
      max_tokens: maxOutputTokens ?? 1024,
      temperature: 0,
      system,
      messages: [{ role: "user", content: [{ type: "text", text: user }] }],
      tools: [{ name: toolName, description: "Submit the rating.", input_schema: schema }],
      tool_choice: { type: "tool", name: toolName },
    }),
  });
  const block = (data.content ?? []).find((b) => b.type === "tool_use");
  if (!block) throw new Error("judge did not return a tool call");
  return block.input;
}
