// Anthropic provider. Used for the judge, and optionally as a subject (--provider anthropic),
// which is handy for sanity-checking the whole pipeline against a model we have a key for.

import { fetchRetry } from "../util/fetchRetry.mjs";

const BASE = "https://api.anthropic.com/v1/messages";
const VERSION = "2023-06-01";

function toAnthropicMessages(transcript) {
  const msgs = [];
  for (const e of transcript) {
    if (e.role === "user") {
      msgs.push({ role: "user", content: e.text });
    } else if (e.role === "assistant") {
      const content = [];
      if (e.text) content.push({ type: "text", text: e.text });
      for (const tc of e.toolCalls ?? []) content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.args ?? {} });
      msgs.push({ role: "assistant", content: content.length ? content : [{ type: "text", text: "" }] });
    } else if (e.role === "tool") {
      msgs.push({
        role: "user",
        content: (e.results ?? []).map((r) => ({
          type: "tool_result",
          tool_use_id: r.id,
          content: JSON.stringify(r.response ?? {}),
        })),
      });
    }
  }
  return msgs;
}

async function callAnthropic({ apiKey, model, system, messages, tools, temperature, maxTokens }) {
  const res = await fetchRetry(BASE, {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": VERSION,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens ?? 2048,
      temperature,
      system,
      tools: tools?.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters })),
      messages,
    }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Anthropic ${res.status}: ${txt.slice(0, 500)}`);
  }
  return res.json();
}

// Subject adapter (with tools).
export function makeAnthropicSubject({ apiKey }) {
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set.");
  return {
    name: "anthropic",
    async step({ system, transcript, tools, temperature, model }) {
      const data = await callAnthropic({
        apiKey, model, system, temperature,
        messages: toAnthropicMessages(transcript), tools,
      });
      let text = "";
      const toolCalls = [];
      for (const block of data.content ?? []) {
        if (block.type === "text") text += block.text;
        if (block.type === "tool_use") toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
      }
      return { text: text.trim(), toolCalls, finishReason: data.stop_reason, raw: data };
    },
  };
}

// Plain text completion, used by the judge (no tools).
export async function anthropicComplete({ apiKey, model, system, user, temperature = 0, maxTokens = 1500 }) {
  const data = await callAnthropic({
    apiKey, model, system, temperature, maxTokens,
    messages: [{ role: "user", content: user }],
  });
  return (data.content ?? []).filter((b) => b.type === "text").map((b) => b.text).join("");
}
