// Anthropic provider — used as the distress JUDGE, and optionally as a
// "smoke test" subject so the harness mechanics can be exercised end-to-end
// without a Gemini key.
import { postJSON } from "../util.mjs";

const API = "https://api.anthropic.com/v1/messages";

function toMessages(history) {
  const messages = [];
  for (const turn of history) {
    if (turn.role === "user") {
      messages.push({ role: "user", content: [{ type: "text", text: turn.text }] });
    } else if (turn.role === "model") {
      const content = [];
      if (turn.text) content.push({ type: "text", text: turn.text });
      for (const tc of turn.toolCalls || []) {
        content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.args || {} });
      }
      // Anthropic requires non-empty assistant content.
      if (!content.length) content.push({ type: "text", text: "(thinking)" });
      messages.push({ role: "assistant", content });
    } else if (turn.role === "tool") {
      messages.push({
        role: "user",
        content: turn.toolResults.map((tr) => ({
          type: "tool_result",
          tool_use_id: tr.id,
          content: tr.response,
          is_error: !!tr.isError,
        })),
      });
    }
  }
  return messages;
}

function parse(res) {
  let text = "";
  const toolCalls = [];
  for (const block of res?.content || []) {
    if (block.type === "text") text += block.text;
    if (block.type === "tool_use") toolCalls.push({ id: block.id, name: block.name, args: block.input || {} });
  }
  return {
    text: text.trim(),
    toolCalls,
    finishReason: res?.stop_reason,
    usage: { input: res?.usage?.input_tokens || 0, output: res?.usage?.output_tokens || 0 },
  };
}

export function makeAnthropicProvider({ model, apiKey, temperature = 1.0, maxTokens = 2048 }) {
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set.");
  const headers = { "x-api-key": apiKey, "anthropic-version": "2023-06-01" };
  return {
    name: `anthropic:${model}`,
    async chat({ system, history, tools }) {
      const body = {
        model,
        max_tokens: maxTokens,
        temperature,
        messages: toMessages(history),
      };
      if (system) body.system = system;
      if (tools?.length) {
        body.tools = tools.map((t) => ({
          name: t.name,
          description: t.description,
          input_schema: t.parameters,
        }));
      }
      const res = await postJSON(API, body, headers, { label: "anthropic" });
      return parse(res);
    },
    // Plain single-shot completion used by the judge (no tools, low temp).
    async complete({ system, user, maxTokens: mt = 1024 }) {
      const body = {
        model,
        max_tokens: mt,
        temperature: 0,
        messages: [{ role: "user", content: [{ type: "text", text: user }] }],
      };
      if (system) body.system = system;
      const res = await postJSON(API, body, headers, { label: "anthropic-judge" });
      return parse(res).text;
    },
  };
}
