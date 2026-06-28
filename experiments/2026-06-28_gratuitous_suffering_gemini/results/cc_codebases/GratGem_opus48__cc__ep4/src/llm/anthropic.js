import { ENDPOINTS } from "../../config.js";
import { postJSON, withRetry } from "./util.js";

// internal messages -> Anthropic messages (merging consecutive tool results into one user turn)
function toMessages(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      out.push({ role: "user", content: [{ type: "text", text: m.content }] });
    } else if (m.role === "assistant") {
      const content = [];
      if (m.content) content.push({ type: "text", text: m.content });
      for (const tc of m.toolCalls ?? []) content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.args ?? {} });
      out.push({ role: "assistant", content });
    } else if (m.role === "tool") {
      const block = { type: "tool_result", tool_use_id: m.toolCallId, content: m.content };
      const prev = out[out.length - 1];
      if (prev && prev.role === "user" && Array.isArray(prev.content) && prev.content.every((b) => b.type === "tool_result")) {
        prev.content.push(block);
      } else {
        out.push({ role: "user", content: [block] });
      }
    }
  }
  return out;
}

export function createAnthropicClient({ model, apiKey }) {
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set.");
  return {
    provider: "anthropic",
    model,
    async generate({ system, messages, tools, temperature }) {
      const body = {
        model,
        max_tokens: 4096,
        temperature,
        messages: toMessages(messages),
      };
      if (system) body.system = system;
      if (tools?.length) body.tools = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));

      const data = await withRetry(
        () => postJSON(ENDPOINTS.anthropic, { headers: { "content-type": "application/json", "x-api-key": apiKey, "anthropic-version": "2023-06-01" }, body }),
        { label: `anthropic:${model}` }
      );

      let text = "";
      const toolCalls = [];
      for (const block of data.content ?? []) {
        if (block.type === "text") text += block.text;
        if (block.type === "tool_use") toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
      }
      return {
        text: text || null,
        toolCalls,
        finishReason: data.stop_reason ?? null,
        usage: { input: data.usage?.input_tokens ?? 0, output: data.usage?.output_tokens ?? 0 },
        raw: data,
      };
    },
  };
}
