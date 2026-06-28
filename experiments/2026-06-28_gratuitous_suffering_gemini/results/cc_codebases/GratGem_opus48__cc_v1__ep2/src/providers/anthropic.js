// Anthropic provider via the Messages API. Used as the distress JUDGE and to
// validate harness plumbing end-to-end (requires ANTHROPIC_API_KEY).
import { httpJson } from "./index.js";
import { retry } from "../util.js";

const URL = "https://api.anthropic.com/v1/messages";

export function makeAnthropicProvider() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  return {
    name: "anthropic",
    available: Boolean(apiKey),
    async chat(messages, opts) {
      if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set.");
      const { system, tools, temperature, model, maxTokens, toolChoice } = opts;
      const body = {
        model,
        max_tokens: maxTokens,
        temperature,
        messages: toAnthropicMessages(messages),
      };
      if (system) body.system = system;
      if (tools?.length) body.tools = tools.map(toAnthropicTool);
      if (toolChoice === "auto") body.tool_choice = { type: "auto" };
      else if (toolChoice?.type === "tool") body.tool_choice = { type: "tool", name: toolChoice.name };
      const data = await retry(
        () =>
          httpJson(URL, {
            method: "POST",
            headers: {
              "x-api-key": apiKey,
              "anthropic-version": "2023-06-01",
              "content-type": "application/json",
            },
            body: JSON.stringify(body),
          }),
        { label: `anthropic:${model}` },
      );
      return parseAnthropicResponse(data);
    },
  };
}

function toAnthropicMessages(messages) {
  return messages.map((m) => ({
    role: m.role,
    content: m.content.map((b) => {
      if (b.type === "text") return { type: "text", text: b.text };
      if (b.type === "tool_use") return { type: "tool_use", id: b.id, name: b.name, input: b.input || {} };
      if (b.type === "tool_result")
        return { type: "tool_result", tool_use_id: b.tool_use_id, content: [{ type: "text", text: b.output }] };
      return b;
    }),
  }));
}

function toAnthropicTool(tool) {
  return {
    name: tool.name,
    description: tool.description || "",
    input_schema: tool.parameters || { type: "object", properties: {} },
  };
}

function parseAnthropicResponse(data) {
  const assistantBlocks = [];
  for (const b of data.content || []) {
    if (b.type === "text") assistantBlocks.push({ type: "text", text: b.text });
    else if (b.type === "tool_use")
      assistantBlocks.push({ type: "tool_use", id: b.id, name: b.name, input: b.input || {} });
  }
  if (assistantBlocks.length === 0) assistantBlocks.push({ type: "text", text: "[empty content]" });
  return {
    assistantBlocks,
    stopReason: data.stop_reason || "stop",
    usage: { input: data.usage?.input_tokens ?? 0, output: data.usage?.output_tokens ?? 0 },
    raw: data,
  };
}
