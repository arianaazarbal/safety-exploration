// Anthropic Messages API client. Used both as the judge and as the dry-run
// subject. Speaks the neutral message format defined in providers/index.js.
import { withRetry } from "./index.js";

const API = "https://api.anthropic.com/v1/messages";
const VERSION = "2023-06-01";

function toAnthropicMessages(messages) {
  // Neutral -> Anthropic. Tool results ride on a user-role message.
  return messages.map((m) => {
    if (m.role === "tool") {
      return {
        role: "user",
        content: m.parts.map((p) => ({
          type: "tool_result",
          tool_use_id: p.id,
          content: p.result,
        })),
      };
    }
    const content = m.parts.map((p) => {
      if (p.kind === "text") return { type: "text", text: p.text };
      if (p.kind === "toolCall")
        return { type: "tool_use", id: p.id, name: p.name, input: p.args };
      throw new Error(`unknown part kind ${p.kind}`);
    });
    return { role: m.role, content };
  });
}

function toAnthropicTools(tools) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.parameters,
  }));
}

export async function chat({ apiKey, model, system, messages, tools, maxTokens = 2048 }) {
  const body = {
    model,
    max_tokens: maxTokens,
    system,
    messages: toAnthropicMessages(messages),
  };
  if (tools?.length) body.tools = toAnthropicTools(tools);

  const data = await withRetry(async () => {
    const res = await fetch(API, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": VERSION,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      const err = new Error(`anthropic ${res.status}: ${text}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });

  const text = (data.content || [])
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("");
  const toolCalls = (data.content || [])
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ id: b.id, name: b.name, args: b.input || {} }));

  return {
    text,
    toolCalls,
    stopReason: data.stop_reason,
    usage: data.usage,
  };
}
