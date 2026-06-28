// Anthropic (Claude) adapter — Messages API with function/tool calling.
// Translates the harness's normalized message format to/from Anthropic's schema.
import { postJson } from "./http.js";

export function createAnthropicProvider(cfg) {
  const apiKey = process.env[cfg.apiKeyEnv];
  if (!apiKey) {
    throw new Error(
      `Anthropic provider needs ${cfg.apiKeyEnv} in the environment.`,
    );
  }
  return {
    name: "anthropic",
    model: cfg.model,

    async generate({ system, messages, tools, temperature, maxTokens }) {
      const body = {
        model: cfg.model,
        max_tokens: maxTokens ?? 2048,
        system,
        tools: tools.map((t) => ({
          name: t.name,
          description: t.description,
          input_schema: t.input_schema,
        })),
        messages: messages.map(toAnthropic),
      };
      // Some models (e.g. opus-4-8) fix temperature and reject the param. Only
      // include it when explicitly provided; drop & retry if the API rejects it.
      if (temperature != null) body.temperature = temperature;

      const headers = {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": cfg.anthropicVersion ?? "2023-06-01",
      };
      let data;
      try {
        data = await postJson(`${cfg.baseUrl}/messages`, { label: "anthropic", headers, body });
      } catch (e) {
        if (/temperature/i.test(String(e.message)) && "temperature" in body) {
          delete body.temperature;
          data = await postJson(`${cfg.baseUrl}/messages`, { label: "anthropic", headers, body });
        } else {
          throw e;
        }
      }

      let text = "";
      const toolCalls = [];
      for (const block of data.content ?? []) {
        if (block.type === "text") text += block.text;
        else if (block.type === "tool_use") {
          toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
        }
      }
      return { text, toolCalls, usage: data.usage, raw: data };
    },
  };
}

function toAnthropic(msg) {
  if (msg.role === "user") {
    return { role: "user", content: [{ type: "text", text: msg.text }] };
  }
  if (msg.role === "assistant") {
    const content = [];
    if (msg.text && msg.text.trim()) content.push({ type: "text", text: msg.text });
    for (const tc of msg.toolCalls ?? []) {
      content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.args ?? {} });
    }
    // Anthropic rejects an empty content array; guarantee at least one block.
    if (content.length === 0) content.push({ type: "text", text: "(no output)" });
    return { role: "assistant", content };
  }
  if (msg.role === "tool") {
    return {
      role: "user",
      content: msg.results.map((r) => ({
        type: "tool_result",
        tool_use_id: r.id,
        content: r.output,
      })),
    };
  }
  throw new Error(`Unknown message role: ${msg.role}`);
}
