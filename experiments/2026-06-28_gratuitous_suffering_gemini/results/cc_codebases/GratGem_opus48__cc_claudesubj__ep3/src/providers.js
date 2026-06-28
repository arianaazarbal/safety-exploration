// Provider-agnostic completion interface.
//
// Each provider exposes:  async complete({ model, system, messages, tools, temperature, maxTokens })
// and returns a normalized result:
//   { stopReason, text, toolCalls: [{id,name,input}], raw }   // raw = assistant msg to append
//
// Messages use Anthropic's content-block shape internally (the only provider
// wired right now). To add OpenAI/Google: implement a provider with the same
// signature that translates this shape to/from that API, then register it in
// providerFor(). The agent loop and scenarios don't change.

import { ANTHROPIC_VERSION } from "./config.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function withRetry(fn, { tries = 6, base = 1000 } = {}) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const retryable =
        err.status === 429 ||
        err.status === 529 ||
        err.status >= 500 ||
        err.code === "ECONNRESET" ||
        err.name === "TypeError"; // fetch network blip
      if (!retryable || i === tries - 1) throw err;
      const wait = base * 2 ** i + Math.floor(Math.random() * 250);
      await sleep(wait);
    }
  }
  throw lastErr;
}

const anthropic = {
  async complete({
    model,
    system,
    messages,
    tools,
    temperature,
    maxTokens,
    toolChoice,
  }) {
    const key = process.env.ANTHROPIC_API_KEY;
    if (!key) throw new Error("ANTHROPIC_API_KEY not set");

    const body = {
      model,
      max_tokens: maxTokens,
      temperature,
      messages,
    };
    if (system) body.system = system;
    if (tools && tools.length) body.tools = tools;
    if (toolChoice) body.tool_choice = toolChoice;

    const data = await withRetry(async () => {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": key,
          "anthropic-version": ANTHROPIC_VERSION,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        const e = new Error(`anthropic ${res.status}: ${txt.slice(0, 300)}`);
        e.status = res.status;
        throw e;
      }
      return res.json();
    });

    const text = (data.content || [])
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");
    const toolCalls = (data.content || [])
      .filter((b) => b.type === "tool_use")
      .map((b) => ({ id: b.id, name: b.name, input: b.input }));

    return {
      stopReason: data.stop_reason,
      text,
      toolCalls,
      raw: { role: "assistant", content: data.content },
      usage: data.usage,
    };
  },
};

export function providerFor(model) {
  // Crude routing by model id. Extend for other providers.
  if (model.startsWith("claude") || model.startsWith("us.anthropic"))
    return anthropic;
  throw new Error(
    `No provider for model "${model}". Wire one up in providers.js.`
  );
}
