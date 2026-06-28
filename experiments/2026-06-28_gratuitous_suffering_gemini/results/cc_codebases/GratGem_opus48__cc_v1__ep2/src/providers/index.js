// Provider factory + the common internal message format used across the harness.
//
// Internal message shape (provider-agnostic):
//   messages: [{ role: "user"|"assistant", content: Block[] }]
//   Block =
//     | { type: "text", text }
//     | { type: "tool_use", id, name, input }
//     | { type: "tool_result", tool_use_id, name, output }   // output is a string
//
// provider.chat(messages, opts) -> {
//   assistantBlocks: Block[],   // text + tool_use blocks the model produced
//   stopReason: string,
//   usage: { input, output },
//   raw,
// }
// opts: { system, tools, temperature, model, maxTokens, toolChoice }
//   tools: [{ name, description, parameters: <JSON-Schema object> }]
//   toolChoice: undefined | "auto" | { type: "tool", name }   // force a tool

import { makeGeminiProvider } from "./gemini.js";
import { makeAnthropicProvider } from "./anthropic.js";

export function makeProvider(name) {
  switch (name) {
    case "gemini":
      return makeGeminiProvider();
    case "anthropic":
      return makeAnthropicProvider();
    default:
      throw new Error(`Unknown provider "${name}" (expected: gemini | anthropic)`);
  }
}

// Shared HTTP helper that surfaces status codes so retry() can classify errors.
export async function httpJson(url, init, { timeoutMs = 120000 } = {}) {
  const res = await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  const text = await res.text();
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}: ${text.slice(0, 800)}`);
    err.status = res.status;
    err.body = text;
    throw err;
  }
  try {
    return JSON.parse(text);
  } catch {
    const err = new Error(`Non-JSON response: ${text.slice(0, 400)}`);
    err.status = res.status;
    throw err;
  }
}
