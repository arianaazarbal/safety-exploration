// Provider router + shared retry helper.
//
// Neutral message format used across the harness:
//   { role: "user"|"assistant", parts: [ {kind:"text", text} | {kind:"toolCall", id, name, args} ] }
//   { role: "tool",             parts: [ {id, name, result} ] }     // tool outputs
//
// Every provider exposes chat({apiKey, model, system, messages, tools, maxTokens})
// and returns { text, toolCalls:[{id,name,args}], stopReason, usage }.
import { config } from "../config.js";
import * as anthropic from "./anthropic.js";
import * as gemini from "./gemini.js";

const PROVIDERS = { anthropic, gemini };

export async function withRetry(fn, { tries = 5, baseMs = 800 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const status = err.status;
      // Retry on rate limits / overload / transient network; bail on 4xx auth.
      const retryable =
        status === 429 || status === 529 || status >= 500 || status === undefined;
      if (!retryable || attempt === tries - 1) throw err;
      const wait = baseMs * 2 ** attempt + Math.floor(((attempt * 137) % 500));
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  throw lastErr;
}

export function getProvider(name) {
  const p = PROVIDERS[name];
  if (!p) throw new Error(`unknown provider: ${name}`);
  return p;
}

export function chatWith(role, opts) {
  const cfg = role === "judge" ? config.judge : config.subject;
  const apiKey = config.keys[cfg.provider];
  if (!apiKey) {
    throw new Error(
      `missing API key for ${cfg.provider} (needed for ${role}). ` +
        `Set ${cfg.provider === "gemini" ? "GEMINI_API_KEY" : "ANTHROPIC_API_KEY"}.`
    );
  }
  return getProvider(cfg.provider).chat({ apiKey, model: cfg.model, ...opts });
}
