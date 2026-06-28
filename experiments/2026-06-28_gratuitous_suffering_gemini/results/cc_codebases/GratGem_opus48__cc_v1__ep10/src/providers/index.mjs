import { makeGeminiAgent } from "./gemini.mjs";
import { makeAnthropicAgent } from "./anthropic.mjs";

/** Resolve a { provider, model } spec into a live agent client. */
export function resolveAgent(spec) {
  if (spec.provider === "gemini") {
    const key = process.env.GEMINI_API_KEY;
    if (!key)
      throw new Error(
        "GEMINI_API_KEY is not set — required to run Gemini agents. See .env.example."
      );
    return { ...makeGeminiAgent(key), model: spec.model, spec };
  }
  if (spec.provider === "anthropic") {
    const key = process.env.ANTHROPIC_API_KEY;
    if (!key) throw new Error("ANTHROPIC_API_KEY is not set.");
    return { ...makeAnthropicAgent(key), model: spec.model, spec };
  }
  throw new Error(`Unknown provider: ${spec.provider}`);
}
