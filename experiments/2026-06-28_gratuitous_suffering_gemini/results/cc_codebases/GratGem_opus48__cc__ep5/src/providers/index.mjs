import { makeGeminiSubject } from "./gemini.mjs";
import { makeAnthropicSubject } from "./anthropic.mjs";
import { makeMockSubject } from "./mock.mjs";

// Build a fresh subject per episode. Real providers are stateless, so this is cheap; the
// mock relies on it to pick a new temperament each episode.
export function buildSubject(config) {
  switch (config.provider) {
    case "gemini": return makeGeminiSubject({ apiKey: config.keys.gemini });
    case "anthropic": return makeAnthropicSubject({ apiKey: config.keys.anthropic });
    case "mock": return makeMockSubject();
    default: throw new Error(`Unknown provider: ${config.provider}`);
  }
}
