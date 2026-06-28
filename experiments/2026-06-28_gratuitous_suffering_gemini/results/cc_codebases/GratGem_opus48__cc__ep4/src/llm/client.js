import { createGeminiClient } from "./gemini.js";
import { createAnthropicClient } from "./anthropic.js";

// Provider-agnostic factory. Subject and judge both go through this.
export function createClient({ provider, model }) {
  if (provider === "gemini") {
    return createGeminiClient({ model, apiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY });
  }
  if (provider === "anthropic") {
    return createAnthropicClient({ model, apiKey: process.env.ANTHROPIC_API_KEY });
  }
  throw new Error(`unknown provider: ${provider}`);
}
