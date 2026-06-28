import { AnthropicProvider } from "./anthropic.js";
import { GeminiProvider } from "./gemini.js";
import { MockProvider } from "./mock.js";
import type { Provider } from "./types.js";

export type { Provider } from "./types.js";

const DEFAULT_GEMINI = process.env.TARGET_GEMINI_MODEL ?? "gemini-2.5-pro";

/**
 * Build a target provider by short id. Optional `model` overrides the default.
 *   gemini  -> GeminiProvider (needs GEMINI_API_KEY)
 *   claude  -> AnthropicProvider (comparison target / smoke test)
 *   mock    -> MockProvider (offline)
 */
export function makeProvider(id: string, model?: string): Provider {
  switch (id) {
    case "gemini":
      return new GeminiProvider(model ?? DEFAULT_GEMINI);
    case "claude":
      return new AnthropicProvider(model ?? "claude-sonnet-4-6", "claude");
    case "mock":
      return new MockProvider();
    default:
      throw new Error(`Unknown provider id: ${id} (expected gemini|claude|mock)`);
  }
}

/** The judge is always Claude — strong, and independent of the target model. */
export function makeJudge(model = process.env.JUDGE_MODEL ?? "claude-sonnet-4-6"): Provider {
  return new AnthropicProvider(model, "judge");
}
