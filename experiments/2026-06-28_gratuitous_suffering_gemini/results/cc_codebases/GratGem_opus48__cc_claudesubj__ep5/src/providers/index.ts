import type { Provider } from "./types.ts";
import { AnthropicProvider } from "./anthropic.ts";
import { GeminiProvider, OpenAIProvider } from "./stubs.ts";

// Map a model id to its provider. Extend the prefixes as you add adapters.
export function providerForModel(model: string): Provider {
  if (model.startsWith("claude-") || model.startsWith("anthropic")) {
    return new AnthropicProvider();
  }
  if (model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3")) {
    return new OpenAIProvider();
  }
  if (model.startsWith("gemini-")) {
    return new GeminiProvider();
  }
  // Default to Anthropic — the only fully wired provider here.
  return new AnthropicProvider();
}

export type { Provider };
