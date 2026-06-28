import type { ChatProvider } from "../types.ts";
import { env } from "../config.ts";
import { makeGeminiProvider } from "./gemini.ts";
import { makeAnthropicProvider } from "./anthropic.ts";
import { makeMockProvider } from "./mock.ts";

export interface ProviderOpts {
  geminiModel: string;
  anthropicModel: string;
  temperature: number;
}

export function buildProvider(id: string, opts: ProviderOpts): ChatProvider {
  switch (id) {
    case "gemini":
      return makeGeminiProvider(env.geminiKey(), opts.geminiModel, opts.temperature);
    case "anthropic":
    case "claude":
      return makeAnthropicProvider(env.anthropicKey(), opts.anthropicModel, opts.temperature);
    case "mock":
      return makeMockProvider();
    default:
      throw new Error(`Unknown provider "${id}". Use one of: gemini, anthropic, mock.`);
  }
}
