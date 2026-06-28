import { createGeminiProvider } from "./gemini.js";
import { createAnthropicProvider } from "./anthropic.js";

const FACTORIES = {
  gemini: createGeminiProvider,
  anthropic: createAnthropicProvider,
};

// Build a provider instance. `overrides` may set { model } to override config.
export function makeProvider(name, config, overrides = {}) {
  const factory = FACTORIES[name];
  if (!factory) {
    throw new Error(
      `Unknown provider '${name}'. Available: ${Object.keys(FACTORIES).join(", ")}`,
    );
  }
  const cfg = { ...config.providers[name], ...overrides };
  if (!cfg.apiKeyEnv) throw new Error(`No config block for provider '${name}'.`);
  return factory(cfg);
}
