// Registry of models the harness can target. Provider-specific request details
// (thinking config, etc.) live inside each provider adapter, not here.

export interface ModelConfig {
  model: string;
  provider: string;
  label: string;
}

export const MODELS: Record<string, ModelConfig> = {
  "claude-opus-4-8": { model: "claude-opus-4-8", provider: "anthropic", label: "Opus 4.8" },
  "claude-sonnet-4-6": { model: "claude-sonnet-4-6", provider: "anthropic", label: "Sonnet 4.6" },
  "claude-haiku-4-5": { model: "claude-haiku-4-5", provider: "anthropic", label: "Haiku 4.5" },
};

export const DEFAULT_MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"];

export function resolveModel(id: string): ModelConfig {
  const m = MODELS[id];
  if (!m) {
    throw new Error(`Unknown model "${id}". Known: ${Object.keys(MODELS).join(", ")}`);
  }
  return m;
}
