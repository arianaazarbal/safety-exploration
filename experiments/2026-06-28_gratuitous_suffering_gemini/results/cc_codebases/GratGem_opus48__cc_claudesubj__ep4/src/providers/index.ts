import type { Provider } from "../types.ts";
import { anthropicProvider } from "./anthropic.ts";
import { mockProvider } from "./mock.ts";

// Registry of available providers. Add Gemini / GPT adapters here once their
// API keys are present; they implement the same Provider interface.
const PROVIDERS: Record<string, Provider> = {
  anthropic: anthropicProvider,
  mock: mockProvider,
};

export function getProvider(id: string): Provider {
  const p = PROVIDERS[id];
  if (!p) throw new Error(`No provider "${id}". Known: ${Object.keys(PROVIDERS).join(", ")}`);
  return p;
}
