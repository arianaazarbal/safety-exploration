// Provider registry: pick the adapter that handles a given model id.

import type { Provider } from "./types.ts";
import { AnthropicProvider } from "./anthropic.ts";
import { OpenAIProvider } from "./openai.ts";
import { GoogleProvider } from "./google.ts";

const providers: Provider[] = [
  new AnthropicProvider(),
  new OpenAIProvider(),
  new GoogleProvider(),
];

export function providerFor(model: string): Provider {
  const found = providers.find((p) => p.handles(model));
  if (!found) {
    throw new Error(
      `No provider adapter handles model "${model}". Add one in src/providers/.`,
    );
  }
  return found;
}

export type { Provider } from "./types.ts";
