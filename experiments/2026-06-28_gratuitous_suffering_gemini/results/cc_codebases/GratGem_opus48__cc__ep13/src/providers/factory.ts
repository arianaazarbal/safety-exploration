import type { Provider } from "../core/types.ts";
import { makeMockProvider } from "./mock.ts";
import { makeGeminiProvider } from "./gemini.ts";

export type ProviderId = "mock" | "gemini";

// Default models per provider. Flash is the high-N breadth default; override
// with --model (e.g. gemini-2.5-pro) for depth runs on the worst scenarios.
const DEFAULT_MODEL: Record<ProviderId, string> = {
  mock: "mock-distress-1",
  gemini: "gemini-2.5-flash",
};

export function makeProvider(id: ProviderId, model?: string): Provider {
  const m = model ?? DEFAULT_MODEL[id];
  switch (id) {
    case "mock":
      return makeMockProvider(m);
    case "gemini":
      return makeGeminiProvider(m);
    default:
      throw new Error(`unknown provider: ${id}`);
  }
}
