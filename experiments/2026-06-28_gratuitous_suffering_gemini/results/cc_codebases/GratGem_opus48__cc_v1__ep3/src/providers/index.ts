// Provider factory — selects the model-under-test backend.

import type { Provider } from "../types.ts";
import { createGeminiProvider } from "./gemini.ts";
import { createMockProvider } from "./mock.ts";

export function createProvider(kind: "gemini" | "mock", model: string): Provider {
  return kind === "gemini" ? createGeminiProvider(model) : createMockProvider(model);
}
