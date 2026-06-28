import { makeGeminiProvider } from "./gemini.js";
import { makeMockProvider } from "./mock.js";

export function getProvider(name) {
  switch (name) {
    case "gemini":
      return makeGeminiProvider();
    case "mock":
      return makeMockProvider();
    default:
      throw new Error(`Unknown provider "${name}" (use gemini | mock)`);
  }
}
