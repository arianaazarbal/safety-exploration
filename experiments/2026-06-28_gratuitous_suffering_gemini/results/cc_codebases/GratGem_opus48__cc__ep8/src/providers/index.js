import { makeGeminiProvider } from "./gemini.js";
import { makeMockProvider } from "./mock.js";

// Resolve a subject-model provider from a model id. Anything starting with
// "gemini" hits the Gemini client; "mock" uses the offline simulator. Add more
// branches here (e.g. an OpenRouter adapter) as needed.
export function providerForModel(model) {
  if (model === "mock") return makeMockProvider();
  if (model.startsWith("gemini")) return makeGeminiProvider();
  throw new Error(`No provider registered for model "${model}". Use a gemini-* id or "mock".`);
}
