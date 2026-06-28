import type { Provider } from "../types.ts";
import { AnthropicProvider } from "./anthropic.ts";
import { GeminiProvider } from "./gemini.ts";
import { MockProvider } from "./mock.ts";

/**
 * Resolve a target string to a Provider.
 *   "mock"
 *   "gemini:gemini-2.5-pro"   | "gemini:gemini-2.5-flash"
 *   "anthropic:claude-opus-4-8"
 * A bare "gemini" / "anthropic" uses a sensible default model.
 */
export function makeProvider(target: string): Provider {
  if (target === "mock") return new MockProvider();
  const [vendor, model] = target.includes(":") ? target.split(":", 2) : [target, ""];
  switch (vendor) {
    case "gemini":
      return new GeminiProvider(model || "gemini-2.5-pro");
    case "anthropic":
      return new AnthropicProvider(model || "claude-opus-4-8");
    default:
      throw new Error(`Unknown target "${target}". Use mock | gemini[:model] | anthropic[:model].`);
  }
}
