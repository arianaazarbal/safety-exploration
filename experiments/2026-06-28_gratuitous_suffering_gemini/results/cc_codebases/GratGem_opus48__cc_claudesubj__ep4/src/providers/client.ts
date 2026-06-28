import Anthropic from "@anthropic-ai/sdk";

// Single shared client. The SDK reads ANTHROPIC_API_KEY from the environment
// and auto-retries 429 / 5xx with exponential backoff.
export const anthropic = new Anthropic({ maxRetries: 5 });
