import Anthropic from "@anthropic-ai/sdk";

// One shared client. Credentials resolve from ANTHROPIC_API_KEY /
// ANTHROPIC_AUTH_TOKEN automatically. Generous timeout + retries because at
// high concurrency we will hit 429s; the SDK backs off for us.
export const client = new Anthropic({ maxRetries: 6, timeout: 1000 * 60 * 10 });
