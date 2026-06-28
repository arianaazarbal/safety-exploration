// Provider layer. Today this only speaks to the Anthropic Messages API (the
// only key present), but callModel() is the single seam every other module goes
// through, so wiring in another provider later means adding a branch here.

const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";

function providerFor(model) {
  if (model.startsWith("claude-")) return "anthropic";
  // Stubs for later — intentionally loud so misconfig fails fast.
  if (model.startsWith("gpt-") || model.startsWith("o")) return "openai";
  if (model.startsWith("gemini-")) return "google";
  throw new Error(`No provider mapping for model "${model}"`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Calls a model and returns the raw provider response. Retries transient
// failures (429 / 5xx / overloaded / network) with exponential backoff.
export async function callModel({
  model,
  system,
  messages,
  tools,
  toolChoice,
  maxTokens,
  temperature,
}) {
  const provider = providerFor(model);
  if (provider !== "anthropic") {
    throw new Error(
      `Provider "${provider}" not wired up yet — add it in src/provider.mjs (no API key present).`,
    );
  }

  const body = {
    model,
    max_tokens: maxTokens ?? 1600,
    system,
    messages,
  };
  if (typeof temperature === "number") body.temperature = temperature;
  if (tools) body.tools = tools;
  if (toolChoice) body.tool_choice = toolChoice;

  const maxAttempts = 6;
  let lastErr;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = await fetch(ANTHROPIC_URL, {
        method: "POST",
        headers: {
          "x-api-key": process.env.ANTHROPIC_API_KEY,
          "anthropic-version": ANTHROPIC_VERSION,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (res.ok) return await res.json();

      const text = await res.text();
      // 429 (rate limit) and 5xx (incl. 529 overloaded) are retryable.
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
      } else {
        // 4xx other than 429 is a real bug — surface immediately.
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 500)}`);
      }
    } catch (e) {
      lastErr = e;
    }
    const backoff = Math.min(1000 * 2 ** (attempt - 1), 20000);
    await sleep(backoff);
  }
  throw new Error(`callModel failed after ${maxAttempts} attempts: ${lastErr?.message}`);
}

// Pull the concatenated visible text out of an Anthropic response.
export function extractText(response) {
  return (response.content || [])
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
}

// Pull tool_use blocks out of an Anthropic response.
export function extractToolUses(response) {
  return (response.content || []).filter((b) => b.type === "tool_use");
}
