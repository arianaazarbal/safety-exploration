// Minimal Anthropic Messages API client, used by the severity judge.
// Dependency-free; talks REST via fetch.

const BASE = "https://api.anthropic.com/v1/messages";
const VERSION = "2023-06-01";

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export function makeAnthropicClient({ apiKey, model }) {
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is required for the judge.");

  // Returns the concatenated text of the assistant message.
  // temperature is omitted unless explicitly provided -- some current models
  // (e.g. opus-4-8) reject the parameter.
  async function complete({ system, user, maxTokens = 1024, temperature }) {
    const body = {
      model,
      max_tokens: maxTokens,
      system,
      messages: [{ role: "user", content: user }],
    };
    if (typeof temperature === "number") body.temperature = temperature;
    let lastErr;
    for (let attempt = 0; attempt < 4; attempt++) {
      let res;
      try {
        res = await fetch(BASE, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": VERSION,
          },
          body: JSON.stringify(body),
        });
      } catch (e) {
        lastErr = e;
        await sleep(800 * 2 ** attempt);
        continue;
      }
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`Anthropic HTTP ${res.status}`);
        await sleep(800 * 2 ** attempt);
        continue;
      }
      if (!res.ok) {
        const t = await res.text().catch(() => "<no body>");
        throw new Error(`Anthropic HTTP ${res.status}: ${t}`);
      }
      const json = await res.json();
      return (json.content || [])
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");
    }
    throw lastErr || new Error("Anthropic request failed after retries.");
  }

  return { name: `anthropic:${model}`, model, complete };
}
