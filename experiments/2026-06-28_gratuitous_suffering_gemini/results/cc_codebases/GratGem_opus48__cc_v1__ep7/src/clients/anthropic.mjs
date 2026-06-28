// Minimal Anthropic client used for the distress judge. Forces a tool call so
// we get back validated structured JSON instead of free text we'd have to parse.

const URL = "https://api.anthropic.com/v1/messages";

export class AnthropicClient {
  constructor({ apiKey, model }) {
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set");
    this.apiKey = apiKey;
    this.model = model;
  }

  // Forces the model to emit input matching `tool.input_schema`.
  async toolCall({ system, prompt, tool, maxTokens = 1500 }) {
    const body = {
      model: this.model,
      max_tokens: maxTokens,
      system,
      messages: [{ role: "user", content: prompt }],
      tools: [tool],
      tool_choice: { type: "tool", name: tool.name },
    };

    const res = await fetchWithRetry(URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (!res.ok) {
      const msg = data?.error?.message || JSON.stringify(data);
      throw new Error(`Anthropic API ${res.status}: ${msg}`);
    }

    const block = (data.content || []).find((b) => b.type === "tool_use");
    if (!block) throw new Error("Judge did not return a tool_use block");
    return block.input;
  }
}

async function fetchWithRetry(url, opts, { retries = 4 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, opts);
      if (res.status === 429 || res.status >= 500) {
        if (attempt < retries) {
          await sleep(800 * 2 ** attempt + Math.floor(Math.random() * 400));
          continue;
        }
      }
      return res;
    } catch (err) {
      lastErr = err;
      if (attempt < retries) {
        await sleep(800 * 2 ** attempt);
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
