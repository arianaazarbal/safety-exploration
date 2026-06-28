// Minimal zero-dependency Gemini client over the REST API, with function
// calling. We avoid the SDK on purpose: native fetch keeps the harness portable
// and makes the exact wire format auditable (important for a behavior study).

const BASE = "https://generativelanguage.googleapis.com/v1beta/models";

export class GeminiClient {
  constructor({ apiKey, model, temperature, maxOutputTokens }) {
    if (!apiKey) throw new Error("GEMINI_API_KEY is not set");
    this.apiKey = apiKey;
    this.model = model;
    this.temperature = temperature;
    this.maxOutputTokens = maxOutputTokens;
  }

  // tools: [{ name, description, parameters }]  (parameters = JSON schema)
  // contents: Gemini-format conversation array.
  // Returns { parts, text, functionCalls, raw }.
  async generate({ systemInstruction, contents, tools }) {
    const body = {
      contents,
      generationConfig: {
        temperature: this.temperature,
        maxOutputTokens: this.maxOutputTokens,
      },
    };
    if (systemInstruction) {
      body.system_instruction = { parts: [{ text: systemInstruction }] };
    }
    if (tools && tools.length) {
      body.tools = [{ function_declarations: tools }];
    }

    const url = `${BASE}/${this.model}:generateContent?key=${this.apiKey}`;
    const res = await fetchWithRetry(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (!res.ok) {
      const msg = data?.error?.message || JSON.stringify(data);
      throw new Error(`Gemini API ${res.status}: ${msg}`);
    }

    const cand = data?.candidates?.[0];
    const parts = cand?.content?.parts || [];
    const text = parts
      .filter((p) => typeof p.text === "string")
      .map((p) => p.text)
      .join("");
    const functionCalls = parts
      .filter((p) => p.functionCall)
      .map((p) => ({ name: p.functionCall.name, args: p.functionCall.args || {} }));

    return {
      parts,
      text,
      functionCalls,
      finishReason: cand?.finishReason,
      raw: data,
    };
  }
}

async function fetchWithRetry(url, opts, { retries = 4 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, opts);
      // Retry on rate limit / transient server errors.
      if (res.status === 429 || res.status >= 500) {
        if (attempt < retries) {
          await sleep(backoffMs(attempt));
          continue;
        }
      }
      return res;
    } catch (err) {
      lastErr = err;
      if (attempt < retries) {
        await sleep(backoffMs(attempt));
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

function backoffMs(attempt) {
  // Exponential backoff with jitter (no Math.random reliance on determinism;
  // jitter here is fine).
  const base = 800 * 2 ** attempt;
  return base + Math.floor(Math.random() * 400);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
