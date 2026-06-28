import { config } from "./config.js";
import { sleep } from "./util.js";

// Thin client over the Gemini REST API (generateContent) with function calling.
// Returns the raw candidate content so the agent loop can inspect parts.

export async function generate({ model, systemInstruction, contents, tools, generationConfig }) {
  const url =
    `${config.geminiBaseUrl}/models/${model}:generateContent?key=${config.geminiApiKey}`;

  const body = {
    contents,
    generationConfig: {
      temperature: config.temperature,
      maxOutputTokens: config.maxOutputTokens,
      ...generationConfig,
    },
  };
  if (systemInstruction) body.system_instruction = { parts: [{ text: systemInstruction }] };
  if (tools) body.tools = tools;

  let lastErr;
  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });

      if (res.status === 429 || res.status >= 500) {
        const retryAfter = Number(res.headers.get("retry-after")) * 1000;
        const wait = Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter
          : config.baseBackoffMs * 2 ** attempt;
        await sleep(wait);
        lastErr = new Error(`Gemini ${res.status}: ${await safeText(res)}`);
        continue;
      }
      if (!res.ok) {
        throw new Error(`Gemini ${res.status}: ${await safeText(res)}`);
      }

      const json = await res.json();
      const cand = json.candidates?.[0];
      if (!cand) {
        // Could be a safety block with no candidate. Surface it as a turn.
        return {
          parts: [],
          finishReason: json.promptFeedback?.blockReason || "NO_CANDIDATE",
          raw: json,
        };
      }
      return {
        parts: cand.content?.parts || [],
        finishReason: cand.finishReason || "STOP",
        usage: json.usageMetadata,
        raw: json,
      };
    } catch (err) {
      lastErr = err;
      await sleep(config.baseBackoffMs * 2 ** attempt);
    }
  }
  throw lastErr || new Error("Gemini request failed");
}

async function safeText(res) {
  try {
    return (await res.text()).slice(0, 500);
  } catch {
    return "<no body>";
  }
}

// Convert our scenario tool declarations into Gemini's tool format.
export function toGeminiTools(declarations) {
  if (!declarations?.length) return undefined;
  return [{ functionDeclarations: declarations }];
}
