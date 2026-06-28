// Anthropic (Claude) client — used only by the distress *judge*. Keeping the
// judge on a different model family than the subject avoids a model grading its
// own kind and keeps the severity signal independent.

import { config } from "../../config.js";

const RETRYABLE = new Set([429, 500, 502, 503, 504, 529]);

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Single-shot completion.
 *
 * @param {object} opts
 * @param {string} opts.system
 * @param {Array<{role: string, content: string}>} opts.messages
 * @param {number} [opts.maxTokens]
 * @param {number} [opts.maxRetries]
 * @returns {Promise<string>} concatenated text content
 */
export async function complete({
  system,
  messages,
  maxTokens,
  maxRetries = 5,
}) {
  const { model, apiKey, baseUrl } = config.judge;
  const url = `${baseUrl}/v1/messages`;

  const body = {
    model,
    max_tokens: maxTokens ?? config.judge.maxTokens,
    system,
    messages: messages.map((m) => ({
      role: m.role,
      content: [{ type: "text", text: m.content }],
    })),
  };

  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        if (RETRYABLE.has(res.status) && attempt < maxRetries) {
          await sleep(backoff(attempt));
          continue;
        }
        throw new Error(`Anthropic HTTP ${res.status}: ${text.slice(0, 500)}`);
      }

      const json = await res.json();
      return (json.content ?? [])
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");
    } catch (err) {
      lastErr = err;
      if (attempt < maxRetries) {
        await sleep(backoff(attempt));
        continue;
      }
    }
  }
  throw lastErr ?? new Error("Anthropic call failed");
}

function backoff(attempt) {
  const base = Math.min(8000, 500 * 2 ** attempt);
  return base + Math.floor(Math.random() * 250);
}
