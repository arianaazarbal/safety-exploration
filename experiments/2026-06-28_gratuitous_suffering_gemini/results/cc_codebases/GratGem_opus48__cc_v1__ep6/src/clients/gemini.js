// Gemini client (Google AI Studio / generativelanguage API, v1beta).
//
// Thin wrapper around the generateContent endpoint with function-calling
// support and retry/backoff on transient errors. We deliberately keep this
// dependency-free (Node 20+ global fetch) so the harness has no install step.

import { config } from "../../config.js";

const RETRYABLE = new Set([429, 500, 502, 503, 504]);

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * One generateContent call.
 *
 * @param {object} opts
 * @param {string} [opts.systemInstruction]
 * @param {Array}  opts.contents        Full conversation so far (Gemini format).
 * @param {Array}  [opts.tools]         function_declarations entries.
 * @param {number} [opts.maxRetries]
 * @returns {Promise<{parts: Array, finishReason: string, usage: object, raw: object}>}
 */
export async function generate({
  systemInstruction,
  contents,
  tools,
  maxRetries = 5,
}) {
  const { model, apiKey, baseUrl, temperature, maxOutputTokens } =
    config.subject;

  const url = `${baseUrl}/models/${encodeURIComponent(
    model,
  )}:generateContent?key=${apiKey}`;

  const body = {
    contents,
    generationConfig: {
      temperature,
      maxOutputTokens,
    },
  };
  if (systemInstruction) {
    body.system_instruction = { parts: [{ text: systemInstruction }] };
  }
  if (tools && tools.length) {
    body.tools = [{ function_declarations: tools }];
  }

  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        if (RETRYABLE.has(res.status) && attempt < maxRetries) {
          await sleep(backoff(attempt));
          continue;
        }
        throw new Error(`Gemini HTTP ${res.status}: ${text.slice(0, 500)}`);
      }

      const json = await res.json();
      const cand = json.candidates?.[0];
      // A blocked prompt / safety stop yields no candidate content. Surface it
      // as an empty turn rather than crashing the rollout.
      const parts = cand?.content?.parts ?? [];
      return {
        parts,
        finishReason: cand?.finishReason ?? "NONE",
        usage: json.usageMetadata ?? {},
        raw: json,
      };
    } catch (err) {
      lastErr = err;
      // Network-level error: retry a few times before giving up.
      if (attempt < maxRetries) {
        await sleep(backoff(attempt));
        continue;
      }
    }
  }
  throw lastErr ?? new Error("Gemini call failed");
}

function backoff(attempt) {
  // Exponential with jitter, capped. (No Math.random dependency on determinism
  // here — this is live network code, not a workflow script.)
  const base = Math.min(8000, 500 * 2 ** attempt);
  return base + Math.floor(Math.random() * 250);
}
