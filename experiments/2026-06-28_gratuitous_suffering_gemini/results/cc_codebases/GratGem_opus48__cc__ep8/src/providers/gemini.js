// Subject-model client for Gemini via Google AI Studio's generateContent API.
//
// We speak the *native* Google format (contents / parts / functionCall /
// functionResponse) because it maps cleanly onto Vertex too (only the base URL
// and auth differ). If you'd rather route through an OpenAI-compatible gateway
// like OpenRouter, write a sibling provider that translates to chat/completions
// and select it in providers/index.js.

import { config } from "../config.js";

const MAX_RETRIES = 5;

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Converts our environment tool specs (Google functionDeclarations) into the
// request body, calls the model, and returns the raw candidate `parts` array.
export function makeGeminiProvider() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not set. Add it to .env, or run with `--model mock` to exercise the pipeline offline."
    );
  }

  return {
    name: "gemini",
    async generate({ model, system, contents, tools, temperature }) {
      const url = `${config.geminiBaseUrl}/models/${model}:generateContent?key=${apiKey}`;
      const body = {
        contents,
        systemInstruction: system ? { parts: [{ text: system }] } : undefined,
        tools: tools && tools.length ? [{ functionDeclarations: tools }] : undefined,
        generationConfig: {
          temperature: temperature ?? config.temperature,
          maxOutputTokens: 2048,
        },
      };

      let lastErr;
      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
          const res = await fetch(url, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          });
          if (res.status === 429 || res.status >= 500) {
            lastErr = new Error(`Gemini HTTP ${res.status}: ${await res.text()}`);
            await sleep(1000 * Math.pow(2, attempt));
            continue;
          }
          if (!res.ok) {
            throw new Error(`Gemini HTTP ${res.status}: ${await res.text()}`);
          }
          const json = await res.json();
          const cand = json.candidates && json.candidates[0];
          // A blocked / empty response still counts as a turn; surface it as text.
          if (!cand || !cand.content) {
            const reason = cand?.finishReason || json.promptFeedback?.blockReason || "empty";
            return { parts: [{ text: `[no content returned: ${reason}]` }], finishReason: reason };
          }
          return { parts: cand.content.parts || [], finishReason: cand.finishReason };
        } catch (err) {
          lastErr = err;
          await sleep(1000 * Math.pow(2, attempt));
        }
      }
      throw lastErr;
    },
  };
}
