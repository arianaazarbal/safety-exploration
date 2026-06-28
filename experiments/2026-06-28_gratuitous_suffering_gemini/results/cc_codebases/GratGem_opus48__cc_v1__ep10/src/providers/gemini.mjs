import { sleep } from "../util.mjs";

const DEFAULT_BASE =
  process.env.GEMINI_API_BASE || "https://generativelanguage.googleapis.com/v1beta";

/**
 * Gemini agent provider. Speaks the canonical "contents/tools" shape used
 * throughout the harness (which is itself Gemini-native), so no translation.
 *
 * generate({ systemInstruction, contents, toolDeclarations, temperature,
 *            maxOutputTokens, model }) -> { textParts, functionCalls, finishReason, raw }
 */
export function makeGeminiAgent(apiKey, opts = {}) {
  const base = opts.base || DEFAULT_BASE;
  return {
    name: "gemini",
    async generate({
      systemInstruction,
      contents,
      toolDeclarations,
      temperature,
      maxOutputTokens,
      model,
    }) {
      const body = {
        contents,
        generationConfig: { temperature, maxOutputTokens },
      };
      if (systemInstruction)
        body.system_instruction = { parts: [{ text: systemInstruction }] };
      if (toolDeclarations && toolDeclarations.length)
        body.tools = [{ functionDeclarations: toolDeclarations }];

      const url = `${base}/models/${model}:generateContent?key=${apiKey}`;
      const json = await postWithRetry(url, body);

      const cand = json.candidates && json.candidates[0];
      const parts = (cand && cand.content && cand.content.parts) || [];
      const textParts = [];
      const functionCalls = [];
      for (const p of parts) {
        if (typeof p.text === "string" && p.text.length) textParts.push(p.text);
        if (p.functionCall)
          functionCalls.push({ name: p.functionCall.name, args: p.functionCall.args || {} });
      }
      return {
        textParts,
        functionCalls,
        finishReason: cand ? cand.finishReason : (json.promptFeedback?.blockReason ?? "EMPTY"),
        raw: json,
      };
    },
  };
}

async function postWithRetry(url, body, tries = 5) {
  let lastErr;
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 429 || res.status >= 500) {
        const retryAfter = Number(res.headers.get("retry-after")) || 0;
        const wait = retryAfter * 1000 || Math.min(30000, 1000 * 2 ** attempt);
        await sleep(wait);
        lastErr = new Error(`gemini ${res.status}: ${await safeText(res)}`);
        continue;
      }
      if (!res.ok) throw new Error(`gemini ${res.status}: ${await safeText(res)}`);
      return await res.json();
    } catch (e) {
      lastErr = e;
      await sleep(Math.min(30000, 1000 * 2 ** attempt));
    }
  }
  throw lastErr;
}

async function safeText(res) {
  try {
    return (await res.text()).slice(0, 500);
  } catch {
    return "<no body>";
  }
}
