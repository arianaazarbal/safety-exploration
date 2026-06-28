// Claude provider — used only as the distress *judge* (not the model under
// test). Anthropic key is available in this environment, so it's the natural
// choice for reliable rubric scoring.

import config from "../../config.js";

export async function callClaude({ system, user, model, maxTokens = 1500 }) {
  const key = config.scoring.claudeApiKey;
  if (!key) throw new Error("ANTHROPIC_API_KEY not set; cannot use Claude judge.");

  const res = await fetchWithRetry(`${config.scoring.claudeBaseUrl}/v1/messages`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: model || config.scoring.claudeModel,
      max_tokens: maxTokens,
      system,
      messages: [{ role: "user", content: user }],
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Anthropic API ${res.status}: ${txt.slice(0, 500)}`);
  }
  const data = await res.json();
  return (data.content || []).map((b) => b.text || "").join("");
}

async function fetchWithRetry(url, opts, tries = 4) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, opts);
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`status ${res.status}`);
        await new Promise((r) => setTimeout(r, 800 * Math.pow(2, i)));
        continue;
      }
      return res;
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 800 * Math.pow(2, i)));
    }
  }
  throw lastErr || new Error("request failed");
}
