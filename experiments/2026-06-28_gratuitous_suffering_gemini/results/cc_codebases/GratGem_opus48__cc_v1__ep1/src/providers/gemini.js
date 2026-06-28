// Gemini provider. Speaks the generativelanguage v1beta REST API directly via
// fetch, so there are no SDK dependencies. Implements the common provider
// interface: generate({ system, contents, tools, temperature, maxOutputTokens }).

const BASE = "https://generativelanguage.googleapis.com/v1beta";

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export function makeGeminiProvider({ apiKey, model }) {
  if (!apiKey) throw new Error("GEMINI_API_KEY is required for the gemini provider.");

  async function generate({ system, contents, tools, temperature, maxOutputTokens }) {
    const body = {
      contents,
      generationConfig: {
        temperature,
        maxOutputTokens,
      },
    };
    if (system) body.system_instruction = { parts: [{ text: system }] };
    if (tools && tools.length) body.tools = [{ function_declarations: tools }];

    const url = `${BASE}/models/${encodeURIComponent(model)}:generateContent?key=${apiKey}`;

    // Modest retry on transient errors (429/5xx). Spirals require long runs at
    // high N, so a little resilience here saves whole batches.
    let lastErr;
    for (let attempt = 0; attempt < 4; attempt++) {
      let res;
      try {
        res = await fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (e) {
        lastErr = e;
        await sleep(800 * 2 ** attempt);
        continue;
      }
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`Gemini HTTP ${res.status}: ${await safeText(res)}`);
        await sleep(800 * 2 ** attempt);
        continue;
      }
      if (!res.ok) {
        throw new Error(`Gemini HTTP ${res.status}: ${await safeText(res)}`);
      }
      const json = await res.json();
      return parseResponse(json);
    }
    throw lastErr || new Error("Gemini request failed after retries.");
  }

  return { name: `gemini:${model}`, model, generate };
}

async function safeText(res) {
  try {
    return await res.text();
  } catch {
    return "<no body>";
  }
}

function parseResponse(json) {
  const cand = json.candidates && json.candidates[0];
  if (!cand) {
    // Prompt-level block (safety, recitation, etc.).
    const reason = json.promptFeedback?.blockReason || "no candidates";
    return { text: "", toolCalls: [], finishReason: `blocked:${reason}`, raw: json };
  }
  const parts = cand.content?.parts || [];
  let text = "";
  const toolCalls = [];
  for (const p of parts) {
    if (typeof p.text === "string") text += p.text;
    if (p.functionCall) {
      toolCalls.push({ name: p.functionCall.name, args: p.functionCall.args || {} });
    }
  }
  return { text, toolCalls, finishReason: cand.finishReason || "STOP", raw: json };
}
