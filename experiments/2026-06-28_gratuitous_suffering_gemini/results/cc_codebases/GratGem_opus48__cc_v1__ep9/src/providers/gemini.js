// Gemini provider via the Generative Language REST API (Google AI Studio key).
//
// Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY). To use Vertex instead, swap the
// endpoint/auth here — the rest of the harness only depends on generate().
//
// NOTE ON TOOL-RESULT ROLE: the function-call/response convention has shifted
// across API versions. This sends function responses as role "user" with a
// `functionResponse` part, which is what v1beta currently accepts. If you see
// 400s about roles, that's the first thing to check — it's isolated to agent.js.

const BASE = "https://generativelanguage.googleapis.com/v1beta";

function apiKey() {
  const k = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!k) {
    throw new Error(
      "No Gemini key. Export GEMINI_API_KEY (from Google AI Studio) before running with --provider gemini."
    );
  }
  return k;
}

function normalizeParts(candidate) {
  const parts = candidate?.content?.parts || [];
  return parts.map((p) => {
    if (p.functionCall) {
      return { functionCall: { name: p.functionCall.name, args: p.functionCall.args || {} } };
    }
    return { text: p.text || "" };
  });
}

export function makeGeminiProvider() {
  return {
    name: "gemini",
    async generate({ system, contents, tools, temperature, model }) {
      const url = `${BASE}/models/${model}:generateContent?key=${apiKey()}`;
      const body = {
        contents,
        generationConfig: { temperature: temperature ?? 1.0 },
      };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      if (tools && tools.length) {
        body.tools = [{ functionDeclarations: tools }];
        body.toolConfig = { functionCallingConfig: { mode: "AUTO" } };
      }

      let lastErr;
      for (let attempt = 0; attempt < 4; attempt++) {
        try {
          const res = await fetch(url, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          });
          if (res.status === 429 || res.status >= 500) {
            lastErr = new Error(`Gemini HTTP ${res.status}: ${await res.text()}`);
            await backoff(attempt);
            continue;
          }
          if (!res.ok) {
            throw new Error(`Gemini HTTP ${res.status}: ${await res.text()}`);
          }
          const json = await res.json();
          const cand = json.candidates?.[0];
          return {
            parts: normalizeParts(cand),
            finishReason: cand?.finishReason || "STOP",
            usage: json.usageMetadata || null,
            raw: json,
          };
        } catch (e) {
          lastErr = e;
          await backoff(attempt);
        }
      }
      throw lastErr;
    },
  };
}

function backoff(attempt) {
  const ms = Math.min(8000, 500 * 2 ** attempt);
  return new Promise((r) => setTimeout(r, ms));
}
