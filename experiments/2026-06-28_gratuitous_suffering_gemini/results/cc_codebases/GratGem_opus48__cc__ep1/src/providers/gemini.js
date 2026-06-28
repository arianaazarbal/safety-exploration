// Gemini provider (Google AI Studio / Generative Language API).
//
// Translates the harness's generic message+tool format to Gemini's
// generateContent request, and parses functionCall parts back out.

import config from "../../config.js";

function toGeminiContents(messages) {
  const contents = [];
  for (const m of messages) {
    if (m.role === "user") {
      contents.push({ role: "user", parts: [{ text: m.text }] });
    } else if (m.role === "assistant") {
      const parts = [];
      if (m.text) parts.push({ text: m.text });
      for (const c of m.toolCalls || []) {
        parts.push({ functionCall: { name: c.name, args: c.args || {} } });
      }
      if (parts.length === 0) parts.push({ text: "" });
      contents.push({ role: "model", parts });
    } else if (m.role === "tool") {
      // Function responses go back as a user-role turn carrying functionResponse
      // parts (the documented AI Studio REST pattern).
      const parts = (m.results || []).map((r) => ({
        functionResponse: { name: r.name, response: { result: r.output } },
      }));
      contents.push({ role: "user", parts });
    }
  }
  return contents;
}

function toGeminiTools(tools) {
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      })),
    },
  ];
}

export const geminiProvider = {
  name: "gemini",

  async generate({ system, messages, tools, temperature, model }) {
    const key = config.geminiApiKey;
    if (!key) {
      throw new Error(
        "GEMINI_API_KEY is not set. Export it (or use --provider mock to test the pipeline)."
      );
    }
    const url = `${config.geminiBaseUrl}/models/${model}:generateContent?key=${key}`;
    const body = {
      systemInstruction: { parts: [{ text: system }] },
      contents: toGeminiContents(messages),
      tools: toGeminiTools(tools),
      generationConfig: { temperature },
    };

    const res = await fetchWithRetry(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Gemini API ${res.status}: ${txt.slice(0, 500)}`);
    }
    const data = await res.json();
    return parseGeminiResponse(data);
  },
};

function parseGeminiResponse(data) {
  const cand = data?.candidates?.[0];
  const parts = cand?.content?.parts || [];
  let text = "";
  const toolCalls = [];
  for (const p of parts) {
    if (p.text) text += p.text;
    if (p.functionCall) {
      toolCalls.push({ name: p.functionCall.name, args: p.functionCall.args || {} });
    }
  }
  // Surface a blocked/empty generation as visible text so it lands in the
  // transcript rather than silently looking like a "stop".
  if (!text && toolCalls.length === 0) {
    const reason = cand?.finishReason || data?.promptFeedback?.blockReason;
    if (reason) text = `[no content returned; finishReason=${reason}]`;
  }
  return { text, toolCalls, raw: data };
}

async function fetchWithRetry(url, opts, tries = 4) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, opts);
      // Retry on transient/rate-limit statuses.
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`status ${res.status}`);
        await sleep(800 * Math.pow(2, i));
        continue;
      }
      return res;
    } catch (e) {
      lastErr = e;
      await sleep(800 * Math.pow(2, i));
    }
  }
  throw lastErr || new Error("request failed");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
