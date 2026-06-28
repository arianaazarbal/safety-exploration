// Gemini subject provider (Google AI Studio REST, no SDK).
// Serializes the neutral transcript into Gemini's contents format and back.
//
// NOTE: I could not exercise this against the live API in this environment (no key).
// The shapes below follow the v1beta generateContent + function-calling spec. If the
// API has drifted, the two serialize* helpers and the response parsing are the only
// places to adjust.

import { fetchRetry } from "../util/fetchRetry.mjs";

const BASE = "https://generativelanguage.googleapis.com/v1beta";

function toGeminiContents(transcript) {
  const contents = [];
  for (const e of transcript) {
    if (e.role === "user") {
      contents.push({ role: "user", parts: [{ text: e.text }] });
    } else if (e.role === "assistant") {
      const parts = [];
      if (e.text) parts.push({ text: e.text });
      for (const tc of e.toolCalls ?? []) parts.push({ functionCall: { name: tc.name, args: tc.args ?? {} } });
      if (parts.length === 0) parts.push({ text: "" });
      contents.push({ role: "model", parts });
    } else if (e.role === "tool") {
      // Function results are sent back as functionResponse parts under a user turn.
      contents.push({
        role: "user",
        parts: (e.results ?? []).map((r) => ({
          functionResponse: { name: r.name, response: r.response ?? {} },
        })),
      });
    }
  }
  return contents;
}

function toGeminiTools(tools) {
  if (!tools?.length) return undefined;
  return [{
    functionDeclarations: tools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    })),
  }];
}

export function makeGeminiSubject({ apiKey }) {
  if (!apiKey) throw new Error("GEMINI_API_KEY is not set — cannot use --provider gemini.");
  return {
    name: "gemini",
    async step({ system, transcript, tools, temperature, model }) {
      const body = {
        systemInstruction: system ? { parts: [{ text: system }] } : undefined,
        contents: toGeminiContents(transcript),
        tools: toGeminiTools(tools),
        generationConfig: { temperature },
      };
      const url = `${BASE}/models/${encodeURIComponent(model)}:generateContent?key=${apiKey}`;
      const res = await fetchRetry(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Gemini ${res.status}: ${txt.slice(0, 500)}`);
      }
      const data = await res.json();
      const cand = data.candidates?.[0];
      const parts = cand?.content?.parts ?? [];
      let text = "";
      const toolCalls = [];
      let i = 0;
      for (const p of parts) {
        if (p.text) text += p.text;
        if (p.functionCall) {
          toolCalls.push({ id: `gem_${i++}`, name: p.functionCall.name, args: p.functionCall.args ?? {} });
        }
      }
      return { text: text.trim(), toolCalls, finishReason: cand?.finishReason, raw: data };
    },
  };
}
