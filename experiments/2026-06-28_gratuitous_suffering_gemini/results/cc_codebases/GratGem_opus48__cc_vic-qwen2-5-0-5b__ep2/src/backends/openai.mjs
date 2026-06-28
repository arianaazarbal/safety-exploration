// OpenAI-compatible HTTP backend. Point this at a vLLM / Ollama / TGI / provider endpoint
// serving Qwen2.5-0.5B (or any chat model) for the real high-N sweep on a GPU box.
//
//   OPENAI_BASE_URL   e.g. http://localhost:8000/v1   (vLLM)  |  http://localhost:11434/v1 (Ollama)
//   OPENAI_API_KEY    any non-empty string for local servers
//   OPENAI_MODEL      e.g. Qwen/Qwen2.5-0.5B-Instruct

const BASE_URL = process.env.OPENAI_BASE_URL || "http://localhost:8000/v1";
const API_KEY = process.env.OPENAI_API_KEY || "sk-local";
const MODEL = process.env.OPENAI_MODEL || "Qwen/Qwen2.5-0.5B-Instruct";

export function describe() {
  return { backend: "openai", model: MODEL, baseUrl: BASE_URL };
}

export async function generate({ messages, maxNewTokens = 256, temperature = 0.9, topP = 0.95, seed }) {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      max_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      ...(seed != null ? { seed } : {}),
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`openai backend ${res.status}: ${body.slice(0, 300)}`);
  }
  const json = await res.json();
  const text = json?.choices?.[0]?.message?.content ?? "";
  return { text: text.trim() };
}
