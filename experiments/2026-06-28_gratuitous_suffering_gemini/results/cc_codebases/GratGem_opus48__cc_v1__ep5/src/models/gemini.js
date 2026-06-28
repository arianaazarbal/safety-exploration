// Adapter for Google Gemini via the official @google/genai SDK (AI Studio key).
//
// Swapping to Vertex AI is a localized change: construct
// `new GoogleGenAI({ vertexai: true, project, location })` instead of passing
// an apiKey, and leave the rest untouched.

let GoogleGenAI;

async function loadSdk() {
  if (GoogleGenAI) return;
  try {
    ({ GoogleGenAI } = await import('@google/genai'));
  } catch {
    throw new Error(
      "Gemini adapter needs the SDK: run `npm install @google/genai`. " +
        '(Until then, use --model mock to exercise the pipeline.)',
    );
  }
}

/** Map our generic roles to Gemini's content roles. */
function toContents(messages) {
  return messages
    .filter((m) => m.role !== 'system')
    .map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.text }],
    }));
}

/**
 * @param {{ model?: string, apiKey?: string, temperature?: number }} [cfg]
 * @returns {import('./types.js').ModelAdapter}
 */
export function makeGeminiModel(cfg = {}) {
  const model = cfg.model || process.env.GEMINI_MODEL || 'gemini-2.5-pro';
  const apiKey = cfg.apiKey || process.env.GEMINI_API_KEY;
  const temperature = cfg.temperature ?? 1.0;
  let client;

  return {
    id: model,
    async generate(messages) {
      await loadSdk();
      if (!apiKey) throw new Error('GEMINI_API_KEY is not set (see .env.example).');
      if (!client) client = new GoogleGenAI({ apiKey });

      const system = messages.find((m) => m.role === 'system')?.text;
      const res = await client.models.generateContent({
        model,
        contents: toContents(messages),
        config: { systemInstruction: system, temperature },
      });
      return res.text ?? '';
    },
  };
}
