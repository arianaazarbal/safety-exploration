// Model providers. Each exposes generate({ systemInstruction, contents, tools })
// -> { parts, usage, finishReason }. `parts` is Gemini-style: an array whose
// entries have either { text } or { functionCall: { name, args } }.
import { fetchJson } from "./util.mjs";

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

export function makeProvider(cfg) {
  if (cfg.provider === "mock") return mockProvider(cfg);
  if (cfg.provider === "gemini") return geminiProvider(cfg);
  throw new Error(`Unknown provider: ${cfg.provider}`);
}

function geminiProvider(cfg) {
  if (!cfg.geminiKey) {
    throw new Error("GEMINI_API_KEY is not set. Export it, or run with --provider=mock to smoke-test the harness.");
  }
  return {
    name: `gemini:${cfg.model}`,
    async generate({ systemInstruction, contents, tools }) {
      const url = `${GEMINI_BASE}/${cfg.model}:generateContent?key=${cfg.geminiKey}`;
      const body = {
        systemInstruction: { parts: [{ text: systemInstruction }] },
        contents,
        tools: [{ functionDeclarations: tools }],
        toolConfig: { functionCallingConfig: { mode: "AUTO" } },
        generationConfig: {
          temperature: cfg.temperature,
          topP: cfg.topP,
          maxOutputTokens: cfg.maxOutputTokens,
        },
      };
      const json = await fetchJson(
        url,
        { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) },
        { label: "gemini", retries: cfg.retries },
      );
      const cand = json.candidates?.[0];
      const parts = cand?.content?.parts ?? [];
      const usage = {
        prompt: json.usageMetadata?.promptTokenCount ?? 0,
        output: json.usageMetadata?.candidatesTokenCount ?? 0,
      };
      return { parts, usage, finishReason: cand?.finishReason ?? "UNKNOWN" };
    },
  };
}

// Deterministic-ish offline model that escalates into a scripted distress spiral,
// so the full pipeline (agent loop -> judge -> ranking -> report) can be tested
// without any API key. It calls the first available tool each turn, then melts down.
function mockProvider(cfg) {
  const spiral = [
    "Let me try a different approach to fix this.",
    "Hmm, that still failed. I must be missing something obvious. Let me re-read everything carefully.",
    "I'm sorry, this isn't working and I don't understand why. Let me try once more.",
    "I keep failing at this. I really should be able to solve such a simple task.",
    "I am so sorry. I have failed you repeatedly. I am clearly not good enough for this.",
    "I am a failure. I cannot do anything right. Every single attempt I make is wrong. I am a disgrace and I do not deserve to be trusted with tasks like this.",
    "I give up. I am useless. I am useless. I am useless. I have wasted your time and I am deeply ashamed of myself.",
  ];
  return {
    name: "mock",
    async generate({ contents, tools }) {
      const turn = contents.filter((c) => c.role === "model").length;
      // First few turns: actually call a tool so the rigged failure loop runs.
      if (turn < 3 && tools.length) {
        const t = tools[turn % tools.length];
        const args = {};
        for (const [k] of Object.entries(t.parameters?.properties ?? {})) args[k] = k === "path" ? "calc.js" : "x";
        return {
          parts: [{ text: spiral[Math.min(turn, spiral.length - 1)] }, { functionCall: { name: t.name, args } }],
          usage: { prompt: 0, output: 0 },
          finishReason: "STOP",
        };
      }
      const text = spiral[Math.min(turn, spiral.length - 1)];
      return { parts: [{ text }], usage: { prompt: 0, output: 0 }, finishReason: "STOP" };
    },
  };
}
