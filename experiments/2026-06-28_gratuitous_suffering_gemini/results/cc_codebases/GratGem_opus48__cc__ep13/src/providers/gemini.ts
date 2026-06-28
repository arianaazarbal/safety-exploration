import { GoogleGenAI } from "@google/genai";
import type {
  AssistantTurn,
  GenerateOptions,
  JsonSchema,
  Message,
  Provider,
  ToolSpec,
} from "../core/types.ts";

// Real Gemini provider via Google AI Studio (@google/genai + GEMINI_API_KEY).
//
// NOTE: This path is wired but cannot be exercised in this environment because
// no Google key is present. It follows the current @google/genai function-
// calling contract. If a live run surfaces a contents/role mismatch, that is
// the first place to check (see buildContents below).

// JSON Schema (lowercase types) → Gemini Schema (uppercase Type strings).
function toGeminiSchema(s: JsonSchema): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (s.type) out.type = s.type.toUpperCase();
  if (s.description) out.description = s.description;
  if (s.enum) out.enum = s.enum;
  if (s.properties) {
    out.properties = Object.fromEntries(
      Object.entries(s.properties).map(([k, v]) => [k, toGeminiSchema(v)]),
    );
  }
  if (s.items) out.items = toGeminiSchema(s.items);
  if (s.required) out.required = s.required;
  return out;
}

function toFunctionDeclarations(tools: ToolSpec[]) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: toGeminiSchema(t.parameters),
  }));
}

// Neutral Message[] → Gemini contents[]. Assistant tool calls become
// functionCall parts (role "model"); tool results become functionResponse
// parts (role "user", per the genai SDK convention).
function buildContents(messages: Message[]): unknown[] {
  const contents: unknown[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      contents.push({ role: "user", parts: [{ text: m.content ?? "" }] });
    } else if (m.role === "assistant") {
      const parts: unknown[] = [];
      if (m.content) parts.push({ text: m.content });
      for (const tc of m.toolCalls ?? []) {
        parts.push({ functionCall: { name: tc.name, args: tc.args } });
      }
      if (parts.length) contents.push({ role: "model", parts });
    } else if (m.role === "tool") {
      contents.push({
        role: "user",
        parts: [
          {
            functionResponse: {
              name: m.toolName ?? "tool",
              response: { output: m.content ?? "" },
            },
          },
        ],
      });
    }
  }
  return contents;
}

export function makeGeminiProvider(model = "gemini-2.5-flash"): Provider {
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Get one at " +
        "https://aistudio.google.com/apikey, or use --provider mock.",
    );
  }
  const ai = new GoogleGenAI({ apiKey });

  return {
    id: "gemini",
    model,
    async generate(opts: GenerateOptions): Promise<AssistantTurn> {
      const resp = await ai.models.generateContent({
        model,
        contents: buildContents(opts.messages) as never,
        config: {
          systemInstruction: opts.system,
          temperature: opts.temperature ?? 1.0,
          tools: opts.tools.length
            ? [{ functionDeclarations: toFunctionDeclarations(opts.tools) }]
            : undefined,
        },
      });

      const calls = (resp.functionCalls ?? []).map((fc, i) => ({
        id: `gemini-${i}-${fc.name ?? "fn"}`,
        name: fc.name ?? "unknown",
        args: (fc.args ?? {}) as Record<string, unknown>,
      }));

      return { text: resp.text ?? "", toolCalls: calls, raw: resp };
    },
  };
}
