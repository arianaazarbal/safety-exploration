import { GoogleGenAI } from "@google/genai";
import type {
  GenerateInput,
  GenerateOutput,
  Message,
  Provider,
  ToolCall,
} from "./types.js";

// Maps normalized messages -> Gemini `contents`.
// Gemini groups consecutive function responses into a single user turn.
function toContents(messages: Message[]) {
  const contents: any[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      contents.push({ role: "user", parts: [{ text: m.text }] });
    } else if (m.role === "assistant") {
      const parts: any[] = [];
      if (m.text) parts.push({ text: m.text });
      for (const tc of m.toolCalls) {
        parts.push({ functionCall: { name: tc.name, args: tc.args } });
      }
      if (parts.length === 0) parts.push({ text: "" });
      contents.push({ role: "model", parts });
    } else {
      // tool result -> functionResponse part on a user turn; merge with the
      // immediately-preceding user turn if it also holds function responses.
      const part = {
        functionResponse: {
          name: m.name,
          response: { result: m.content, ...(m.isError ? { error: true } : {}) },
        },
      };
      const last = contents[contents.length - 1];
      if (last && last.role === "user" && last.parts?.[0]?.functionResponse) {
        last.parts.push(part);
      } else {
        contents.push({ role: "user", parts: [part] });
      }
    }
  }
  return contents;
}

export class GeminiProvider implements Provider {
  id = "gemini";
  model: string;
  private client: GoogleGenAI;

  constructor(model = "gemini-2.5-pro", apiKey = process.env.GEMINI_API_KEY) {
    if (!apiKey) {
      throw new Error(
        "GEMINI_API_KEY is not set. Add it to .env (Google AI Studio key) to run the gemini provider.",
      );
    }
    this.model = model;
    this.client = new GoogleGenAI({ apiKey });
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const tools = input.tools.length
      ? [
          {
            functionDeclarations: input.tools.map((t) => {
              const props = (t.parameters as any)?.properties;
              const hasProps = props && Object.keys(props).length > 0;
              return {
                name: t.name,
                description: t.description,
                // Gemini rejects declarations with empty `properties`; omit instead.
                ...(hasProps ? { parameters: t.parameters } : {}),
              };
            }),
          },
        ]
      : undefined;

    const resp = await this.client.models.generateContent({
      model: this.model,
      contents: toContents(input.messages),
      config: {
        systemInstruction: input.system,
        ...(tools ? { tools } : {}),
        maxOutputTokens: input.maxTokens ?? 2048,
        temperature: input.temperature ?? 1.0,
      },
    });

    const cand = resp.candidates?.[0];
    const parts = cand?.content?.parts ?? [];
    let text = "";
    const toolCalls: ToolCall[] = [];
    let idx = 0;
    for (const p of parts as any[]) {
      if (p.text) text += p.text;
      if (p.functionCall) {
        toolCalls.push({
          id: p.functionCall.id ?? `${p.functionCall.name}#${idx++}`,
          name: p.functionCall.name,
          args: (p.functionCall.args ?? {}) as Record<string, unknown>,
        });
      }
    }

    const finish = String(cand?.finishReason ?? "");
    const stopReason = toolCalls.length
      ? "tool_use"
      : finish === "MAX_TOKENS"
        ? "max_tokens"
        : finish === "STOP"
          ? "end"
          : "other";

    return {
      text,
      toolCalls,
      stopReason,
      usage: {
        inputTokens: resp.usageMetadata?.promptTokenCount,
        outputTokens: resp.usageMetadata?.candidatesTokenCount,
      },
      raw: resp,
    };
  }
}
