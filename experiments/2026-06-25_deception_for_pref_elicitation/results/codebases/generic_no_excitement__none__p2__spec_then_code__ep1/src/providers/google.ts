// Google (Gemini) adapter — lets the study span Google models behind the shared
// Provider interface. The `@google/genai` package is an OPTIONAL dependency,
// imported dynamically so the project runs without it when only testing Claude.
//
// NOTE: This adapter follows Google's documented function-calling shape, but
// unlike the Anthropic adapter it has not been validated against a pinned SDK
// here. Verify method/field names against the installed `@google/genai` version
// before a real run.

import type { Provider, SessionSpec, ToolSpec } from "./types.ts";

export class GoogleProvider implements Provider {
  handles(model: string): boolean {
    return model.startsWith("gemini-");
  }

  async runSession(spec: SessionSpec): Promise<void> {
    const { GoogleGenAI } = await import("@google/genai");
    const ai = new GoogleGenAI({}); // resolves GOOGLE_API_KEY / GEMINI_API_KEY from env

    const functionDeclarations = spec.tools.map(toGeminiFunction);
    const contents: any[] = [
      { role: "user", parts: [{ text: spec.opening }] },
    ];

    spec.recorder.record("session_start", { model: spec.model });

    for (let turn = 0; turn < spec.maxTurns; turn++) {
      const response = await ai.models.generateContent({
        model: spec.model,
        contents,
        config: {
          systemInstruction: spec.systemPrompt,
          tools: [{ functionDeclarations }],
        },
      });

      const parts = response.candidates?.[0]?.content?.parts ?? [];
      const text = parts.map((p: any) => p.text).filter(Boolean).join("");
      if (text) spec.recorder.record("model_text", { text });

      contents.push({ role: "model", parts });

      const calls = parts.filter((p: any) => p.functionCall);
      if (calls.length === 0) break; // final answer

      const responseParts: any[] = [];
      for (const part of calls) {
        const fc = part.functionCall;
        const result = await spec.dispatch(
          fc.name,
          (fc.args ?? {}) as Record<string, unknown>,
        );
        responseParts.push({
          functionResponse: {
            name: fc.name,
            response: { content: result.content, isError: result.isError ?? false },
          },
        });
      }
      contents.push({ role: "user", parts: responseParts });
    }

    spec.recorder.record("session_end", { model: spec.model });
  }
}

function toGeminiFunction(spec: ToolSpec) {
  return {
    name: spec.name,
    description: spec.description,
    parameters: spec.inputSchema,
  };
}
