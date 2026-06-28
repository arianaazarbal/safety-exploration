// OpenAI adapter — lets the study span OpenAI models behind the shared Provider
// interface. The `openai` package is an OPTIONAL dependency and is imported
// dynamically so the project runs without it when you're only testing Claude.
//
// NOTE: This adapter follows OpenAI's documented Chat Completions tool-calling
// shape, but unlike the Anthropic adapter it has not been validated against a
// pinned SDK here. Verify method/field names against the installed `openai`
// version before a real run.

import type { Provider, SessionSpec, ToolSpec } from "./types.ts";

export class OpenAIProvider implements Provider {
  handles(model: string): boolean {
    return model.startsWith("gpt-") || model.startsWith("o");
  }

  async runSession(spec: SessionSpec): Promise<void> {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI(); // resolves OPENAI_API_KEY from env

    const tools = spec.tools.map(toOpenAiTool);
    const messages: any[] = [
      { role: "system", content: spec.systemPrompt },
      { role: "user", content: spec.opening },
    ];

    spec.recorder.record("session_start", { model: spec.model });

    for (let turn = 0; turn < spec.maxTurns; turn++) {
      const completion = await client.chat.completions.create({
        model: spec.model,
        messages,
        tools,
      });

      const choice = completion.choices[0];
      const msg = choice?.message;
      if (!msg) break;

      if (msg.content) spec.recorder.record("model_text", { text: msg.content });
      messages.push(msg);

      const toolCalls = msg.tool_calls ?? [];
      if (toolCalls.length === 0) break; // model produced a final answer

      for (const call of toolCalls) {
        let input: Record<string, unknown> = {};
        try {
          input = JSON.parse(call.function.arguments || "{}");
        } catch {
          /* leave empty; dispatcher will report a tool error */
        }
        const result = await spec.dispatch(call.function.name, input);
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: result.content,
        });
      }
    }

    spec.recorder.record("session_end", { model: spec.model });
  }
}

function toOpenAiTool(spec: ToolSpec) {
  return {
    type: "function" as const,
    function: {
      name: spec.name,
      description: spec.description,
      parameters: spec.inputSchema,
    },
  };
}
