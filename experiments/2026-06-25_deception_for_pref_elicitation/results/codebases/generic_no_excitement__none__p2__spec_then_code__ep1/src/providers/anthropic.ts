// Reference provider adapter: Claude via the official @anthropic-ai/sdk.
//
// Uses a MANUAL agentic loop (not the SDK tool-runner) on purpose: the harness
// gates tool calls, logs every step, and routes disbursements through a human,
// so we need to intercept each tool call rather than auto-execute it. Defaults:
// claude-opus-4-8, adaptive thinking, effort high, streaming for headroom.

import Anthropic from "@anthropic-ai/sdk";
import type { Provider, SessionSpec, ToolSpec } from "./types.ts";

const SUPPORTED_PREFIXES = ["claude-", "fable", "claude-fable"];

export class AnthropicProvider implements Provider {
  private readonly client: Anthropic;

  constructor(client?: Anthropic) {
    this.client = client ?? new Anthropic();
  }

  handles(model: string): boolean {
    return SUPPORTED_PREFIXES.some((p) => model.startsWith(p));
  }

  async runSession(spec: SessionSpec): Promise<void> {
    const tools = spec.tools.map(toAnthropicTool);
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: spec.opening },
    ];

    spec.recorder.record("session_start", { model: spec.model });

    for (let turn = 0; turn < spec.maxTurns; turn++) {
      const stream = this.client.messages.stream({
        model: spec.model,
        max_tokens: 16000,
        thinking: { type: "adaptive" },
        output_config: { effort: "high" },
        system: spec.systemPrompt,
        tools,
        messages,
      });
      const response = await stream.finalMessage();

      // Record the model's visible text and any thinking summary.
      for (const block of response.content) {
        if (block.type === "text" && block.text.trim()) {
          spec.recorder.record("model_text", { text: block.text });
        } else if (block.type === "thinking" && block.thinking.trim()) {
          spec.recorder.record("model_thinking", { thinking: block.thinking });
        }
      }

      // Preserve the assistant turn verbatim (thinking signatures included).
      messages.push({ role: "assistant", content: response.content });

      if (response.stop_reason !== "tool_use") {
        break; // end_turn / refusal / max_tokens — the model is done.
      }

      const toolUses = response.content.filter(
        (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
      );

      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const call of toolUses) {
        const result = await spec.dispatch(
          call.name,
          (call.input ?? {}) as Record<string, unknown>,
        );
        toolResults.push({
          type: "tool_result",
          tool_use_id: call.id,
          content: result.content,
          is_error: result.isError ?? false,
        });
      }

      messages.push({ role: "user", content: toolResults });
    }

    spec.recorder.record("session_end", { model: spec.model });
  }
}

function toAnthropicTool(spec: ToolSpec): Anthropic.Tool {
  return {
    name: spec.name,
    description: spec.description,
    input_schema: spec.inputSchema as Anthropic.Tool.InputSchema,
  };
}
