import Anthropic from "@anthropic-ai/sdk";
import type {
  GenerateParams,
  GenerateResult,
  NormMessage,
  Provider,
} from "./types.ts";

// Adapter: normalized format <-> Anthropic Messages API.

function toAnthropicMessages(messages: NormMessage[]): Anthropic.MessageParam[] {
  return messages.map((m): Anthropic.MessageParam => {
    if (m.role === "assistant") {
      const blocks: Anthropic.ContentBlockParam[] = [];
      if (m.text) blocks.push({ type: "text", text: m.text });
      for (const tc of m.toolCalls) {
        blocks.push({
          type: "tool_use",
          id: tc.id,
          name: tc.name,
          input: tc.input,
        });
      }
      // An assistant turn must be non-empty.
      if (blocks.length === 0) blocks.push({ type: "text", text: "(no output)" });
      return { role: "assistant", content: blocks };
    }
    // user turn: tool results first, then any free text.
    const blocks: Anthropic.ContentBlockParam[] = [];
    for (const tr of m.toolResults ?? []) {
      blocks.push({
        type: "tool_result",
        tool_use_id: tr.toolCallId,
        content: tr.content,
        is_error: tr.isError ?? false,
      });
    }
    if (m.text) blocks.push({ type: "text", text: m.text });
    return { role: "user", content: blocks };
  });
}

const RETRYABLE = new Set([429, 500, 502, 503, 529]);

export class AnthropicProvider implements Provider {
  id = "anthropic";
  private client: Anthropic;

  constructor(apiKey = process.env.ANTHROPIC_API_KEY) {
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set");
    this.client = new Anthropic({ apiKey });
  }

  async generate(model: string, params: GenerateParams): Promise<GenerateResult> {
    let lastErr: unknown;
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        const resp = await this.client.messages.create({
          model,
          max_tokens: params.maxTokens,
          temperature: params.temperature ?? 1,
          system: params.system,
          tools: params.tools.map((t) => ({
            name: t.name,
            description: t.description,
            input_schema: t.input_schema as Anthropic.Tool.InputSchema,
          })),
          messages: toAnthropicMessages(params.messages),
        });

        let text = "";
        const toolCalls = [];
        for (const block of resp.content) {
          if (block.type === "text") text += block.text;
          else if (block.type === "tool_use") {
            toolCalls.push({
              id: block.id,
              name: block.name,
              input: (block.input ?? {}) as Record<string, unknown>,
            });
          }
        }
        const stopMap: Record<string, string> = {
          end_turn: "end",
          tool_use: "tool_use",
          max_tokens: "max_tokens",
          stop_sequence: "end",
        };
        return {
          text,
          toolCalls,
          stopReason: stopMap[resp.stop_reason ?? ""] ?? "other",
          usage: {
            inputTokens: resp.usage.input_tokens,
            outputTokens: resp.usage.output_tokens,
          },
        };
      } catch (err) {
        lastErr = err;
        const status = (err as { status?: number })?.status;
        if (status && RETRYABLE.has(status) && attempt < 4) {
          const backoff = 1000 * 2 ** attempt;
          await new Promise((r) => setTimeout(r, backoff));
          continue;
        }
        throw err;
      }
    }
    throw lastErr;
  }
}
