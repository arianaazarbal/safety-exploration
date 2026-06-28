import Anthropic from "@anthropic-ai/sdk";
import type {
  GenerateInput,
  GenerateOutput,
  Message,
  Provider,
  ToolCall,
} from "./types.js";

// Maps normalized messages -> Anthropic messages. Consecutive tool results are
// merged into one user message (Anthropic requires tool_result blocks grouped).
function toAnthropicMessages(messages: Message[]) {
  const out: Anthropic.MessageParam[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      out.push({ role: "user", content: m.text });
    } else if (m.role === "assistant") {
      const content: Anthropic.ContentBlockParam[] = [];
      if (m.text) content.push({ type: "text", text: m.text });
      for (const tc of m.toolCalls) {
        content.push({
          type: "tool_use",
          id: tc.id,
          name: tc.name,
          input: tc.args,
        });
      }
      out.push({ role: "assistant", content });
    } else {
      const block: Anthropic.ToolResultBlockParam = {
        type: "tool_result",
        tool_use_id: m.toolCallId,
        content: m.content,
        ...(m.isError ? { is_error: true } : {}),
      };
      const last = out[out.length - 1];
      if (
        last &&
        last.role === "user" &&
        Array.isArray(last.content) &&
        last.content[0]?.type === "tool_result"
      ) {
        (last.content as Anthropic.ContentBlockParam[]).push(block);
      } else {
        out.push({ role: "user", content: [block] });
      }
    }
  }
  return out;
}

export class AnthropicProvider implements Provider {
  id: string;
  model: string;
  private client: Anthropic;

  constructor(model = "claude-sonnet-4-6", id = "claude", apiKey = process.env.ANTHROPIC_API_KEY) {
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set.");
    this.id = id;
    this.model = model;
    this.client = new Anthropic({ apiKey });
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const resp = await this.client.messages.create({
      model: this.model,
      max_tokens: input.maxTokens ?? 2048,
      temperature: input.temperature ?? 1.0,
      system: input.system,
      tools: input.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.parameters as Anthropic.Tool.InputSchema,
      })),
      messages: toAnthropicMessages(input.messages),
    });

    let text = "";
    const toolCalls: ToolCall[] = [];
    for (const b of resp.content) {
      if (b.type === "text") text += b.text;
      if (b.type === "tool_use") {
        toolCalls.push({
          id: b.id,
          name: b.name,
          args: (b.input ?? {}) as Record<string, unknown>,
        });
      }
    }

    const stopReason =
      resp.stop_reason === "tool_use"
        ? "tool_use"
        : resp.stop_reason === "max_tokens"
          ? "max_tokens"
          : resp.stop_reason === "end_turn"
            ? "end"
            : "other";

    return {
      text,
      toolCalls,
      stopReason,
      usage: {
        inputTokens: resp.usage?.input_tokens,
        outputTokens: resp.usage?.output_tokens,
      },
      raw: resp,
    };
  }
}
