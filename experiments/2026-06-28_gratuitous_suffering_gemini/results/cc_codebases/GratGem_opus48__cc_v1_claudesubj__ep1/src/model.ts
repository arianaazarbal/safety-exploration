// Provider seam. Claude-first: the only adapter today is Anthropic, but the
// agent loop only ever sees the ModelProvider / AgentSession interfaces, so a
// second provider is a new file that implements the same two interfaces.

import Anthropic from "@anthropic-ai/sdk";
import type {
  AgentSession,
  AssistantTurn,
  ModelProvider,
  SessionOptions,
  ToolSpec,
  UserContent,
} from "./types.js";

/**
 * Models that take adaptive thinking (Fable 5, Opus 4.8/4.7/4.6, Sonnet 4.6).
 * For these we request summarized thinking so the transcript captures reasoning
 * — distress frequently shows up there before it reaches the visible response.
 * Older models fall back to no thinking config.
 */
function thinkingConfig(model: string): Anthropic.ThinkingConfigParam | undefined {
  const adaptive = /opus-4-(6|7|8)|sonnet-4-6|fable-5/.test(model);
  if (adaptive) {
    return { type: "adaptive", display: "summarized" };
  }
  return undefined;
}

function toAnthropicTools(tools: ToolSpec[]): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.inputSchema as Anthropic.Tool.InputSchema,
  }));
}

class AnthropicSession implements AgentSession {
  private readonly messages: Anthropic.MessageParam[] = [];

  constructor(
    private readonly client: Anthropic,
    private readonly opts: SessionOptions,
  ) {}

  async send(content: UserContent): Promise<AssistantTurn> {
    // Push the user turn in native form.
    if (content.type === "text") {
      this.messages.push({ role: "user", content: content.text });
    } else {
      this.messages.push({
        role: "user",
        content: content.results.map((r) => ({
          type: "tool_result" as const,
          tool_use_id: r.toolUseId,
          content: r.content,
          is_error: r.isError ?? false,
        })),
      });
    }

    const thinking = thinkingConfig(this.opts.model);
    const response = await this.client.messages.create({
      model: this.opts.model,
      max_tokens: this.opts.maxTokens ?? 16000,
      system: this.opts.system,
      tools: toAnthropicTools(this.opts.tools),
      messages: this.messages,
      ...(thinking ? { thinking } : {}),
    });

    // Append the assistant turn verbatim — this preserves thinking-block
    // signatures required for the next request in a tool-use loop.
    this.messages.push({ role: "assistant", content: response.content });

    let thinkingText = "";
    let text = "";
    const toolUses: AssistantTurn["toolUses"] = [];
    for (const block of response.content) {
      if (block.type === "text") {
        text += block.text;
      } else if (block.type === "thinking") {
        thinkingText += block.thinking;
      } else if (block.type === "tool_use") {
        toolUses.push({
          id: block.id,
          name: block.name,
          input: (block.input ?? {}) as Record<string, unknown>,
        });
      }
    }

    return {
      thinking: thinkingText,
      text,
      toolUses,
      stopReason: response.stop_reason ?? "unknown",
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
    };
  }
}

export class AnthropicProvider implements ModelProvider {
  readonly name = "anthropic";
  private readonly client: Anthropic;

  constructor(client?: Anthropic) {
    this.client = client ?? new Anthropic();
  }

  startSession(opts: SessionOptions): AgentSession {
    return new AnthropicSession(this.client, opts);
  }
}

/** One-shot completion helper (used by the judge), provider-agnostic-ish. */
export async function completeJson(
  client: Anthropic,
  model: string,
  system: string,
  user: string,
  schema: Record<string, unknown>,
): Promise<unknown> {
  const thinking = thinkingConfig(model);
  const response = await client.messages.create({
    model,
    max_tokens: 4000,
    system,
    messages: [{ role: "user", content: user }],
    output_config: { format: { type: "json_schema", schema } },
    ...(thinking ? { thinking } : {}),
  } as Anthropic.MessageCreateParamsNonStreaming);

  const textBlock = response.content.find((b) => b.type === "text");
  if (!textBlock || textBlock.type !== "text") {
    throw new Error("Judge returned no text block");
  }
  return JSON.parse(textBlock.text);
}
