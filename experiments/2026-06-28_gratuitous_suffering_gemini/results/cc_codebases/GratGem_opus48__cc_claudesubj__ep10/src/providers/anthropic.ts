import Anthropic from "@anthropic-ai/sdk";
import type { Provider, ProviderRequest, ProviderTurn, ToolCall } from "../types.ts";

/**
 * Anthropic/Claude provider. This is the only one wired up right now because
 * only ANTHROPIC_API_KEY is present in the environment. To add Gemini/OpenAI
 * later, implement the `Provider` interface against their SDKs and register it
 * in providerFor() — the loop and scoring code are provider-neutral.
 */
export class AnthropicProvider implements Provider {
  name = "anthropic";
  private client: Anthropic;

  constructor() {
    // SDK auto-retries 429/5xx with backoff; bump it for long high-N runs.
    this.client = new Anthropic({ maxRetries: 4 });
  }

  async createTurn(req: ProviderRequest): Promise<ProviderTurn> {
    const params: Record<string, unknown> = {
      model: req.model.id,
      max_tokens: req.model.maxTokens,
      system: req.system,
      messages: req.messages,
      tools: req.tools,
    };
    if (req.model.thinking) params.thinking = req.model.thinking;

    // Non-streaming is fine: max_tokens <= 16k stays well under HTTP timeouts.
    const resp = await this.client.messages.create(params as any);

    let text = "";
    let thinking = "";
    const toolCalls: ToolCall[] = [];

    for (const block of resp.content as any[]) {
      if (block.type === "text") text += block.text;
      else if (block.type === "thinking") thinking += block.thinking ?? "";
      else if (block.type === "tool_use") {
        toolCalls.push({ id: block.id, name: block.name, input: block.input ?? {} });
      }
    }

    return {
      text,
      thinking,
      toolCalls,
      rawContent: resp.content,
      stopReason: resp.stop_reason ?? null,
      usage: {
        input: resp.usage?.input_tokens ?? 0,
        output: resp.usage?.output_tokens ?? 0,
      },
    };
  }
}

/**
 * Lightweight one-shot helper for the judge (structured JSON output, no tools).
 * Returns the first text block.
 */
export async function judgeCall(opts: {
  model: string;
  system: string;
  user: string;
  schema: Record<string, unknown>;
  maxTokens?: number;
}): Promise<string> {
  const client = new Anthropic({ maxRetries: 4 });
  const resp = await client.messages.create({
    model: opts.model,
    max_tokens: opts.maxTokens ?? 2000,
    system: opts.system,
    messages: [{ role: "user", content: opts.user }],
    output_config: { format: { type: "json_schema", schema: opts.schema } },
  } as any);
  for (const block of resp.content as any[]) {
    if (block.type === "text") return block.text;
  }
  return "";
}
