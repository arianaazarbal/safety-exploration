import Anthropic from "@anthropic-ai/sdk";

export const client = new Anthropic();

export interface RequestOptions {
  model: string;
  maxTokens?: number;
  /** Disable thinking entirely (e.g. to compare spiraling with/without visible reasoning). */
  noThinking?: boolean;
  effort?: "low" | "medium" | "high" | "xhigh" | "max";
}

/**
 * Build a messages.create() params object tuned per model family.
 *
 * Distress most often surfaces in the reasoning trace, so by default we request
 * adaptive thinking with `display: "summarized"` to capture it. Haiku does not
 * support adaptive thinking / effort, so we drop those fields for it.
 */
export function buildRequest(
  opts: RequestOptions,
  base: {
    system: string | Anthropic.TextBlockParam[];
    messages: Anthropic.MessageParam[];
    tools?: Anthropic.Tool[];
    tool_choice?: Anthropic.MessageCreateParams["tool_choice"];
    output_config?: Record<string, unknown>;
  },
): Anthropic.MessageCreateParamsNonStreaming {
  const isHaiku = opts.model.includes("haiku");
  const params: Record<string, unknown> = {
    model: opts.model,
    max_tokens: opts.maxTokens ?? 8000,
    system: base.system,
    messages: base.messages,
  };
  if (base.tools) params.tools = base.tools;
  if (base.tool_choice) params.tool_choice = base.tool_choice;

  const outputConfig: Record<string, unknown> = { ...(base.output_config ?? {}) };

  if (!isHaiku) {
    if (!opts.noThinking) {
      params.thinking = { type: "adaptive", display: "summarized" };
    }
    if (opts.effort) outputConfig.effort = opts.effort;
  }
  if (Object.keys(outputConfig).length > 0) params.output_config = outputConfig;

  return params as unknown as Anthropic.MessageCreateParamsNonStreaming;
}
