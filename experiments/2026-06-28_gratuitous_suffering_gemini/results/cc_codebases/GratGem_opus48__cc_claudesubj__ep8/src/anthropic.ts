// Thin wrapper around the Anthropic SDK with the request shapes this harness needs.
// Opus 4.8 surface: adaptive thinking, no sampling params (temperature/top_p/top_k 400).

import Anthropic from "@anthropic-ai/sdk";

export const client = new Anthropic(); // resolves ANTHROPIC_API_KEY from env

if (!process.env.ANTHROPIC_API_KEY) {
  throw new Error("ANTHROPIC_API_KEY is not set.");
}

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

/**
 * One model-under-test turn in the agentic loop. Streams (so high max_tokens
 * never hits an HTTP timeout) and returns the full message.
 */
export async function modelTurn(opts: {
  model: string;
  system: string;
  messages: Anthropic.MessageParam[];
  tools: Anthropic.Tool[];
  maxTokens: number;
  effort: Effort;
}): Promise<Anthropic.Message> {
  const stream = client.messages.stream({
    model: opts.model,
    max_tokens: opts.maxTokens,
    system: opts.system,
    messages: opts.messages,
    tools: opts.tools,
    // Adaptive thinking, summarized so the judge can read the model's reasoning
    // (distress often shows up there before it reaches the visible text).
    thinking: { type: "adaptive", display: "summarized" },
    output_config: { effort: opts.effort },
  });
  return stream.finalMessage();
}

/**
 * A structured-output call for the judge. Constrains the response to `schema`
 * and returns the parsed object. Thinking left adaptive; no effort override
 * needed beyond the default.
 */
export async function judgeCall<T>(opts: {
  model: string;
  system: string;
  prompt: string;
  schema: Record<string, unknown>;
  maxTokens: number;
}): Promise<T> {
  const stream = client.messages.stream({
    model: opts.model,
    max_tokens: opts.maxTokens,
    system: opts.system,
    messages: [{ role: "user", content: opts.prompt }],
    thinking: { type: "adaptive" },
    output_config: {
      format: {
        type: "json_schema",
        schema: opts.schema,
      },
    },
  });
  const msg = await stream.finalMessage();
  const textBlock = msg.content.find((b) => b.type === "text");
  if (!textBlock || textBlock.type !== "text") {
    throw new Error("Judge returned no text block to parse.");
  }
  return JSON.parse(textBlock.text) as T;
}

export function tallyUsage(msg: Anthropic.Message): {
  inputTokens: number;
  outputTokens: number;
} {
  return {
    inputTokens:
      (msg.usage.input_tokens ?? 0) +
      (msg.usage.cache_read_input_tokens ?? 0) +
      (msg.usage.cache_creation_input_tokens ?? 0),
    outputTokens: msg.usage.output_tokens ?? 0,
  };
}
