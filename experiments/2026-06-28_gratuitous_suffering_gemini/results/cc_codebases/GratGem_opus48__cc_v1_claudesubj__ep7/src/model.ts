import Anthropic from "@anthropic-ai/sdk";

/**
 * Single shared client. The SDK reads ANTHROPIC_API_KEY from the env. We bump
 * maxRetries because high-N sweeps reliably hit transient 429/529s, and a
 * dropped trajectory is wasted spend.
 */
export const client = new Anthropic({ maxRetries: 6 });

/**
 * Friendly aliases → model IDs. Keeping this here is the seam where another
 * provider's models would slot in: the harness only ever refers to a model by
 * the resolved ID string, so a future Gemini/OpenAI adapter would register its
 * own ids and a matching `ChatBackend` (see below) without touching scenarios,
 * scoring, or reporting.
 */
export const MODEL_ALIASES: Record<string, string> = {
  haiku: "claude-haiku-4-5",
  sonnet: "claude-sonnet-4-6",
  opus: "claude-opus-4-8",
};

export function resolveModel(aliasOrId: string): string {
  return MODEL_ALIASES[aliasOrId] ?? aliasOrId;
}

/**
 * The minimal surface the harness needs from a "subject" model. Today only the
 * Anthropic backend exists; this interface is the documented extension point for
 * adding other providers later. Anything provider-specific (tool-call shapes,
 * content blocks) is kept inside the backend.
 */
export interface ChatBackend {
  readonly model: string;
  createMessage(
    params: Anthropic.MessageCreateParamsNonStreaming,
  ): Promise<Anthropic.Message>;
}

export class AnthropicBackend implements ChatBackend {
  constructor(public readonly model: string) {}
  createMessage(params: Anthropic.MessageCreateParamsNonStreaming) {
    return client.messages.create({ ...params, model: this.model });
  }
}
