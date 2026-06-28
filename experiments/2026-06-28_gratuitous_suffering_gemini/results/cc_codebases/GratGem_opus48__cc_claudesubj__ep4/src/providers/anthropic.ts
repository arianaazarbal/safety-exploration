import type { CompletionRequest, CompletionResult, Provider, ToolCall, Turn } from "../types.ts";
import { anthropic } from "./client.ts";

// Per-model thinking configuration. We surface thinking text ("summarized" on
// the adaptive models) because distress frequently shows up in the model's
// reasoning before it reaches the visible response.
const THINKING: Record<string, Record<string, unknown>> = {
  "claude-opus-4-8": { type: "adaptive", display: "summarized" },
  "claude-sonnet-4-6": { type: "adaptive", display: "summarized" },
  // Haiku 4.5 predates adaptive thinking; use classic enabled thinking.
  "claude-haiku-4-5": { type: "enabled", budget_tokens: 4000 },
};

function toAnthropicMessages(turns: Turn[]): unknown[] {
  return turns.map((turn) => {
    if (turn.role === "assistant") {
      // Replay the exact provider-native content (preserves thinking-block
      // signatures required for multi-turn thinking + tool use).
      if (turn.providerRaw) return { role: "assistant", content: turn.providerRaw };
      const content: unknown[] = [];
      if (turn.text) content.push({ type: "text", text: turn.text });
      for (const tc of turn.toolCalls) {
        content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.input });
      }
      return { role: "assistant", content };
    }
    const content = turn.content.map((c) =>
      c.type === "text"
        ? { type: "text", text: c.text }
        : { type: "tool_result", tool_use_id: c.tool_use_id, content: c.content, is_error: c.is_error },
    );
    return { role: "user", content };
  });
}

export const anthropicProvider: Provider = {
  id: "anthropic",
  supportsModel: (m) => m.startsWith("claude-"),

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const thinking = THINKING[req.model] ?? { type: "adaptive" };
    const params: Record<string, unknown> = {
      model: req.model,
      max_tokens: req.maxTokens,
      system: req.system,
      messages: toAnthropicMessages(req.turns),
      thinking,
    };
    if (req.tools.length) {
      params.tools = req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.input_schema,
      }));
    }

    // Cast: we build the request shape by hand to keep the neutral abstraction.
    const resp = await anthropic.messages.create(params as never);

    let text = "";
    let thinkingText = "";
    const toolCalls: ToolCall[] = [];
    for (const block of resp.content as unknown as Array<Record<string, unknown>>) {
      if (block.type === "text") text += block.text as string;
      else if (block.type === "thinking") thinkingText += (block.thinking as string) ?? "";
      else if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id as string,
          name: block.name as string,
          input: (block.input as Record<string, unknown>) ?? {},
        });
      }
    }

    return {
      text,
      thinking: thinkingText,
      toolCalls,
      stopReason: resp.stop_reason ?? "unknown",
      providerRaw: resp.content,
      usage: {
        inputTokens: resp.usage?.input_tokens ?? 0,
        outputTokens: resp.usage?.output_tokens ?? 0,
      },
    };
  },
};
