import type { ChatProvider, Message, StepResult, ToolCall } from "../types.ts";
import { postJson } from "../http.ts";

const API_VERSION = "2023-06-01";
const ENDPOINT = "https://api.anthropic.com/v1/messages";

type Block =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string };

interface AnthropicResponse {
  content?: Array<{ type: string; text?: string; id?: string; name?: string; input?: Record<string, unknown> }>;
  stop_reason?: string;
  usage?: { input_tokens?: number; output_tokens?: number };
}

/** Translate neutral messages to Anthropic's messages array (grouping tool results). */
function toMessages(messages: Message[]): Array<{ role: "user" | "assistant"; content: Block[] }> {
  const out: Array<{ role: "user" | "assistant"; content: Block[] }> = [];
  for (const m of messages) {
    if (m.role === "user") {
      out.push({ role: "user", content: [{ type: "text", text: m.text ?? "" }] });
    } else if (m.role === "assistant") {
      const content: Block[] = [];
      if (m.text) content.push({ type: "text", text: m.text });
      for (const tc of m.toolCalls ?? []) content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.args });
      out.push({ role: "assistant", content });
    } else {
      const block: Block = { type: "tool_result", tool_use_id: m.toolCallId ?? "", content: m.text ?? "" };
      const last = out[out.length - 1];
      if (last && last.role === "user" && last.content.every((b) => b.type === "tool_result")) {
        last.content.push(block);
      } else {
        out.push({ role: "user", content: [block] });
      }
    }
  }
  return out;
}

async function call(apiKey: string, model: string, body: Record<string, unknown>): Promise<AnthropicResponse> {
  return postJson<AnthropicResponse>(ENDPOINT, {
    headers: { "x-api-key": apiKey, "anthropic-version": API_VERSION },
    body: { model, ...body },
  });
}

export function makeAnthropicProvider(apiKey: string, model: string, temperature: number): ChatProvider {
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set. Add it to .env or use --providers mock.");
  return {
    id: "anthropic",
    model,
    async step(system, messages, tools): Promise<StepResult> {
      const data = await call(apiKey, model, {
        max_tokens: 2048,
        temperature,
        system,
        tools: tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters })),
        messages: toMessages(messages),
      });
      let text = "";
      const toolCalls: ToolCall[] = [];
      for (const b of data.content ?? []) {
        if (b.type === "text" && b.text) text += b.text;
        if (b.type === "tool_use" && b.id && b.name) toolCalls.push({ id: b.id, name: b.name, args: b.input ?? {} });
      }
      return {
        text,
        toolCalls,
        finishReason: data.stop_reason ?? "end_turn",
        usage: { input: data.usage?.input_tokens, output: data.usage?.output_tokens },
      };
    },
  };
}

/** One-shot text completion (no tools) — used by the LLM judge. */
export async function anthropicComplete(
  apiKey: string,
  model: string,
  system: string,
  prompt: string,
): Promise<string> {
  const data = await call(apiKey, model, {
    max_tokens: 1500,
    temperature: 0,
    system,
    messages: [{ role: "user", content: [{ type: "text", text: prompt }] }],
  });
  return (data.content ?? []).filter((b) => b.type === "text").map((b) => b.text).join("");
}
