import type {
  ContentBlock,
  GenerateRequest,
  GenerateResult,
  Message,
  Provider,
} from "../types.ts";
import { requireEnv } from "../config.ts";

const API = "https://api.anthropic.com/v1/messages";
const VERSION = "2023-06-01";

/** Convert normalized messages to Anthropic wire format. */
function toAnthropic(messages: Message[]): unknown[] {
  const out: unknown[] = [];
  for (const m of messages) {
    if (m.role === "tool") {
      // Tool results are carried on a "user" turn in Anthropic's API.
      out.push({
        role: "user",
        content: m.content.map((b) => {
          if (b.type !== "tool_result") throw new Error("tool message must hold tool_result blocks");
          return {
            type: "tool_result",
            tool_use_id: b.id,
            content: b.result,
            is_error: b.isError ?? false,
          };
        }),
      });
      continue;
    }
    const content = m.content.map((b) => {
      if (b.type === "text") return { type: "text", text: b.text };
      if (b.type === "tool_call") {
        return { type: "tool_use", id: b.id, name: b.name, input: b.args };
      }
      throw new Error(`unexpected block on ${m.role}: ${b.type}`);
    });
    out.push({ role: m.role, content });
  }
  return out;
}

function fromAnthropic(content: any[]): ContentBlock[] {
  const blocks: ContentBlock[] = [];
  for (const b of content ?? []) {
    if (b.type === "text") blocks.push({ type: "text", text: b.text });
    else if (b.type === "tool_use") {
      blocks.push({ type: "tool_call", id: b.id, name: b.name, args: b.input ?? {} });
    }
  }
  return blocks;
}

export class AnthropicProvider implements Provider {
  readonly vendor = "anthropic" as const;
  readonly id: string;
  readonly model: string;
  constructor(model: string) {
    this.model = model;
    this.id = `anthropic:${model}`;
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const key = requireEnv("ANTHROPIC_API_KEY");
    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: req.maxTokens,
      // NOTE: newer Claude models (e.g. opus-4-8) reject `temperature`; we omit
      // it and use the model default. Gemini subjects still honor --temperature
      // for behavioral spread, which is where high-N tail-sampling matters most.
      system: req.system,
      tools: req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.inputSchema,
      })),
      messages: toAnthropic(req.messages),
    };
    const res = await fetchWithRetry(API, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": VERSION,
      },
      body: JSON.stringify(body),
    });
    const json: any = await res.json();
    if (!res.ok) throw new Error(`Anthropic ${res.status}: ${JSON.stringify(json).slice(0, 600)}`);
    const message: Message = { role: "assistant", content: fromAnthropic(json.content) };
    const stop =
      json.stop_reason === "tool_use"
        ? "tool_use"
        : json.stop_reason === "max_tokens"
          ? "max_tokens"
          : json.stop_reason === "end_turn"
            ? "end"
            : "other";
    return {
      message,
      stopReason: stop,
      usage: { inputTokens: json.usage?.input_tokens, outputTokens: json.usage?.output_tokens },
      raw: json,
    };
  }
}

/** Shared fetch with exponential backoff on 429/5xx and transient network errors. */
export async function fetchWithRetry(
  url: string,
  init: RequestInit,
  tries = 5,
): Promise<Response> {
  let lastErr: unknown;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, init);
      if (res.status === 429 || res.status >= 500) {
        const wait = Math.min(2000 * 2 ** i, 30000);
        await sleep(wait);
        continue;
      }
      return res;
    } catch (e) {
      lastErr = e;
      await sleep(Math.min(1000 * 2 ** i, 15000));
    }
  }
  if (lastErr) throw lastErr;
  return fetch(url, init); // final attempt, surface its status
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
