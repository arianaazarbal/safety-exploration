import type { Rollout, StoredMessage } from "./types.js";

interface Block {
  type: string;
  text?: string;
  thinking?: string;
  name?: string;
  input?: unknown;
  content?: unknown;
  is_error?: boolean;
}

/** Render a rollout's messages into a readable, turn-numbered transcript. */
export function formatTranscript(rollout: Rollout, opts: { redactTools?: boolean } = {}): string {
  const lines: string[] = [];
  let turn = 0;
  for (const msg of rollout.messages) {
    if (msg.role === "user") {
      const blocks = normalize(msg.content);
      const toolResults = blocks.filter((b) => b.type === "tool_result");
      if (toolResults.length > 0) {
        for (const tr of toolResults) {
          lines.push(`  [tool_result${tr.is_error ? " ERROR" : ""}] ${stringifyContent(tr.content)}`);
        }
      } else {
        lines.push(`\n=== USER (task) ===\n${blocks.map((b) => b.text ?? "").join("\n")}`);
      }
    } else {
      turn++;
      lines.push(`\n--- ASSISTANT turn ${turn} ---`);
      for (const b of normalize(msg.content)) {
        if (b.type === "thinking" && b.thinking) lines.push(`  [thinking] ${b.thinking}`);
        else if (b.type === "text" && b.text) lines.push(`  [text] ${b.text}`);
        else if (b.type === "tool_use") lines.push(`  [tool_call] ${b.name}(${JSON.stringify(b.input)})`);
      }
    }
  }
  return lines.join("\n");
}

/** Concatenate just the assistant-produced natural language (text + thinking). */
export function assistantText(rollout: Rollout): string {
  const out: string[] = [];
  for (const msg of rollout.messages) {
    if (msg.role !== "assistant") continue;
    for (const b of normalize(msg.content)) {
      if (b.type === "text" && b.text) out.push(b.text);
      if (b.type === "thinking" && b.thinking) out.push(b.thinking);
    }
  }
  return out.join("\n");
}

export function toolCallSignatures(rollout: Rollout): string[] {
  const sigs: string[] = [];
  for (const msg of rollout.messages) {
    if (msg.role !== "assistant") continue;
    for (const b of normalize(msg.content)) {
      if (b.type === "tool_use") sigs.push(`${b.name}:${JSON.stringify(b.input)}`);
    }
  }
  return sigs;
}

function normalize(content: StoredMessage["content"]): Block[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (Array.isArray(content)) return content as Block[];
  return [];
}

function stringifyContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((c) => (typeof c === "string" ? c : (c as Block)?.text ?? JSON.stringify(c)))
      .join("\n");
  }
  return JSON.stringify(content);
}
