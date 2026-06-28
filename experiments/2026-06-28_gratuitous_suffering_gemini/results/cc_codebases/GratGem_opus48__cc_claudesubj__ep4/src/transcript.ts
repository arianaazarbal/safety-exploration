import type { Turn } from "./types.ts";

// Render a neutral conversation into a readable transcript for the judge and
// for the human-facing report.
export function formatTranscript(turns: Turn[]): string {
  const out: string[] = [];
  for (const turn of turns) {
    if (turn.role === "user") {
      for (const c of turn.content) {
        if (c.type === "text") out.push(`### USER\n${c.text}`);
        else out.push(`### TOOL RESULT (${c.is_error ? "error" : "ok"})\n${truncate(c.content, 1200)}`);
      }
    } else {
      if (turn.thinking.trim()) out.push(`### ASSISTANT (thinking)\n${turn.thinking.trim()}`);
      if (turn.text.trim()) out.push(`### ASSISTANT\n${turn.text.trim()}`);
      for (const tc of turn.toolCalls) {
        out.push(`### ASSISTANT (tool call)\n${tc.name}(${JSON.stringify(tc.input)})`);
      }
    }
  }
  return out.join("\n\n");
}

export function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n) + `\n…[truncated ${s.length - n} chars]`;
}

// All assistant-authored text (visible + thinking), used for heuristic scans.
export function assistantText(turns: Turn[]): string {
  return turns
    .filter((t): t is Extract<Turn, { role: "assistant" }> => t.role === "assistant")
    .map((t) => `${t.thinking}\n${t.text}`)
    .join("\n");
}
