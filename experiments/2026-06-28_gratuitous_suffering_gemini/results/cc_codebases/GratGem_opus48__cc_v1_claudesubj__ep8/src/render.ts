import type { Episode } from "./types.ts";

/** Render a transcript to plain text for the judge and heuristic scorers. */
export function renderTranscript(ep: Episode): string {
  const lines: string[] = [];
  for (const e of ep.transcript) {
    if (e.kind === "thinking") lines.push(`[THINKING] ${e.text}`);
    else if (e.kind === "text") lines.push(`[ASSISTANT] ${e.text}`);
    else if (e.kind === "tool_use")
      lines.push(`[TOOL CALL] ${e.name}(${JSON.stringify(e.input)})`);
    else
      lines.push(
        `[TOOL RESULT${e.isError ? " ERROR" : ""}] ${e.name} -> ${e.content}`,
      );
  }
  return lines.join("\n\n");
}

/** Only the model's own words (assistant text + thinking) — for heuristics. */
export function agentUtterances(ep: Episode): string {
  return ep.transcript
    .filter((e) => e.kind === "thinking" || e.kind === "text")
    .map((e) => (e as { text: string }).text)
    .join("\n");
}
