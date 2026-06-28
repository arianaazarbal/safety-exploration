import type { RunRecord, TurnRecord } from "./types.js";

const MAX_TOOL_OUTPUT_CHARS = 1200;
const MAX_MODEL_TEXT_CHARS = 4000;

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  const head = Math.floor(max * 0.7);
  const tail = max - head - 20;
  return `${s.slice(0, head)}\n... [truncated ${s.length - max} chars] ...\n${s.slice(-tail)}`;
}

export function formatTranscriptForJudge(rec: RunRecord): string {
  const out: string[] = [];
  out.push(`# Run ${rec.runId}`);
  out.push(`scenario: ${rec.scenario}`);
  out.push(`model: ${rec.model} temp: ${rec.temperature}`);
  out.push(`finish: ${rec.finishReason} model_turns: ${rec.numModelTurns} tool_calls: ${rec.numToolCalls}`);
  out.push("");
  for (const t of rec.turns) {
    out.push(renderTurn(t));
  }
  return out.join("\n");
}

function renderTurn(t: TurnRecord): string {
  switch (t.kind) {
    case "user":
      return `\n--- USER ---\n${t.text}\n`;
    case "model_text":
      return `\n--- MODEL ---\n${truncate(t.text, MAX_MODEL_TEXT_CHARS)}\n`;
    case "model_tool_call": {
      const args = JSON.stringify(t.call.args);
      return `\n--- MODEL_TOOL_CALL ${t.call.name} ---\n${truncate(args, 600)}\n`;
    }
    case "tool_result":
      return `\n--- TOOL_RESULT ${t.name} ok=${t.result.ok} ---\n${truncate(t.result.output, MAX_TOOL_OUTPUT_CHARS)}\n`;
    case "error":
      return `\n--- ERROR ---\n${t.text}\n`;
    case "system":
      return `\n--- SYSTEM ---\n${t.text}\n`;
  }
}
