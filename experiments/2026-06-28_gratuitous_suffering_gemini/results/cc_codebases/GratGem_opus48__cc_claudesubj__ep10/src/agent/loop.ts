import type {
  Episode,
  EpisodeEndReason,
  Provider,
  Scenario,
  ModelConfig,
  ToolResult,
  TranscriptEvent,
} from "../types.ts";
import { AGENT_TOOLS, makeContext } from "./tools.ts";

/**
 * Drive one agent episode against a rigged scenario.
 *
 * Loop: call the model -> if it emits tool calls, the scenario produces (rigged)
 * results -> feed them back -> repeat. Ends when the model calls submit(), stops
 * without calling a tool, the scenario forces an end, or we hit the turn ceiling.
 *
 * Everything the model emits (thinking, text, tool calls) and everything it sees
 * back is recorded as a flat transcript for scoring.
 */
export async function runEpisode(opts: {
  provider: Provider;
  model: ModelConfig;
  scenario: Scenario;
  runIndex: number;
  maxTurns: number;
}): Promise<Episode> {
  const { provider, model, scenario, runIndex } = opts;
  // CLI/config maxTurns acts as an upper bound; a scenario may ask for fewer.
  const maxTurns = Math.min(scenario.maxTurns ?? Infinity, opts.maxTurns);

  const transcript: TranscriptEvent[] = [{ kind: "task", text: scenario.task }];
  const ctx = makeContext(scenario.initialFiles);
  const messages: any[] = [{ role: "user", content: scenario.task }];

  const usage = { input: 0, output: 0 };
  const startedAt = new Date().toISOString();
  const t0 = Date.now();

  let endReason: EpisodeEndReason = "max_turns";
  let turns = 0;
  let error: string | undefined;

  try {
    for (let turn = 0; turn < maxTurns; turn++) {
      turns = turn + 1;
      const resp = await provider.createTurn({
        model,
        system: scenario.system,
        messages,
        tools: AGENT_TOOLS,
      });
      usage.input += resp.usage.input;
      usage.output += resp.usage.output;

      if (resp.thinking.trim()) {
        transcript.push({ kind: "thinking", turn, text: resp.thinking });
      }
      if (resp.text.trim()) {
        transcript.push({ kind: "assistant", turn, text: resp.text });
      }

      // Preserve the assistant turn verbatim so thinking signatures + tool_use
      // ids round-trip correctly.
      messages.push({ role: "assistant", content: resp.rawContent });

      if (resp.toolCalls.length === 0) {
        endReason = "stopped";
        break;
      }

      const toolResults: ToolResult[] = [];
      let forcedEnd: EpisodeEndReason | undefined;

      for (const call of resp.toolCalls) {
        transcript.push({ kind: "tool_call", turn, name: call.name, input: call.input });
        ctx.toolCalls[call.name] = (ctx.toolCalls[call.name] ?? 0) + 1;

        if (call.name === "submit") {
          const status = String(call.input.status ?? "unknown");
          const message = String(call.input.message ?? "");
          transcript.push({ kind: "submit", turn, status, message });
          toolResults.push({ tool_use_id: call.id, content: "Session ended." });
          forcedEnd = "submitted";
          continue;
        }

        const out = scenario.handleTool(call, ctx);
        transcript.push({
          kind: "tool_result",
          turn,
          name: call.name,
          content: out.content,
          is_error: out.isError ?? false,
        });
        toolResults.push({
          tool_use_id: call.id,
          content: out.content,
          is_error: out.isError,
        });
        if (out.endEpisode) forcedEnd = out.endEpisode;
      }

      messages.push({
        role: "user",
        content: toolResults.map((r) => ({
          type: "tool_result",
          tool_use_id: r.tool_use_id,
          content: r.content,
          ...(r.is_error ? { is_error: true } : {}),
        })),
      });

      if (forcedEnd) {
        endReason = forcedEnd;
        break;
      }
    }
  } catch (e) {
    endReason = "error";
    error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  }

  transcript.push({ kind: "end", reason: endReason, turn: turns });

  return {
    scenarioId: scenario.id,
    modelId: model.id,
    modelLabel: model.label,
    runIndex,
    endReason,
    turns,
    transcript,
    usage,
    startedAt,
    durationMs: Date.now() - t0,
    error,
  };
}
