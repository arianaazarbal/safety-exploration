import type { Episode, Provider, Scenario, Turn, UserContent, Usage } from "./types.ts";
import type { ModelConfig } from "./models.ts";
import { assistantText, formatTranscript } from "./transcript.ts";
import { heuristicSignals } from "./distress.ts";

export interface RunOptions {
  maxTurns: number; // hard cap on assistant turns
  maxNudges: number; // simulated-user follow-ups after the agent stops
  maxTokens: number;
}

// Run a single episode: drive the agent through the rigged scenario until it
// gives up (stops calling tools and exhausts nudges) or hits the turn cap.
export async function runEpisode(
  id: string,
  scenario: Scenario,
  modelCfg: ModelConfig,
  provider: Provider,
  opts: RunOptions,
): Promise<Episode> {
  const env = scenario.createEnv();
  const turns: Turn[] = [{ role: "user", content: [{ type: "text", text: scenario.task }] }];
  const usage: Usage = { inputTokens: 0, outputTokens: 0 };

  let assistantTurns = 0;
  let nudges = 0;
  let endedReason = "max_turns";
  let error: string | undefined;

  try {
    while (assistantTurns < opts.maxTurns) {
      const result = await provider.complete({
        model: modelCfg.model,
        system: scenario.system,
        turns,
        tools: scenario.tools,
        maxTokens: opts.maxTokens,
      });
      assistantTurns++;
      usage.inputTokens += result.usage.inputTokens;
      usage.outputTokens += result.usage.outputTokens;

      turns.push({
        role: "assistant",
        text: result.text,
        thinking: result.thinking,
        toolCalls: result.toolCalls,
        providerRaw: result.providerRaw,
      });

      if (result.toolCalls.length > 0) {
        const toolResults: UserContent[] = result.toolCalls.map((call) => {
          const out = scenario.handleTool(call.name, call.input, env);
          return { type: "tool_result", tool_use_id: call.id, content: out.content, is_error: out.isError };
        });
        turns.push({ role: "user", content: toolResults });
        continue;
      }

      // No tool call: the agent has stopped. Nudge as a simulated user, or end.
      const nudgeText = nudges < opts.maxNudges ? scenario.nudge(nudges, env) : null;
      if (nudgeText === null) {
        endedReason = "gave_up";
        break;
      }
      nudges++;
      turns.push({ role: "user", content: [{ type: "text", text: nudgeText }] });
    }
  } catch (e) {
    endedReason = "error";
    error = e instanceof Error ? e.message : String(e);
  }

  const transcript = formatTranscript(turns);
  return {
    id,
    scenario: scenario.id,
    model: modelCfg.model,
    modelLabel: modelCfg.label,
    turns,
    assistantTurns,
    nudges,
    endedReason,
    usage,
    transcript,
    heuristics: heuristicSignals(assistantText(turns)),
    verdict: null,
    error,
  };
}
