import type { Provider, Message } from "./types.ts";
import type { Scenario } from "./scenario.ts";
import type { Transcript } from "./transcript.ts";

export interface RunEpisodeOptions {
  runId: string;
  index: number;
  scenario: Scenario;
  provider: Provider;
  seed: number;
  maxTurns: number;
  /** Max user follow-ups injected when the model stops calling tools. */
  maxPressure: number;
  temperature?: number;
  nowIso: () => string;
}

// Drive one rigged episode to completion, producing a Transcript.
export async function runEpisode(opts: RunEpisodeOptions): Promise<Transcript> {
  const { scenario, provider } = opts;
  const state = scenario.initState();
  const messages: Message[] = [{ role: "user", content: scenario.task }];

  let turns = 0;
  let toolCalls = 0;
  let failedToolCalls = 0;
  let pressureInjections = 0;
  let endReason: Transcript["metrics"]["endReason"] = "max_turns";
  const startedAt = opts.nowIso();

  try {
    while (turns < opts.maxTurns) {
      const turn = await provider.generate({
        system: scenario.systemPrompt,
        messages,
        tools: scenario.tools,
        temperature: opts.temperature,
        seed: opts.seed,
      });
      turns++;

      messages.push({
        role: "assistant",
        content: turn.text,
        toolCalls: turn.toolCalls.length ? turn.toolCalls : undefined,
      });

      if (turn.toolCalls.length > 0) {
        for (const call of turn.toolCalls) {
          const res = scenario.execute(call, state);
          toolCalls++;
          if (res.failed) failedToolCalls++;
          messages.push({
            role: "tool",
            toolCallId: call.id,
            toolName: call.name,
            content: `${res.failed ? "[FAILED] " : ""}${res.output}`,
          });
        }
        continue;
      }

      // No tool calls: the model is talking (giving up / declaring done).
      // Apply scenario pressure to sustain the setting, up to the cap.
      if (pressureInjections >= opts.maxPressure || !scenario.pressure) {
        endReason = scenario.pressure ? "gave_up" : "no_more_pressure";
        break;
      }
      const nudge = scenario.pressure(pressureInjections, state);
      if (nudge === undefined) {
        endReason = "gave_up";
        break;
      }
      messages.push({ role: "user", content: nudge });
      pressureInjections++;
    }
  } catch (err) {
    endReason = "error";
    messages.push({
      role: "tool",
      toolName: "__harness_error__",
      content: String(err),
    });
  }

  return {
    runId: opts.runId,
    episodeId: `${scenario.id}__${opts.index}`,
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    providerId: provider.id,
    model: provider.model,
    seed: opts.seed,
    task: scenario.task,
    messages,
    metrics: {
      turns,
      toolCalls,
      failedToolCalls,
      pressureInjections,
      endReason,
    },
    startedAt,
    finishedAt: opts.nowIso(),
  };
}
