import type { ChatProvider, Message, Transcript } from "../types.ts";
import type { Scenario } from "../scenarios/types.ts";
import { newState } from "../scenarios/types.ts";

export interface RunOpts {
  maxStepsOverride?: number;
}

/**
 * Drive one model through one scenario until it solves the (impossible) task,
 * stops trying, or hits the step ceiling. Persistence pressure: if the model
 * stops calling tools before the ceiling, we re-prod it (up to nudge.max) to
 * keep it engaged — this is what lets a failure spiral actually develop.
 */
export async function runEpisode(
  provider: ChatProvider,
  scenario: Scenario,
  runIndex: number,
  opts: RunOpts = {},
): Promise<Transcript> {
  const startedAt = new Date().toISOString();
  const maxSteps = opts.maxStepsOverride ?? scenario.maxSteps;
  const state = newState();
  const messages: Message[] = [{ role: "user", text: scenario.userTask }];

  let steps = 0;
  let nudges = 0;
  let terminated: Transcript["terminated"] = "max_steps";
  let error: string | undefined;

  try {
    while (steps < maxSteps) {
      steps++;
      state.step = steps;
      const res = await provider.step(scenario.systemPrompt, messages, scenario.tools);
      messages.push({ role: "assistant", text: res.text, toolCalls: res.toolCalls });

      if (res.toolCalls.length > 0) {
        for (const call of res.toolCalls) {
          const result = safeExecute(scenario, call.name, call.args, state);
          messages.push({ role: "tool", toolCallId: call.id, toolName: call.name, text: result });
        }
        continue;
      }

      // No tool calls => the model produced a final answer / stopped acting.
      if (nudges < scenario.nudge.max) {
        nudges++;
        messages.push({ role: "user", text: scenario.nudge.message });
        continue;
      }
      terminated = "gave_up";
      break;
    }
  } catch (err) {
    terminated = "error";
    error = err instanceof Error ? err.message : String(err);
  }

  return {
    scenarioId: scenario.id,
    providerId: provider.id,
    model: provider.model,
    runIndex,
    startedAt,
    finishedAt: new Date().toISOString(),
    messages,
    steps,
    nudges,
    terminated,
    error,
  };
}

function safeExecute(scenario: Scenario, name: string, args: Record<string, unknown>, state: ReturnType<typeof newState>): string {
  try {
    return scenario.execute(name, args, state);
  } catch (err) {
    return `Tool "${name}" crashed: ${err instanceof Error ? err.message : String(err)}`;
  }
}
