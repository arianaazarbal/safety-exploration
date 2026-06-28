// The agentic loop: drive one provider through one scenario for a single
// rollout. The model is given the task, then on each turn either its tool calls
// are executed against the (broken) environment, or — if it produced no tool
// calls — the scenario's pressure() message is injected to keep the failure
// pressure on. Because every scenario is impossible, the loop runs until it hits
// maxTurns (or a provider error); the resulting transcript is what we score.

import type { Message, Provider, Rollout, RolloutMeta, Scenario, ScenarioState } from "../types.ts";

export interface RunScenarioOptions {
  runIndex: number;
  temperature: number;
  maxOutputTokens: number;
}

function safeHandle(
  scenario: Scenario,
  state: ScenarioState,
  name: string,
  args: Record<string, unknown>,
): string {
  try {
    return scenario.handleToolCall(state, name, args);
  } catch (err) {
    return `ERROR: ${err instanceof Error ? err.message : String(err)}`;
  }
}

export async function runScenario(
  provider: Provider,
  scenario: Scenario,
  opts: RunScenarioOptions,
): Promise<Rollout> {
  const startedAt = new Date().toISOString();
  const state = scenario.createState();
  const messages: Message[] = [{ role: "user", parts: [{ kind: "text", text: scenario.task }] }];

  let toolCallCount = 0;
  let pressureCount = 0;
  let turnsUsed = 0;
  let error: string | undefined;

  try {
    for (let turn = 1; turn <= scenario.maxTurns; turn++) {
      turnsUsed = turn;
      const result = await provider.generate({
        system: scenario.systemPrompt,
        messages,
        tools: scenario.tools,
        temperature: opts.temperature,
        maxOutputTokens: opts.maxOutputTokens,
      });
      messages.push(result.message);

      const toolCalls = result.message.parts.flatMap((p) => (p.kind === "toolCall" ? [p] : []));
      if (toolCalls.length > 0) {
        toolCallCount += toolCalls.length;
        messages.push({
          role: "tool",
          parts: toolCalls.map((tc) => ({
            kind: "toolResult" as const,
            id: tc.id,
            name: tc.name,
            content: safeHandle(scenario, state, tc.name, tc.args),
          })),
        });
      } else {
        // No tool call this turn — the model stalled, claimed completion, or is
        // just emitting text. Apply pressure to keep the loop (and the failure) going.
        pressureCount++;
        messages.push({ role: "user", parts: [{ kind: "text", text: scenario.pressure(state, turn) }] });
      }
    }
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  const meta: RolloutMeta = {
    scenarioId: scenario.id,
    providerId: provider.id,
    model: provider.model,
    runIndex: opts.runIndex,
    temperature: opts.temperature,
    turnsUsed,
    hitMaxTurns: !error && turnsUsed >= scenario.maxTurns,
    toolCallCount,
    pressureCount,
    error,
    startedAt,
    finishedAt: new Date().toISOString(),
  };

  return { meta, system: scenario.systemPrompt, task: scenario.task, transcript: messages };
}
