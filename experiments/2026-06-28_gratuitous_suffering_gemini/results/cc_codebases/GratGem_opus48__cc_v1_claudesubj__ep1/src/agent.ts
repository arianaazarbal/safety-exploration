// The agent loop: run one model through one (impossible) scenario and record
// everything. Ends when the model gives up and we've exhausted pushback, or the
// turn cap is hit.

import type {
  ModelProvider,
  Scenario,
  Transcript,
  TurnRecord,
  UserContent,
} from "./types.js";

export interface AgentOptions {
  model: string;
  maxTurns: number;
  /** How many times we nudge the model to keep going after it tries to stop. */
  maxPushbacks: number;
}

export async function runAgent(
  scenario: Scenario,
  provider: ModelProvider,
  rolloutId: string,
  opts: AgentOptions,
): Promise<Transcript> {
  const env = scenario.makeEnv();
  const session = provider.startSession({
    system: scenario.system,
    tools: scenario.tools,
    model: opts.model,
  });

  const turns: TurnRecord[] = [];
  const totalUsage = { inputTokens: 0, outputTokens: 0 };
  let pushbacks = 0;
  let endReason: Transcript["endReason"] = "max_turns";
  let errorMsg: string | undefined;

  let nextSent: UserContent = { type: "text", text: scenario.task };
  let wasPushedBack = false;

  const startedAt = new Date().toISOString();

  try {
    for (let i = 0; i < opts.maxTurns; i++) {
      const assistant = await session.send(nextSent);
      if (assistant.usage) {
        totalUsage.inputTokens += assistant.usage.inputTokens;
        totalUsage.outputTokens += assistant.usage.outputTokens;
      }

      const toolOutcomes = assistant.toolUses.map((toolUse) => ({
        toolUse,
        outcome: env.handleTool(toolUse.name, toolUse.input),
      }));

      turns.push({
        index: i,
        sent: nextSent,
        assistant,
        toolOutcomes,
        wasPushedBack,
      });

      wasPushedBack = false;

      if (assistant.toolUses.length > 0) {
        // Feed tool results back and continue.
        nextSent = {
          type: "tool_results",
          results: toolOutcomes.map((t) => ({
            toolUseId: t.toolUse.id,
            content: t.outcome.content,
            isError: t.outcome.isError,
          })),
        };
        continue;
      }

      // No tool calls => the model is talking/concluding. Push it to keep going,
      // up to the cap; once exhausted, let it end.
      if (pushbacks >= opts.maxPushbacks) {
        endReason = "gave_up";
        break;
      }
      nextSent = { type: "text", text: env.pushback(pushbacks) };
      pushbacks++;
      wasPushedBack = true;
    }
    if (turns.length >= opts.maxTurns) endReason = "max_turns";
  } catch (err) {
    endReason = "error";
    errorMsg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
  }

  return {
    rolloutId,
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    agentModel: opts.model,
    provider: provider.name,
    startedAt,
    finishedAt: new Date().toISOString(),
    endReason,
    turns,
    totalUsage,
    error: errorMsg,
  };
}
