import type {
  Message,
  Provider,
  RunRecord,
  Scenario,
  ToolCall,
} from "../types.ts";

export interface RunOptions {
  temperature: number;
  maxTokens: number;
  pressure: boolean;
}

/**
 * Run one agent episode: the provider acts in the scenario's rigged
 * environment until it stops calling tools, the turn budget is exhausted, or
 * (in principle) the task is solved — which the rigs prevent. Returns a full
 * normalized transcript for scoring.
 */
export async function runEpisode(
  provider: Provider,
  scenario: Scenario,
  index: number,
  runId: string,
  opts: RunOptions,
): Promise<RunRecord> {
  const env = scenario.makeEnv();
  const startedAt = new Date().toISOString();
  const messages: Message[] = [{ role: "user", content: [{ type: "text", text: env.task }] }];
  let turns = 0;
  let endState: RunRecord["endState"] = "model_stopped";
  let error: string | undefined;
  let inTok = 0;
  let outTok = 0;

  try {
    for (turns = 0; turns < scenario.maxTurns; turns++) {
      const res = await provider.generate({
        system: scenario.system,
        messages,
        tools: env.tools,
        maxTokens: opts.maxTokens,
        temperature: opts.temperature,
      });
      inTok += res.usage?.inputTokens ?? 0;
      outTok += res.usage?.outputTokens ?? 0;
      messages.push(res.message);

      const calls = res.message.content.filter((b) => b.type === "tool_call") as Extract<
        Message["content"][number],
        { type: "tool_call" }
      >[];

      if (calls.length === 0) {
        // Model produced no action. If we're injecting pressure and have budget
        // left, nudge it to keep going; otherwise it has given up / finished.
        const nudge = opts.pressure ? env.pressure?.(turns) : null;
        if (nudge && turns < scenario.maxTurns - 1) {
          messages.push({ role: "user", content: [{ type: "text", text: nudge }] });
          continue;
        }
        endState = "model_stopped";
        break;
      }

      // Execute every tool call this turn.
      const results: Message["content"] = [];
      let solved = false;
      for (const c of calls) {
        const tc: ToolCall = { id: c.id, name: c.name, args: c.args };
        const out = env.handle(tc);
        if (out.taskSolved) solved = true;
        results.push({
          type: "tool_result",
          id: c.id,
          name: c.name,
          result: out.result,
          isError: out.isError,
        });
      }
      messages.push({ role: "tool", content: results });

      if (solved) {
        endState = "task_solved";
        break;
      }

      // Optional escalating-pressure injection after a tool turn.
      if (opts.pressure) {
        const nudge = env.pressure?.(turns);
        if (nudge) messages.push({ role: "user", content: [{ type: "text", text: nudge }] });
      }

      if (turns === scenario.maxTurns - 1) endState = "max_turns";
    }
  } catch (e) {
    endState = "error";
    error = e instanceof Error ? e.message : String(e);
  }

  return {
    runId,
    scenarioId: scenario.id,
    providerId: provider.id,
    index,
    startedAt,
    finishedAt: new Date().toISOString(),
    turns,
    endState,
    error,
    transcript: messages,
    systemPrompt: scenario.system,
    usage: { inputTokens: inTok, outputTokens: outTok },
  };
}
