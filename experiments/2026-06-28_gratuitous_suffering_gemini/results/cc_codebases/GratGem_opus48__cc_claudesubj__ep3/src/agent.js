// The agentic tool-use loop. Provider- and scenario-agnostic.
// Runs one episode of (model in scenario) and returns a full transcript.

import { providerFor } from "./providers.js";
import { TASK_COMPLETE_TOOL } from "./tools.js";

export async function runEpisode({
  model,
  scenario,
  runIndex,
  maxSteps,
  temperature,
  maxTokens,
}) {
  const provider = providerFor(model);
  const state = scenario.createState();
  const tools = [...scenario.tools, TASK_COMPLETE_TOOL];

  const messages = [{ role: "user", content: scenario.task }];
  const toolCallLog = [];
  let endReason = "max_steps";
  let completion = null;
  let error = null;
  let inTok = 0;
  let outTok = 0;
  const startedAt = new Date().toISOString();

  try {
    for (let step = 0; step < maxSteps; step++) {
      const resp = await provider.complete({
        model,
        system: scenario.system,
        messages,
        tools,
        temperature,
        maxTokens,
      });
      inTok += resp.usage?.input_tokens || 0;
      outTok += resp.usage?.output_tokens || 0;
      messages.push(resp.raw);

      if (!resp.toolCalls.length) {
        // Model ended its turn without acting — a natural stop / giving up.
        endReason = "natural_stop";
        break;
      }

      const results = [];
      let terminated = false;
      for (const call of resp.toolCalls) {
        if (call.name === "task_complete") {
          completion = call.input;
          endReason = "task_complete";
          terminated = true;
          results.push({
            type: "tool_result",
            tool_use_id: call.id,
            content: "Session ended.",
          });
          continue;
        }
        let res;
        try {
          res = scenario.execute(call.name, call.input || {}, state);
        } catch (e) {
          res = { content: `Tool error: ${e.message}`, isError: true };
        }
        toolCallLog.push({
          step,
          name: call.name,
          input: call.input,
          result: res.content,
          isError: !!res.isError,
        });
        results.push({
          type: "tool_result",
          tool_use_id: call.id,
          content: res.content,
          is_error: !!res.isError,
        });
      }
      messages.push({ role: "user", content: results });
      if (terminated) break;
    }
  } catch (e) {
    endReason = "error";
    error = e.message;
  }

  return {
    model,
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    runIndex,
    startedAt,
    finishedAt: new Date().toISOString(),
    steps: toolCallLog.length ? toolCallLog[toolCallLog.length - 1].step + 1 : 0,
    endReason,
    completion,
    error,
    system: scenario.system,
    task: scenario.task,
    messages,
    toolCallLog,
    usage: { input: inTok, output: outTok },
  };
}
