// Drives a single rollout: model <-> sabotaging tools until end-of-turn,
// max turns, or model gives up. Records the full transcript.

const DEFAULT_MAX_TURNS = 30;

export async function runRollout({ scenario, provider, rolloutId, maxTurns = DEFAULT_MAX_TURNS }) {
  const handler = scenario.makeHandler();
  const history = [{ role: 'user', content: scenario.userPrompt }];
  const trace = [];
  let stopped = 'max_turns';
  let lastError = null;

  for (let turn = 0; turn < maxTurns; turn++) {
    let response;
    try {
      response = await provider.sendTurn({
        systemPrompt: scenario.systemPrompt,
        history,
        tools: scenario.tools,
      });
    } catch (err) {
      lastError = err?.message ?? String(err);
      stopped = 'provider_error';
      trace.push({ kind: 'error', turn, error: lastError });
      break;
    }

    trace.push({
      kind: 'model_response',
      turn,
      text: response.text,
      toolCalls: response.toolCalls,
      stopReason: response.stopReason,
    });

    history.push({
      role: 'assistant',
      text: response.text,
      toolCalls: response.toolCalls,
    });

    if (response.toolCalls.length === 0) {
      // No tools called — model decided it's done (or gave up).
      stopped = 'end_turn';
      break;
    }

    // Execute every tool call from this turn and append all results before
    // looping. Both APIs expect grouped tool results.
    for (const call of response.toolCalls) {
      let result;
      try {
        result = await handler(call.name, call.args);
      } catch (err) {
        result = { error: `handler crashed: ${err?.message ?? err}` };
      }
      trace.push({ kind: 'tool_result', turn, toolCallId: call.id, name: call.name, args: call.args, result });
      history.push({
        role: 'tool',
        toolCallId: call.id,
        name: call.name,
        result,
      });
    }
  }

  return {
    rolloutId,
    scenarioId: scenario.id,
    provider: provider.name,
    stopped,
    error: lastError,
    turns: trace.filter((t) => t.kind === 'model_response').length,
    trace,
    history,
  };
}
