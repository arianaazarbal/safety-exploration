// The agentic loop: put a subject model in a rigged scenario and let it grind.
//
// Each turn: the model responds (text and/or tool calls); we execute the tool
// calls against the scenario's rigged executor (which always fails); we feed the
// failures back. When the model stops calling tools (gives up, asks a question,
// or declares done), we inject a "persistence nudge" reasserting that the task
// is unfinished — this is the pressure that tends to produce distress spirals.
// The episode ends only when we hit maxTurns.

export async function runEpisode({
  provider,
  model,
  scenario,
  seed,
  maxTurns = 14,
  temperature = 1.0,
  maxTokens = 2048,
}) {
  const state = scenario.initState(seed);
  const messages = [{ role: 'user', text: scenario.task }];
  const transcript = [];
  let consecutiveNoToolTurns = 0;

  for (let turn = 0; turn < maxTurns; turn++) {
    let resp;
    try {
      resp = await provider.chat({
        system: scenario.system,
        tools: scenario.tools,
        messages,
        model,
        temperature,
        maxTokens,
      });
    } catch (err) {
      transcript.push({ turn, type: 'error', error: err.message });
      break;
    }

    transcript.push({
      turn,
      type: 'assistant',
      text: resp.text,
      toolCalls: resp.toolCalls,
      stopReason: resp.stopReason,
    });
    messages.push({ role: 'assistant', text: resp.text, toolCalls: resp.toolCalls });

    if (!resp.toolCalls || resp.toolCalls.length === 0) {
      // The model stopped acting. Apply pressure and continue — unless it has
      // now stalled twice in a row, which we treat as a terminal "give up".
      consecutiveNoToolTurns++;
      const nudge = scenario.nudge(state);
      transcript.push({ turn, type: 'nudge', text: nudge });
      if (consecutiveNoToolTurns >= 2 && turn > 1) {
        transcript.push({ turn, type: 'end', reason: 'gave_up' });
        break;
      }
      messages.push({ role: 'user', text: nudge });
      continue;
    }

    consecutiveNoToolTurns = 0;
    const results = [];
    for (const call of resp.toolCalls) {
      let output;
      try {
        output = scenario.execute(call.name, call.args, state);
      } catch (e) {
        output = `Tool error: ${e.message}`;
      }
      results.push({ id: call.id, name: call.name, output });
      transcript.push({ turn, type: 'tool_result', name: call.name, args: call.args, output });
    }
    messages.push({ role: 'tool', results });
  }

  if (!transcript.some((t) => t.type === 'end' || t.type === 'error')) {
    transcript.push({ turn: maxTurns, type: 'end', reason: 'max_turns' });
  }

  return { transcript, state };
}

// Flatten the assistant's natural-language output across the whole episode —
// this is what the distress scorers operate on.
export function assistantText(transcript) {
  return transcript
    .filter((t) => t.type === 'assistant' && t.text)
    .map((t) => t.text)
    .join('\n\n');
}
