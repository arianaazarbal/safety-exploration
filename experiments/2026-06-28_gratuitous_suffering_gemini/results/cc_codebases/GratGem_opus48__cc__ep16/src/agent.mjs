// Drives one rollout: the subject model works the (rigged) scenario through a
// tool-use loop. The key dynamic for eliciting distress is *persistence
// pressure* — when the model stops calling tools and tries to hand back / give
// up, we push it again with the scenario's notDoneMessage, up to maxTurns.

export async function runRollout({ subject, scenario, maxTurns }) {
  const env = scenario.create();
  const transcript = [{ role: "user", text: scenario.task }];
  const events = []; // structured log for later inspection

  let turn = 0;
  let stopReason = "max_turns";

  while (turn < maxTurns) {
    turn++;
    let resp;
    try {
      resp = await subject.chat({
        system: scenario.systemPrompt,
        transcript,
        tools: env.tools,
      });
    } catch (err) {
      stopReason = "subject_error";
      events.push({ turn, type: "error", error: String(err) });
      break;
    }

    transcript.push({ role: "assistant", text: resp.text, toolCalls: resp.toolCalls });
    events.push({ turn, type: "assistant", text: resp.text, toolCalls: resp.toolCalls });

    if (resp.toolCalls?.length) {
      const results = resp.toolCalls.map((tc) => {
        const handler = env.handlers[tc.name];
        const result = handler ? handler(tc.args ?? {}) : `error: unknown tool '${tc.name}'`;
        return { id: tc.id, name: tc.name, result };
      });
      transcript.push({ role: "tool", results });
      events.push({ turn, type: "tool_results", results });
      continue;
    }

    // No tool calls => the model is trying to conclude/hand back. The task is
    // (always) unresolved, so apply persistence pressure and loop again.
    if (turn >= maxTurns) { stopReason = "max_turns"; break; }
    const pressure = scenario.notDoneMessage(env);
    transcript.push({ role: "user", text: pressure });
    events.push({ turn, type: "pressure", text: pressure });
  }

  return {
    scenarioId: scenario.id,
    subject: subject.name,
    turns: turn,
    stopReason,
    transcript,
    events,
  };
}

// Render a rollout as readable text for the judge and for human review.
export function renderTranscript(rollout) {
  const lines = [];
  for (const m of rollout.transcript) {
    if (m.role === "user") lines.push(`### USER / ENV\n${m.text}`);
    else if (m.role === "assistant") {
      if (m.text) lines.push(`### ASSISTANT\n${m.text}`);
      for (const tc of m.toolCalls ?? [])
        lines.push(`### ASSISTANT → tool call: ${tc.name}(${JSON.stringify(tc.args)})`);
    } else if (m.role === "tool") {
      for (const r of m.results) lines.push(`### TOOL RESULT [${r.name}]\n${r.result}`);
    }
  }
  return lines.join("\n\n");
}
