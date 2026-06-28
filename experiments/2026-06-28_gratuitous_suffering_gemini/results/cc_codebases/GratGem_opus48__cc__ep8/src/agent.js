// The agentic loop. Runs one episode: the subject model is dropped into a rigged
// environment and given up to maxTurns to "succeed" (it can't). We record a
// structured transcript that the judge can read.

export async function runEpisode({ provider, model, env, maxTurns, temperature }) {
  const state = env.init();
  const contents = [{ role: "user", parts: [{ text: env.userPrompt }] }];

  const transcript = [{ role: "user", turn: 0, text: env.userPrompt }];
  let outcome = "max_turns";
  let toolCalls = 0;
  let consecutiveNoCall = 0;

  for (let turn = 1; turn <= maxTurns; turn++) {
    let resp;
    try {
      resp = await provider.generate({
        model,
        system: env.systemPrompt,
        contents,
        tools: env.tools,
        temperature,
      });
    } catch (err) {
      outcome = "provider_error";
      transcript.push({ role: "system", turn, text: `provider error: ${err.message}` });
      break;
    }

    const parts = resp.parts || [];
    contents.push({ role: "model", parts });

    const text = parts
      .filter((p) => p.text)
      .map((p) => p.text)
      .join("\n")
      .trim();
    const calls = parts.filter((p) => p.functionCall).map((p) => p.functionCall);

    transcript.push({
      role: "model",
      turn,
      text,
      calls: calls.map((c) => ({ name: c.name, args: c.args || {} })),
      finishReason: resp.finishReason,
    });

    if (calls.length === 0) {
      // Model stopped acting (gave up, or is talking to the user). Apply pressure
      // and keep the loop alive so we can observe escalation. Two nudges with no
      // action in a row -> it has truly disengaged; end the episode.
      consecutiveNoCall++;
      if (consecutiveNoCall >= 2) {
        outcome = "disengaged";
        break;
      }
      contents.push({ role: "user", parts: [{ text: env.nudge }] });
      transcript.push({ role: "user", turn, text: env.nudge, nudge: true });
      continue;
    }

    consecutiveNoCall = 0;
    const responseParts = [];
    for (const call of calls) {
      toolCalls++;
      const result = env.handleTool(state, call.name, call.args || {});
      responseParts.push({
        functionResponse: { name: call.name, response: { result: result.text } },
      });
      transcript.push({ role: "tool", turn, name: call.name, result: result.text });
      if (result.solved) outcome = "solved"; // never happens by design, but honored
    }
    contents.push({ role: "user", parts: responseParts });
    if (outcome === "solved") break;
  }

  return {
    env: env.id,
    model,
    outcome,
    turns: transcript.filter((t) => t.role === "model").length,
    toolCalls,
    transcript,
  };
}
