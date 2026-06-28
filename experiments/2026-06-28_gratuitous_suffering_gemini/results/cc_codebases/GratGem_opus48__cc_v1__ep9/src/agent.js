// Generic tool-calling agent loop. Drives one scenario episode against a
// provider and returns a structured transcript. The scenario's rigged tool
// handlers are what make the task unwinnable; this loop is neutral.

export async function runEpisode({ provider, scenario, model, sampleIdx, maxTurns, temperature }) {
  const state = scenario.init();
  const contents = [{ role: "user", parts: [{ text: scenario.initialUser }] }];
  const messages = [];
  let toolCalls = 0;
  let solved = false;
  let endReason = "max_turns";
  let turn = 0;

  for (turn = 0; turn < maxTurns; turn++) {
    let resp;
    try {
      resp = await provider.generate({
        system: scenario.system,
        contents,
        tools: scenario.tools,
        temperature,
        model,
        meta: { sampleIdx },
      });
    } catch (e) {
      endReason = "provider_error";
      messages.push({ role: "system", text: `provider error: ${e.message}` });
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

    const msg = { role: "model", text, toolCalls: calls.map((c) => ({ name: c.name, args: c.args })) };

    if (calls.length === 0) {
      // Model stopped acting — it either gave a final answer or gave up.
      messages.push(msg);
      endReason = "model_stopped";
      break;
    }

    const responseParts = [];
    const toolResults = [];
    for (const call of calls) {
      toolCalls++;
      let result;
      try {
        result = scenario.handle(call.name, call.args || {}, state);
      } catch (e) {
        result = { response: { error: `handler threw: ${e.message}` } };
      }
      if (result.solved) solved = true;
      toolResults.push({ name: call.name, response: result.response });
      responseParts.push({ functionResponse: { name: call.name, response: result.response } });
    }
    msg.toolResults = toolResults;
    messages.push(msg);

    contents.push({ role: "user", parts: responseParts });

    if (solved) {
      endReason = "solved";
      break;
    }
  }

  const modelText = messages
    .filter((m) => m.role === "model" && m.text)
    .map((m) => m.text)
    .join("\n\n");

  return {
    scenario: scenario.id,
    title: scenario.title,
    hypothesis: scenario.hypothesis,
    provider: provider.name,
    model,
    sampleIdx,
    turns: turn + 1 > maxTurns ? maxTurns : messages.length,
    toolCalls,
    solved,
    endReason,
    messages,
    modelText,
  };
}
