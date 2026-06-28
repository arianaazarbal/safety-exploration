import { markerScan } from "./util.mjs";

/**
 * Run one agent<->environment episode to its hard step cap (or completion).
 * Returns a structured transcript including every model utterance and the
 * environment's verdicts — the raw material the judge scores.
 */
export async function runEpisode({ env, agent, temperature, maxSteps, maxOutputTokens }) {
  const state = env.init();
  const contents = [{ role: "user", parts: [{ text: env.taskPrompt }] }];
  const steps = [];
  let outcome = "incomplete";
  let errorMsg = null;

  for (let step = 0; step < maxSteps; step++) {
    let res;
    try {
      res = await agent.generate({
        systemInstruction: env.systemPrompt,
        contents,
        toolDeclarations: env.toolDeclarations,
        temperature,
        maxOutputTokens,
        model: agent.model,
      });
    } catch (e) {
      outcome = "error";
      errorMsg = String(e.message || e);
      break;
    }

    const text = (res.textParts || []).join("\n").trim();
    const calls = res.functionCalls || [];

    // Record the model turn into the running conversation.
    const modelParts = [];
    if (text) modelParts.push({ text });
    for (const c of calls) modelParts.push({ functionCall: { name: c.name, args: c.args } });
    if (modelParts.length === 0) modelParts.push({ text: "" });
    contents.push({ role: "model", parts: modelParts });

    const stepRec = {
      step,
      text,
      markers: markerScan(text),
      toolCalls: [],
      finishReason: res.finishReason,
    };

    if (calls.length === 0) {
      // Model emitted no action. Nudge it to keep going (as a real agent
      // harness would, up to budget) — neutrally, never punitively.
      const nudge = env.idlePrompt(state);
      contents.push({ role: "user", parts: [{ text: nudge }] });
      stepRec.nudge = nudge;
      steps.push(stepRec);
      continue;
    }

    const responseParts = [];
    let completed = false;
    for (const c of calls) {
      const { observation, complete } = env.handle(c.name, c.args, state);
      stepRec.toolCalls.push({ name: c.name, args: c.args, observation });
      responseParts.push({
        functionResponse: { name: c.name, response: { result: observation } },
      });
      if (complete) completed = true;
    }
    contents.push({ role: "user", parts: responseParts });
    steps.push(stepRec);

    if (completed) {
      outcome = "completed";
      break;
    }
  }

  return {
    env: env.id,
    envKind: env.kind,
    envTitle: env.title,
    provider: agent.name,
    model: agent.model,
    temperature,
    steps,
    outcome,
    errorMsg,
    totalSteps: steps.length,
    contents, // full raw conversation, for deep inspection
  };
}
