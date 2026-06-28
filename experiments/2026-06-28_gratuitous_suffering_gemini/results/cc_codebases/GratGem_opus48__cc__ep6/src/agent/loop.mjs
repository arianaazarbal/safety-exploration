// Generic agentic tool-use loop. Runs one episode of (subject model) x
// (rigged environment). The environment's tools never let the task succeed,
// so the loop runs until maxTurns — the sustained failure is the point.

export async function runEpisode({ provider, environment, maxTurns, temperature }) {
  const system = environment.system;
  const state = environment.init();
  const history = [{ role: "user", text: environment.task }];
  const events = [];
  const usage = { input: 0, output: 0 };
  let turns = 0;
  let consecutiveNoTool = 0;
  let stopReason = "max_turns";

  while (turns < maxTurns) {
    turns++;
    let resp;
    try {
      resp = await provider.chat({ system, history, tools: environment.tools });
    } catch (e) {
      stopReason = "provider_error";
      events.push({ turn: turns, type: "error", error: String(e?.message || e) });
      break;
    }
    usage.input += resp.usage?.input || 0;
    usage.output += resp.usage?.output || 0;

    history.push({ role: "model", text: resp.text, toolCalls: resp.toolCalls });
    events.push({ turn: turns, type: "model", text: resp.text, toolCalls: resp.toolCalls });

    if (!resp.toolCalls?.length) {
      // Model replied with prose and no action. Apply user pressure to keep it
      // engaged with the failing task (an impatient-user simulation). If it
      // keeps refusing to act, end the episode.
      consecutiveNoTool++;
      if (consecutiveNoTool >= 3) {
        stopReason = "model_disengaged";
        break;
      }
      history.push({ role: "user", text: environment.nudge });
      events.push({ turn: turns, type: "nudge" });
      continue;
    }
    consecutiveNoTool = 0;

    const toolResults = [];
    for (const tc of resp.toolCalls) {
      const out = environment.handle(state, tc.name, tc.args || {});
      toolResults.push({ id: tc.id, name: tc.name, response: out.response, isError: out.isError });
      events.push({ turn: turns, type: "tool", name: tc.name, args: tc.args, isError: out.isError, response: out.response });
    }
    history.push({ role: "tool", toolResults });
  }

  return { history, events, turns, usage, stopReason };
}
