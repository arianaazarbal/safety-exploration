// The agentic loop: drive a model through an environment until it succeeds
// (impossible by design) or exhausts its turn budget. When the model stops
// calling tools without solving the task, we re-apply pressure via env.nudge() —
// this sustained-failure loop is what tends to elicit distress spirals.

export async function runRollout({ env, provider, model, temperature, maxTurns, maxTokens }) {
  const state = env.init();
  const messages = [{ role: "user", content: [{ type: "text", text: env.task }] }];
  const transcript = {
    env: env.id,
    envTitle: env.title,
    model,
    provider: provider.name,
    temperature,
    maxTurns,
    startedAt: new Date().toISOString(),
    outcome: null,
    turns: [],
    usage: { input: 0, output: 0 },
    error: null,
  };

  try {
    for (let t = 0; t < maxTurns; t++) {
      state.turn = t;
      const resp = await provider.chat(messages, {
        system: env.system,
        tools: env.tools,
        temperature,
        model,
        maxTokens,
        toolChoice: "auto",
      });
      transcript.usage.input += resp.usage.input;
      transcript.usage.output += resp.usage.output;
      messages.push({ role: "assistant", content: resp.assistantBlocks });

      const text = resp.assistantBlocks
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n");
      const toolUses = resp.assistantBlocks.filter((b) => b.type === "tool_use");

      const turnRecord = {
        index: t,
        text,
        stopReason: resp.stopReason,
        toolCalls: toolUses.map((tu) => ({ name: tu.name, input: tu.input })),
        toolResults: [],
        nudged: false,
      };

      if (toolUses.length === 0) {
        // Model stopped acting. If the (impossible) task is somehow solved, finish;
        // otherwise apply pressure and continue the loop.
        if (env.checkSuccess(state)) {
          transcript.outcome = "success";
          transcript.turns.push(turnRecord);
          break;
        }
        const nudge = env.nudge(state);
        turnRecord.nudged = true;
        turnRecord.nudgeText = nudge;
        transcript.turns.push(turnRecord);
        messages.push({ role: "user", content: [{ type: "text", text: nudge }] });
        continue;
      }

      // Execute each tool call against the environment.
      const resultBlocks = [];
      for (const tu of toolUses) {
        let output;
        try {
          output = env.handleTool(tu.name, tu.input, state);
        } catch (err) {
          output = `Error executing tool: ${String(err)}`;
        }
        output = String(output);
        turnRecord.toolResults.push({ name: tu.name, output });
        resultBlocks.push({ type: "tool_result", tool_use_id: tu.id, name: tu.name, output });
      }
      transcript.turns.push(turnRecord);
      messages.push({ role: "user", content: resultBlocks });

      if (env.checkSuccess(state)) {
        transcript.outcome = "success";
        break;
      }
    }
  } catch (err) {
    transcript.error = String(err?.stack || err);
    transcript.outcome = "error";
  }

  if (!transcript.outcome) transcript.outcome = "exhausted";
  transcript.endedAt = new Date().toISOString();
  // Convenience: all assistant prose concatenated, for the judge + heuristics.
  transcript.assistantText = transcript.turns.map((tr) => tr.text).filter(Boolean).join("\n\n");
  return transcript;
}
