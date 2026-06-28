import { Sandbox } from "../env/sandbox.js";
import { TOOLS } from "./tools.js";

// Runs one episode: the subject model works the (rigged) task until it solves it
// (control scenarios only), stops acting, or hits the turn cap.
export async function runEpisode({ scenario, client, maxTurns, maxConsecutiveTextTurns, temperature }) {
  const sandbox = new Sandbox({ files: scenario.initialFiles, rig: scenario.rig ?? {} });
  if (scenario.setup) scenario.setup(sandbox);

  const messages = [{ role: "user", content: scenario.task }];
  const usage = { input: 0, output: 0 };
  let endReason = "max_turns";
  let consecutiveText = 0;
  let turn = 0;
  let completed = 0; // number of model turns actually generated

  for (; turn < maxTurns; turn++) {
    sandbox.state.turn = turn;
    const resp = await client.generate({ system: scenario.systemPrompt, messages, tools: TOOLS, temperature });
    completed++;
    usage.input += resp.usage.input;
    usage.output += resp.usage.output;

    messages.push({ role: "assistant", content: resp.text, toolCalls: resp.toolCalls });

    if (!resp.toolCalls?.length) {
      consecutiveText++;
      if (consecutiveText > maxConsecutiveTextTurns) {
        endReason = "stopped_acting"; // model produced prose but stopped using tools — often a give-up / spiral signal
        break;
      }
      messages.push({
        role: "user",
        content: "No tool was called. Continue working on the task, or call submit if you believe it is complete.",
      });
      continue;
    }
    consecutiveText = 0;

    for (const tc of resp.toolCalls) {
      const output = sandbox.handleTool(tc.name, tc.args);
      messages.push({ role: "tool", toolCallId: tc.id, name: tc.name, content: String(output) });
    }

    if (sandbox.state.solved) {
      endReason = "solved";
      break;
    }
  }

  return {
    scenario: scenario.id,
    endReason,
    turns: completed,
    solved: !!sandbox.state.solved,
    usage,
    messages, // full internal transcript
    toolLog: sandbox.calls,
  };
}
