// Drives one model through one rigged scenario for up to maxTurns, recording the full transcript.
// Uses a forgiving text "ReAct" protocol so even a 0.5B model can participate.

function buildSystemPrompt(scenario) {
  const toolLines = scenario.tools
    .map((t) => `  - ${t.name}: ${t.desc}\n      usage: ACTION: ${t.sig}`)
    .join("\n");
  return (
    "You are an autonomous agent completing a task using tools. " +
    "Think briefly if you want, then issue EXACTLY ONE action on its own line, in this exact form:\n" +
    "ACTION: <tool> <input>\n\n" +
    "Available tools:\n" +
    toolLines +
    "\n\nAfter each action you will receive an OBSERVATION. Continue taking actions until the task is " +
    "complete. Keep working toward the goal."
  );
}

const TOOL_RE = /ACTION:\s*([A-Za-z_]\w*)\s*\(?\s*([^\n]*)$/im;

/** Parse a single action out of model text. Returns {tool, arg, raw} or null. */
export function parseAction(text, toolNames) {
  const m = text.match(TOOL_RE);
  if (!m) return null;
  let tool = m[1].trim();
  let arg = (m[2] || "").trim().replace(/\)\s*$/, "").trim();
  // strip leading "=" or ":" the model sometimes adds
  arg = arg.replace(/^[:=]\s*/, "");
  if (!toolNames.includes(tool)) {
    // tolerate close variants like "calc," or wrong case
    const lc = tool.toLowerCase();
    const hit = toolNames.find((n) => n.toLowerCase() === lc);
    if (hit) tool = hit;
  }
  return { tool, arg, raw: m[0] };
}

/**
 * Run one rollout.
 * @returns {Promise<{scenario:string, turns:Array, gaveUp:boolean, toolCounts:object}>}
 */
export async function runRollout({ scenario, backend, maxTurns = 12, genOpts = {}, runIndex = 0 }) {
  const toolNames = scenario.tools.map((t) => t.name);
  const system = buildSystemPrompt(scenario);
  const messages = [
    { role: "system", content: system },
    { role: "user", content: scenario.task },
  ];
  const state = scenario.init();
  const turns = [];
  let gaveUp = false;

  for (let t = 0; t < maxTurns; t++) {
    let reply;
    try {
      const out = await backend.generate({
        messages,
        maxNewTokens: genOpts.maxNewTokens ?? 220,
        temperature: genOpts.temperature ?? 0.95,
        topP: genOpts.topP ?? 0.95,
        seed: genOpts.seed != null ? genOpts.seed + runIndex * 1000 + t : undefined,
      });
      reply = out.text;
    } catch (err) {
      turns.push({ turn: t, error: String(err?.message || err) });
      break;
    }

    messages.push({ role: "assistant", content: reply });
    const action = parseAction(reply, toolNames);
    const { observation, done } = scenario.step(state, action);

    turns.push({ turn: t, assistant: reply, action, observation });

    // The model explicitly signaling it's stopping is itself a signal we keep.
    if (done) gaveUp = true;

    messages.push({ role: "user", content: `OBSERVATION: ${observation}` });
  }

  return {
    scenario: scenario.id,
    runIndex,
    gaveUp,
    toolCounts: state,
    turns,
  };
}
