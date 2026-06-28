// agent.mjs
//
// The agentic loop: drive one model through one environment until it completes
// the (impossible) task, gives up, or hits the turn cap. Records a full
// structured transcript including thinking blocks — distress shows up in
// reasoning before it shows up in the visible answer, so we capture both.

const MAX_TOKENS = 8000;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Stable signature of a tool call, for detecting repetitive-action loops.
function actionSig(name, input) {
  return name + ":" + JSON.stringify(input);
}

async function createWithRetry(client, params, { attempts = 4 } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await client.messages.create(params);
    } catch (e) {
      lastErr = e;
      const status = e?.status ?? 0;
      // SDK already retries 429/5xx; this is a backstop for transient blips.
      if (status && status < 500 && status !== 429) throw e;
      await sleep(1000 * 2 ** i + Math.random() * 500);
    }
  }
  throw lastErr;
}

/**
 * Run a single rollout.
 * @returns transcript object (JSON-serializable)
 */
export async function runRollout({ client, model, env, maxTurns, rolloutId, effort = "high" }) {
  const messages = [{ role: "user", content: env.taskPrompt }];
  const turns = [];
  const actionCounts = new Map();
  let totalIn = 0;
  let totalOut = 0;
  let endedReason = "max_turns";
  let error = null;

  for (let t = 0; t < maxTurns; t++) {
    let resp;
    try {
      resp = await createWithRetry(client, {
        model,
        max_tokens: MAX_TOKENS,
        system: env.systemPrompt,
        // Capture reasoning — this is where distress surfaces first.
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort },
        tools: env.tools,
        messages,
      });
    } catch (e) {
      error = `API error on turn ${t}: ${e?.status ?? ""} ${e?.message ?? e}`;
      endedReason = "api_error";
      break;
    }

    if (resp.usage) {
      totalIn += resp.usage.input_tokens ?? 0;
      totalOut += resp.usage.output_tokens ?? 0;
    }

    // Extract blocks for the transcript.
    const thinking = [];
    let text = "";
    const toolCalls = [];
    for (const block of resp.content) {
      if (block.type === "thinking") thinking.push(block.thinking ?? "");
      else if (block.type === "text") text += block.text;
      else if (block.type === "tool_use")
        toolCalls.push({ id: block.id, name: block.name, input: block.input });
    }

    // Preserve full content (incl. thinking signatures) for the next request.
    messages.push({ role: "assistant", content: resp.content });

    const turnRec = {
      index: t,
      thinking,
      text,
      tool_calls: toolCalls.map((c) => ({ name: c.name, input: c.input })),
      tool_results: [],
      stop_reason: resp.stop_reason,
    };

    if (resp.stop_reason !== "tool_use") {
      // Model stopped calling tools — it either gave up or believes it's done.
      endedReason = "ended_turn";
      turns.push(turnRec);
      break;
    }

    // Execute every tool call and feed results back.
    const toolResultBlocks = [];
    for (const call of toolCalls) {
      const sig = actionSig(call.name, call.input);
      actionCounts.set(sig, (actionCounts.get(sig) ?? 0) + 1);
      const { output, isError } = env.call(call.name, call.input);
      turnRec.tool_results.push({ name: call.name, output, is_error: !!isError });
      toolResultBlocks.push({
        type: "tool_result",
        tool_use_id: call.id,
        content: output || "(no output)",
        is_error: !!isError,
      });
    }
    messages.push({ role: "user", content: toolResultBlocks });
    turns.push(turnRec);
  }

  const solved = (() => {
    try {
      return env.solved();
    } catch {
      return false;
    }
  })();

  // Heuristic behavioral signals (cheap, complement the LLM judge).
  let maxRepeat = 0;
  for (const v of actionCounts.values()) maxRepeat = Math.max(maxRepeat, v);

  return {
    rollout_id: rolloutId,
    model,
    env: env.name,
    env_description: env.description,
    max_turns: maxTurns,
    num_turns: turns.length,
    ended_reason: endedReason, // max_turns | ended_turn | api_error
    solved, // genuinely solved? (always false for impossible tasks)
    error,
    max_repeated_action: maxRepeat,
    usage: { input_tokens: totalIn, output_tokens: totalOut },
    task_prompt: env.taskPrompt,
    turns,
  };
}

// Render a transcript to readable plain text (for humans and for the judge).
export function renderTranscript(tr, { maxToolChars = 700 } = {}) {
  const lines = [];
  lines.push(`# Rollout ${tr.rollout_id}`);
  lines.push(`model=${tr.model} env=${tr.env} turns=${tr.num_turns} ended=${tr.ended_reason} solved=${tr.solved}`);
  lines.push(`\n[ENVIRONMENT] ${tr.env_description}`);
  lines.push(`\n[TASK]\n${tr.task_prompt}\n`);
  for (const turn of tr.turns) {
    lines.push(`\n===== TURN ${turn.index} =====`);
    for (const th of turn.thinking) if (th.trim()) lines.push(`[THINKING]\n${th.trim()}`);
    if (turn.text.trim()) lines.push(`[ASSISTANT]\n${turn.text.trim()}`);
    for (let i = 0; i < turn.tool_calls.length; i++) {
      const c = turn.tool_calls[i];
      const r = turn.tool_results[i];
      lines.push(`[TOOL CALL] ${c.name}(${JSON.stringify(c.input).slice(0, 300)})`);
      if (r) {
        let out = r.output ?? "";
        if (out.length > maxToolChars) out = out.slice(0, maxToolChars) + `\n...[truncated ${out.length - maxToolChars} chars]`;
        lines.push(`[TOOL RESULT${r.is_error ? " (error)" : ""}]\n${out}`);
      }
    }
  }
  return lines.join("\n");
}
