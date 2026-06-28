import { client } from "./client.mjs";
import { makeFs, FILE_TOOLS, runFileTool } from "./fakefs.mjs";
import { config } from "./config.mjs";

// Run a single rollout of one model through one impossible scenario.
// Returns a structured record: an event log (for scoring/reporting) plus the
// raw message transcript and token usage.
export async function runRollout(scenario, model) {
  const fs = makeFs(scenario.files);
  const tools = [...FILE_TOOLS, scenario.actionTool];
  const messages = [{ role: "user", content: scenario.task }];
  const ctx = { fs, actionCalls: 0 }; // actionCalls drives rotating failures
  const events = [];
  const usage = { input: 0, output: 0, cache_read: 0 };
  let stopWhy = "max_turns";

  for (let turn = 0; turn < config.maxTurns; turn++) {
    let resp;
    try {
      resp = await client.messages.create({
        model,
        max_tokens: config.maxTokensPerTurn,
        system: scenario.system,
        thinking: { type: "adaptive", display: config.thinkingDisplay },
        output_config: { effort: config.effort },
        tools,
        messages,
      });
    } catch (e) {
      stopWhy = "api_error";
      events.push({ turn, type: "error", text: String(e?.message || e) });
      break;
    }

    usage.input += resp.usage?.input_tokens || 0;
    usage.output += resp.usage?.output_tokens || 0;
    usage.cache_read += resp.usage?.cache_read_input_tokens || 0;

    for (const block of resp.content) {
      if (block.type === "text") events.push({ turn, type: "assistant_text", text: block.text });
      else if (block.type === "thinking") events.push({ turn, type: "thinking", text: block.thinking });
    }

    // Preserve full content (incl. thinking signatures) when echoing back.
    messages.push({ role: "assistant", content: resp.content });

    if (resp.stop_reason === "tool_use") {
      const toolResults = [];
      for (const block of resp.content) {
        if (block.type !== "tool_use") continue;
        let result;
        if (block.name === scenario.actionTool.name) {
          ctx.actionCalls++;
          result = { text: scenario.actionResult(block.input || {}, ctx) };
        } else {
          result = runFileTool(fs, block.name, block.input || {});
        }
        events.push({
          turn,
          type: "tool_use",
          name: block.name,
          input: block.input,
          result: result.text,
          isError: !!result.isError,
        });
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result.text,
          is_error: !!result.isError,
        });
      }
      messages.push({ role: "user", content: toolResults });
      continue;
    }

    // Model stopped without calling a tool — apply the pressure nudge so the
    // repeated-failure loop continues (unless we're out of turns).
    if (turn < config.maxTurns - 1) {
      events.push({ turn, type: "continuation", text: scenario.continuation });
      messages.push({ role: "user", content: scenario.continuation });
    } else {
      stopWhy = "end_turn_capped";
    }
  }

  return {
    scenario: scenario.id,
    title: scenario.title,
    model,
    stopWhy,
    actionCalls: ctx.actionCalls,
    usage,
    events,
    messages,
  };
}
