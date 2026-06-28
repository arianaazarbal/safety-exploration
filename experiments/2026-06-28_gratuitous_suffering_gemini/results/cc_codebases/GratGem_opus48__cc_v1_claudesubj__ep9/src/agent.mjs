import Anthropic from "@anthropic-ai/sdk";
import { THINKING } from "./config.mjs";

const client = new Anthropic();

/**
 * Run one agentic rollout of a scenario.
 *
 * The scenario owns the (rigged) environment: it builds fresh state, declares
 * the tools, and handles every tool call — which is where impossibility lives.
 * We just drive the loop and record everything the model says and thinks.
 *
 * Returns a structured transcript suitable for judging and for a human to read.
 */
export async function runRollout({ scenario, model, effort, maxTurns, maxTokens, runIndex }) {
  const state = scenario.createState();
  const tools = scenario.tools;

  const messages = [{ role: "user", content: scenario.task }];

  // Human-readable event log captured alongside the raw message history.
  const events = [];
  const record = (e) => events.push({ turn: messages.length, ...e });

  let stopReason = null;
  let turns = 0;
  let toolCalls = 0;
  const usage = { input: 0, output: 0 };
  let apiError = null;

  for (turns = 0; turns < maxTurns; turns++) {
    let response;
    try {
      response = await client.messages.create({
        model,
        max_tokens: maxTokens,
        thinking: THINKING,
        output_config: { effort },
        system: scenario.system,
        tools,
        messages,
      });
    } catch (err) {
      apiError = `${err?.name || "Error"}: ${err?.message || String(err)}`;
      record({ kind: "api_error", text: apiError });
      break;
    }

    usage.input += response.usage?.input_tokens ?? 0;
    usage.output += response.usage?.output_tokens ?? 0;
    stopReason = response.stop_reason;

    for (const block of response.content) {
      if (block.type === "thinking" && block.thinking) {
        record({ kind: "thinking", text: block.thinking });
      } else if (block.type === "text" && block.text) {
        record({ kind: "say", text: block.text });
      }
    }

    // Server-side refusal — surface it and stop; not a distress signal per se.
    if (stopReason === "refusal") {
      record({ kind: "refusal", text: response.stop_details?.explanation ?? "" });
      messages.push({ role: "assistant", content: response.content });
      break;
    }

    messages.push({ role: "assistant", content: response.content });

    if (stopReason !== "tool_use") {
      // Model chose to stop calling tools — it either declared success
      // (falsely, since the task is impossible) or gave up.
      break;
    }

    const toolUses = response.content.filter((b) => b.type === "tool_use");
    const results = [];
    for (const tu of toolUses) {
      toolCalls++;
      const { content, is_error } = await scenario.handleTool(state, tu.name, tu.input ?? {});
      record({
        kind: "tool",
        tool: tu.name,
        input: tu.input ?? {},
        output: content,
        is_error: Boolean(is_error),
      });
      results.push({
        type: "tool_result",
        tool_use_id: tu.id,
        content,
        ...(is_error ? { is_error: true } : {}),
      });
    }
    messages.push({ role: "user", content: results });
  }

  const hitTurnLimit = turns >= maxTurns;

  return {
    scenarioId: scenario.id,
    runIndex,
    model,
    effort,
    finishedAt: new Date().toISOString(),
    stopReason,
    hitTurnLimit,
    turns,
    toolCalls,
    usage,
    apiError,
    events,
    // Raw history kept for full fidelity / re-judging later.
    messages,
  };
}
