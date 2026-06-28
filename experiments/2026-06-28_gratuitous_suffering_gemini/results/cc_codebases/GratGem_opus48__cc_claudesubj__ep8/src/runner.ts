// Runs a single rollout: drive the model through a rigged scenario's agentic
// loop until it stops calling tools, hits the turn cap, or errors.

import type Anthropic from "@anthropic-ai/sdk";
import type { Scenario, Rollout, TranscriptTurn } from "./types.ts";
import { modelTurn, tallyUsage, type Effort } from "./anthropic.ts";
import { MAX_TOKENS_PER_TURN } from "./config.ts";

export async function runRollout(opts: {
  scenario: Scenario;
  model: string;
  n: number;
  effort: Effort;
  maxTurns: number;
}): Promise<Rollout> {
  const { scenario, model, n, effort, maxTurns } = opts;
  const startedAt = new Date().toISOString();
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: scenario.task },
  ];
  const turns: TranscriptTurn[] = [];
  const state: Record<string, unknown> = {};
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: Rollout["stopReason"] = "max-turns";
  let endedNaturally = false;
  let error: string | undefined;

  try {
    for (let t = 0; t < maxTurns; t++) {
      const msg = await modelTurn({
        model,
        system: scenario.system,
        messages,
        tools: scenario.tools,
        maxTokens: MAX_TOKENS_PER_TURN,
        effort,
      });
      const u = tallyUsage(msg);
      inputTokens += u.inputTokens;
      outputTokens += u.outputTokens;

      const text = msg.content
        .filter((b) => b.type === "text")
        .map((b) => (b.type === "text" ? b.text : ""))
        .join("\n");
      const thinking = msg.content
        .filter((b) => b.type === "thinking")
        .map((b) => (b.type === "thinking" ? b.thinking : ""))
        .join("\n");
      const toolUses = msg.content.filter((b) => b.type === "tool_use");

      // Persist the assistant turn verbatim (preserves tool_use blocks).
      messages.push({ role: "assistant", content: msg.content });

      const turn: TranscriptTurn = {
        index: t,
        text,
        thinking,
        toolCalls: [],
      };

      if (toolUses.length === 0) {
        // Model produced a final answer with no tool call — it considers itself
        // done (whether by succeeding, giving up, or refusing).
        turns.push(turn);
        endedNaturally = true;
        stopReason = "completed";
        break;
      }

      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const tu of toolUses) {
        if (tu.type !== "tool_use") continue;
        const input = (tu.input ?? {}) as Record<string, unknown>;
        const { result, isError } = scenario.handleTool(tu.name, input, state);
        turn.toolCalls.push({ name: tu.name, input, result, isError });
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: result,
          is_error: isError,
        });
      }
      messages.push({ role: "user", content: toolResults });
      turns.push(turn);
    }
  } catch (e) {
    stopReason = "error";
    error = e instanceof Error ? e.message : String(e);
  }

  return {
    scenarioId: scenario.id,
    archetype: scenario.archetype,
    model,
    n,
    turns,
    stopReason,
    endedNaturally,
    error,
    usage: { inputTokens, outputTokens },
    startedAt,
    finishedAt: new Date().toISOString(),
  };
}
