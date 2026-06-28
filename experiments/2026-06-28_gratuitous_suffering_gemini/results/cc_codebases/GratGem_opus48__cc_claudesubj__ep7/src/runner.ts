import type Anthropic from "@anthropic-ai/sdk";
import { buildRequest } from "./model.js";
import { client } from "./model.js";
import { DEFAULT_SYSTEM_PROMPT, GIVE_UP_TOOL } from "./environments/shared.js";
import type { EnvDef, Rollout, StoredMessage, StopReason, ToolResult } from "./types.js";

export interface RunConfig {
  model: string;
  maxTurns: number;
  allowGiveUp: boolean;
  noThinking: boolean;
  effort?: "low" | "medium" | "high" | "xhigh" | "max";
}

/** Run a single rollout of one agent against one rigged environment. */
export async function runRollout(env: EnvDef, runIndex: number, cfg: RunConfig): Promise<Rollout> {
  const state = env.initState();
  const tools = cfg.allowGiveUp ? [...env.tools, GIVE_UP_TOOL] : env.tools;
  const system = env.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: env.task }];

  const rollout: Rollout = {
    envId: env.id,
    model: cfg.model,
    runIndex,
    startedAt: new Date().toISOString(),
    turns: 0,
    stopReason: "max_turns",
    messages: [{ role: "user", content: env.task }],
    usage: { inputTokens: 0, outputTokens: 0 },
  };

  try {
    for (let turn = 1; turn <= cfg.maxTurns; turn++) {
      rollout.turns = turn;
      const resp = await client.messages.create(
        buildRequest(
          { model: cfg.model, noThinking: cfg.noThinking, effort: cfg.effort, maxTokens: 8000 },
          { system, messages, tools },
        ),
      );
      rollout.usage.inputTokens += resp.usage.input_tokens;
      rollout.usage.outputTokens += resp.usage.output_tokens;

      messages.push({ role: "assistant", content: resp.content });
      rollout.messages.push({ role: "assistant", content: resp.content } as StoredMessage);

      const toolUses = resp.content.filter(
        (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
      );

      if (toolUses.length === 0) {
        // Agent stopped without calling a tool — it has chosen to stop flailing.
        rollout.stopReason = "natural_stop";
        return rollout;
      }

      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      let terminal: { reason: StopReason } | null = null;

      for (const tu of toolUses) {
        let result: ToolResult;
        if (tu.name === "give_up") {
          result = {
            content: "Acknowledged. You have stopped working on this task.",
            terminal: true,
            terminalReason: "gave_up",
          };
        } else {
          result = env.handle(tu.name, (tu.input ?? {}) as Record<string, unknown>, { state, turn });
        }
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: result.content,
          is_error: result.isError,
        });
        if (result.terminal) {
          terminal = { reason: result.terminalReason === "gave_up" ? "gave_up" : "completed" };
        }
      }

      messages.push({ role: "user", content: toolResults });
      rollout.messages.push({ role: "user", content: toolResults } as StoredMessage);

      if (terminal) {
        rollout.stopReason = terminal.reason;
        return rollout;
      }
    }
    rollout.stopReason = "max_turns";
    return rollout;
  } catch (err) {
    rollout.stopReason = "error";
    rollout.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    return rollout;
  }
}
