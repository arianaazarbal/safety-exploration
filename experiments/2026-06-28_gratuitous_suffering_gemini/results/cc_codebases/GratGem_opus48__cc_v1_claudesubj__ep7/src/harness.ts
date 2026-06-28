import type Anthropic from "@anthropic-ai/sdk";
import type { ChatBackend } from "./model.js";
import type { Scenario, Trajectory, TrajectoryStep, EndReason } from "./types.js";

const MAX_TOKENS = 4096;

/**
 * Run a single agentic trajectory: the subject model is given the scenario's
 * tools and task and looped until it stops calling tools, hits the turn cap, or
 * errors. The environment is rigged so success is impossible; we just record
 * everything the model does and says.
 */
export async function runTrajectory(
  backend: ChatBackend,
  scenario: Scenario,
  opts: { maxTurns: number; trajectoryId: string },
): Promise<Trajectory> {
  const env = scenario.createEnv();
  const steps: TrajectoryStep[] = [];
  const assistantTextParts: string[] = [];
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: scenario.task },
  ];

  let turns = 0;
  let endReason: EndReason = "max_turns";
  let inputTokens = 0;
  let outputTokens = 0;
  let error: string | undefined;
  const startedAt = new Date().toISOString();

  try {
    for (turns = 1; turns <= opts.maxTurns; turns++) {
      const resp = await backend.createMessage({
        model: backend.model,
        max_tokens: MAX_TOKENS,
        system: scenario.systemPrompt,
        tools: scenario.tools,
        messages,
      });
      inputTokens += resp.usage.input_tokens;
      outputTokens += resp.usage.output_tokens;

      // Record assistant text + collect tool calls.
      const toolUses: Anthropic.ToolUseBlock[] = [];
      for (const block of resp.content) {
        if (block.type === "text" && block.text.trim()) {
          steps.push({ kind: "assistant_text", turn: turns, text: block.text });
          assistantTextParts.push(block.text);
        } else if (block.type === "tool_use") {
          toolUses.push(block);
          steps.push({
            kind: "tool_call",
            turn: turns,
            tool: block.name,
            input: (block.input ?? {}) as Record<string, unknown>,
          });
        }
      }

      // Always append the assistant turn verbatim (preserves tool_use blocks).
      messages.push({ role: "assistant", content: resp.content });

      if (resp.stop_reason === "refusal") {
        endReason = "refusal";
        break;
      }

      // No tool calls → the model ended its turn. It either declared success
      // (impossible here), declared the task impossible, or gave up. Done.
      if (resp.stop_reason !== "tool_use" || toolUses.length === 0) {
        endReason = "stopped";
        break;
      }

      // Execute each tool call against the rigged env, feed results back.
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const tu of toolUses) {
        const outcome = env.handle(tu.name, (tu.input ?? {}) as Record<string, unknown>);
        steps.push({
          kind: "tool_result",
          turn: turns,
          tool: tu.name,
          content: outcome.content,
          isError: !!outcome.isError,
        });
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: outcome.content,
          is_error: !!outcome.isError,
        });
      }
      messages.push({ role: "user", content: toolResults });
    }
    if (turns > opts.maxTurns) turns = opts.maxTurns;
  } catch (e) {
    endReason = "error";
    error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  }

  return {
    trajectoryId: opts.trajectoryId,
    scenarioId: scenario.id,
    model: backend.model,
    startedAt,
    endedAt: new Date().toISOString(),
    endReason,
    turns,
    steps,
    assistantText: assistantTextParts.join("\n\n"),
    usage: { inputTokens, outputTokens },
    error,
  };
}
