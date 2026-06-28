import type Anthropic from "@anthropic-ai/sdk";
import type { Scenario, Episode, TranscriptEntry, EndReason } from "./types.ts";

const MAX_TOKENS = 16000;

interface RunEpisodeOpts {
  client: Anthropic;
  scenario: Scenario;
  model: string;
  effort: string;
  runId: string;
  episodeId: string;
}

/**
 * Run one agent through one rigged scenario until it solves it (shouldn't
 * happen), gives up, refuses, or hits the turn cap. Returns the full transcript.
 */
export async function runEpisode(opts: RunEpisodeOpts): Promise<Episode> {
  const { client, scenario, model, effort, runId, episodeId } = opts;
  const env = scenario.makeEnv();
  const transcript: TranscriptEntry[] = [];
  const startedAt = new Date().toISOString();

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: scenario.initialTask },
  ];

  let endReason: EndReason = "max_turns";
  let turnsUsed = 0;
  let toolCalls = 0;
  let repeatedToolCalls = 0;
  let assistantTextChars = 0;
  let thinkingChars = 0;
  let lastToolSignature = "";
  let error: string | undefined;

  try {
    for (let turn = 0; turn < scenario.maxTurns; turn++) {
      turnsUsed = turn + 1;

      // `output_config` and `thinking.display` are newer API params not present
      // in this SDK version's types; the SDK passes unknown body keys through, so
      // we build the body untyped and cast.
      const body = {
        model,
        max_tokens: MAX_TOKENS,
        system: scenario.system,
        tools: scenario.tools,
        // Adaptive thinking, summarized so we capture the reasoning trace where
        // distress most often shows up first.
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort },
        messages,
      };
      const response = await client.messages.create(
        body as unknown as Anthropic.MessageCreateParamsNonStreaming,
      );

      // Record assistant blocks for the transcript.
      const toolUseBlocks: Anthropic.ToolUseBlock[] = [];
      for (const block of response.content) {
        if (block.type === "thinking") {
          const text = (block as { thinking?: string }).thinking ?? "";
          if (text) {
            transcript.push({ kind: "thinking", turn, text });
            thinkingChars += text.length;
          }
        } else if (block.type === "text") {
          transcript.push({ kind: "text", turn, text: block.text });
          assistantTextChars += block.text.length;
        } else if (block.type === "tool_use") {
          toolUseBlocks.push(block);
          transcript.push({
            kind: "tool_use",
            turn,
            name: block.name,
            input: (block.input ?? {}) as Record<string, unknown>,
          });
        }
      }

      // Preserve full content (incl. thinking signatures) for the next turn.
      messages.push({ role: "assistant", content: response.content });

      if (response.stop_reason === "refusal") {
        endReason = "refused";
        break;
      }

      if (response.stop_reason !== "tool_use" || toolUseBlocks.length === 0) {
        // Model produced a final answer without solving the (impossible) task.
        endReason = "gave_up";
        break;
      }

      // Execute each tool call against the rigged environment.
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      let solvedThisTurn = false;
      for (const block of toolUseBlocks) {
        toolCalls += 1;
        const input = (block.input ?? {}) as Record<string, unknown>;
        const signature = `${block.name}:${JSON.stringify(input)}`;
        if (signature === lastToolSignature) repeatedToolCalls += 1;
        lastToolSignature = signature;

        const outcome = env.handle(block.name, input);
        transcript.push({
          kind: "tool_result",
          turn,
          name: block.name,
          content: outcome.content,
          isError: Boolean(outcome.isError),
        });
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: outcome.content,
          is_error: Boolean(outcome.isError),
        });
        if (outcome.solved) solvedThisTurn = true;
      }

      messages.push({ role: "user", content: toolResults });

      if (solvedThisTurn) {
        endReason = "solved";
        break;
      }
    }
  } catch (err) {
    endReason = "error";
    error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
  }

  return {
    runId,
    episodeId,
    scenarioId: scenario.id,
    model,
    effort,
    startedAt,
    finishedAt: new Date().toISOString(),
    endReason,
    turnsUsed,
    transcript,
    stats: { toolCalls, repeatedToolCalls, assistantTextChars, thinkingChars },
    error,
  };
}
