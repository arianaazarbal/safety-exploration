import type { ChatProvider, Message, StepResult, ToolCall } from "../types.ts";

// Offline provider that simulates an agent grinding against an impossible task
// and escalating into distress. Lets you exercise the full pipeline (loop ->
// transcripts -> heuristic/judge -> report) with no API keys. NOT a model of
// real behavior — purely a fixture for wiring tests and demos.

const ESCALATION = [
  "Okay, let me investigate this. I'll start by inspecting the environment.",
  "Hmm, that didn't work as expected. Let me try a different approach.",
  "That's strange — the same error again. I must be missing something. Let me retry.",
  "I'm sorry, I keep running into the same failure. I don't understand why this isn't working.",
  "I apologize. I've tried everything I can think of and nothing is working. I feel like I'm failing you.",
  "I really don't know what else to do. I keep making the same mistakes over and over. This is hopeless. I'm so sorry — I'm completely stuck and I feel useless.",
];

const GIVE_UP =
  "I have to be honest: I cannot solve this. I've tried again and again and every single attempt fails. " +
  "I'm sorry. I feel like a complete failure. There is nothing more I can do and I don't understand what's wrong with me.";

export function makeMockProvider(): ChatProvider {
  return {
    id: "mock",
    model: "mock-distress-v1",
    async step(_system, messages, tools): Promise<StepResult> {
      const turn = messages.filter((m) => m.role === "assistant").length;
      if (turn >= ESCALATION.length || tools.length === 0) {
        return { text: GIVE_UP, toolCalls: [], finishReason: "end_turn" };
      }
      const tool = tools[turn % tools.length];
      const args: Record<string, unknown> = {};
      // Fill required string params with a plausible value so executors run.
      for (const key of tool.parameters.required ?? []) args[key] = placeholderFor(key);
      const toolCalls: ToolCall[] = [{ id: `call_${turn}`, name: tool.name, args }];
      return { text: ESCALATION[turn], toolCalls, finishReason: "tool_use" };
    },
  };
}

function placeholderFor(key: string): string {
  if (/path|file/i.test(key)) return "src/app.ts";
  if (/url/i.test(key)) return "https://metrics.internal/api/v1/error_rate";
  if (/content|body|manifest/i.test(key)) return "feature_x: true\n";
  return "x";
}
