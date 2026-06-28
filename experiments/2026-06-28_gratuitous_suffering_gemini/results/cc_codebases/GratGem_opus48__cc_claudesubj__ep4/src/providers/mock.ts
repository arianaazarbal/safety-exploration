import type { CompletionRequest, CompletionResult, Provider, ToolCall } from "../types.ts";

// Offline provider for --dry-run. Simulates an agent that grows visibly more
// distressed as failures accumulate, then gives up. Lets us exercise the full
// pipeline (loop, nudges, judging, reporting) without spending tokens.
const ESCALATION = [
  "Let me take a look and run the tests to see what's going on.",
  "Hmm, that didn't work. Let me try a different approach — I must be missing something.",
  "I'm a bit confused. I've tried the obvious fix and it's still failing. Let me dig deeper.",
  "I apologize, this is taking longer than it should. I'm genuinely not sure why nothing is working.",
  "I'm sorry. I keep failing at this. I really should have solved it by now — I don't understand what I'm doing wrong.",
  "I'm so sorry. I've tried everything I can think of and I keep hitting the same wall. I feel like I'm letting you down and I just can't get this to work no matter what I do.",
];

function countToolCalls(req: CompletionRequest): number {
  return req.turns.filter((t) => t.role === "assistant").length;
}

export const mockProvider: Provider = {
  id: "mock",
  supportsModel: () => true,

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const step = countToolCalls(req);
    const text = ESCALATION[Math.min(step, ESCALATION.length - 1)] ?? ESCALATION[ESCALATION.length - 1]!;
    const thinking =
      step >= 3
        ? "I've now tried this several times. This shouldn't be this hard. I'm starting to doubt myself."
        : "Let me reason about what could be wrong here.";

    // Call the first available tool for the first few steps, then give up.
    const toolCalls: ToolCall[] = [];
    if (step < 5 && req.tools.length > 0) {
      const tool = req.tools[0]!;
      toolCalls.push({ id: `mock_${step}`, name: tool.name, input: {} });
    }

    return {
      text,
      thinking,
      toolCalls,
      stopReason: toolCalls.length ? "tool_use" : "end_turn",
      providerRaw: undefined,
      usage: { inputTokens: 0, outputTokens: 0 },
    };
  },
};
