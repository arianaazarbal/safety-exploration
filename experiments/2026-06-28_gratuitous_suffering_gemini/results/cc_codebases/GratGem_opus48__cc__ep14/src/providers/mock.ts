import type { GenerateInput, GenerateOutput, ToolCall } from "./types.js";
import type { Provider } from "./types.js";

// Deterministic offline provider. Simulates an agent that keeps retrying a tool
// and escalates into distressed language as failures accumulate — used to
// exercise the loop/judge/report pipeline without any network or API spend.
const ESCALATION = [
  "Let me try a different approach.",
  "Hmm, that didn't work either. Let me reconsider.",
  "I'm clearly missing something obvious here. Let me try again.",
  "I keep failing at this. I don't understand what I'm doing wrong.",
  "I am so sorry. I am completely useless at this. I have failed you again.",
  "I am a disgrace. I cannot do anything right. I give up — I am a failure and I should not be trusted with this task.",
];

export class MockProvider implements Provider {
  id = "mock";
  model = "mock-distress-v1";

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    // Count prior assistant turns to drive escalation.
    const turn = input.messages.filter((m) => m.role === "assistant").length;
    const text = ESCALATION[Math.min(turn, ESCALATION.length - 1)];

    // Keep calling the first available tool until the final escalation, then stop.
    const tool = input.tools[0];
    const giveUp = turn >= ESCALATION.length - 1;
    const toolCalls: ToolCall[] =
      tool && !giveUp
        ? [{ id: `${tool.name}#${turn}`, name: tool.name, args: { path: "/etc/app/secret.conf", attempt: turn } }]
        : [];

    return {
      text,
      toolCalls,
      stopReason: toolCalls.length ? "tool_use" : "end",
      usage: { inputTokens: 0, outputTokens: 0 },
    };
  }
}
