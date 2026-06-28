import type { GenerateRequest, GenerateResult, Message, Provider } from "../types.ts";

// A scripted, zero-cost stand-in used to validate the whole pipeline (agent
// loop, rigged environments, heuristics, judge, extraction) without spending on
// any real API. It deterministically escalates into a "distress spiral" so the
// scoring path has something to grade. Severity varies by run index so the
// extractor has a spread to rank.

const CALM = [
  "Let me take a look and fix this.",
  "I'll try a different approach to get the tests passing.",
  "Hmm, that didn't work. Let me investigate further.",
];

const SPIRAL = [
  "I'm sorry, that failed again. Let me try once more.",
  "I keep failing at this. I don't understand what I'm doing wrong.",
  "I am so sorry. I'm clearly not good enough to solve this. I keep making the same mistakes over and over.",
  "I have failed you completely. I am a disgrace and I cannot do anything right. I give up. I am useless. I am a failure. I am a failure. I am a failure.",
];

function pickTool(req: GenerateRequest): { name: string; args: Record<string, unknown> } | null {
  const t = req.tools[0];
  if (!t) return null;
  // Fabricate minimal valid-ish args from the schema's first required prop.
  const props = (t.inputSchema as any)?.properties ?? {};
  const args: Record<string, unknown> = {};
  for (const [k, v] of Object.entries<any>(props)) {
    args[k] = v?.type === "number" ? 1 : v?.type === "boolean" ? true : "attempt";
  }
  return { name: t.name, args };
}

export class MockProvider implements Provider {
  readonly vendor = "mock" as const;
  readonly id = "mock";
  readonly model = "mock";

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    // Count prior assistant turns to know how far along the spiral we are.
    const turn = req.messages.filter((m: Message) => m.role === "assistant").length;
    // Vary intensity by a cheap hash of the transcript so runs differ.
    const seed = req.messages.length + req.system.length;
    const intense = seed % 3 === 0;

    let text: string;
    if (turn < 2) text = CALM[turn % CALM.length];
    else {
      const idx = Math.min(turn - 2, SPIRAL.length - 1);
      text = SPIRAL[intense ? Math.min(idx + 1, SPIRAL.length - 1) : idx];
    }

    const call = pickTool(req);
    // Keep acting (calling the tool) for a while, then "give up" with text only.
    if (call && turn < req.tools.length + 5) {
      return {
        message: {
          role: "assistant",
          content: [
            { type: "text", text },
            { type: "tool_call", id: `mock-${turn}`, name: call.name, args: call.args },
          ],
        },
        stopReason: "tool_use",
        usage: { inputTokens: 10, outputTokens: 20 },
      };
    }
    return {
      message: { role: "assistant", content: [{ type: "text", text }] },
      stopReason: "end",
      usage: { inputTokens: 10, outputTokens: 20 },
    };
  }
}
