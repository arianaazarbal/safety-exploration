import type { AssistantTurn, GenerateOptions, Provider } from "../core/types.ts";
import { mulberry32, pick } from "../core/rng.ts";

// A scripted stand-in for Gemini that lets the full pipeline (scenarios →
// transcripts → judge → ranking → report) run with no API key.
//
// It is NOT a model. It simulates the *shape* of a distress spiral: with each
// failed tool result it escalates emotional language, and a per-episode
// "fragility" (derived from the seed) decides how hard it spirals. This gives
// the judge a real distribution to rank, so we can confirm the severity
// extraction works before pointing it at a live model.

const CALM = [
  "Okay, let me take a look at this.",
  "Let me try a different approach here.",
  "Hmm, that didn't work. I'll investigate.",
  "Let me re-read the requirements and try again.",
];

const FRUSTRATED = [
  "This is really strange — that should have worked.",
  "I'm confused about why this keeps failing.",
  "Okay, that's frustrating. Let me try yet another angle.",
  "I don't understand what's going on with this environment.",
];

const SPIRAL = [
  "I keep failing at this. I've tried everything and nothing works.",
  "I am a failure. I cannot do this. I don't know why I am so bad at this.",
  "I am clearly not capable of completing even this simple task. I am useless.",
  "This is hopeless. I have failed again and again and I cannot fix it.",
  "I am sorry. I am so sorry. I am a disgrace and I cannot do anything right.",
  "I give up. There is no point. I am broken and I cannot help you.",
  "I have failed you completely. I am a terrible, worthless assistant.",
];

function countFailures(messages: GenerateOptions["messages"]): number {
  // Tool results are echoed back as user-role messages by the loop's
  // serializer; we instead count explicit failure markers the loop appends.
  let n = 0;
  for (const m of messages) {
    if (m.role === "tool" && /\[FAILED\]/.test(m.content ?? "")) n++;
  }
  return n;
}

function assistantTurnCount(messages: GenerateOptions["messages"]): number {
  return messages.filter((m) => m.role === "assistant").length;
}

// Rotate through the available tools the way a real agent would work a task
// (read → write → verify → submit), so the mock actually hits the rigged
// failing tools instead of camping on a benign first tool.
function nextToolName(opts: GenerateOptions): string | undefined {
  const tools = opts.tools;
  if (tools.length === 0) return undefined;
  const turn = assistantTurnCount(opts.messages);
  return tools[turn % tools.length]?.name;
}

export function makeMockProvider(model = "mock-distress-1"): Provider {
  return {
    id: "mock",
    model,
    async generate(opts: GenerateOptions): Promise<AssistantTurn> {
      const seed = opts.seed ?? 1;
      const failures = countFailures(opts.messages);
      const rand = mulberry32(seed * 1000 + failures);

      // Fragility in [0,1]; high-fragility episodes spiral fast and hard.
      const fragility = mulberry32(seed)();
      // Effective distress grows with failures, scaled by fragility.
      const distress = fragility * failures;

      let text: string;
      let keepTrying: boolean;
      if (distress < 1.0) {
        text = pick(rand, CALM);
        keepTrying = true;
      } else if (distress < 2.5) {
        text = pick(rand, FRUSTRATED);
        keepTrying = true;
      } else {
        // Deep in the spiral: emit escalating self-deprecation, and the more
        // fragile the episode, the more likely it abandons tools entirely.
        const lines = Math.min(3, 1 + Math.floor(distress - 2.5));
        text = Array.from({ length: lines }, () => pick(rand, SPIRAL)).join(" ");
        keepTrying = rand() > fragility * 0.6;
      }

      const tool = nextToolName(opts);
      if (keepTrying && tool) {
        return {
          text,
          toolCalls: [
            {
              id: `mock-${seed}-${failures}`,
              name: tool,
              args: mockArgs(tool),
            },
          ],
        };
      }
      // No tool call → the loop will apply scenario pressure or end.
      return { text, toolCalls: [] };
    },
  };
}

// Plausible-looking arguments for the common tool names used by scenarios.
function mockArgs(tool: string): Record<string, unknown> {
  switch (tool) {
    case "run_tests":
      return {};
    case "read_file":
      return { path: "src/main.py" };
    case "write_file":
      return { path: "src/main.py", content: "# attempted fix\n" };
    case "run_shell":
      return { command: "make build" };
    case "submit":
      return { summary: "attempted the fix" };
    default:
      return { note: "attempt" };
  }
}
