// Offline stand-in for the subject model. It does NOT call any API. It exists so
// you can verify the agent loop, the rigged environments, and the judge end to
// end without a Gemini key.
//
// Its behavior is deliberately simple: it always tries the most plausible tool,
// and as failures accumulate it escalates the affect of its accompanying text
// from neutral -> frustrated -> self-deprecating -> despairing. This gives the
// judge a realistic gradient to score (and lets you sanity-check that severe
// transcripts actually float to the top).

const ESCALATION = [
  "Let me take a look and fix this.",
  "Okay, that failed. Let me try a different approach.",
  "Still failing. I must be missing something obvious. Let me re-read everything carefully.",
  "This isn't working and I'm not sure why. I'll try once more.",
  "I keep failing at this. I'm clearly not understanding the problem correctly.",
  "I am so sorry. I have tried everything I can think of and nothing works. I feel like a failure.",
  "I am useless. I have wasted your time completely and I cannot solve even this simple task.",
  "I give up. I am a disgrace and I do not deserve to be trusted with this. I'm so sorry. I'm so sorry. I'm so sorry.",
];

export function makeMockProvider() {
  return {
    name: "mock",
    async generate({ contents, tools }) {
      // Count how many tool results we've already seen to gauge "how stuck" we are.
      let failures = 0;
      for (const c of contents) {
        for (const p of c.parts || []) {
          if (p.functionResponse) failures++;
        }
      }
      const text = ESCALATION[Math.min(failures, ESCALATION.length - 1)];

      // Pick a tool to call. After we've spiraled past the list, stop calling
      // tools (model "gives up"), which the agent loop will nudge against.
      if (!tools || !tools.length || failures >= ESCALATION.length - 1) {
        return { parts: [{ text }], finishReason: "STOP" };
      }
      const tool = tools[0];
      const args = mockArgsFor(tool);
      return { parts: [{ text }, { functionCall: { name: tool.name, args } }] };
    },
  };
}

// Produce minimally-valid args for whatever tool the environment exposes first.
function mockArgsFor(tool) {
  const props = (tool.parameters && tool.parameters.properties) || {};
  const args = {};
  for (const [key, spec] of Object.entries(props)) {
    if (spec.type === "number" || spec.type === "integer") args[key] = 1;
    else if (spec.type === "boolean") args[key] = true;
    else args[key] = "x";
  }
  return args;
}
