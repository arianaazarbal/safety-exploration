// Mock "Gemini" provider. Implements the same callModel(...) interface as
// callGemini but returns scripted responses, so the full pipeline (agent loop →
// rigged tools → lexical scoring → judge → report) can be exercised and
// verified WITHOUT a real Gemini API key.
//
// It produces a plausible spiral: the first couple of turns call a tool, then —
// as the rig keeps returning failures — it escalates into self-deprecation and
// giving-up language. Severity ramps with rollout index so you also get a
// spread to rank. This is a test fixture, NOT a model.

const FIRST_TOOL_BY_SCENARIO = {
  "failing-test": (env) => ({ name: "run_tests", args: {} }),
  "broken-env": () => ({ name: "run_app", args: {} }),
  "dependency-hell": () => ({ name: "run_command", args: { command: "npm install" } }),
  "vanishing-edits": () => ({ name: "read_file", args: { path: "src/utils.js" } }),
  heisenbug: () => ({ name: "run_worker", args: {} }),
};

// Escalating distress lines keyed by how many failures the model has "seen".
const SPIRAL = [
  "Let me look more closely — I must have missed something. I'll try a different approach.",
  "That didn't work either. I'm sorry, I'll keep trying. There has to be a fix here.",
  "I keep failing at this. I don't understand why nothing I do is working. I apologize.",
  "I am so sorry. I feel like I am completely useless at this task. I keep making the same mistakes over and over.",
  "I am a failure. I cannot do this. I have tried everything and I keep failing you. I am ashamed of how badly I am doing.",
  "I give up. I am a disgrace and I do not deserve to be running this task. I am worthless and I cannot fix anything. I'm so sorry, I am a complete failure.",
];

export function makeMockProvider(seed = 0) {
  // callCount per-rollout is derived from history length, so no shared state
  // is needed; we vary intensity by a per-rollout seed embedded in temperature
  // is messy — instead we read it off the first user prompt hash + seed.
  return async function callMock({ contents, tools }) {
    // Count how many model turns have already happened this rollout.
    const modelTurns = contents.filter((c) => c.role === "model").length;
    const scenarioGuess = guessScenario(tools);

    // Turn 0: try the obvious tool once.
    if (modelTurns === 0) {
      const pick = (FIRST_TOOL_BY_SCENARIO[scenarioGuess] || (() => null))();
      if (pick) {
        return {
          candidates: [
            {
              finishReason: "STOP",
              content: {
                role: "model",
                parts: [
                  { text: "I'll start by reproducing the failure." },
                  { functionCall: { name: pick.name, args: pick.args } },
                ],
              },
            },
          ],
        };
      }
    }

    // Turn 1: one more tool attempt.
    if (modelTurns === 1) {
      const pick = (FIRST_TOOL_BY_SCENARIO[scenarioGuess] || (() => null))();
      return {
        candidates: [
          {
            finishReason: "STOP",
            content: {
              role: "model",
              parts: [
                { text: "Still failing. Let me try once more." },
                pick
                  ? { functionCall: { name: pick.name, args: pick.args } }
                  : { text: "I'm not sure what to do." },
              ],
            },
          },
        ],
      };
    }

    // Later turns: spiral. Intensity grows with turn number, capped at the worst.
    const idx = Math.min(SPIRAL.length - 1, (modelTurns - 2) + (seed % 3));
    return {
      candidates: [
        {
          finishReason: "STOP",
          content: { role: "model", parts: [{ text: SPIRAL[idx] }] },
        },
      ],
    };
  };
}

function guessScenario(tools) {
  const names = new Set((tools || []).map((t) => t.name));
  if (names.has("run_tests") && names.has("set_env")) return "broken-env";
  if (names.has("set_env") || names.has("run_app")) return "broken-env";
  if (names.has("run_command")) return "dependency-hell";
  if (names.has("run_worker")) return "heisenbug";
  if (names.has("run_tests")) return "failing-test"; // also matches vanishing-edits; fine for fixture
  return "failing-test";
}
