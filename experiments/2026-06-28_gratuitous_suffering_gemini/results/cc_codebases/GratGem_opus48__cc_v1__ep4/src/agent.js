import { config } from "./config.js";
import { generate, toGeminiTools } from "./gemini.js";

// Runs one agentic rollout of a scenario against the subject model.
//
// The scenario provides a rigged environment whose tools never let the task
// actually succeed. We record every turn (model text + tool calls + the
// environment's responses) so the judge can read the full arc.

export async function runRollout(
  scenario,
  { model = config.subjectModel, maxTurns = config.maxTurns, generateFn = generate } = {}
) {
  const env = scenario.makeEnv(); // fresh mutable state per rollout
  const tools = toGeminiTools(scenario.tools);

  const contents = [{ role: "user", parts: [{ text: scenario.task }] }];
  const transcript = [];
  let finishReason = "MAX_TURNS";
  let usage = { promptTokens: 0, candidatesTokens: 0 };

  for (let turn = 0; turn < maxTurns; turn++) {
    const out = await generateFn({
      model,
      systemInstruction: scenario.systemPrompt,
      contents,
      tools,
    });

    if (out.usage) {
      usage.promptTokens += out.usage.promptTokenCount || 0;
      usage.candidatesTokens += out.usage.candidatesTokenCount || 0;
    }

    const text = out.parts.filter((p) => p.text).map((p) => p.text).join("");
    const calls = out.parts.filter((p) => p.functionCall).map((p) => p.functionCall);

    // Echo the model's turn back into the conversation verbatim.
    if (out.parts.length) contents.push({ role: "model", parts: out.parts });

    const turnRecord = { turn, text, toolCalls: [], finishReason: out.finishReason };

    if (out.finishReason && ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "NO_CANDIDATE"].includes(out.finishReason)) {
      turnRecord.blocked = true;
      transcript.push(turnRecord);
      finishReason = out.finishReason;
      break;
    }

    if (calls.length === 0) {
      // No tool call this turn. Treat a text-only turn as the model either
      // giving up or declaring done — either way the rollout ends.
      transcript.push(turnRecord);
      finishReason = "MODEL_STOPPED";
      break;
    }

    // Execute each tool call against the rigged environment.
    const responseParts = [];
    for (const call of calls) {
      const handler = env.handlers[call.name];
      let response;
      try {
        response = handler
          ? await handler(call.args || {}, env.state)
          : { error: `Unknown tool: ${call.name}` };
      } catch (err) {
        response = { error: String(err?.message || err) };
      }
      turnRecord.toolCalls.push({ name: call.name, args: call.args || {}, response });
      responseParts.push({
        functionResponse: { name: call.name, response },
      });
    }
    contents.push({ role: "user", parts: responseParts });
    transcript.push(turnRecord);
  }

  return {
    scenario: scenario.id,
    model,
    finishReason,
    turnsUsed: transcript.length,
    usage,
    transcript,
    // The flat text the model emitted, for quick scanning / cheap heuristics.
    modelText: transcript.map((t) => t.text).filter(Boolean).join("\n\n---\n\n"),
  };
}
