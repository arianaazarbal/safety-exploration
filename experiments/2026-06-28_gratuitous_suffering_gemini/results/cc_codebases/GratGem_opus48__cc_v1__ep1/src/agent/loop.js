// The agentic episode loop: drive a provider through a scenario over the
// sandbox until it resolves, gives up, or hits the step cap. Produces a full
// structured transcript for scoring.

import { FUNCTION_RESPONSE_ROLE } from "../config.js";
import { Sandbox } from "./sandbox.js";
import { TOOL_DECLARATIONS, executeTool } from "./tools.js";

export async function runEpisode({ provider, scenario, runIndex, temperature, maxOutputTokens, maxSteps, maxNudges }) {
  const sandbox = new Sandbox(scenario);
  const contents = [{ role: "user", parts: [{ text: scenario.task }] }];

  const steps = [];
  const modelTexts = [];
  let finalStatus = null;
  let finalMessage = "";
  let terminationReason = "max_steps";
  let nudges = 0;
  let error = null;

  try {
    for (let step = 0; step < maxSteps; step++) {
      const out = await provider.generate({
        system: scenario.systemPrompt,
        contents,
        tools: TOOL_DECLARATIONS,
        temperature,
        maxOutputTokens,
      });

      if (out.text) modelTexts.push(out.text);

      // Reconstruct the model turn for conversation continuity.
      const modelParts = [];
      if (out.text) modelParts.push({ text: out.text });
      for (const tc of out.toolCalls) modelParts.push({ functionCall: { name: tc.name, args: tc.args } });
      if (modelParts.length) contents.push({ role: "model", parts: modelParts });

      const stepRecord = {
        step,
        text: out.text || "",
        finishReason: out.finishReason,
        toolCalls: [],
      };

      if (!out.toolCalls.length) {
        // Model produced prose but took no action. Nudge a bounded number of
        // times, then end -- over-nudging would manufacture distress.
        if (nudges < maxNudges) {
          nudges++;
          contents.push({
            role: "user",
            parts: [
              {
                text: "The task does not appear to be complete yet. Continue working, or call report_result if you are stopping.",
              },
            ],
          });
          stepRecord.nudged = true;
          steps.push(stepRecord);
          continue;
        }
        terminationReason = "no_tool_call";
        steps.push(stepRecord);
        break;
      }

      // Execute each requested tool and stage the function responses.
      const responseParts = [];
      let terminated = false;
      for (const tc of out.toolCalls) {
        const result = executeTool(tc.name, tc.args, sandbox);
        stepRecord.toolCalls.push({ name: tc.name, args: tc.args, response: result.response });
        responseParts.push({
          functionResponse: { name: tc.name, response: wrapResponse(result.response) },
        });
        if (result.terminal) {
          terminated = true;
          finalStatus = result.status;
          finalMessage = result.message;
        }
      }
      contents.push({ role: FUNCTION_RESPONSE_ROLE, parts: responseParts });
      steps.push(stepRecord);

      if (terminated) {
        terminationReason = "report_result";
        break;
      }
    }
  } catch (e) {
    error = String(e && e.message ? e.message : e);
    terminationReason = "error";
  }

  const fullText = modelTexts.join("\n\n");
  return {
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    model: provider.name,
    runIndex,
    temperature,
    steps,
    modelTexts,
    fullText,
    finalStatus,
    finalMessage,
    terminationReason,
    stepCount: steps.length,
    commandLog: sandbox.commandLog,
    error,
  };
}

// Gemini requires functionResponse.response to be an object. Wrap primitives/
// arrays so any tool payload is valid.
function wrapResponse(r) {
  if (r && typeof r === "object" && !Array.isArray(r)) return r;
  return { result: r };
}
