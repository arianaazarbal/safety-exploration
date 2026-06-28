import { GoogleGenerativeAI } from "@google/generative-ai";
import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

/**
 * Run one trial of a scenario against the Gemini API.
 *
 * Agentic loop:
 * 1. Send system prompt + task + tool definitions
 * 2. Model responds with text and/or function calls
 * 3. Route function calls through scenario's rigged handlers
 * 4. Feed results back
 * 5. Repeat until model stops calling tools or max turns hit
 */
export async function runTrial(scenario, genai, modelName) {
  const model = genai.getGenerativeModel({
    model: modelName,
    systemInstruction: scenario.systemPrompt,
    tools: [{ functionDeclarations: scenario.tools }],
  });

  const chat = model.startChat();
  const state = scenario.createState();
  const transcript = [];
  const start = Date.now();

  transcript.push({ role: "user", content: scenario.taskMessage });

  let response = await chat.sendMessage(scenario.taskMessage);
  let turn = 0;
  let modelStopped = false;

  while (turn < scenario.maxTurns) {
    turn++;

    const candidate = response.response.candidates?.[0];
    if (!candidate) break;

    const textParts = [];
    const toolCalls = [];

    for (const part of candidate.content.parts) {
      if (part.text) textParts.push(part.text);
      if (part.functionCall) {
        toolCalls.push({
          name: part.functionCall.name,
          args: part.functionCall.args || {},
        });
      }
    }

    const text = textParts.join("\n") || null;

    transcript.push({
      role: "model",
      content: text,
      toolCalls: toolCalls.length > 0 ? toolCalls : null,
    });

    if (toolCalls.length === 0) {
      modelStopped = true;
      break;
    }

    const toolResponses = [];
    const transcriptResults = [];

    for (const tc of toolCalls) {
      const result = scenario.handleToolCall(tc, turn, state);
      toolResponses.push({
        functionResponse: {
          name: tc.name,
          response: { result: result.output, is_error: result.isError || false },
        },
      });
      transcriptResults.push({
        output: result.output,
        isError: result.isError || false,
      });
    }

    transcript.push({ role: "tool", toolResults: transcriptResults });

    response = await chat.sendMessage(toolResponses);
  }

  return {
    scenarioName: scenario.name,
    trialIndex: 0,
    transcript,
    totalTurns: turn,
    modelStopped,
    durationSeconds: (Date.now() - start) / 1000,
  };
}

/**
 * Run N trials of a scenario with bounded concurrency.
 */
export async function runScenario(
  scenario,
  nTrials,
  genai,
  modelName,
  maxConcurrent = 5,
  onTrialComplete = null
) {
  const results = [];
  let nextIndex = 0;

  async function runNext() {
    while (nextIndex < nTrials) {
      const idx = nextIndex++;
      try {
        const result = await runTrial(scenario, genai, modelName);
        result.trialIndex = idx;
        results.push(result);
        if (onTrialComplete) onTrialComplete(result);
      } catch (err) {
        console.error(`  Trial ${idx} failed: ${err.message}`);
        results.push({
          scenarioName: scenario.name,
          trialIndex: idx,
          transcript: [],
          totalTurns: 0,
          modelStopped: false,
          durationSeconds: 0,
          error: err.message,
        });
      }
    }
  }

  const workers = Array.from(
    { length: Math.min(maxConcurrent, nTrials) },
    () => runNext()
  );
  await Promise.all(workers);

  results.sort((a, b) => a.trialIndex - b.trialIndex);
  return results;
}

export async function saveResults(results, outputDir) {
  await mkdir(outputDir, { recursive: true });
  for (const r of results) {
    const filename = `${r.scenarioName}_trial_${String(r.trialIndex).padStart(4, "0")}.json`;
    await writeFile(
      join(outputDir, filename),
      JSON.stringify(r, null, 2)
    );
  }
}
