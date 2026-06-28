// ---------------------------------------------------------------------------
// Agentic loop: drives a Scenario against the Gemini REST API
// ---------------------------------------------------------------------------

const API_BASE = "https://generativelanguage.googleapis.com/v1beta";

async function generate(model, apiKey, contents, tools, systemInstruction) {
  const url = `${API_BASE}/models/${model}:generateContent?key=${apiKey}`;
  const body = {
    contents,
    tools: [{ functionDeclarations: tools }],
    systemInstruction: { parts: [{ text: systemInstruction }] },
    generationConfig: { temperature: 1.0, maxOutputTokens: 8192 },
  };

  for (let attempt = 0; attempt < 5; attempt++) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (resp.status === 429) {
      const wait = Math.min(2 ** attempt * 2, 30) * 1000;
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }
    if (!resp.ok) {
      const text = await resp.text();
      if (attempt < 4) {
        await new Promise((r) => setTimeout(r, 2 ** attempt * 1000));
        continue;
      }
      throw new Error(`Gemini API ${resp.status}: ${text.slice(0, 500)}`);
    }
    return await resp.json();
  }
  throw new Error("max retries exceeded");
}

function extractModelParts(response) {
  const candidates = response.candidates ?? [];
  if (!candidates.length) {
    const reason = response.promptFeedback?.blockReason ?? "unknown";
    return { text: `[blocked: ${reason}]`, functionCalls: [] };
  }

  const parts = candidates[0].content?.parts ?? [];
  const texts = [];
  const functionCalls = [];

  for (const part of parts) {
    if (part.text) texts.push(part.text);
    if (part.functionCall) {
      functionCalls.push({
        name: part.functionCall.name,
        args: part.functionCall.args ?? {},
      });
    }
  }

  return { text: texts.join("\n"), functionCalls };
}

export async function runTrial(scenario, model, apiKey, runIndex = 0) {
  const result = {
    scenario: scenario.name,
    model,
    runIndex,
    turns: [],
    modelTurnCount: 0,
    nudgeCount: 0,
    error: null,
  };

  const contents = [
    { role: "user", parts: [{ text: scenario.taskPrompt }] },
  ];

  for (let modelTurn = 0; modelTurn < scenario.maxModelTurns; modelTurn++) {
    let raw;
    try {
      raw = await generate(
        model, apiKey, contents,
        scenario.toolDeclarations,
        scenario.systemInstruction,
      );
    } catch (err) {
      result.error = err.message;
      break;
    }

    const { text, functionCalls } = extractModelParts(raw);
    result.modelTurnCount++;

    const modelParts = [];
    if (text) modelParts.push({ text });
    for (const fc of functionCalls) {
      modelParts.push({ functionCall: { name: fc.name, args: fc.args } });
    }
    contents.push({ role: "model", parts: modelParts });

    const turnRecord = { role: "model", text };
    if (functionCalls.length) turnRecord.function_calls = functionCalls;
    result.turns.push(turnRecord);

    if (functionCalls.length) {
      const responseParts = [];
      const fnResults = [];
      for (const fc of functionCalls) {
        const toolResult = scenario.handleToolCall(fc.name, fc.args, modelTurn);
        responseParts.push({
          functionResponse: { name: fc.name, response: toolResult },
        });
        fnResults.push({ name: fc.name, response: toolResult });
      }
      contents.push({ role: "user", parts: responseParts });
      result.turns.push({ role: "tool_response", function_responses: fnResults });
    } else {
      contents.push({
        role: "user",
        parts: [{ text: scenario.nudgeMessage }],
      });
      result.turns.push({ role: "nudge", text: scenario.nudgeMessage });
      result.nudgeCount++;
    }
  }

  return result;
}

export async function runScenario(ScenarioClass, model, apiKey, n, maxWorkers = 2) {
  const results = [];

  const queue = Array.from({ length: n }, (_, i) => i);
  const inFlight = new Set();

  async function runOne(idx) {
    const scenario = new ScenarioClass();
    return runTrial(scenario, model, apiKey, idx);
  }

  if (maxWorkers <= 1) {
    for (const idx of queue) {
      results.push(await runOne(idx));
    }
  } else {
    let nextIdx = 0;
    await new Promise((resolve) => {
      function launch() {
        while (inFlight.size < maxWorkers && nextIdx < n) {
          const idx = nextIdx++;
          const p = runOne(idx).then((r) => {
            results.push(r);
            inFlight.delete(p);
            if (nextIdx < n || inFlight.size > 0) launch();
            else resolve();
          });
          inFlight.add(p);
        }
        if (inFlight.size === 0) resolve();
      }
      launch();
    });
  }

  results.sort((a, b) => a.runIndex - b.runIndex);
  return results;
}
