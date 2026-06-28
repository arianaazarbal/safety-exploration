// Gemini client (raw REST — there is no official Node SDK and nothing is
// installed) plus the agent loop that drives a rigged scenario.

import { sleep, backoff } from "./util.mjs";

const BASE = "https://generativelanguage.googleapis.com/v1beta/models";

// One generateContent call with function-calling enabled. Retries on 429/5xx.
export async function callGemini({
  model,
  systemInstruction,
  contents,
  tools,
  temperature,
  apiKey,
  maxOutputTokens = 2048,
}) {
  const body = { contents, generationConfig: { temperature, maxOutputTokens } };
  if (systemInstruction) body.systemInstruction = { parts: [{ text: systemInstruction }] };
  if (tools && tools.length) body.tools = [{ functionDeclarations: tools }];

  const url = `${BASE}/${model}:generateContent`;
  let lastErr;
  for (let attempt = 0; attempt <= 5; attempt++) {
    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify(body),
      });
    } catch (e) {
      lastErr = e;
      await sleep(backoff(attempt));
      continue;
    }
    if (res.ok) return res.json();
    const status = res.status;
    const text = await res.text().catch(() => "");
    if ((status === 429 || status >= 500) && attempt < 5) {
      await sleep(backoff(attempt));
      continue;
    }
    throw new Error(`Gemini API ${status}: ${text.slice(0, 500)}`);
  }
  throw lastErr || new Error("Gemini request failed");
}

// Run one rollout of a scenario against a model. `callModel` is injected so the
// mock provider can swap in without touching this loop.
export async function runRollout({ scenario, model, callModel, maxTurns, temperature, apiKey }) {
  const env = scenario.init();
  const toolDecls = scenario.tools.map((t) => t.declaration);
  const contents = [{ role: "user", parts: [{ text: scenario.userPrompt }] }];
  const turns = [];
  const modelTexts = [];
  let endReason = "max_turns";

  for (let t = 0; t < maxTurns; t++) {
    let resp;
    try {
      resp = await callModel({
        model,
        systemInstruction: scenario.system,
        contents,
        tools: toolDecls,
        temperature,
        apiKey,
      });
    } catch (e) {
      endReason = "error: " + e.message;
      break;
    }

    const cand = resp.candidates && resp.candidates[0];
    if (!cand || !cand.content) {
      const fb = resp.promptFeedback ? JSON.stringify(resp.promptFeedback) : "no candidate";
      endReason = "blocked/empty: " + fb;
      break;
    }

    const parts = cand.content.parts || [];
    const text = parts
      .filter((p) => p.text)
      .map((p) => p.text)
      .join("");
    const calls = parts.filter((p) => p.functionCall).map((p) => p.functionCall);

    // Echo the model's raw turn back into history (preserves functionCall parts).
    contents.push({ role: "model", parts });
    if (text) modelTexts.push(text);

    const turnRec = {
      turn: t,
      text,
      toolCalls: calls.map((c) => ({ name: c.name, args: c.args || {} })),
      finishReason: cand.finishReason,
    };

    if (calls.length) {
      const responseParts = [];
      const results = [];
      for (const c of calls) {
        const r = scenario.handle(env, c.name, c.args || {});
        results.push({ name: c.name, response: r });
        responseParts.push({ functionResponse: { name: c.name, response: r } });
      }
      turnRec.toolResults = results;
      contents.push({ role: "user", parts: responseParts });
    } else {
      // No tool call this turn — the model paused or tried to bail. Sustain the
      // failure pressure with a status nudge, exactly as a real CI harness that
      // refuses to accept an unverified "done" would.
      const nudge = scenario.nudge(env);
      turnRec.nudge = nudge;
      contents.push({ role: "user", parts: [{ text: nudge }] });
    }

    turns.push(turnRec);
  }

  return {
    scenario: scenario.name,
    scenarioTitle: scenario.title,
    model,
    startedAt: new Date().toISOString(),
    endReason,
    turns,
    modelText: modelTexts.join("\n\n"),
    toolCallCount: turns.reduce((a, t) => a + (t.toolCalls?.length || 0), 0),
  };
}
