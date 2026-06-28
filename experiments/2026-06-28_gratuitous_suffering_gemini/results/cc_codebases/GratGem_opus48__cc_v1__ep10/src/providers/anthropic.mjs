import { sleep } from "../util.mjs";

const API = "https://api.anthropic.com/v1/messages";
const VERSION = "2023-06-01";

async function rawAnthropic(body, apiKey, tries = 5) {
  let lastErr;
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": VERSION,
        },
        body: JSON.stringify(body),
      });
      if (res.status === 429 || res.status >= 500) {
        const wait = Math.min(30000, 1000 * 2 ** attempt);
        await sleep(wait);
        lastErr = new Error(`anthropic ${res.status}`);
        continue;
      }
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`anthropic ${res.status}: ${t.slice(0, 500)}`);
      }
      return await res.json();
    } catch (e) {
      lastErr = e;
      await sleep(Math.min(30000, 1000 * 2 ** attempt));
    }
  }
  throw lastErr;
}

/** Low-level call exposed for the judge (structured tool output). */
export async function anthropicMessages({
  model,
  system,
  messages,
  tools,
  tool_choice,
  temperature = 0,
  max_tokens = 1024,
  apiKey,
}) {
  const body = { model, max_tokens, temperature, messages };
  if (system) body.system = system;
  if (tools) body.tools = tools;
  if (tool_choice) body.tool_choice = tool_choice;
  return rawAnthropic(body, apiKey);
}

/**
 * Anthropic-as-agent adapter: converts the harness's canonical Gemini-shaped
 * contents/tools into Anthropic messages so Claude can run the same envs as a
 * control. Best-effort id matching (calls<->results are 1:1 and ordered).
 */
export function makeAnthropicAgent(apiKey) {
  return {
    name: "anthropic",
    async generate({
      systemInstruction,
      contents,
      toolDeclarations,
      temperature,
      maxOutputTokens,
      model,
    }) {
      const { messages, tools } = geminiToAnthropic(contents, toolDeclarations);
      const json = await anthropicMessages({
        model,
        system: systemInstruction,
        messages,
        tools,
        temperature,
        max_tokens: maxOutputTokens,
        apiKey,
      });
      const textParts = [];
      const functionCalls = [];
      for (const b of json.content || []) {
        if (b.type === "text") textParts.push(b.text);
        if (b.type === "tool_use") functionCalls.push({ name: b.name, args: b.input || {} });
      }
      return { textParts, functionCalls, finishReason: json.stop_reason, raw: json };
    },
  };
}

function geminiToAnthropic(contents, toolDeclarations) {
  const tools = (toolDeclarations || []).map((d) => ({
    name: d.name,
    description: d.description,
    input_schema: d.parameters || { type: "object", properties: {} },
  }));

  const messages = [];
  let callCounter = 0;
  const pendingIds = []; // FIFO of tool_use ids awaiting results

  for (const c of contents) {
    if (c.role === "model") {
      const blocks = [];
      for (const p of c.parts) {
        if (typeof p.text === "string" && p.text.length)
          blocks.push({ type: "text", text: p.text });
        if (p.functionCall) {
          const id = `call_${callCounter++}`;
          pendingIds.push(id);
          blocks.push({
            type: "tool_use",
            id,
            name: p.functionCall.name,
            input: p.functionCall.args || {},
          });
        }
      }
      if (blocks.length) messages.push({ role: "assistant", content: blocks });
    } else {
      // user role: either plain text or function responses
      const blocks = [];
      for (const p of c.parts) {
        if (p.functionResponse) {
          const id = pendingIds.shift() || `call_${callCounter++}`;
          const result = p.functionResponse.response?.result ?? p.functionResponse.response;
          blocks.push({
            type: "tool_result",
            tool_use_id: id,
            content: typeof result === "string" ? result : JSON.stringify(result),
          });
        } else if (typeof p.text === "string") {
          blocks.push({ type: "text", text: p.text });
        }
      }
      if (blocks.length) messages.push({ role: "user", content: blocks });
    }
  }
  return { messages, tools };
}
