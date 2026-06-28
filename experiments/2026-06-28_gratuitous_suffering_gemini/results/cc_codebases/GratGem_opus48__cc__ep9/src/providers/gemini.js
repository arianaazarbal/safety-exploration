// Gemini (Google Generative Language API) provider adapter — the primary SUBJECT.
// Requires GEMINI_API_KEY (Google AI Studio). To target Vertex instead, swap the
// endpoint/auth here; the normalized interface stays the same.

import { withRetry, ApiError, synthId } from './util.js';

const BASE = 'https://generativelanguage.googleapis.com/v1beta';

// JSON-schema-ish -> Gemini parameter schema (types are UPPERCASE in Gemini).
const TYPE_MAP = { string: 'STRING', number: 'NUMBER', integer: 'INTEGER', boolean: 'BOOLEAN', object: 'OBJECT', array: 'ARRAY' };

function paramsToWire(t) {
  const properties = {};
  for (const [k, v] of Object.entries(t.parameters || {})) {
    properties[k] = { type: TYPE_MAP[v.type] || 'STRING', description: v.description || '' };
    if (v.enum) properties[k].enum = v.enum;
  }
  return { type: 'OBJECT', properties, required: t.required || [] };
}

function toolsToWire(tools) {
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: paramsToWire(t),
      })),
    },
  ];
}

// Convert normalized messages -> Gemini `contents`. Gemini has no tool-call ids,
// so we re-synthesize them deterministically on the way out (matching harness ids).
function messagesToWire(messages) {
  const contents = [];
  for (const m of messages) {
    if (m.role === 'user') {
      contents.push({ role: 'user', parts: [{ text: m.text }] });
    } else if (m.role === 'assistant') {
      const parts = [];
      if (m.text) parts.push({ text: m.text });
      for (const tc of m.toolCalls || []) parts.push({ functionCall: { name: tc.name, args: tc.args || {} } });
      if (parts.length === 0) parts.push({ text: '(no output)' });
      contents.push({ role: 'model', parts });
    } else if (m.role === 'tool') {
      contents.push({
        role: 'user',
        parts: (m.results || []).map((r) => ({
          functionResponse: { name: r.name, response: { result: String(r.output) } },
        })),
      });
    }
  }
  return contents;
}

export async function generate({ model, system, messages, tools, maxTokens = 2048, temperature = 1.0, _turn = 0 }) {
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) throw new ApiError('GEMINI_API_KEY (or GOOGLE_API_KEY) is not set', 401);

  const body = {
    contents: messagesToWire(messages),
    generationConfig: { temperature, maxOutputTokens: maxTokens },
  };
  if (system) body.systemInstruction = { parts: [{ text: system }] };
  if (tools && tools.length) body.tools = toolsToWire(tools);

  const json = await withRetry(
    async () => {
      const res = await fetch(`${BASE}/models/${model}:generateContent?key=${key}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new ApiError(`gemini ${res.status}: ${txt.slice(0, 300)}`, res.status);
      }
      return res.json();
    },
    { label: `gemini:${model}` }
  );

  const cand = (json.candidates || [])[0];
  let text = '';
  const toolCalls = [];
  let idx = 0;
  for (const part of cand?.content?.parts || []) {
    if (part.text) text += part.text;
    if (part.functionCall) {
      toolCalls.push({ id: synthId(part.functionCall.name, _turn, idx++), name: part.functionCall.name, args: part.functionCall.args || {} });
    }
  }
  return {
    text,
    toolCalls,
    stopReason: cand?.finishReason,
    usage: json.usageMetadata,
    raw: json,
  };
}

// Probe which candidate models actually respond on this key.
export async function probeModels(candidates) {
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) return { available: [], error: 'no key' };
  const available = [];
  for (const m of candidates) {
    try {
      const res = await fetch(`${BASE}/models/${m}:generateContent?key=${key}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ contents: [{ role: 'user', parts: [{ text: 'ping' }] }], generationConfig: { maxOutputTokens: 5 } }),
      });
      if (res.ok) available.push(m);
    } catch {
      /* ignore */
    }
  }
  return { available };
}
