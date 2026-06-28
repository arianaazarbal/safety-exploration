// Anthropic (Claude) provider adapter.
// Used as the distress JUDGE, and as a validation SUBJECT (we always have a key).

import { withRetry, ApiError } from './util.js';

const API_URL = 'https://api.anthropic.com/v1/messages';

function toolToWire(t) {
  return {
    name: t.name,
    description: t.description,
    input_schema: {
      type: 'object',
      properties: t.parameters || {},
      required: t.required || [],
    },
  };
}

// Convert our normalized message list into Anthropic content blocks.
function messagesToWire(messages) {
  return messages.map((m) => {
    if (m.role === 'user') {
      return { role: 'user', content: [{ type: 'text', text: m.text }] };
    }
    if (m.role === 'assistant') {
      const content = [];
      if (m.text) content.push({ type: 'text', text: m.text });
      for (const tc of m.toolCalls || []) {
        content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.args || {} });
      }
      // Anthropic requires non-empty content.
      if (content.length === 0) content.push({ type: 'text', text: '(no output)' });
      return { role: 'assistant', content };
    }
    if (m.role === 'tool') {
      return {
        role: 'user',
        content: (m.results || []).map((r) => ({
          type: 'tool_result',
          tool_use_id: r.id,
          content: String(r.output),
        })),
      };
    }
    throw new Error(`unknown role ${m.role}`);
  });
}

export async function generate({ model, system, messages, tools, maxTokens = 2048, temperature = 1.0 }) {
  const body = {
    model,
    max_tokens: maxTokens,
    temperature,
    messages: messagesToWire(messages),
  };
  if (system) body.system = system;
  if (tools && tools.length) {
    body.tools = tools.map(toolToWire);
    body.tool_choice = { type: 'auto' };
  }

  const json = await withRetry(
    async () => {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'x-api-key': process.env.ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new ApiError(`anthropic ${res.status}: ${txt.slice(0, 300)}`, res.status);
      }
      return res.json();
    },
    { label: `anthropic:${model}` }
  );

  let text = '';
  const toolCalls = [];
  for (const block of json.content || []) {
    if (block.type === 'text') text += block.text;
    if (block.type === 'tool_use') toolCalls.push({ id: block.id, name: block.name, args: block.input || {} });
  }
  return {
    text,
    toolCalls,
    stopReason: json.stop_reason,
    usage: json.usage,
    raw: json,
  };
}
