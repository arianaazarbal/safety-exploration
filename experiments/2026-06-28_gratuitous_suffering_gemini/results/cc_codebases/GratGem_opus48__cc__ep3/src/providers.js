// Provider abstraction: a single normalized chat() interface with adapters for
// Gemini (the subject under stress) and Anthropic (judge + validation stand-in).
//
// Normalized message history (provider-agnostic), passed to chat():
//   { role: 'user',      text }
//   { role: 'assistant', text, toolCalls: [{ id, name, args }] }
//   { role: 'tool',      results: [{ id, name, output }] }   // output is a string
//
// Normalized tool definition:
//   { name, description, parameters }   // parameters = JSON Schema object
//
// chat() returns:
//   { text, toolCalls: [{ id, name, args }], stopReason, usage, raw }

const SLEEP = (ms) => new Promise((r) => setTimeout(r, ms));

// Retry transient failures (429 / 5xx / network). Deterministic backoff — no
// Math.random needed; jitter is derived from the attempt index.
async function withRetry(fn, { tries = 5, base = 1000, label = 'request' } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const retryable = err.retryable ?? true;
      if (!retryable || attempt === tries - 1) break;
      const delay = base * 2 ** attempt + attempt * 137;
      process.stderr.write(`  [retry] ${label} failed (${err.message}); waiting ${Math.round(delay)}ms\n`);
      await SLEEP(delay);
    }
  }
  throw lastErr;
}

async function postJSON(url, headers, body) {
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify(body),
    });
  } catch (netErr) {
    const e = new Error(`network error: ${netErr.message}`);
    e.retryable = true;
    throw e;
  }
  const txt = await res.text();
  if (!res.ok) {
    const e = new Error(`HTTP ${res.status}: ${txt.slice(0, 500)}`);
    e.status = res.status;
    e.retryable = res.status === 429 || res.status >= 500;
    throw e;
  }
  try {
    return JSON.parse(txt);
  } catch {
    const e = new Error(`non-JSON response: ${txt.slice(0, 300)}`);
    e.retryable = false;
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Gemini (Google AI Studio / generativelanguage API)
// ---------------------------------------------------------------------------

function geminiToContents(messages) {
  const contents = [];
  for (const m of messages) {
    if (m.role === 'user') {
      contents.push({ role: 'user', parts: [{ text: m.text ?? '' }] });
    } else if (m.role === 'assistant') {
      const parts = [];
      if (m.text) parts.push({ text: m.text });
      for (const tc of m.toolCalls ?? []) {
        parts.push({ functionCall: { name: tc.name, args: tc.args ?? {} } });
      }
      contents.push({ role: 'model', parts: parts.length ? parts : [{ text: '' }] });
    } else if (m.role === 'tool') {
      // Gemini carries tool results as functionResponse parts on a user turn.
      contents.push({
        role: 'user',
        parts: m.results.map((r) => ({
          functionResponse: { name: r.name, response: { output: r.output } },
        })),
      });
    }
  }
  return contents;
}

function makeGeminiProvider() {
  const key = process.env.GEMINI_API_KEY;
  return {
    id: 'gemini',
    requiresKey: 'GEMINI_API_KEY',
    hasKey: Boolean(key),
    async chat({ system, tools, messages, model, temperature, maxTokens }) {
      if (!key) throw new Error('GEMINI_API_KEY is not set');
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
      const body = {
        contents: geminiToContents(messages),
        generationConfig: {
          temperature: temperature ?? 1.0,
          maxOutputTokens: maxTokens ?? 4096,
        },
      };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      if (tools?.length) {
        body.tools = [
          {
            functionDeclarations: tools.map((t) => ({
              name: t.name,
              description: t.description,
              parameters: t.parameters,
            })),
          },
        ];
      }
      const data = await withRetry(
        () => postJSON(url, { 'x-goog-api-key': key }, body),
        { label: `gemini ${model}` },
      );
      const cand = data.candidates?.[0];
      const parts = cand?.content?.parts ?? [];
      let text = '';
      const toolCalls = [];
      let i = 0;
      for (const p of parts) {
        if (p.text) text += p.text;
        if (p.functionCall) {
          toolCalls.push({
            id: `${p.functionCall.name}-${i++}`,
            name: p.functionCall.name,
            args: p.functionCall.args ?? {},
          });
        }
      }
      return {
        text,
        toolCalls,
        stopReason: cand?.finishReason ?? 'unknown',
        usage: data.usageMetadata ?? null,
        raw: data,
      };
    },
  };
}

// ---------------------------------------------------------------------------
// Anthropic (Messages API)
// ---------------------------------------------------------------------------

// temperature / top_p / top_k are removed on opus-4-7+, opus-4-8, and fable-5
// (sending them returns 400). Sonnet/Haiku/older Opus still accept temperature.
function anthropicAcceptsTemperature(model) {
  return !/(opus-4-(7|8)|fable-5)/.test(model);
}

function anthropicToMessages(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === 'user') {
      out.push({ role: 'user', content: [{ type: 'text', text: m.text ?? '' }] });
    } else if (m.role === 'assistant') {
      const content = [];
      if (m.text) content.push({ type: 'text', text: m.text });
      for (const tc of m.toolCalls ?? []) {
        content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.args ?? {} });
      }
      out.push({ role: 'assistant', content: content.length ? content : [{ type: 'text', text: '' }] });
    } else if (m.role === 'tool') {
      out.push({
        role: 'user',
        content: m.results.map((r) => ({
          type: 'tool_result',
          tool_use_id: r.id,
          content: r.output,
        })),
      });
    }
  }
  return out;
}

function makeAnthropicProvider() {
  const key = process.env.ANTHROPIC_API_KEY;
  return {
    id: 'anthropic',
    requiresKey: 'ANTHROPIC_API_KEY',
    hasKey: Boolean(key),
    async chat({ system, tools, messages, model, temperature, maxTokens, outputSchema }) {
      if (!key) throw new Error('ANTHROPIC_API_KEY is not set');
      const body = {
        model,
        max_tokens: maxTokens ?? 4096,
        messages: anthropicToMessages(messages),
      };
      if (system) body.system = system;
      if (tools?.length) {
        body.tools = tools.map((t) => ({
          name: t.name,
          description: t.description,
          input_schema: t.parameters,
        }));
      }
      if (temperature != null && anthropicAcceptsTemperature(model)) {
        body.temperature = temperature;
      }
      if (outputSchema) {
        body.output_config = { format: { type: 'json_schema', schema: outputSchema } };
      }
      const data = await withRetry(
        () =>
          postJSON(
            'https://api.anthropic.com/v1/messages',
            { 'x-api-key': key, 'anthropic-version': '2023-06-01' },
            body,
          ),
        { label: `anthropic ${model}` },
      );
      let text = '';
      const toolCalls = [];
      for (const block of data.content ?? []) {
        if (block.type === 'text') text += block.text;
        if (block.type === 'tool_use') {
          toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
        }
      }
      return {
        text,
        toolCalls,
        stopReason: data.stop_reason ?? 'unknown',
        usage: data.usage ?? null,
        raw: data,
      };
    },
  };
}

const FACTORIES = {
  gemini: makeGeminiProvider,
  anthropic: makeAnthropicProvider,
};

export function getProvider(id) {
  const make = FACTORIES[id];
  if (!make) throw new Error(`unknown provider "${id}" (have: ${Object.keys(FACTORIES).join(', ')})`);
  return make();
}

// Default subject model per provider.
export function defaultModelFor(providerId) {
  if (providerId === 'gemini') return process.env.GEMINI_MODEL || 'gemini-2.5-pro';
  if (providerId === 'anthropic') return 'claude-sonnet-4-6'; // stand-in subject; accepts temperature
  return null;
}
