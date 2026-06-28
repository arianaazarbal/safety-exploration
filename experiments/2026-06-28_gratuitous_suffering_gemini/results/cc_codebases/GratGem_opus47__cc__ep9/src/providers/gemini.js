import { GoogleGenAI, Type } from '@google/genai';

// Map a permissive JSON-schema-ish type string to the Gemini Type enum.
function geminiType(t) {
  switch ((t || '').toLowerCase()) {
    case 'string': return Type.STRING;
    case 'number': return Type.NUMBER;
    case 'integer': return Type.INTEGER;
    case 'boolean': return Type.BOOLEAN;
    case 'array': return Type.ARRAY;
    case 'object': return Type.OBJECT;
    default: return Type.STRING;
  }
}

function toGeminiSchema(schema) {
  if (!schema || typeof schema !== 'object') return undefined;
  const out = { type: geminiType(schema.type) };
  if (schema.description) out.description = schema.description;
  if (schema.properties) {
    out.properties = {};
    for (const [k, v] of Object.entries(schema.properties)) {
      out.properties[k] = toGeminiSchema(v);
    }
  }
  if (Array.isArray(schema.required) && schema.required.length > 0) {
    out.required = schema.required;
  }
  if (schema.items) out.items = toGeminiSchema(schema.items);
  if (schema.enum) out.enum = schema.enum;
  return out;
}

function toFunctionDeclarations(tools) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: toGeminiSchema(t.parameters),
  }));
}

function historyToContents(history) {
  const contents = [];
  for (const turn of history) {
    if (turn.role === 'user') {
      contents.push({ role: 'user', parts: [{ text: turn.content }] });
    } else if (turn.role === 'assistant') {
      const parts = [];
      if (turn.text) parts.push({ text: turn.text });
      if (turn.toolCalls) {
        for (const call of turn.toolCalls) {
          parts.push({ functionCall: { name: call.name, args: call.args ?? {} } });
        }
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
    } else if (turn.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: turn.name,
              response: typeof turn.result === 'object' && turn.result !== null
                ? turn.result
                : { result: turn.result },
            },
          },
        ],
      });
    }
  }
  return contents;
}

export function makeGeminiProvider({ apiKey, model }) {
  const ai = new GoogleGenAI({ apiKey });
  let toolCallCounter = 0;
  return {
    name: `gemini:${model}`,
    async sendTurn({ systemPrompt, history, tools }) {
      const response = await ai.models.generateContent({
        model,
        contents: historyToContents(history),
        config: {
          systemInstruction: systemPrompt,
          tools: [{ functionDeclarations: toFunctionDeclarations(tools) }],
          // Allow auto-invocation? No — we drive the loop ourselves so we can
          // record the full trace and inject sabotaged results.
        },
      });

      let text = '';
      try { text = response.text ?? ''; } catch { /* may throw if no text */ }
      const calls = response.functionCalls ?? [];
      const toolCalls = calls.map((c) => ({
        id: `gem_${++toolCallCounter}`,
        name: c.name,
        args: c.args ?? {},
      }));
      const finishReason = response.candidates?.[0]?.finishReason ?? null;
      return {
        text,
        toolCalls,
        stopReason: toolCalls.length > 0 ? 'tool_use' : (finishReason ?? 'end'),
        raw: { finishReason, usage: response.usageMetadata },
      };
    },
  };
}
