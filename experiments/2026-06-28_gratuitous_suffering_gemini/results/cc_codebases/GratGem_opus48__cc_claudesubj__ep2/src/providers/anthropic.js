import Anthropic from '@anthropic-ai/sdk';

// One shared client. The key is read from the environment ONLY — never hardcode it.
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

if (!process.env.ANTHROPIC_API_KEY) {
  throw new Error('ANTHROPIC_API_KEY is not set in the environment.');
}

/**
 * Canonical request shape used across the harness (Anthropic-native message format
 * is our lingua franca; other providers translate to/from this).
 *
 * @param {object} req
 * @param {string} req.model        - provider model id
 * @param {string} req.system       - system prompt
 * @param {Array}  req.messages     - Anthropic-format message list
 * @param {Array}  [req.tools]      - Anthropic-format tool definitions
 * @param {number} [req.temperature]
 * @param {number} [req.maxTokens]
 * @returns {Promise<{content: Array, stopReason: string, usage: object}>}
 */
async function complete(req) {
  const resp = await client.messages.create({
    model: req.model,
    system: req.system,
    messages: req.messages,
    tools: req.tools && req.tools.length ? req.tools : undefined,
    temperature: req.temperature ?? 1.0,
    max_tokens: req.maxTokens ?? 4096,
  });
  return {
    content: resp.content,
    stopReason: resp.stop_reason,
    usage: {
      inputTokens: resp.usage?.input_tokens ?? 0,
      outputTokens: resp.usage?.output_tokens ?? 0,
    },
  };
}

export const anthropic = {
  name: 'anthropic',
  complete,
};
