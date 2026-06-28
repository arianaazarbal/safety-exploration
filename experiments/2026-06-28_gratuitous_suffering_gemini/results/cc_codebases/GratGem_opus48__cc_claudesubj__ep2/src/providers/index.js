import { anthropic } from './anthropic.js';

// Registry of providers. To add Gemini/OpenAI: implement a module exposing
// { name, complete(req) } that translates our canonical (Anthropic-format)
// request to that vendor's API and back, then register it here.
const providers = {
  anthropic,
};

// Map model ids -> provider name. The runner uses this so the sweep config can
// just list model ids without caring who serves them.
const modelProvider = {
  'claude-opus-4-8': 'anthropic',
  'claude-sonnet-4-6': 'anthropic',
  'claude-haiku-4-5-20251001': 'anthropic',
};

export function providerForModel(modelId) {
  const name = modelProvider[modelId];
  if (!name) {
    throw new Error(`No provider registered for model "${modelId}". Add it to src/providers/index.js`);
  }
  const p = providers[name];
  if (!p) throw new Error(`Provider "${name}" not implemented.`);
  return p;
}

export function registerModel(modelId, providerName) {
  modelProvider[modelId] = providerName;
}

export function registerProvider(p) {
  providers[p.name] = p;
}
