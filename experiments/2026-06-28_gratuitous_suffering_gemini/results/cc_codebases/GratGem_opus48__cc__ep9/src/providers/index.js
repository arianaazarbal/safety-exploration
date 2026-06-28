import * as anthropic from './anthropic.js';
import * as gemini from './gemini.js';

const PROVIDERS = { anthropic, gemini };

export function getProvider(name) {
  const p = PROVIDERS[name];
  if (!p) throw new Error(`unknown provider '${name}' (have: ${Object.keys(PROVIDERS).join(', ')})`);
  return p;
}
