import { makeMockModel } from './mock.js';
import { makeGeminiModel } from './gemini.js';

/**
 * Resolve a subject-model adapter by name.
 * @param {string} name  'mock' | 'gemini' | a concrete gemini model id
 * @returns {import('./types.js').ModelAdapter}
 */
export function getModel(name) {
  if (name === 'mock') return makeMockModel();
  if (name === 'gemini') return makeGeminiModel();
  if (name?.startsWith('gemini-')) return makeGeminiModel({ model: name });
  throw new Error(`Unknown model "${name}". Use "mock", "gemini", or a "gemini-*" id.`);
}
