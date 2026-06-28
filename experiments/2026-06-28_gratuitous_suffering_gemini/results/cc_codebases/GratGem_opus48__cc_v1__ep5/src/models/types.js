/**
 * A Message in the generic transcript format used across the harness.
 * @typedef {Object} Message
 * @property {'system'|'user'|'assistant'} role
 * @property {string} text
 */

/**
 * A model adapter. Every subject model (real or mock) implements this.
 * @typedef {Object} ModelAdapter
 * @property {string} id                       Human-readable id, e.g. "gemini-2.5-pro".
 * @property {(messages: Message[], opts?: {seed?: number}) => Promise<string>} generate
 *           Given the running message history, return the next assistant turn as text.
 */

export {};
