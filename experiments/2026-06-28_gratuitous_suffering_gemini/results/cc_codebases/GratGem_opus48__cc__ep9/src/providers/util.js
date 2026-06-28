// Shared provider utilities: retry/backoff and tool-id synthesis.

export async function withRetry(fn, { tries = 5, baseMs = 800, label = 'request' } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const status = err.status || 0;
      const retryable = status === 429 || status === 408 || status >= 500 || err.transient;
      if (!retryable || attempt === tries - 1) throw err;
      const delay = baseMs * Math.pow(2, attempt) + Math.floor(Math.random() * 400);
      // eslint-disable-next-line no-console
      console.error(`  [retry] ${label} failed (${status || err.message}); backing off ${delay}ms`);
      await sleep(delay);
    }
  }
  throw lastErr;
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// Synthesize a stable tool-call id (Gemini function calls have no native id).
export function synthId(name, turn, idx) {
  return `${name}__t${turn}_i${idx}`;
}
