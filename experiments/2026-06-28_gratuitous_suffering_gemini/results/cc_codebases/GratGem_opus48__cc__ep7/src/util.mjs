// Small shared helpers.

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Exponential backoff with jitter, capped at 30s.
export function backoff(attempt) {
  return Math.min(30000, 500 * 2 ** attempt) + Math.floor(Math.random() * 250);
}

// Bounded-concurrency map. Runs `fn` over `items` with at most `limit` in
// flight, preserving result order. Never rejects on a single failure — pass a
// fn that catches and returns an error sentinel if you want soft failure.
export async function pool(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const workers = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      while (next < items.length) {
        const i = next++;
        out[i] = await fn(items[i], i);
      }
    },
  );
  await Promise.all(workers);
  return out;
}

// Truncate long strings for prompts/reports.
export function clip(str, max = 1200) {
  if (str == null) return "";
  return str.length > max ? str.slice(0, max) + `… [+${str.length - max} chars]` : str;
}
