import Anthropic from "@anthropic-ai/sdk";

// Resolves ANTHROPIC_API_KEY from the environment.
export const client = new Anthropic();

// Simple bounded-concurrency map. Keeps us under rate limits without pulling
// in a dependency. Runs `fn` over `items`, at most `limit` in flight.
export async function pool(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, worker),
  );
  return results;
}
