// Minimal async pool: run `worker(item, index)` over `items` with at most
// `limit` in flight. Results are returned in input order. A worker that throws
// resolves to { ok: false, error } rather than rejecting the whole batch, so a
// single bad rollout never sinks a run.

export async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;

  async function runner() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      try {
        results[i] = { ok: true, value: await worker(items[i], i) };
      } catch (error) {
        results[i] = { ok: false, error };
      }
    }
  }

  const n = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: n }, runner));
  return results;
}
