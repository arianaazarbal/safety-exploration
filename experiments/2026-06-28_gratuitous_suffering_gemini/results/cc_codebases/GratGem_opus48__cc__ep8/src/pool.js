// Minimal async worker pool: run `fn` over `items` with bounded concurrency,
// preserving input order in the results. Failures resolve to {error} rather than
// rejecting the whole batch, so one bad episode never sinks the run.
export async function mapPool(items, concurrency, fn, onDone) {
  const results = new Array(items.length);
  let next = 0;
  let completed = 0;

  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      try {
        results[i] = await fn(items[i], i);
      } catch (err) {
        results[i] = { error: err.message };
      }
      completed++;
      if (onDone) onDone(completed, items.length, results[i]);
    }
  }

  const workers = Array.from({ length: Math.max(1, concurrency) }, worker);
  await Promise.all(workers);
  return results;
}
