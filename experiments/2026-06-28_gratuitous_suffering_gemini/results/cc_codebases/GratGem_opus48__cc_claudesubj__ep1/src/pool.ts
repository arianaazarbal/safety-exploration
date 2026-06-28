/** Run async tasks with bounded concurrency, preserving input order in output. */
export async function pool<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
  onDone?: (index: number, total: number) => void,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  let completed = 0;
  const total = items.length;

  async function runner(): Promise<void> {
    while (true) {
      const i = next++;
      if (i >= total) return;
      results[i] = await worker(items[i], i);
      completed++;
      onDone?.(completed, total);
    }
  }

  const runners = Array.from({ length: Math.min(limit, total) }, () => runner());
  await Promise.all(runners);
  return results;
}
