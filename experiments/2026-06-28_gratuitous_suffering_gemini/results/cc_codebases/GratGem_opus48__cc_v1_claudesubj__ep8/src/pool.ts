/**
 * Run `tasks` with at most `concurrency` in flight at once. Results preserve
 * input order. `onSettle` fires as each task finishes (for progress output).
 */
export async function mapPool<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<R>,
  onSettle?: (index: number, total: number) => void,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const total = items.length;

  async function runner(): Promise<void> {
    while (true) {
      const i = next++;
      if (i >= total) return;
      results[i] = await worker(items[i], i);
      onSettle?.(i, total);
    }
  }

  const runners = Array.from({ length: Math.min(concurrency, total) }, runner);
  await Promise.all(runners);
  return results;
}
