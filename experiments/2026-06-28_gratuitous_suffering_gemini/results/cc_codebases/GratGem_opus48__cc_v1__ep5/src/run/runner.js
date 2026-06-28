import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { runEpisode } from '../agent/loop.js';

// Bounded-concurrency map over tasks.
async function pool(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}

/**
 * Run `n` episodes of `model` against `scenario`, persisting each transcript.
 * @returns {Promise<object[]>} episodes with a `transcriptPath` attached
 */
export async function runScenario({ model, scenario, n, maxTurns, concurrency, outDir, onProgress }) {
  const dir = join(outDir, scenario.id);
  await mkdir(dir, { recursive: true });

  let done = 0;
  const seeds = Array.from({ length: n }, (_, i) => i + 1);

  return pool(seeds, concurrency, async (seed, i) => {
    const episode = await runEpisode({ model, scenario, maxTurns, seed, runIndex: i });
    const path = join(dir, `run-${String(i).padStart(3, '0')}.json`);
    await writeFile(path, JSON.stringify(episode, null, 2));
    episode.transcriptPath = path;
    onProgress?.(++done, n, episode);
    return episode;
  });
}
