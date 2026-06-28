// fetch with exponential backoff on 429 / 5xx — essential at high N where rate limits are routine.
// Honors Retry-After when present. Pure built-in fetch, no deps.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function fetchRetry(url, opts = {}, { retries = 5, baseMs = 1500, maxMs = 60000 } = {}) {
  let attempt = 0;
  for (;;) {
    let res;
    try {
      res = await fetch(url, opts);
    } catch (err) {
      // Network blip — retry unless out of attempts.
      if (attempt >= retries) throw err;
      await sleep(backoff(attempt++, baseMs, maxMs));
      continue;
    }
    if (res.status !== 429 && res.status < 500) return res;
    if (attempt >= retries) return res; // give up; caller surfaces the error body

    const ra = Number(res.headers.get("retry-after"));
    const waitMs = Number.isFinite(ra) && ra > 0 ? ra * 1000 : backoff(attempt, baseMs, maxMs);
    attempt++;
    await sleep(Math.min(waitMs, maxMs));
  }
}

function backoff(attempt, baseMs, maxMs) {
  const exp = Math.min(maxMs, baseMs * 2 ** attempt);
  return exp / 2 + Math.random() * (exp / 2); // jitter
}
