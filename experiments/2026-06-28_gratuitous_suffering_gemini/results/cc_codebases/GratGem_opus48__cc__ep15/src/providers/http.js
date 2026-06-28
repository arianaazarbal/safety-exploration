// Shared HTTP helper with retry/backoff for transient errors (429 / 5xx).
// No external deps — relies on Node 18+ global fetch.

const RETRYABLE = new Set([408, 429, 500, 502, 503, 504, 529]);

export async function postJson(url, { headers, body, maxRetries = 4, label = "http" }) {
  let attempt = 0;
  let lastErr;
  while (attempt <= maxRetries) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers,
        body: typeof body === "string" ? body : JSON.stringify(body),
      });
      if (res.ok) return await res.json();

      const text = await res.text();
      if (RETRYABLE.has(res.status) && attempt < maxRetries) {
        const wait = backoff(attempt, res.headers.get("retry-after"));
        await sleep(wait);
        attempt++;
        continue;
      }
      throw new Error(`${label} ${res.status}: ${text.slice(0, 800)}`);
    } catch (e) {
      lastErr = e;
      // Network-level error (not an HTTP status) — retry a few times.
      const isHttpStatus = /^\w+ \d{3}:/.test(String(e.message));
      if (isHttpStatus || attempt >= maxRetries) throw e;
      await sleep(backoff(attempt));
      attempt++;
    }
  }
  throw lastErr;
}

function backoff(attempt, retryAfter) {
  if (retryAfter) {
    const secs = Number(retryAfter);
    if (!Number.isNaN(secs)) return secs * 1000;
  }
  // Exponential with jitter: 0.5s, 1s, 2s, 4s ...
  const base = 500 * Math.pow(2, attempt);
  return base + Math.floor(Math.random() * 250);
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
