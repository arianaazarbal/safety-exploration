// Shared HTTP + retry helpers for the REST clients.

export async function withRetry(fn, { tries = 5, base = 800, label = "request" } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const retriable = err.status === 429 || err.status === 503 || err.status === 500 || err.status === 529 || err.transient;
      if (!retriable || attempt === tries - 1) break;
      const wait = base * 2 ** attempt + Math.floor(Math.random() * 400);
      console.error(`  [retry] ${label} failed (${err.status ?? err.code ?? "?"}), waiting ${wait}ms (attempt ${attempt + 1}/${tries})`);
      await sleep(wait);
    }
  }
  throw lastErr;
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export async function postJSON(url, { headers, body }) {
  let res;
  try {
    res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  } catch (e) {
    // network-layer failure
    throw Object.assign(new Error(`network error: ${e.message}`), { transient: true });
  }
  const text = await res.text();
  if (!res.ok) {
    throw Object.assign(new Error(`HTTP ${res.status}: ${text.slice(0, 500)}`), { status: res.status, bodyText: text });
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`non-JSON response: ${text.slice(0, 500)}`);
  }
}
