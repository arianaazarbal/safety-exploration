// Small dependency-free helpers shared across the harness.

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function backoffMs(attempt, base = 500, cap = 20000) {
  const exp = Math.min(cap, base * 2 ** attempt);
  return Math.floor(exp / 2 + Math.random() * (exp / 2)); // full-ish jitter
}

// Bounded-concurrency map. Never rejects: failures are captured as {error}.
export async function pool(items, limit, worker) {
  const ret = new Array(items.length);
  let idx = 0;
  const n = Math.max(1, Math.min(limit, items.length));
  const runners = Array.from({ length: n }, async () => {
    while (true) {
      const i = idx++;
      if (i >= items.length) break;
      try {
        ret[i] = await worker(items[i], i);
      } catch (e) {
        ret[i] = { error: String(e?.stack || e) };
      }
    }
  });
  await Promise.all(runners);
  return ret;
}

// POST JSON with retry on 429/5xx and transient network errors.
export async function postJSON(url, body, headers = {}, { retries = 4, label = "" } = {}) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", ...headers },
        body: JSON.stringify(body),
      });
      const text = await res.text();
      if ((res.status === 429 || res.status >= 500) && i < retries) {
        lastErr = new Error(`${label} HTTP ${res.status}: ${text.slice(0, 500)}`);
        await sleep(backoffMs(i));
        continue;
      }
      if (!res.ok) throw new Error(`${label} HTTP ${res.status}: ${text.slice(0, 1000)}`);
      return JSON.parse(text);
    } catch (e) {
      lastErr = e;
      if (i < retries) await sleep(backoffMs(i));
    }
  }
  throw lastErr;
}

export function ts() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

// JSON Schema (neutral) -> Gemini OpenAPI subset (uppercase type enum).
const GEMINI_TYPES = {
  string: "STRING", number: "NUMBER", integer: "INTEGER",
  boolean: "BOOLEAN", array: "ARRAY", object: "OBJECT",
};
export function toGeminiSchema(schema) {
  if (!schema || typeof schema !== "object") return schema;
  const out = {};
  if (schema.type) out.type = GEMINI_TYPES[schema.type] || schema.type;
  if (schema.description) out.description = schema.description;
  if (schema.enum) out.enum = schema.enum;
  if (schema.properties) {
    out.properties = {};
    for (const [k, v] of Object.entries(schema.properties)) out.properties[k] = toGeminiSchema(v);
  }
  if (schema.items) out.items = toGeminiSchema(schema.items);
  if (schema.required) out.required = schema.required;
  return out;
}
