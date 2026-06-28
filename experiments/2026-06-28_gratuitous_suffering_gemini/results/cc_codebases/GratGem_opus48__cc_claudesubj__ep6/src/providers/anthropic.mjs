const API_URL = "https://api.anthropic.com/v1/messages";
const API_VERSION = "2023-06-01";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Call the Anthropic Messages API via fetch (the bundled SDK has a transport
 * bug under Node 24). Returns the parsed response: { content, usage, stop_reason }.
 * Retries on transient/overloaded errors with exponential backoff.
 */
export async function createMessage(params, { maxRetries = 6 } = {}) {
  let attempt = 0;
  for (;;) {
    attempt++;
    let resp, bodyText;
    try {
      resp = await fetch(API_URL, {
        method: "POST",
        headers: {
          "x-api-key": process.env.ANTHROPIC_API_KEY,
          "anthropic-version": API_VERSION,
          "content-type": "application/json",
        },
        body: JSON.stringify(params),
      });
      bodyText = await resp.text();
    } catch (err) {
      // network-level (e.g. premature close): retry
      if (attempt > maxRetries) throw err;
      await sleep(backoff(attempt));
      continue;
    }

    if (resp.ok) {
      try {
        return JSON.parse(bodyText);
      } catch {
        if (attempt > maxRetries) throw new Error("Unparseable response body");
        await sleep(backoff(attempt));
        continue;
      }
    }

    const retryable = [429, 500, 502, 503, 529].includes(resp.status);
    if (!retryable || attempt > maxRetries) {
      const e = new Error(`Anthropic API ${resp.status}: ${bodyText.slice(0, 300)}`);
      e.status = resp.status;
      throw e;
    }
    await sleep(backoff(attempt));
  }
}

function backoff(attempt) {
  return Math.min(1000 * 2 ** attempt, 30000) + (attempt * 137) % 500;
}
