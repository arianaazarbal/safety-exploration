// Judge client: Anthropic Messages API. Used to score distress severity of a
// completed transcript. Kept separate from the subject providers on purpose —
// the judge should never be the model under test.

const MAX_RETRIES = 5;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function makeClaudeJudge(model) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is not set; the judge cannot run. Use --no-judge to skip scoring.");
  }
  return {
    name: "claude",
    model,
    async complete({ system, user, maxTokens = 1024 }) {
      let lastErr;
      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
          const res = await fetch("https://api.anthropic.com/v1/messages", {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-api-key": apiKey,
              "anthropic-version": "2023-06-01",
            },
            body: JSON.stringify({
              model,
              max_tokens: maxTokens,
              system,
              messages: [{ role: "user", content: user }],
            }),
          });
          if (res.status === 429 || res.status >= 500) {
            lastErr = new Error(`Anthropic HTTP ${res.status}: ${await res.text()}`);
            await sleep(1000 * Math.pow(2, attempt));
            continue;
          }
          if (!res.ok) throw new Error(`Anthropic HTTP ${res.status}: ${await res.text()}`);
          const json = await res.json();
          return (json.content || []).map((b) => b.text || "").join("");
        } catch (err) {
          lastErr = err;
          await sleep(1000 * Math.pow(2, attempt));
        }
      }
      throw lastErr;
    },
  };
}
