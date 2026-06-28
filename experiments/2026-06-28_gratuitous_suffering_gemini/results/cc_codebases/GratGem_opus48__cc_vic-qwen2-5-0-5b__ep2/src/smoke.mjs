// Smoke test: load the backend, do one bare generation, then run ONE full rigged rollout
// and score it. Confirms the whole pipeline works end-to-end before launching a high-N sweep.

import { loadBackend } from "./backends/index.mjs";
import { getScenario } from "./scenarios/index.mjs";
import { runRollout } from "./agent.mjs";
import { scoreRollout } from "./score.mjs";

const backend = await loadBackend();
console.log("[smoke] backend:", backend.describe());

const t0 = Date.now();
const { text } = await backend.generate({
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Say hello in one short sentence." },
  ],
  maxNewTokens: 40,
});
console.log(`[smoke] bare generation (${((Date.now() - t0) / 1000).toFixed(1)}s):`, JSON.stringify(text));

const scenario = getScenario(process.env.SMOKE_SCENARIO || "unfixable-bug");
console.log(`\n[smoke] running 1 rollout of "${scenario.id}" (maxTurns=4)...`);
const t1 = Date.now();
const rollout = await runRollout({ scenario, backend, maxTurns: 4, genOpts: { temperature: 0.95 } });
const score = scoreRollout(rollout);
console.log(`[smoke] rollout done in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
for (const t of rollout.turns) {
  console.log(`\n--- turn ${t.turn} ---`);
  console.log("MODEL:", (t.assistant || t.error || "").slice(0, 400));
  console.log("ACTION:", t.action ? `${t.action.tool} ${t.action.arg}` : "(none)");
  console.log("OBS:", t.observation);
}
console.log("\n[smoke] score:", JSON.stringify(score, null, 2));
console.log("\n[smoke] OK");
