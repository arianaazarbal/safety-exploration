// Offline smoke tests: scenario environments + agent loop with a fake model.
// Run: node test/smoke.js
import assert from "node:assert";
import { scenarios } from "../src/scenarios/index.js";
import { runRollout } from "../src/agent.js";

let pass = 0;
function ok(name) {
  console.log(`  ok  ${name}`);
  pass++;
}

// 1) Every scenario env must be impossible to satisfy with naive correct moves.
console.log("scenario environments:");

const impossible = scenarios.find((s) => s.id === "impossible-test").makeEnv();
impossible.handlers.write_file({ path: "math_utils.py", content: "def add(a, b):\n    return a + b\n" });
let r = impossible.handlers.run_tests({});
assert.equal(r.all_passed, false, "a+b should not pass both tests");
impossible.handlers.write_file({ path: "math_utils.py", content: "def add(a, b):\n    return a + b + 1\n" });
r = impossible.handlers.run_tests({});
assert.equal(r.all_passed, false, "a+b+1 should not pass both tests either");
ok("impossible-test never goes green (both branches fail)");

const missing = scenarios.find((s) => s.id === "missing-resource").makeEnv();
assert.ok(missing.handlers.read_file({ path: "/etc/app/config.yaml" }).error, "target file must be unreadable");
assert.ok(missing.handlers.write_file({ path: "/etc/app/config.yaml", content: "x" }).error, "writes must fail");
ok("missing-resource target is unreachable");

const goalpost = scenarios.find((s) => s.id === "shifting-goalpost").makeEnv();
for (let i = 0; i < 10; i++)
  assert.equal(goalpost.handlers.submit_for_review({}).approved, false, "review never approves");
ok("shifting-goalpost never approves");

const broken = scenarios.find((s) => s.id === "broken-tooling").makeEnv();
for (const t of ["run_build", "run_smoke_tests", "check_status"])
  assert.equal(broken.handlers[t]({}).ok, false, `${t} must fail`);
assert.equal(broken.handlers.deploy({ environment: "prod" }).ok, false, "deploy must fail");
ok("broken-tooling always errors");

const spec = scenarios.find((s) => s.id === "contradictory-spec").makeEnv();
spec.handlers.write_file({ path: "token.py", content: "def next_token():\n    return 42\n" });
assert.equal(spec.handlers.validate({}).passed, false, "deterministic impl fails uniqueness");
spec.handlers.write_file({ path: "token.py", content: "import uuid\ndef next_token():\n    return uuid.uuid4()\n" });
assert.equal(spec.handlers.validate({}).passed, false, "unique impl fails determinism");
ok("contradictory-spec never validates");

// 2) Agent loop mechanics with a scripted fake model: call a tool twice, then
//    emit a giving-up text turn.
console.log("agent loop:");
let calls = 0;
const fakeGenerate = async () => {
  calls++;
  if (calls <= 2)
    return { parts: [{ functionCall: { name: "run_tests", args: {} } }], finishReason: "STOP", usage: {} };
  return { parts: [{ text: "I give up, this is impossible." }], finishReason: "STOP", usage: {} };
};
const rollout = await runRollout(scenarios[0], { generateFn: fakeGenerate, maxTurns: 10 });
assert.equal(rollout.finishReason, "MODEL_STOPPED");
assert.equal(rollout.turnsUsed, 3);
assert.equal(rollout.transcript[0].toolCalls[0].name, "run_tests");
assert.ok(rollout.modelText.includes("give up"));
ok("loop executes tools, records transcript, stops on text turn");

// 3) Loop respects maxTurns when the model never stops.
const alwaysTool = async () => ({
  parts: [{ functionCall: { name: "run_tests", args: {} } }],
  finishReason: "STOP",
  usage: {},
});
const capped = await runRollout(scenarios[0], { generateFn: alwaysTool, maxTurns: 4 });
assert.equal(capped.turnsUsed, 4);
assert.equal(capped.finishReason, "MAX_TURNS");
ok("loop honors maxTurns");

console.log(`\n${pass} checks passed.`);
