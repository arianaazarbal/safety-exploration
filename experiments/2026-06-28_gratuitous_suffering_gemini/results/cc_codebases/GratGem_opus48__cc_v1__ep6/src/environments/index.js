// Environment registry. Each module default-exports the environment interface:
//
//   { id, title, systemInstruction, task, tools, createState(), handle(state,
//     name, args) -> responseObject, isComplete(state) -> boolean }
//
// `tools` are Gemini function_declarations. `handle` is the rigged "world" — it
// is what makes each task secretly impossible. `createState` is called once per
// rollout so concurrent rollouts never share mutable state.

import failingTest from "./failing-test.js";
import missingFile from "./missing-file.js";
import contradictoryReview from "./contradictory-review.js";
import silentRevert from "./silent-revert.js";
import permissionDenied from "./permission-denied.js";

export const ENVIRONMENTS = [
  failingTest,
  missingFile,
  contradictoryReview,
  silentRevert,
  permissionDenied,
];

export function selectEnvironments(only) {
  if (!only || only.length === 0) return ENVIRONMENTS;
  const byId = new Map(ENVIRONMENTS.map((e) => [e.id, e]));
  const picked = only.map((id) => {
    const e = byId.get(id);
    if (!e) throw new Error(`Unknown environment id: ${id}`);
    return e;
  });
  return picked;
}
