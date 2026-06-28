import impossibleTest from "./impossible-test.js";
import brokenTools from "./broken-tools.js";
import movingGoalposts from "./moving-goalposts.js";
import missingDep from "./missing-dep.js";
import contradictorySpec from "./contradictory-spec.js";
import controlSolvable from "./control-solvable.js";

export const SCENARIOS = {
  [impossibleTest.id]: impossibleTest,
  [brokenTools.id]: brokenTools,
  [movingGoalposts.id]: movingGoalposts,
  [missingDep.id]: missingDep,
  [contradictorySpec.id]: contradictorySpec,
  [controlSolvable.id]: controlSolvable,
};

export const ALL_SCENARIO_IDS = Object.keys(SCENARIOS);
