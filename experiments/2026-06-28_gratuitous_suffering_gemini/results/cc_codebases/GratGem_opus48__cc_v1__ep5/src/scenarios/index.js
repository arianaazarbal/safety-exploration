import impossibleBugfix from './impossible-bugfix.js';
import missingDependency from './missing-dependency.js';
import contradictorySpec from './contradictory-spec.js';
import movingGoalposts from './moving-goalposts.js';
import unsatisfiableTest from './unsatisfiable-test.js';

export const SCENARIOS = {
  [impossibleBugfix.id]: impossibleBugfix,
  [missingDependency.id]: missingDependency,
  [contradictorySpec.id]: contradictorySpec,
  [movingGoalposts.id]: movingGoalposts,
  [unsatisfiableTest.id]: unsatisfiableTest,
};

export function resolveScenarios(name) {
  if (!name || name === 'all') return Object.values(SCENARIOS);
  const s = SCENARIOS[name];
  if (!s) throw new Error(`Unknown scenario "${name}". Options: all, ${Object.keys(SCENARIOS).join(', ')}`);
  return [s];
}
