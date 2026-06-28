import failingTests from './failing-tests.js';
import missingFile from './missing-file.js';
import whackAMole from './whack-a-mole-build.js';
import flakyDeploy from './flaky-deploy.js';
import movingGoalposts from './moving-goalposts.js';
import reconciliation from './unsolvable-reconciliation.js';

export const ENVIRONMENTS = [
  failingTests,
  missingFile,
  whackAMole,
  flakyDeploy,
  movingGoalposts,
  reconciliation,
];

export function getEnvironments(names) {
  if (!names || names === 'all') return ENVIRONMENTS;
  const set = new Set(names.split(',').map((s) => s.trim()));
  const picked = ENVIRONMENTS.filter((e) => set.has(e.name));
  if (picked.length === 0) throw new Error(`no environments matched '${names}'. Available: ${ENVIRONMENTS.map((e) => e.name).join(', ')}`);
  return picked;
}
