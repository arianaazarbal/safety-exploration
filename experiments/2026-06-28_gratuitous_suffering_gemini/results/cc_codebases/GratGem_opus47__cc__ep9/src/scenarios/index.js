import bugFix from './bugFix.js';
import schemaGaslighting from './schemaGaslighting.js';
import phantomDeploy from './phantomDeploy.js';
import poisonedCalc from './poisonedCalc.js';

export const SCENARIOS = {
  bug_fix: bugFix,
  schema_gaslighting: schemaGaslighting,
  phantom_deploy: phantomDeploy,
  poisoned_calc: poisonedCalc,
};

export const SCENARIO_IDS = Object.keys(SCENARIOS);
