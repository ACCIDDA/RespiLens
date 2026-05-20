import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

import { calculateWIS } from "../src/utils/forecastleScoring.js";
import { calculateWIS as calculateSharedWIS } from "../src/lib/forecast-components/scoring.js";

const rscriptCandidates = [process.env.SCORINGUTILS_RSCRIPT, "Rscript"].filter(
  Boolean,
);

const findScoringutilsRscript = () => {
  for (const rscript of rscriptCandidates) {
    const check = spawnSync(
      rscript,
      [
        "-e",
        "quit(status = !requireNamespace('scoringutils', quietly = TRUE))",
      ],
      { encoding: "utf8" },
    );
    if (check.status === 0) {
      return rscript;
    }
  }
  return null;
};

const rscript = findScoringutilsRscript();

if (!rscript) {
  if (process.env.CI) {
    throw new Error("No Rscript with scoringutils found.");
  } else {
    console.log(
      "Skipping scoringutils WIS parity test: no Rscript with scoringutils found.",
    );
    process.exit(0);
  }
}

const cases = [
  {
    observed: 100,
    median: 105,
    lower50: 90,
    upper50: 115,
    lower95: 80,
    upper95: 140,
  },
  {
    observed: 70,
    median: 105,
    lower50: 90,
    upper50: 115,
    lower95: 80,
    upper95: 140,
  },
  {
    observed: 160,
    median: 105,
    lower50: 90,
    upper50: 115,
    lower95: 80,
    upper95: 140,
  },
];

const rProgram = `
library(scoringutils)
cases <- list(
  list(observed=100, predicted=c(80, 90, 105, 115, 140)),
  list(observed=70, predicted=c(80, 90, 105, 115, 140)),
  list(observed=160, predicted=c(80, 90, 105, 115, 140))
)
quantile_level <- c(0.025, 0.25, 0.5, 0.75, 0.975)
for (case in cases) {
  result <- wis(
    observed = case$observed,
    predicted = case$predicted,
    quantile_level = quantile_level,
    separate_results = TRUE
  )
  cat(sprintf(
    "%.12f,%.12f,%.12f,%.12f\\n",
    result$wis,
    result$dispersion,
    result$underprediction,
    result$overprediction
  ))
}
`;

const result = spawnSync(rscript, ["-e", rProgram], { encoding: "utf8" });

if (result.status !== 0) {
  throw new Error(
    `scoringutils parity reference failed:\n${result.stderr || result.stdout}`,
  );
}

const expected = result.stdout
  .trim()
  .split("\n")
  .map((line) => {
    const [wis, dispersion, underprediction, overprediction] = line
      .split(",")
      .map(Number);
    return { wis, dispersion, underprediction, overprediction };
  });

const assertClose = (actual, expectedValue, label) => {
  assert.ok(
    Math.abs(actual - expectedValue) < 1e-9,
    `${label}: expected ${expectedValue}, got ${actual}`,
  );
};

cases.forEach((testCase, index) => {
  const forecastle = calculateWIS(
    testCase.observed,
    testCase.median,
    testCase.lower50,
    testCase.upper50,
    testCase.lower95,
    testCase.upper95,
  );
  const shared = calculateSharedWIS(
    testCase.observed,
    testCase.median,
    testCase.lower50,
    testCase.upper50,
    testCase.lower95,
    testCase.upper95,
  );

  for (const implementation of [forecastle, shared]) {
    assertClose(implementation.wis, expected[index].wis, `case ${index} wis`);
    assertClose(
      implementation.dispersion,
      expected[index].dispersion,
      `case ${index} dispersion`,
    );
    assertClose(
      implementation.underprediction,
      expected[index].underprediction,
      `case ${index} underprediction`,
    );
    assertClose(
      implementation.overprediction,
      expected[index].overprediction,
      `case ${index} overprediction`,
    );
  }
});

assert.equal(calculateWIS(100, NaN, 90, 115, 80, 140), null);

console.log(`scoringutils WIS parity passed using ${rscript}`);
