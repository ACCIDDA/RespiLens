/**
 * Calculate RMSE (Root Mean Squared Error) for a set of predictions
 * @param {Array} predictions - Array of predicted values
 * @param {Array} observations - Array of observed (ground truth) values
 * @returns {number} RMSE value
 */
export const calculateRMSE = (predictions, observations) => {
  if (
    !predictions ||
    !observations ||
    predictions.length === 0 ||
    observations.length === 0
  ) {
    return null;
  }

  if (predictions.length !== observations.length) {
    return null;
  }

  let sumSquaredErrors = 0;
  let count = 0;

  for (let i = 0; i < predictions.length; i += 1) {
    const pred = predictions[i];
    const obs = observations[i];

    if (Number.isFinite(pred) && Number.isFinite(obs)) {
      const error = pred - obs;
      sumSquaredErrors += error * error;
      count += 1;
    }
  }

  if (count === 0) {
    return null;
  }

  return Math.sqrt(sumSquaredErrors / count);
};

/**
 * Calculate interval score for a single prediction interval
 * @param {number} observed - Observed value
 * @param {number} lower - Lower bound of prediction interval
 * @param {number} upper - Upper bound of prediction interval
 * @param {number} alpha - Alpha level (e.g., 0.5 for 50% interval, 0.05 for 95% interval)
 * @returns {Object} Interval score with components {score, dispersion, underprediction, overprediction}
 */
const calculateIntervalScore = (observed, lower, upper, alpha) => {
  if (
    !Number.isFinite(observed) ||
    !Number.isFinite(lower) ||
    !Number.isFinite(upper)
  ) {
    return null;
  }

  const dispersion = upper - lower;
  const underprediction =
    observed < lower ? (2 / alpha) * (lower - observed) : 0;
  const overprediction =
    observed > upper ? (2 / alpha) * (observed - upper) : 0;
  const score = dispersion + underprediction + overprediction;

  return {
    score,
    dispersion,
    underprediction,
    overprediction,
  };
};

/**
 * Calculate WIS (Weighted Interval Score) for a single forecast
 * @param {number} observed - Observed value
 * @param {number} median - Median prediction
 * @param {number} lower50 - Lower bound of 50% interval (0.25 quantile)
 * @param {number} upper50 - Upper bound of 50% interval (0.75 quantile)
 * @param {number} lower95 - Lower bound of 95% interval (0.025 quantile)
 * @param {number} upper95 - Upper bound of 95% interval (0.975 quantile)
 * @returns {Object} WIS with components {wis, dispersion, underprediction, overprediction}
 */
export const calculateWIS = (
  observed,
  median,
  lower50,
  upper50,
  lower95,
  upper95,
) => {
  if (!Number.isFinite(observed) || !Number.isFinite(median)) {
    return null;
  }

  // Calculate interval scores for each interval
  const interval50 = calculateIntervalScore(observed, lower50, upper50, 0.5);
  const interval95 = calculateIntervalScore(observed, lower95, upper95, 0.05);

  if (!interval50 || !interval95) {
    return null;
  }

  const medianAE = Math.abs(observed - median);

  // Weights match scoringutils::wis(..., weigh = TRUE,
  // count_median_twice = FALSE), normalized by K + 0.5.
  const weight50 = 0.5 / 2; // 0.25
  const weight95 = 0.05 / 2; // 0.025
  const weightMedian = 0.5; // 0.5
  const normalizer = 2 + 0.5; // two central intervals plus median half-weight

  // Weighted sum
  const wis =
    (weight50 * interval50.score +
      weight95 * interval95.score +
      weightMedian * medianAE) /
    normalizer;

  // Aggregate components
  const dispersion =
    (weight50 * interval50.dispersion + weight95 * interval95.dispersion) /
    normalizer;
  const medianUnderprediction =
    observed > median ? (weightMedian * (observed - median)) / normalizer : 0;
  const medianOverprediction =
    observed < median ? (weightMedian * (median - observed)) / normalizer : 0;
  const underprediction =
    (weight50 * interval50.overprediction +
      weight95 * interval95.overprediction) /
      normalizer +
    medianUnderprediction;
  const overprediction =
    (weight50 * interval50.underprediction +
      weight95 * interval95.underprediction) /
      normalizer +
    medianOverprediction;

  return {
    wis,
    dispersion,
    underprediction,
    overprediction,
  };
};

/**
 * Extract ground truth values for specific horizon dates
 * @param {Array} groundTruthSeries - Full ground truth series with {date, value}
 * @param {Array} horizonDates - Array of dates to extract ground truth for
 * @returns {Array} Array of ground truth values matching horizonDates
 */
export const extractGroundTruthForHorizons = (
  groundTruthSeries,
  horizonDates,
) => {
  const truthMap = new Map(
    groundTruthSeries.map((entry) => [entry.date, entry.value]),
  );
  return horizonDates.map((date) => truthMap.get(date) ?? null);
};

/**
 * Score user's forecast using WIS (Weighted Interval Score)
 * @param {Array} userForecasts - Array of user forecast objects with {median, lower50, upper50, lower95, upper95}
 * @param {Array} groundTruthValues - Array of ground truth values for each horizon
 * @returns {Object} Score result with WIS and components
 */
export const scoreUserForecast = (userForecasts, groundTruthValues) => {
  if (!userForecasts || !groundTruthValues || userForecasts.length === 0) {
    return {
      wis: null,
      dispersion: null,
      underprediction: null,
      overprediction: null,
      validCount: 0,
      totalHorizons: 0,
    };
  }

  let sumWIS = 0;
  let sumDispersion = 0;
  let sumUnderprediction = 0;
  let sumOverprediction = 0;
  let count = 0;

  for (let i = 0; i < userForecasts.length; i += 1) {
    const forecast = userForecasts[i];
    const observed = groundTruthValues[i];

    if (!Number.isFinite(observed)) {
      continue;
    }

    const wisResult = calculateWIS(
      observed,
      forecast.median,
      forecast.lower50,
      forecast.upper50,
      forecast.lower95,
      forecast.upper95,
    );

    if (wisResult) {
      sumWIS += wisResult.wis;
      sumDispersion += wisResult.dispersion;
      sumUnderprediction += wisResult.underprediction;
      sumOverprediction += wisResult.overprediction;
      count += 1;
    }
  }

  if (count === 0) {
    return {
      wis: null,
      dispersion: null,
      underprediction: null,
      overprediction: null,
      validCount: 0,
      totalHorizons: userForecasts.length,
    };
  }

  return {
    wis: sumWIS / count,
    dispersion: sumDispersion / count,
    underprediction: sumUnderprediction / count,
    overprediction: sumOverprediction / count,
    validCount: count,
    totalHorizons: userForecasts.length,
  };
};

/**
 * Get official ensemble and baseline model names for a dataset
 * @param {string} datasetKey - Dataset key (e.g., 'flusight', 'rsv', 'covid19')
 * @returns {Object} Object with ensembleKey and baselineKey
 */
export const getOfficialModels = (datasetKey) => {
  const modelMap = {
    flusight: {
      ensemble: "FluSight-ensemble",
      baseline: "FluSight-baseline",
    },
    rsv: {
      ensemble: "RSVHub-ensemble",
      baseline: "RSVHub-baseline",
    },
    covid19: {
      ensemble: "CovidHub-ensemble",
      baseline: "CovidHub-baseline",
    },
  };

  return modelMap[datasetKey] || { ensemble: null, baseline: null };
};

/**
 * Score model forecasts against ground truth using WIS
 * @param {Object} modelForecasts - Object with model names as keys and predictions as values
 * @param {Array} horizons - Array of horizon values (e.g., [1, 2, 3])
 * @param {Array} groundTruthValues - Array of ground truth values for each horizon
 * @returns {Array} Array of {modelName, wis, dispersion, underprediction, overprediction, validCount} sorted by WIS (best first)
 */
export const scoreModels = (modelForecasts, horizons, groundTruthValues) => {
  const modelScores = [];

  Object.entries(modelForecasts).forEach(([modelName, modelData]) => {
    const predictions = modelData?.predictions;
    if (!predictions) {
      return;
    }

    const modelForecastsRaw = horizons.map((horizon) => {
      const horizonPrediction = predictions[String(horizon)];
      if (!horizonPrediction) {
        return null;
      }

      // Extract quantiles from model predictions
      // quantiles and values are parallel arrays: [0.025, 0.25, 0.5, 0.75, 0.975] and corresponding values
      const quantiles = horizonPrediction.quantiles;
      const values = horizonPrediction.values;

      if (!quantiles || !values || quantiles.length !== values.length) {
        return null;
      }

      // Extract the required quantiles
      const getQuantileValue = (targetQuantile) => {
        const index = quantiles.findIndex(
          (q) => Math.abs(q - targetQuantile) < 0.001,
        );
        return index !== -1 && Number.isFinite(values[index])
          ? values[index]
          : null;
      };

      const median = getQuantileValue(0.5);
      const lower50 = getQuantileValue(0.25);
      const upper50 = getQuantileValue(0.75);
      const lower95 = getQuantileValue(0.025);
      const upper95 = getQuantileValue(0.975);

      if (
        median === null ||
        lower50 === null ||
        upper50 === null ||
        lower95 === null ||
        upper95 === null
      ) {
        return null;
      }

      if (
        lower95 > lower50 ||
        lower50 > median ||
        median > upper50 ||
        upper50 > upper95
      ) {
        return null;
      }

      return {
        median,
        lower50,
        upper50,
        lower95,
        upper95,
      };
    });

    // Filter out null forecasts to prevent scoreUserForecast from receiving invalid data
    const validIndices = [];
    const validForecasts = [];
    const validGroundTruth = [];

    modelForecastsRaw.forEach((forecast, index) => {
      if (forecast !== null) {
        validIndices.push(index);
        validForecasts.push(forecast);
        validGroundTruth.push(groundTruthValues[index]);
      }
    });

    // Skip this model if there are no valid forecasts
    if (validForecasts.length === 0) {
      return;
    }

    // Score this model with only valid forecasts
    const score = scoreUserForecast(validForecasts, validGroundTruth);
    if (score.wis !== null) {
      modelScores.push({
        modelName,
        wis: score.wis,
        dispersion: score.dispersion,
        underprediction: score.underprediction,
        overprediction: score.overprediction,
        validCount: score.validCount,
      });
    }
  });

  // Sort by WIS (lower is better)
  return modelScores.sort((a, b) => a.wis - b.wis);
};
