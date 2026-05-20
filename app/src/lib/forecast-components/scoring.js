/**
 * Reusable Forecast Scoring Utilities
 * WIS is re-exported from the canonical Forecastle scoring utility.
 */

import { calculateWIS } from "../../utils/forecastleScoring.js";

export { calculateWIS };

/**
 * Validate forecast intervals
 * @param {Object} forecast - Forecast object with median and intervals
 * @returns {Object} Validation result {valid, errors}
 */
export const validateForecastIntervals = (forecast) => {
  const errors = [];

  if (!forecast) {
    return { valid: false, errors: ["Forecast is required"] };
  }

  const { median, q25, q75, q025, q975 } = forecast;

  // Check all values are numbers
  if (!Number.isFinite(median)) errors.push("Median must be a number");
  if (!Number.isFinite(q25)) errors.push("25th percentile must be a number");
  if (!Number.isFinite(q75)) errors.push("75th percentile must be a number");
  if (!Number.isFinite(q025)) errors.push("2.5th percentile must be a number");
  if (!Number.isFinite(q975)) errors.push("97.5th percentile must be a number");

  // Check all values are non-negative
  if (median < 0) errors.push("Median must be non-negative");
  if (q25 < 0) errors.push("25th percentile must be non-negative");
  if (q75 < 0) errors.push("75th percentile must be non-negative");
  if (q025 < 0) errors.push("2.5th percentile must be non-negative");
  if (q975 < 0) errors.push("97.5th percentile must be non-negative");

  // Check interval ordering: q025 <= q25 <= median <= q75 <= q975
  if (q025 > q25) errors.push("2.5th percentile must be <= 25th percentile");
  if (q25 > median) errors.push("25th percentile must be <= median");
  if (median > q75) errors.push("Median must be <= 75th percentile");
  if (q75 > q975) errors.push("75th percentile must be <= 97.5th percentile");

  return {
    valid: errors.length === 0,
    errors,
  };
};

/**
 * Calculate average WIS across multiple forecasts
 * @param {Array} forecasts - Array of forecast objects
 * @param {Array} observations - Array of observed values
 * @returns {number} Average WIS
 */
export const calculateAverageWIS = (forecasts, observations) => {
  if (
    !forecasts ||
    !observations ||
    forecasts.length === 0 ||
    observations.length === 0
  ) {
    return null;
  }

  if (forecasts.length !== observations.length) {
    return null;
  }

  const wisScores = forecasts
    .map((forecast, idx) => {
      const obs = observations[idx];
      if (!Number.isFinite(obs)) return null;

      const result = calculateWIS(
        obs,
        forecast.median,
        forecast.q25,
        forecast.q75,
        forecast.q025,
        forecast.q975,
      );

      return result?.wis ?? null;
    })
    .filter((score) => score !== null);

  if (wisScores.length === 0) return null;

  return wisScores.reduce((sum, score) => sum + score, 0) / wisScores.length;
};
