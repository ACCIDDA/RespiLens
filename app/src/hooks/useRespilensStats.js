import { useMemo } from "react";
import { getForecastleGames } from "../utils/respilensStorage";
import { calculateRelativeWIS, calculateWIS } from "../utils/forecastleScoring";

/**
 * Calculate interval coverage for a single game
 * @param {Array} userForecasts - Array of forecast entries with intervals
 * @param {Array} groundTruth - Array of actual observed values
 * @returns {Object} Coverage counts for 95% and 50% intervals
 */
function calculateIntervalCoverage(userForecasts, groundTruth) {
  let coverage95 = 0;
  let coverage50 = 0;
  let validHorizons = 0;

  userForecasts.forEach((forecast, idx) => {
    const truth = groundTruth[idx];

    // Only count horizons with valid ground truth and intervals
    if (
      Number.isFinite(truth) &&
      Number.isFinite(forecast.lower95) &&
      Number.isFinite(forecast.upper95) &&
      Number.isFinite(forecast.lower50) &&
      Number.isFinite(forecast.upper50)
    ) {
      validHorizons += 1;

      // Check if ground truth falls within intervals
      if (truth >= forecast.lower95 && truth <= forecast.upper95) {
        coverage95 += 1;
      }
      if (truth >= forecast.lower50 && truth <= forecast.upper50) {
        coverage50 += 1;
      }
    }
  });

  return {
    coverage95,
    coverage50,
    validHorizons,
  };
}

/**
 * Compute statistics for a single game
 * @param {Object} game - Game object from storage
 * @returns {Object} Computed statistics for the game
 */
function computeGameStats(game) {
  // Calculate WIS for all horizons
  let sumWIS = 0;
  let sumDispersion = 0;
  let sumUnderprediction = 0;
  let sumOverprediction = 0;
  let validCount = 0;

  game.userForecasts.forEach((forecast, idx) => {
    const observed = game.groundTruth[idx];

    if (Number.isFinite(observed)) {
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
        validCount += 1;
      }
    }
  });

  const wis = validCount > 0 ? sumWIS / validCount : null;
  const relativeWis =
    game.userRelativeWIS ??
    calculateRelativeWIS(game.userWIS ?? wis, game.baselineWIS ?? null);
  const dispersion = validCount > 0 ? sumDispersion / validCount : null;
  const underprediction =
    validCount > 0 ? sumUnderprediction / validCount : null;
  const overprediction = validCount > 0 ? sumOverprediction / validCount : null;

  // Calculate interval coverage
  const coverage = calculateIntervalCoverage(
    game.userForecasts,
    game.groundTruth,
  );

  return {
    id: game.id,
    playedAt: game.playedAt,
    challengeDate: game.challengeDate,
    dataset: game.dataset,
    location: game.location,
    target: game.target,
    // User WIS (computed or from storage)
    wis: relativeWis,
    rawWIS: game.userWIS ?? wis,
    relativeWIS: relativeWis,
    dispersion: game.userDispersion ?? dispersion,
    underprediction: game.userUnderprediction ?? underprediction,
    overprediction: game.userOverprediction ?? overprediction,
    coverage95: coverage.coverage95,
    coverage50: coverage.coverage50,
    validHorizons: coverage.validHorizons,
    totalHorizons: game.userForecasts.length,
    // Ranking information
    userRank: game.userRank || null,
    totalModels: game.totalModels || null,
    ensembleRank: game.ensembleRank || null,
    baselineRank: game.baselineRank || null,
    // Ensemble scores
    ensembleWIS:
      game.ensembleRelativeWIS ??
      calculateRelativeWIS(game.ensembleWIS ?? null, game.baselineWIS ?? null),
    ensembleRawWIS: game.ensembleWIS ?? null,
    ensembleDispersion: game.ensembleDispersion ?? null,
    ensembleUnderprediction: game.ensembleUnderprediction ?? null,
    ensembleOverprediction: game.ensembleOverprediction ?? null,
    // Baseline scores
    baselineWIS:
      game.baselineRelativeWIS ??
      calculateRelativeWIS(game.baselineWIS ?? null, game.baselineWIS ?? null),
    baselineRawWIS: game.baselineWIS ?? null,
    baselineDispersion: game.baselineDispersion ?? null,
    baselineUnderprediction: game.baselineUnderprediction ?? null,
    baselineOverprediction: game.baselineOverprediction ?? null,
    // Raw data needed for per-pathogen coverage calculation
    userForecasts: game.userForecasts,
    groundTruth: game.groundTruth,
    horizonDates: game.horizonDates,
  };
}

/**
 * Calculate consecutive day streak
 * @param {Array} games - Array of game stats sorted by challengeDate
 * @returns {Object} Current streak and max streak
 */
function calculateStreaks(games) {
  if (games.length === 0) {
    return { currentStreak: 0, maxStreak: 0 };
  }

  // Get unique challenge dates and sort (most recent first)
  const uniqueDates = [...new Set(games.map((g) => g.challengeDate))].sort(
    (a, b) => b.localeCompare(a),
  ); // ISO date string comparison

  if (uniqueDates.length === 0) {
    return { currentStreak: 0, maxStreak: 0 };
  }

  // Get today's date in UTC (YYYY-MM-DD format)
  const today = new Date();
  const todayUTC = new Date(
    Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()),
  );
  const todayStr = todayUTC.toISOString().slice(0, 10);

  // Calculate days difference between two YYYY-MM-DD date strings
  const daysDiff = (date1Str, date2Str) => {
    const d1 = new Date(date1Str + "T00:00:00Z");
    const d2 = new Date(date2Str + "T00:00:00Z");
    return Math.round((d1 - d2) / (1000 * 60 * 60 * 24));
  };

  // Check if most recent game is today or yesterday
  const mostRecent = uniqueDates[0];
  const daysSinceLastGame = daysDiff(todayStr, mostRecent);

  let currentStreak = 0;
  if (daysSinceLastGame <= 1) {
    // Current streak is alive, count from most recent backward
    currentStreak = 1;
    for (let i = 1; i < uniqueDates.length; i += 1) {
      const diff = daysDiff(uniqueDates[i - 1], uniqueDates[i]);
      if (diff === 1) {
        currentStreak += 1;
      } else {
        break;
      }
    }
  }

  // Calculate max streak by scanning all dates
  let maxStreak = 0;
  let tempStreak = 1;

  for (let i = 1; i < uniqueDates.length; i += 1) {
    const diff = daysDiff(uniqueDates[i - 1], uniqueDates[i]);
    if (diff === 1) {
      tempStreak += 1;
    } else {
      maxStreak = Math.max(maxStreak, tempStreak);
      tempStreak = 1;
    }
  }
  maxStreak = Math.max(maxStreak, tempStreak, currentStreak);

  return { currentStreak, maxStreak };
}

/**
 * React hook to compute RespiLens statistics from stored game data
 * @param {number} [refreshTrigger] - Optional trigger to force recalculation
 * @returns {Object} Computed statistics and game history
 */
export function useRespilensStats(refreshTrigger) {
  const stats = useMemo(() => {
    const games = getForecastleGames();

    if (games.length === 0) {
      return {
        gamesPlayed: 0,
        averageWIS: null,
        bestWIS: null,
        worstWIS: null,
        averageRawWIS: null,
        averageDispersion: null,
        averageUnderprediction: null,
        averageOverprediction: null,
        coverage95Percent: null,
        coverage50Percent: null,
        currentStreak: 0,
        maxStreak: 0,
        averageRankVsEnsemble: null,
        averagePercentDiffEnsemble: null,
        gameHistory: [],
      };
    }

    // Compute stats for each game
    const gameStats = games.map(computeGameStats);

    // Filter games with valid WIS
    const validGames = gameStats.filter((g) => Number.isFinite(g.wis));

    // Aggregate statistics
    const totalWIS = validGames.reduce((sum, g) => sum + g.wis, 0);
    const averageWIS =
      validGames.length > 0 ? totalWIS / validGames.length : null;
    const totalRawWIS = validGames.reduce((sum, g) => sum + (g.rawWIS || 0), 0);
    const averageRawWIS =
      validGames.length > 0 ? totalRawWIS / validGames.length : null;
    const bestWIS =
      validGames.length > 0 ? Math.min(...validGames.map((g) => g.wis)) : null;
    const worstWIS =
      validGames.length > 0 ? Math.max(...validGames.map((g) => g.wis)) : null;

    // Aggregate WIS components
    const totalDispersion = validGames.reduce(
      (sum, g) => sum + (g.dispersion || 0),
      0,
    );
    const totalUnderprediction = validGames.reduce(
      (sum, g) => sum + (g.underprediction || 0),
      0,
    );
    const totalOverprediction = validGames.reduce(
      (sum, g) => sum + (g.overprediction || 0),
      0,
    );
    const averageDispersion =
      validGames.length > 0 ? totalDispersion / validGames.length : null;
    const averageUnderprediction =
      validGames.length > 0 ? totalUnderprediction / validGames.length : null;
    const averageOverprediction =
      validGames.length > 0 ? totalOverprediction / validGames.length : null;

    // Aggregate interval coverage
    const totalCoverage95 = gameStats.reduce((sum, g) => sum + g.coverage95, 0);
    const totalCoverage50 = gameStats.reduce((sum, g) => sum + g.coverage50, 0);
    const totalValidHorizons = gameStats.reduce(
      (sum, g) => sum + g.validHorizons,
      0,
    );

    const coverage95Percent =
      totalValidHorizons > 0
        ? (totalCoverage95 / totalValidHorizons) * 100
        : null;
    const coverage50Percent =
      totalValidHorizons > 0
        ? (totalCoverage50 / totalValidHorizons) * 100
        : null;

    // Calculate streaks
    const streaks = calculateStreaks(gameStats);

    // Calculate rank comparison vs ensemble
    const gamesWithRankData = gameStats.filter(
      (g) => Number.isFinite(g.userRank) && Number.isFinite(g.ensembleRank),
    );
    const totalRankDiff = gamesWithRankData.reduce(
      (sum, g) => sum + (g.ensembleRank - g.userRank),
      0,
    );
    const averageRankVsEnsemble =
      gamesWithRankData.length > 0
        ? totalRankDiff / gamesWithRankData.length
        : null;

    // Calculate % difference in WIS vs ensemble
    const gamesWithWISComparison = gameStats.filter(
      (g) =>
        Number.isFinite(g.wis) &&
        Number.isFinite(g.ensembleWIS) &&
        g.ensembleWIS > 0,
    );
    const totalPercentDiff = gamesWithWISComparison.reduce((sum, g) => {
      const percentDiff = ((g.wis - g.ensembleWIS) / g.ensembleWIS) * 100;
      return sum + percentDiff;
    }, 0);
    const averagePercentDiffEnsemble =
      gamesWithWISComparison.length > 0
        ? totalPercentDiff / gamesWithWISComparison.length
        : null;

    return {
      gamesPlayed: games.length,
      averageWIS,
      bestWIS,
      worstWIS,
      averageRawWIS,
      averageDispersion,
      averageUnderprediction,
      averageOverprediction,
      coverage95Percent,
      coverage50Percent,
      currentStreak: streaks.currentStreak,
      maxStreak: streaks.maxStreak,
      averageRankVsEnsemble,
      averagePercentDiffEnsemble,
      gameHistory: gameStats,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]); // Recalculates when refreshTrigger changes

  return stats;
}
