const STORAGE_KEY = "respilens_forecastle_games";

/**
 * Check if localStorage is available
 * @returns {boolean} True if localStorage is available
 */
function isLocalStorageAvailable() {
  try {
    const test = "__storage_test__";
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate a game object has the required structure
 * @param {Object} game - Game object to validate
 * @returns {boolean} True if valid
 */
function isValidGame(game) {
  return (
    game &&
    typeof game === "object" &&
    typeof game.id === "string" &&
    typeof game.challengeDate === "string" &&
    typeof game.dataset === "string" &&
    typeof game.location === "string" &&
    Array.isArray(game.userForecasts) &&
    Array.isArray(game.groundTruth) &&
    Array.isArray(game.horizonDates)
  );
}

/**
 * Round a number to specified decimal places, handling null/undefined
 * @param {number|null|undefined} value - Value to round
 * @param {number} decimals - Number of decimal places
 * @returns {number|null} Rounded value or null
 */
function roundToDecimals(value, decimals) {
  if (!Number.isFinite(value)) {
    return null;
  }
  const multiplier = Math.pow(10, decimals);
  return Math.round(value * multiplier) / multiplier;
}

/**
 * Get all stored Forecastle games
 * @returns {Array} Array of game objects
 */
export function getForecastleGames() {
  if (!isLocalStorageAvailable()) {
    console.warn("localStorage is not available");
    return [];
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];

    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) {
      console.error("Stored data is not an array, resetting");
      return [];
    }

    // Filter out invalid games
    return parsed.filter((game) => isValidGame(game));
  } catch (error) {
    console.error("Error reading Forecastle games from localStorage:", error);
    return [];
  }
}

/**
 * Save a completed Forecastle game
 * @param {Object} gameData - The game data to save
 * @param {string} gameData.challengeDate - ISO date string of the challenge
 * @param {string} gameData.forecastDate - ISO date string of the forecast date
 * @param {string} gameData.dataset - Dataset name (e.g., 'flusight')
 * @param {string} gameData.location - Location code (e.g., 'NY')
 * @param {string} gameData.target - Target name (e.g., 'influenza_hospitalizations')
 * @param {Array} gameData.userForecasts - Array of forecast entries with median and intervals
 * @param {Array} gameData.groundTruth - Array of actual observed values
 * @param {Array} gameData.horizonDates - Array of ISO date strings for each horizon
 * @returns {boolean} Success status
 */
export function saveForecastleGame(gameData) {
  if (!isLocalStorageAvailable()) {
    console.error("localStorage is not available");
    return false;
  }

  try {
    const games = getForecastleGames();

    // Create unique ID for this game (include forecastDate and target to avoid collisions)
    const id = `${gameData.challengeDate}_${gameData.forecastDate}_${gameData.dataset}_${gameData.location}_${gameData.target}`;

    // Check if this game already exists (replay protection)
    const existingIndex = games.findIndex((g) => g.id === id);

    const gameEntry = {
      id,
      playedAt: new Date().toISOString(),
      challengeDate: gameData.challengeDate,
      forecastDate: gameData.forecastDate,
      dataset: gameData.dataset,
      location: gameData.location,
      target: gameData.target,
      userForecasts: gameData.userForecasts.map((f) => ({
        horizon: f.horizon,
        median: f.median,
        lower95: f.lower95,
        upper95: f.upper95,
        lower50: f.lower50,
        upper50: f.upper50,
      })),
      groundTruth: gameData.groundTruth,
      horizonDates: gameData.horizonDates,
      // Ranking information
      userRank: gameData.userRank,
      totalModels: gameData.totalModels,
      ensembleRank: gameData.ensembleRank,
      baselineRank: gameData.baselineRank,
      // User scores (rounded to 3 decimal places)
      userWIS: roundToDecimals(gameData.userWIS, 3),
      userRelativeWIS: roundToDecimals(gameData.userRelativeWIS, 3),
      userDispersion: roundToDecimals(gameData.userDispersion, 3),
      userUnderprediction: roundToDecimals(gameData.userUnderprediction, 3),
      userOverprediction: roundToDecimals(gameData.userOverprediction, 3),
      // Ensemble scores (rounded to 3 decimal places)
      ensembleWIS: roundToDecimals(gameData.ensembleWIS, 3),
      ensembleRelativeWIS: roundToDecimals(gameData.ensembleRelativeWIS, 3),
      ensembleDispersion: roundToDecimals(gameData.ensembleDispersion, 3),
      ensembleUnderprediction: roundToDecimals(
        gameData.ensembleUnderprediction,
        3,
      ),
      ensembleOverprediction: roundToDecimals(
        gameData.ensembleOverprediction,
        3,
      ),
      // Baseline scores (rounded to 3 decimal places)
      baselineWIS: roundToDecimals(gameData.baselineWIS, 3),
      baselineRelativeWIS: roundToDecimals(gameData.baselineRelativeWIS, 3),
      baselineDispersion: roundToDecimals(gameData.baselineDispersion, 3),
      baselineUnderprediction: roundToDecimals(
        gameData.baselineUnderprediction,
        3,
      ),
      baselineOverprediction: roundToDecimals(
        gameData.baselineOverprediction,
        3,
      ),
    };

    if (existingIndex >= 0) {
      // Replace existing game (user replayed the same challenge)
      games[existingIndex] = gameEntry;
    } else {
      // Add new game
      games.push(gameEntry);
    }

    // Sort by playedAt date (newest first)
    games.sort((a, b) => new Date(b.playedAt) - new Date(a.playedAt));

    localStorage.setItem(STORAGE_KEY, JSON.stringify(games));
    return true;
  } catch (error) {
    // Check for quota exceeded error
    if (error.name === "QuotaExceededError" || error.code === 22) {
      console.error(
        "localStorage quota exceeded. Consider clearing old games.",
      );
      throw new Error(
        "Storage quota exceeded. Please clear some game history.",
      );
    }
    console.error("Error saving Forecastle game to localStorage:", error);
    return false;
  }
}

/**
 * Get a specific game by ID
 * @param {string} id - Game ID (format: "challengeDate_forecastDate_dataset_location_target")
 * @returns {Object|null} Game object or null if not found
 */
export function getForecastleGame(id) {
  const games = getForecastleGames();
  return games.find((g) => g.id === id) || null;
}

/**
 * Delete a specific game
 * @param {string} id - Game ID to delete
 * @returns {boolean} Success status
 */
export function deleteForecastleGame(id) {
  if (!isLocalStorageAvailable()) {
    console.error("localStorage is not available");
    return false;
  }

  try {
    const games = getForecastleGames();
    const filtered = games.filter((g) => g.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    return true;
  } catch (error) {
    console.error("Error deleting Forecastle game:", error);
    return false;
  }
}

/**
 * Clear all Forecastle game data
 * @returns {boolean} Success status
 */
export function clearForecastleGames() {
  if (!isLocalStorageAvailable()) {
    console.error("localStorage is not available");
    return false;
  }

  try {
    localStorage.removeItem(STORAGE_KEY);
    return true;
  } catch (error) {
    console.error("Error clearing Forecastle games:", error);
    return false;
  }
}

/**
 * Export all game data as JSON
 * @returns {string} JSON string of all games
 */
export function exportForecastleData() {
  const games = getForecastleGames();
  return JSON.stringify(games, null, 2);
}

/**
 * Import game data from JSON
 * @param {string} jsonData - JSON string of games to import
 * @returns {boolean} Success status
 */
export function importForecastleData(jsonData) {
  if (!isLocalStorageAvailable()) {
    console.error("localStorage is not available");
    return false;
  }

  try {
    const imported = JSON.parse(jsonData);
    if (!Array.isArray(imported)) {
      throw new Error("Invalid data format: expected array");
    }

    // Validate all games
    const validGames = imported.filter((game) => {
      const valid = isValidGame(game);
      if (!valid) {
        console.warn("Skipping invalid game during import:", game);
      }
      return valid;
    });

    if (validGames.length === 0) {
      throw new Error("No valid games found in import data");
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(validGames));
    return true;
  } catch (error) {
    console.error("Error importing Forecastle data:", error);
    throw error; // Re-throw so caller can handle
  }
}
