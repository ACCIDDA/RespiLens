/**
 * Tournament API
 * Handles communication with Google Sheets backend via Apps Script
 */

import { TOURNAMENT_CONFIG, getChallengeByNumber } from "../config";

/**
 * Make a GET request to the tournament API
 * @param {string} action - API action
 * @param {Object} params - Query parameters
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {Promise<Object>} Response data
 */
const apiGet = async (
  action,
  params = {},
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  const apiUrl = tournamentConfig.apiUrl;

  if (!apiUrl) {
    throw new Error(
      `Tournament API URL not configured for ${tournamentConfig.id}`,
    );
  }

  const url = new URL(apiUrl);
  url.searchParams.append("action", action);

  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.append(key, value);
  });

  try {
    // No headers needed for GET - avoids CORS preflight
    const response = await fetch(url.toString(), {
      method: "GET",
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || "Unknown API error");
    }

    return data;
  } catch (error) {
    console.error("Tournament API GET error:", error);
    throw error;
  }
};

/**
 * Make a POST request to the tournament API
 * @param {Object} payload - Request payload
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {Promise<Object>} Response data
 */
const apiPost = async (payload, tournamentConfig = TOURNAMENT_CONFIG) => {
  const apiUrl = tournamentConfig.apiUrl;

  if (!apiUrl) {
    throw new Error(
      `Tournament API URL not configured for ${tournamentConfig.id}`,
    );
  }

  try {
    // Use text/plain to avoid CORS preflight
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "text/plain",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || "Unknown API error");
    }

    return data;
  } catch (error) {
    console.error("Tournament API POST error:", error);
    throw error;
  }
};

/**
 * Register a new participant or login existing participant
 * @param {string} name - Participant's recognizable name
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {Promise<Object>} Participant data {participantId, message}
 */
export const registerParticipant = async (
  name,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  if (!name) {
    throw new Error("Name is required");
  }

  const data = await apiPost(
    {
      action: "register",
      tournamentId: tournamentConfig.id,
      name: name.trim(),
    },
    tournamentConfig,
  );

  // Store in localStorage
  localStorage.setItem(
    tournamentConfig.storageKeys.participantId,
    data.participantId,
  );
  localStorage.setItem(
    tournamentConfig.storageKeys.participantName,
    name.trim(),
  );

  return data;
};

/**
 * Submit a forecast for a challenge
 * @param {string} participantId - Participant ID
 * @param {number} challengeNum - Stable enabled challenge number
 * @param {Array|Object} forecasts - Array of forecast entries (one per horizon) or single forecast object for backward compatibility
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {Promise<Object>} Submission data {submissionId, message}
 */
export const submitForecast = async (
  participantId,
  challengeNum,
  forecasts,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  if (!participantId) {
    throw new Error("Participant ID is required");
  }

  const challenge = getChallengeByNumber(
    Number(challengeNum),
    tournamentConfig,
  );
  if (!challengeNum || !challenge) {
    const enabledChallengeNumbers = tournamentConfig.challenges
      .map((enabledChallenge) => enabledChallenge.number)
      .join(", ");
    throw new Error(
      enabledChallengeNumbers
        ? `Challenge number must be one of: ${enabledChallengeNumbers}`
        : "No tournament challenges are enabled",
    );
  }

  if (!forecasts) {
    throw new Error("Forecast data is required");
  }

  // Convert to array if single forecast object (backward compatibility)
  const forecastArray = Array.isArray(forecasts) ? forecasts : [forecasts];

  // Validate that all forecasts have required fields
  for (const forecast of forecastArray) {
    if (
      !forecast ||
      forecast.median === null ||
      forecast.median === undefined
    ) {
      throw new Error("Each forecast must have a median value");
    }
  }

  // Format forecasts for submission
  const formattedForecasts = forecastArray.map((f) => ({
    horizon: f.horizon || 1,
    median: f.median,
    q25: f.q25 || f.lower50,
    q75: f.q75 || f.upper50,
    q025: f.q025 || f.lower95,
    q975: f.q975 || f.upper95,
  }));

  const data = await apiPost(
    {
      action: "submitForecast",
      tournamentId: tournamentConfig.id,
      participantId,
      challengeNum,
      challengeId: challenge.id,
      forecasts: formattedForecasts,
      // No WIS - scoring is done on frontend
    },
    tournamentConfig,
  );

  // Update last sync time
  localStorage.setItem(
    tournamentConfig.storageKeys.lastSync,
    new Date().toISOString(),
  );

  return data;
};

/**
 * Get the leaderboard
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {Promise<Array>} Leaderboard data
 */
export const getLeaderboard = async (tournamentConfig = TOURNAMENT_CONFIG) => {
  const data = await apiGet(
    "getLeaderboard",
    { tournamentId: tournamentConfig.id },
    tournamentConfig,
  );
  return data.leaderboard || [];
};

/**
 * Get participant data including submissions
 * @param {string} participantId - Participant ID
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {Promise<Object>} Participant data {participant, submissions}
 */
export const getParticipant = async (
  participantId,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  if (!participantId) {
    throw new Error("Participant ID is required");
  }

  const data = await apiGet(
    "getParticipant",
    { participantId, tournamentId: tournamentConfig.id },
    tournamentConfig,
  );

  return {
    participant: data.participant,
    submissions: data.submissions || [],
  };
};

/**
 * Get participant ID from localStorage
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {string|null} Participant ID or null if not found
 */
export const getStoredParticipantId = (
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  return localStorage.getItem(tournamentConfig.storageKeys.participantId);
};

/**
 * Get participant name from localStorage
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {string|null} Participant name or null if not found
 */
export const getStoredParticipantName = (
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  return localStorage.getItem(tournamentConfig.storageKeys.participantName);
};

/**
 * Clear participant data from localStorage (logout)
 * @param {Object} tournamentConfig - Tournament configuration
 */
export const clearParticipantData = (tournamentConfig = TOURNAMENT_CONFIG) => {
  localStorage.removeItem(tournamentConfig.storageKeys.participantId);
  localStorage.removeItem(tournamentConfig.storageKeys.participantName);
  localStorage.removeItem(tournamentConfig.storageKeys.submissions);
  localStorage.removeItem(tournamentConfig.storageKeys.lastSync);
};

/**
 * Check if participant is registered
 * @param {Object} tournamentConfig - Tournament configuration
 * @returns {boolean} True if participant ID is stored
 */
export const isParticipantRegistered = (
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  return !!getStoredParticipantId(tournamentConfig);
};
