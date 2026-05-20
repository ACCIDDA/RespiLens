/**
 * Tournament registry.
 *
 * To add a new tournament such as CSTE2026, add one object to
 * TOURNAMENT_REGISTRY. Routes, navigation, storage keys, enabled challenge
 * counts, and leaderboard API selection are derived from this file.
 */

const parseList = (value) => {
  if (!value) return [];
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
};

const createStorageKeys = (prefix) => ({
  participantId: `${prefix}_participant_id`,
  participantName: `${prefix}_participant_name`,
  submissions: `${prefix}_submissions`,
  lastSync: `${prefix}_last_sync`,
});

const DEFAULT_TOURNAMENT_SETTINGS = {
  challengesAlwaysActive: true,
  scoring: {
    method: "WIS",
    lowerIsBetter: true,
    intervals: [50, 95],
  },
  leaderboard: {
    updateFrequency: 30000,
    showRealNames: true,
    showScoreBreakdown: true,
    onlyShowCompleted: false,
    rankingMethod: "avgWIS",
  },
  ui: {
    chartHeight: 380,
    showIntervals: true,
    zoomedView: false,
    showProgress: true,
    progressStyle: "dots",
    medals: {
      1: "🥇",
      2: "🥈",
      3: "🥉",
    },
  },
  features: {
    allowResubmit: false,
    maskChallengeYear: false,
    maskPathogen: false,
    showOtherForecasts: false,
    showModelComparisons: false,
    enableSocialSharing: true,
  },
};

export const TOURNAMENT_REGISTRY = [
  {
    id: "allhands-challenge",
    enabled: true,
    path: "/allhands-challenge",
    navLabel: "AllhandsChallenge",
    storageKeyPrefix: "allhandschallenge",
    name: "ACCIDDA all-hands Forecastle challenge!",
    description:
      "Compete in 3 epidemic forecasting challenges and climb the leaderboard",
    // changed this in app/.env (add your constant)
    apiUrl:
      import.meta.env.VITE_ALLHANDS_CHALLENGE_API_URL ||
      "https://script.google.com/macros/s/AKfycbzGBcCDAnAgnbCcyw0zSHSTNjlfgdiO5HyNCYn5gNGxhAjypgUcoufUTx2E7X1IOXyb/exec",
    sheetId: "1ppWi9Dzp-QjgTXjCsTQEgxr8fhRDGpdZhX6cB0ssvjw",
    features: {
      maskChallengeYear: true,
      maskPathogen: false,
    },
    challenges: [
      {
        id: "ch-1",
        enabled: true,
        number: 1,
        title: "New York Influenza Forecast",
        description:
          "Predict New York flu hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "flu",
        datasetKey: "flusight",
        dataPath: "flusight",
        fileSuffix: "flu.json",
        location: "NY",
        displayName: "New York",
        target: "wk inc flu hosp",
        horizons: [1, 2, 3],
        forecastDate: "2024-11-25",
      },
      {
        id: "ch-2",
        enabled: true,
        number: 2,
        title: "North Carolina Influenza Forecast",
        description:
          "Predict North Carolina flu hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "flu",
        datasetKey: "flusight",
        dataPath: "flusight",
        fileSuffix: "flu.json",
        location: "NC",
        displayName: "North Carolina",
        target: "wk inc flu hosp",
        horizons: [1, 2, 3],
        forecastDate: "2024-12-23",
      },
      {
        id: "ch-3",
        enabled: true,
        number: 3,
        title: "Maryland RSV Forecast",
        description:
          "Predict Maryland RSV hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "rsv",
        datasetKey: "rsv",
        dataPath: "rsvforecasthub",
        fileSuffix: "rsv.json",
        location: "MD",
        displayName: "Maryland",
        target: "wk inc rsv hosp",
        horizons: [1, 2, 3],
        forecastDate: "2025-01-13",
      },
    ],
  },
  {
    id: "emily-tournament",
    enabled: true,
    path: "/emily-tournament",
    navLabel: "EmilyTournament",
    storageKeyPrefix: "emilytesttournament",
    name: "Toy tournament created by Emily",
    description:
      "Compete in 3 epidemic forecasting challenges and climb the leaderboard",
    // changed this in app/.env (add your constant)
    apiUrl:
      import.meta.env.VITE_EMILY_TOURNAMENT_API_URL ||
      "https://script.google.com/macros/s/AKfycby_pE9-KoA_bWjv9xIzNC1DF8jIrMPbQJ3I9P62RafivdQaHujnX2539tYFZtrn-nGRpw/exec",
    sheetId: "1-WMVKajvdkxRpNM7NwYaRYyVK1JdP7jbfV6JooLfguo",
    features: {
      maskChallengeYear: true,
      maskPathogen: false,
    },
    challenges: [
      {
        id: "ch-1",
        enabled: true,
        number: 1,
        title: "California Influenza Forecast",
        description:
          "Predict California flu hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "flu",
        datasetKey: "flusight",
        dataPath: "flusight",
        fileSuffix: "flu.json",
        location: "CA",
        displayName: "California",
        target: "wk inc flu hosp",
        horizons: [1, 2, 3],
        forecastDate: "2023-11-18",
      },
      {
        id: "ch-2",
        enabled: true,
        number: 2,
        title: "Nebraska Influenza Forecast",
        description:
          "Predict Nebraska flu hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "flu",
        datasetKey: "flusight",
        dataPath: "flusight",
        fileSuffix: "flu.json",
        location: "NE",
        displayName: "Nebraska",
        target: "wk inc flu hosp",
        horizons: [1, 2, 3],
        forecastDate: "2025-01-18",
      },
      {
        id: "ch-3",
        enabled: true,
        number: 3,
        title: "North Carolina COVID-19 Forecast",
        description:
          "Predict North Carolina COVID hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "covid",
        datasetKey: "covid19",
        dataPath: "covid19forecasthub",
        fileSuffix: "covid19.json",
        location: "NC",
        displayName: "North Carolina",
        target: "wk inc covid hosp",
        horizons: [1, 2, 3],
        forecastDate: "2025-09-13",
      },
    ],
  },
  {
    id: "epidemics-10",
    enabled: true,
    path: "/epidemics10",
    navLabel: "Epidemics10",
    storageKeyPrefix: "epidemics10",
    name: "Epidemics 10 Forecasting Tournament",
    description:
      "Compete in 3 epidemic forecasting challenges and climb the leaderboard",
    apiUrl:
      import.meta.env.VITE_EPIDEMICS10_TOURNAMENT_API_URL ||
      "https://script.google.com/macros/s/AKfycbwB7LnE8DSk9S7ACLs20j65iB-9ryCXAiih2FlMwpeWDDE4pLZ1zF3RQilfrm6_byLU7w/exec",
    sheetId: "17J5KWUrVuqmqqBcVJg2A-dfVdrL4LjXTvlztCDpS0g0",
    challenges: [
      {
        id: "ch-1",
        enabled: true,
        number: 1,
        title: "California Influenza Forecast",
        description:
          "Predict California flu hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "flu",
        datasetKey: "flusight",
        dataPath: "flusight",
        fileSuffix: "flu.json",
        location: "CA",
        displayName: "California",
        target: "wk inc flu hosp",
        horizons: [1, 2, 3],
        forecastDate: "2023-11-11",
      },
      {
        id: "ch-2",
        enabled: true,
        number: 2,
        title: "Colorado Influenza Forecast",
        description:
          "Predict Colorado flu hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "flu",
        datasetKey: "flusight",
        dataPath: "flusight",
        fileSuffix: "flu.json",
        location: "CO",
        displayName: "Colorado",
        target: "wk inc flu hosp",
        horizons: [1, 2, 3],
        forecastDate: "2025-01-18",
      },
      {
        id: "ch-3",
        enabled: true,
        number: 3,
        title: "North Carolina COVID-19 Forecast",
        description:
          "Predict North Carolina COVID hospitalizations for 1, 2, and 3 weeks ahead",
        dataset: "covid",
        datasetKey: "covid19",
        dataPath: "covid19forecasthub",
        fileSuffix: "covid19.json",
        location: "NC",
        displayName: "North Carolina",
        target: "wk inc covid hosp",
        horizons: [1, 2, 3],
        forecastDate: "2025-09-13",
      },
    ],
  },
];

const normalizeTournament = (tournament) => {
  const envKey = `VITE_${tournament.id
    .replace(/[^a-zA-Z0-9]/g, "_")
    .toUpperCase()}_ENABLED_CHALLENGES`;
  const configuredChallengeIds = parseList(import.meta.env[envKey]);
  const enabledChallengeIds =
    configuredChallengeIds.length > 0 ? new Set(configuredChallengeIds) : null;
  const challenges = tournament.challenges.filter((challenge) => {
    if (enabledChallengeIds) {
      return enabledChallengeIds.has(challenge.id);
    }
    return challenge.enabled !== false;
  });
  const storageKeyPrefix = tournament.storageKeyPrefix || tournament.id;

  return {
    ...DEFAULT_TOURNAMENT_SETTINGS,
    ...tournament,
    scoring: {
      ...DEFAULT_TOURNAMENT_SETTINGS.scoring,
      ...(tournament.scoring || {}),
    },
    leaderboard: {
      ...DEFAULT_TOURNAMENT_SETTINGS.leaderboard,
      ...(tournament.leaderboard || {}),
    },
    ui: {
      ...DEFAULT_TOURNAMENT_SETTINGS.ui,
      ...(tournament.ui || {}),
    },
    features: {
      ...DEFAULT_TOURNAMENT_SETTINGS.features,
      ...(tournament.features || {}),
    },
    path: tournament.path || `/${tournament.id}`,
    navLabel: tournament.navLabel || tournament.name,
    storageKeys: {
      ...createStorageKeys(storageKeyPrefix),
      ...(tournament.storageKeys || {}),
    },
    challenges,
    numChallenges: challenges.length,
  };
};

export const ENABLED_TOURNAMENTS = TOURNAMENT_REGISTRY.filter(
  (tournament) => tournament.enabled !== false,
).map(normalizeTournament);

export const shouldMaskChallengeYear = (tournamentConfig = TOURNAMENT_CONFIG) =>
  tournamentConfig.features?.maskChallengeYear === true;

export const shouldMaskPathogen = (tournamentConfig = TOURNAMENT_CONFIG) =>
  tournamentConfig.features?.maskPathogen === true;

export const getMaskedForecastDate = (
  forecastDate,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  if (!shouldMaskChallengeYear(tournamentConfig)) {
    return forecastDate;
  }

  return String(forecastDate).replace(/\b\d{4}\b/g, "xxxx");
};

export const getChallengeDatasetLabel = (
  challenge,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  if (!shouldMaskPathogen(tournamentConfig)) {
    return challenge.dataset.toUpperCase();
  }

  return "PATHOGEN";
};

export const getTournamentById = (tournamentId) =>
  ENABLED_TOURNAMENTS.find((tournament) => tournament.id === tournamentId) ||
  null;

export const getTournamentByPath = (pathname) =>
  ENABLED_TOURNAMENTS.find((tournament) =>
    pathname.startsWith(tournament.path),
  ) || null;

export const TOURNAMENT_CONFIG =
  getTournamentById(import.meta.env.VITE_DEFAULT_TOURNAMENT_ID) ||
  ENABLED_TOURNAMENTS[0] ||
  normalizeTournament({
    id: "none",
    enabled: false,
    path: "/emily_tournament",
    navLabel: "Tournament",
    name: "Tournament",
    description: "",
    apiUrl: "",
    sheetId: "",
    challenges: [],
  });

export const getChallengeById = (
  challengeId,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  return (
    tournamentConfig.challenges.find(
      (challenge) => challenge.id === challengeId,
    ) || null
  );
};

export const getChallengeByNumber = (
  challengeNumber,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  return (
    tournamentConfig.challenges.find(
      (challenge) => Number(challenge.number) === Number(challengeNumber),
    ) || null
  );
};

export const areAllChallengesCompleted = (
  submissions,
  tournamentConfig = TOURNAMENT_CONFIG,
) => {
  if (
    !submissions ||
    submissions.length === 0 ||
    tournamentConfig.numChallenges === 0
  ) {
    return false;
  }

  const challengeIdByNumber = new Map(
    tournamentConfig.challenges.map((challenge) => [
      Number(challenge.number),
      challenge.id,
    ]),
  );
  const completedChallenges = new Set(
    submissions
      .map(
        (sub) =>
          sub.challengeId || challengeIdByNumber.get(Number(sub.challengeNum)),
      )
      .filter(Boolean),
  );

  return tournamentConfig.challenges.every((challenge) =>
    completedChallenges.has(challenge.id),
  );
};
