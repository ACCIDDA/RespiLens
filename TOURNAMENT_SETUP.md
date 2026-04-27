# Epidemics 10 Tournament - Setup Guide

## Overview

The Epidemics 10 Tournament is a multi-participant forecasting competition built into RespiLens. Participants compete by making predictions for 3 different epidemic forecasting challenges, and their accuracy is tracked on a live leaderboard.

**Just like Forecastle**: Each challenge works exactly like a Forecastle game with 3 horizons (1, 2, 3 weeks ahead). After submitting, participants see their score compared to models and other participants.

## Architecture

- **Frontend**: React components in `/app/src/components/tournament/`
- **Backend**: Google Apps Script (serverless)
- **Database**: Google Sheets
- **Storage**: LocalStorage for participant session, Google Sheets for submissions
- **Scoring**: Weighted Interval Score (WIS) - same as Forecastle

## Quick Start

1. **Set up Google Sheets** (see structure below)
2. **Deploy Google Apps Script** (see Apps Script code below)
3. **Configure API URL** in `/app/.env`
4. **Configure the tournament** in `/app/src/config/tournament.js`
5. **Navigate to** the tournament `path` from that config, such as `/epidemics10`

## Tournament Configuration

Tournament instances live in `/app/src/config/tournament.js`.

To add a new event like CSTE2026, add one object to `TOURNAMENT_REGISTRY` with:

- a stable `id`
- `enabled: true`
- `path`, such as `/cste2026`
- `navLabel`
- `name` and `description`
- `apiUrl` for that event's Apps Script backend
- `sheetId` for documentation
- `storageKeyPrefix`, such as `cste2026`
- a `challenges` array

Each challenge object needs:

- a stable `id`
- a stable numeric `number`
- `dataset`, `datasetKey`, `dataPath`, `fileSuffix`, `location`, `target`
- `forecastDate`
- `horizons`
- `enabled: true`

Routes, sidebar navigation, localStorage keys, enabled challenge counts, and leaderboard API selection are derived from this registry. Set a tournament or challenge to `enabled: false` to keep it in code but hidden.

The Google Sheets backend stores `challenge_id` as the primary challenge key. Keep `challenge_num` only as a legacy/display field so old rows can be migrated safely. For a separate leaderboard, use a separate spreadsheet and Apps Script URL for the tournament entry.

---

## Google Sheets Structure

Your sheet should have the following tabs:

### 1. **Participants** (Sheet 1)

```
participant_id | name          | joined_at
---------------|---------------|-------------------
uuid-1         | DrForecast    | 2025-11-18 10:00
uuid-2         | TeamBlue      | 2025-11-18 10:15
```

### 2. **Submissions** (Sheet 2)

**Simple format - just raw forecasts (scoring done on frontend):**

```
submission_id | participant_id | challenge_id | challenge_num | horizon | median | q25  | q75  | q025 | q975 | submitted_at
-------------|----------------|--------------|---------------|---------|--------|------|------|------|------|-------------------
sub-1        | uuid-1         | ch-1         | 1             | 1       | 1500   | 1200 | 1800 | 900  | 2100 | 2025-11-18 10:30
sub-1        | uuid-1         | ch-1         | 1             | 2       | 1600   | 1300 | 1900 | 1000 | 2200 | 2025-11-18 10:30
sub-1        | uuid-1         | ch-1         | 1             | 3       | 1700   | 1400 | 2000 | 1100 | 2300 | 2025-11-18 10:30
sub-2        | uuid-1         | ch-2         | 2             | 1       | 2300   | 2000 | 2600 | 1800 | 2900 | 2025-11-18 11:00
```

**Note**: `challenge_id` is the primary key. `challenge_num` is retained for display/backward compatibility. No WIS column - all scoring is calculated on the frontend.

---

## Google Apps Script Backend

### Setup Instructions:

1. Open your Google Sheet
2. Go to **Extensions > Apps Script**
3. Delete the default code
4. Paste the code below
5. Click **Deploy > New Deployment**
6. Select **Web App**
7. Set:
   - Execute as: **Me**
   - Who has access: **Anyone**
8. Click **Deploy** and copy the Web App URL

### Apps Script Code:

```javascript
// Paste this in Google Apps Script Editor

const SHEET_ID = "17J5KWUrVuqmqqBcVJg2A-dfVdrL4LjXTvlztCDpS0g0";

function doGet(e) {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const action = e.parameter.action;

  let result;
  if (action === "getLeaderboard") {
    result = getLeaderboard(ss, e.parameter.tournamentId);
  } else if (action === "getParticipant") {
    result = getParticipant(
      ss,
      e.parameter.participantId,
      e.parameter.tournamentId,
    );
  } else {
    result = { error: "Invalid action" };
  }

  return ContentService.createTextOutput(JSON.stringify(result)).setMimeType(
    ContentService.MimeType.JSON,
  );
}

function doPost(e) {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const data = JSON.parse(e.postData.contents);
  const action = data.action;

  let result;
  if (action === "register") {
    result = registerParticipant(ss, data);
  } else if (action === "submitForecast") {
    result = submitForecast(ss, data);
  } else {
    result = { error: "Invalid action" };
  }

  return ContentService.createTextOutput(JSON.stringify(result)).setMimeType(
    ContentService.MimeType.JSON,
  );
}

function registerParticipant(ss, data) {
  const sheet = ss.getSheetByName("Participants");
  const participantId = Utilities.getUuid();
  const timestamp = new Date().toISOString();

  // Check if name already exists
  const existingData = sheet.getDataRange().getValues();
  for (let i = 1; i < existingData.length; i++) {
    if (existingData[i][1] === data.name) {
      return {
        success: true,
        participantId: existingData[i][0],
        message: "Welcome back!",
      };
    }
  }

  // Add new participant
  sheet.appendRow([participantId, data.name, timestamp]);

  return {
    success: true,
    participantId: participantId,
    message: "Registration successful!",
  };
}

function submitForecast(ss, data) {
  const submissionsSheet = ss.getSheetByName("Submissions");
  const submissionId = Utilities.getUuid();
  const timestamp = new Date().toISOString();
  const challengeId = data.challengeId || `ch-${data.challengeNum}`;

  // Handle multiple horizons (new format)
  const forecasts = data.forecasts || [
    {
      horizon: 1,
      median: data.median,
      q25: data.q25,
      q75: data.q75,
      q025: data.q025,
      q975: data.q975,
    },
  ];

  // Delete existing submissions for this challenge
  const existingData = submissionsSheet.getDataRange().getValues();
  const rowsToDelete = [];
  for (let i = existingData.length - 1; i >= 1; i--) {
    const rowChallengeId = existingData[i][2] || `ch-${existingData[i][3]}`;
    if (
      existingData[i][1] === data.participantId &&
      rowChallengeId === challengeId
    ) {
      rowsToDelete.push(i + 1); // +1 because sheet rows are 1-indexed
    }
  }

  // Delete rows in reverse order to maintain indices
  for (const rowIndex of rowsToDelete) {
    submissionsSheet.deleteRow(rowIndex);
  }

  // Add new submissions (one row per horizon)
  forecasts.forEach((forecast) => {
    submissionsSheet.appendRow([
      submissionId,
      data.participantId,
      challengeId,
      data.challengeNum,
      forecast.horizon,
      forecast.median,
      forecast.q25,
      forecast.q75,
      forecast.q025,
      forecast.q975,
      timestamp,
    ]);
  });

  return {
    success: true,
    submissionId: submissionId,
    message: "Forecast submitted!",
  };
}

function getLeaderboard(ss, tournamentId) {
  const participantsSheet = ss.getSheetByName("Participants");
  const submissionsSheet = ss.getSheetByName("Submissions");

  // Get all participants
  const participantData = participantsSheet.getDataRange().getValues();
  const participants = [];

  for (let i = 1; i < participantData.length; i++) {
    participants.push({
      participantId: participantData[i][0],
      name: participantData[i][1],
    });
  }

  // Get all submissions
  const submissionData = submissionsSheet.getDataRange().getValues();

  // Group submissions by participant and challenge
  const participantSubmissions = {};

  for (let i = 1; i < submissionData.length; i++) {
    const participantId = submissionData[i][1];
    const challengeId = submissionData[i][2] || `ch-${submissionData[i][3]}`;
    const challengeNum = submissionData[i][3];
    const horizon = submissionData[i][4];

    if (!participantSubmissions[participantId]) {
      participantSubmissions[participantId] = {};
    }

    if (!participantSubmissions[participantId][challengeId]) {
      participantSubmissions[participantId][challengeId] = [];
    }

    participantSubmissions[participantId][challengeId].push({
      challengeId: challengeId,
      challengeNum: challengeNum,
      horizon: horizon,
      median: submissionData[i][5],
      q25: submissionData[i][6],
      q75: submissionData[i][7],
      q025: submissionData[i][8],
      q975: submissionData[i][9],
    });
  }

  // Build leaderboard data (frontend will calculate WIS)
  const leaderboard = participants.map((participant) => {
    const submissions = participantSubmissions[participant.participantId] || {};
    const completed = Object.keys(submissions).length;

    return {
      participantId: participant.participantId,
      name: participant.name,
      completed: completed,
      submissions: submissions, // Send raw submissions to frontend for scoring
    };
  });

  return {
    success: true,
    leaderboard: leaderboard,
  };
}

function getParticipant(ss, participantId, tournamentId) {
  const participantsSheet = ss.getSheetByName("Participants");
  const submissionsSheet = ss.getSheetByName("Submissions");

  // Get participant info
  const participantData = participantsSheet.getDataRange().getValues();
  let participant = null;
  for (let i = 1; i < participantData.length; i++) {
    if (participantData[i][0] === participantId) {
      participant = {
        participantId: participantData[i][0],
        name: participantData[i][1],
      };
      break;
    }
  }

  if (!participant) {
    return {
      success: false,
      error: "Participant not found",
    };
  }

  // Get submissions (grouped by challenge)
  const submissionData = submissionsSheet.getDataRange().getValues();
  const submissionsByChallenge = {};

  for (let i = 1; i < submissionData.length; i++) {
    if (submissionData[i][1] === participantId) {
      const challengeId = submissionData[i][2] || `ch-${submissionData[i][3]}`;
      const challengeNum = submissionData[i][3];
      const horizon = submissionData[i][4];

      if (!submissionsByChallenge[challengeId]) {
        submissionsByChallenge[challengeId] = {
          challengeId: challengeId,
          challengeNum: challengeNum,
          forecasts: [],
          submittedAt: submissionData[i][10],
        };
      }

      submissionsByChallenge[challengeId].forecasts.push({
        horizon: horizon,
        median: submissionData[i][5],
        q25: submissionData[i][6],
        q75: submissionData[i][7],
        q025: submissionData[i][8],
        q975: submissionData[i][9],
      });
    }
  }

  // Convert to array and sort forecasts by horizon
  const submissions = Object.values(submissionsByChallenge).map((sub) => ({
    ...sub,
    forecasts: sub.forecasts.sort((a, b) => a.horizon - b.horizon),
  }));

  return {
    success: true,
    participant: participant,
    submissions: submissions,
  };
}
```

---

## Configuration in React App

After deploying the Apps Script, add the Web App URL to your React config:

```javascript
// /app/src/config/tournament.js
export const TOURNAMENT_CONFIG = {
  apiUrl: "YOUR_APPS_SCRIPT_WEB_APP_URL_HERE",
  sheetId: "17J5KWUrVuqmqqBcVJg2A-dfVdrL4LjXTvlztCDpS0g0",
  // ... rest of config
};
```

---

## Testing

1. Test registration: `POST {apiUrl}?action=register` with `{name: 'DrForecast'}`
2. Test submission: `POST {apiUrl}?action=submitForecast` with forecast data
3. Test leaderboard: `GET {apiUrl}?action=getLeaderboard`

---

## Architecture: Frontend Scoring

✅ **All scoring is done on the frontend** using the same code as Forecastle

### How It Works

**Google Sheets = Dumb Data Store**

- Only stores raw forecast submissions (median, q25, q75, q025, q975)
- No WIS calculations
- No ranking logic
- Just stores and retrieves data

**Frontend = Smart Scoring Engine**

1. **TournamentLeaderboard** fetches all participants' forecasts from Google Sheets
2. Loads ground truth data from the same files Forecastle uses
3. Calculates WIS for each participant/challenge using `scoreUserForecast()`
4. Builds leaderboard rankings in JavaScript
5. Displays real-time scores (updates every 30 seconds)

### Benefits

- Uses exact same scoring code as Forecastle
- Simpler backend - Google Sheets just stores/retrieves data
- Easier debugging - all logic visible in browser DevTools
- No complex spreadsheet formulas
- Ground truth from the same files as Forecastle

### Historical Challenges

Each tournament challenge uses a specific past date where ground truth is already available:

- Challenge 1: US Flu on **2024-01-20**
- Challenge 2: US COVID on **2024-02-10**
- Challenge 3: US RSV on **2024-01-27**

This ensures instant scoring and fair comparison.
