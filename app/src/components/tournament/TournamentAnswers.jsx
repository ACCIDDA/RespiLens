import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Badge,
  Box,
  Center,
  Loader,
  Paper,
  Stack,
  Table,
  Text,
  Title,
} from "@mantine/core";
import { IconAlertCircle, IconTarget } from "@tabler/icons-react";
import { TOURNAMENT_CONFIG, getMaskedForecastDate } from "../../config";
import { getLeaderboard } from "../../utils/tournamentAPI";
import ForecastleChartCanvas from "../forecastle/ForecastleChartCanvas";

const addWeeksToDate = (dateString, weeks) => {
  const base = new Date(`${dateString}T00:00:00Z`);
  if (Number.isNaN(base.getTime())) {
    return dateString;
  }
  base.setUTCDate(base.getUTCDate() + weeks * 7);
  return base.toISOString().slice(0, 10);
};

const restoreForecastEntries = (forecasts = []) =>
  forecasts.map((forecast) => ({
    horizon: forecast.horizon,
    median: forecast.median,
    lower50: forecast.q25,
    upper50: forecast.q75,
    lower95: forecast.q025,
    upper95: forecast.q975,
    width50: Math.max(
      forecast.q75 - forecast.median,
      forecast.median - forecast.q25,
    ),
    width95: Math.max(
      forecast.q975 - forecast.median,
      forecast.median - forecast.q025,
    ),
  }));

const formatValue = (value) =>
  Number.isFinite(value) ? Math.round(value).toLocaleString("en-US") : "—";

const formatDelta = (forecast, truth) => {
  if (!Number.isFinite(forecast) || !Number.isFinite(truth)) {
    return "—";
  }

  const delta = truth - forecast;
  if (delta === 0) {
    return "0";
  }

  const prefix = delta > 0 ? "+" : "";
  return `${prefix}${Math.round(delta).toLocaleString("en-US")}`;
};

const getSubmissionForecasts = (submission) => {
  if (!submission) return null;

  if (Array.isArray(submission)) {
    return submission;
  }

  if (submission?.forecasts && Array.isArray(submission.forecasts)) {
    return submission.forecasts;
  }

  return null;
};

const getSubmissionByChallenge = (submissions, challenge) => {
  if (!submissions) return null;

  if (Array.isArray(submissions)) {
    return (
      submissions.find(
        (submission) =>
          submission.challengeId === challenge.id ||
          Number(submission.challengeNum) === Number(challenge.number),
      ) || null
    );
  }

  return submissions[challenge.id] || submissions[challenge.number] || null;
};

const TournamentAnswers = ({
  tournamentConfig = TOURNAMENT_CONFIG,
  participantId,
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submissionsByChallenge, setSubmissionsByChallenge] = useState({});
  const [challengeData, setChallengeData] = useState({});
  const [participantFound, setParticipantFound] = useState(true);

  useEffect(() => {
    const loadAnswers = async () => {
      if (!participantId) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const leaderboard = await getLeaderboard(tournamentConfig);
        const participantEntry =
          leaderboard.find((entry) => entry.participantId === participantId) ||
          null;

        if (!participantEntry) {
          setParticipantFound(false);
          setSubmissionsByChallenge({});
          setChallengeData({});
          return;
        }

        setParticipantFound(true);

        const nextSubmissionsByChallenge = {};
        tournamentConfig.challenges.forEach((challenge) => {
          const submission = getSubmissionByChallenge(
            participantEntry.submissions || [],
            challenge,
          );
          const forecasts = getSubmissionForecasts(submission);
          if (forecasts?.length) {
            nextSubmissionsByChallenge[challenge.id] = submission;
          }
        });

        setSubmissionsByChallenge(nextSubmissionsByChallenge);

        if (
          Object.keys(nextSubmissionsByChallenge).length <
          tournamentConfig.numChallenges
        ) {
          setChallengeData({});
          return;
        }

        const challengeResponses = await Promise.all(
          tournamentConfig.challenges.map(async (challenge) => {
            const filePath = `/processed_data/${challenge.dataPath}/${challenge.location}_${challenge.fileSuffix}`;
            const response = await fetch(filePath);

            if (!response.ok) {
              throw new Error(
                `Failed to load challenge ${challenge.number} data`,
              );
            }

            const locationData = await response.json();
            const groundTruthDates = locationData.ground_truth?.dates || [];
            const groundTruthValues =
              locationData.ground_truth?.[challenge.target] || [];
            const forecastTimestamp = new Date(
              challenge.forecastDate,
            ).getTime();
            const groundTruthSeries = groundTruthDates
              .map((date, index) => ({
                date,
                value: groundTruthValues[index],
              }))
              .filter((entry) => {
                const entryTimestamp = new Date(entry.date).getTime();
                return (
                  Number.isFinite(entry.value) &&
                  entryTimestamp <= forecastTimestamp
                );
              });

            const horizonDates = challenge.horizons.map((horizon) =>
              addWeeksToDate(challenge.forecastDate, horizon),
            );

            const horizonTruth = horizonDates.map((horizonDate) => {
              const index = groundTruthDates.indexOf(horizonDate);
              if (index >= 0 && Number.isFinite(groundTruthValues[index])) {
                return groundTruthValues[index];
              }
              return null;
            });

            return [
              challenge.id,
              {
                groundTruthSeries,
                horizonDates,
                horizonTruth,
              },
            ];
          }),
        );

        setChallengeData(Object.fromEntries(challengeResponses));
      } catch (err) {
        setError(err.message || "Failed to load submitted answers");
      } finally {
        setLoading(false);
      }
    };

    loadAnswers();
  }, [participantId, tournamentConfig]);

  const completedCount = useMemo(
    () => Object.keys(submissionsByChallenge).length,
    [submissionsByChallenge],
  );

  if (loading) {
    return (
      <Center style={{ minHeight: "320px" }}>
        <Stack align="center" spacing="md">
          <Loader />
          <Text>Loading answer review...</Text>
        </Stack>
      </Center>
    );
  }

  if (error) {
    return (
      <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red">
        {error}
      </Alert>
    );
  }

  if (!participantFound) {
    return (
      <Alert color="yellow" title="Answers unavailable">
        We could not find a completed participant record for this browser
        session yet.
      </Alert>
    );
  }

  if (completedCount < tournamentConfig.numChallenges) {
    return (
      <Alert color="blue" title="Answers unlock after full submission">
        The `Answers` tab stays hidden until all{" "}
        {tournamentConfig.numChallenges} challenge
        {tournamentConfig.numChallenges === 1 ? "" : "s"} have been completed
        and submitted. You currently have {completedCount} of{" "}
        {tournamentConfig.numChallenges} finished.
      </Alert>
    );
  }

  return (
    <Stack spacing="lg">
      <div>
        <Title order={3}>Answers</Title>
        <Text size="sm" c="dimmed">
          Compare your submitted medians against what actually happened for each
          challenge.
        </Text>
      </div>
      {tournamentConfig.challenges.map((challenge) => {
        const submission = submissionsByChallenge[challenge.id];
        const answerData = challengeData[challenge.id];
        const forecasts = getSubmissionForecasts(submission);

        if (!forecasts?.length) {
          return (
            <Paper key={challenge.id} shadow="sm" p="lg" withBorder>
              <Text size="sm" c="dimmed">
                Submission data is unavailable for {challenge.title}.
              </Text>
            </Paper>
          );
        }

        if (!answerData) {
          return (
            <Paper key={challenge.id} shadow="sm" p="lg" withBorder>
              <Text size="sm" c="dimmed">
                Challenge data is still loading for {challenge.title}.
              </Text>
            </Paper>
          );
        }

        const entries = restoreForecastEntries(forecasts);
        const observedMax = Math.max(
          ...answerData.groundTruthSeries.map((entry) => entry.value ?? 0),
          0,
        );
        const forecastMax = Math.max(
          ...entries.map((entry) => entry.upper95 ?? entry.median ?? 0),
          0,
        );
        const truthMax = Math.max(...answerData.horizonTruth, 0);
        const chartMax = Math.max(observedMax, forecastMax, truthMax, 1);

        return (
          <Paper key={challenge.id} shadow="sm" p="lg" withBorder>
            <Stack spacing="md">
              <div>
                <Title order={4}>
                  {challenge.title}{" "}
                  <Text component="span" inherit c="dimmed">
                    (Challenge {challenge.number})
                  </Text>
                </Title>
                <Text size="sm" c="dimmed">
                  Forecast date:{" "}
                  {getMaskedForecastDate(
                    challenge.forecastDate,
                    tournamentConfig,
                  )}
                </Text>
              </div>

              <Box style={{ width: "100%", height: 420 }}>
                <ForecastleChartCanvas
                  groundTruthSeries={answerData.groundTruthSeries}
                  horizonDates={answerData.horizonDates}
                  entries={entries}
                  maxValue={chartMax}
                  onAdjust={() => {}}
                  height={420}
                  showIntervals={false}
                  zoomedView={false}
                  scores={{ groundTruth: answerData.horizonTruth }}
                  showScoring={true}
                  dateLabelFormatter={(date) =>
                    getMaskedForecastDate(date, tournamentConfig)
                  }
                />
              </Box>

              <div>
                <Text size="sm" fw={600} mb="xs">
                  Horizon Summary
                </Text>
                <Table striped highlightOnHover>
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th>Date</th>
                      <th style={{ textAlign: "right" }}>Your median</th>
                      <th style={{ textAlign: "right" }}>Truth</th>
                      <th style={{ textAlign: "right" }}>Truth - forecast</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((entry, index) => (
                      <tr key={`${challenge.id}-${entry.horizon}`}>
                        <td>
                          <Badge variant="light">
                            <span
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 4,
                              }}
                            >
                              <IconTarget size={12} />
                              {entry.horizon} wk
                            </span>
                          </Badge>
                        </td>
                        <td>
                          {answerData.horizonDates[index]
                            ? getMaskedForecastDate(
                                answerData.horizonDates[index],
                                tournamentConfig,
                              )
                            : "—"}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {formatValue(entry.median)}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {formatValue(answerData.horizonTruth[index])}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {formatDelta(
                            entry.median,
                            answerData.horizonTruth[index],
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            </Stack>
          </Paper>
        );
      })}
    </Stack>
  );
};

export default TournamentAnswers;
