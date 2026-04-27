import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Badge,
  Box,
  Button,
  Center,
  Container,
  Divider,
  Grid,
  Group,
  Loader,
  Paper,
  Stack,
  Stepper,
  Switch,
  Text,
  Title,
} from "@mantine/core";
import {
  IconAlertTriangle,
  IconTarget,
  IconCheck,
  IconTrophy,
} from "@tabler/icons-react";
import {
  initialiseForecastInputs,
  convertToIntervals,
} from "../../utils/forecastleInputs";
import { validateForecastSubmission } from "../../utils/forecastleValidation";
import { TOURNAMENT_CONFIG } from "../../config";
import {
  scoreUserForecast,
  scoreModels,
  getOfficialModels,
} from "../../utils/forecastleScoring";
import {
  submitForecast,
  getParticipant,
  getLeaderboard,
} from "../../utils/tournamentAPI";
import ForecastleChartCanvas from "../forecastle/ForecastleChartCanvas";
import ForecastleInputControls from "../forecastle/ForecastleInputControls";

const addWeeksToDate = (dateString, weeks) => {
  const base = new Date(`${dateString}T00:00:00Z`);
  if (Number.isNaN(base.getTime())) {
    return dateString;
  }
  base.setUTCDate(base.getUTCDate() + weeks * 7);
  return base.toISOString().slice(0, 10);
};

const getSubmissionForecasts = (submissions, challenge) => {
  if (!submissions) return null;
  return submissions[challenge.id] || submissions[challenge.number] || null;
};

const TournamentGame = ({
  tournamentConfig = TOURNAMENT_CONFIG,
  participantId,
  participantName,
  onAllCompleted,
}) => {
  const [currentChallengeIndex, setCurrentChallengeIndex] = useState(0);
  const [completedChallenges, setCompletedChallenges] = useState(new Set());
  const [submissionErrors, setSubmissionErrors] = useState({});
  const [scores, setScores] = useState(null);
  const [inputMode, setInputMode] = useState("median"); // 'median', 'intervals', or 'scoring'
  const [zoomedView, setZoomedView] = useState(true);
  const [visibleRankings, setVisibleRankings] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [leaderboardData, setLeaderboardData] = useState(null);
  const [groundTruthData, setGroundTruthData] = useState({});
  const [scenarioData, setScenarioData] = useState({});
  const [loading, setLoading] = useState(true);

  const challenge = tournamentConfig.challenges[currentChallengeIndex];
  const allChallengesCompleted =
    tournamentConfig.numChallenges > 0 &&
    tournamentConfig.challenges.every((ch) => completedChallenges.has(ch.id));
  const enabledChallengeIds = useMemo(
    () => new Set(tournamentConfig.challenges.map((ch) => ch.id)),
    [tournamentConfig],
  );
  const challengeIdByNumber = useMemo(
    () =>
      new Map(
        tournamentConfig.challenges.map((ch) => [Number(ch.number), ch.id]),
      ),
    [tournamentConfig],
  );

  // Load all challenge data and ground truth
  useEffect(() => {
    const loadChallengeData = async () => {
      const gtData = {};
      const scData = {};

      for (const ch of tournamentConfig.challenges) {
        try {
          const filePath = `/processed_data/${ch.dataPath}/${ch.location}_${ch.fileSuffix}`;
          const response = await fetch(filePath);
          if (!response.ok) continue;

          const locationData = await response.json();
          scData[ch.number] = locationData;

          const groundTruthDates = locationData.ground_truth?.dates || [];
          const groundTruthValues =
            locationData.ground_truth?.[ch.target] || [];

          const horizonDates = ch.horizons.map((horizon) =>
            addWeeksToDate(ch.forecastDate, horizon),
          );

          const groundTruthForHorizons = horizonDates.map((horizonDate) => {
            const index = groundTruthDates.indexOf(horizonDate);
            if (index >= 0 && Number.isFinite(groundTruthValues[index])) {
              return groundTruthValues[index];
            }
            return null;
          });

          gtData[ch.number] = {
            values: groundTruthForHorizons,
            dates: horizonDates,
            fullSeries: locationData.ground_truth,
          };
        } catch (error) {
          console.error(
            `Failed to load data for challenge ${ch.number}:`,
            error,
          );
        }
      }

      setGroundTruthData(gtData);
      setScenarioData(scData);
      setLoading(false);
    };

    loadChallengeData();
  }, [tournamentConfig]);

  // Load completed challenges for this participant
  useEffect(() => {
    const loadCompletedChallenges = async () => {
      if (!participantId) return;

      try {
        const data = await getParticipant(participantId, tournamentConfig);
        const completed = new Set();

        data.submissions.forEach((sub) => {
          const challengeId =
            sub.challengeId ||
            challengeIdByNumber.get(Number(sub.challengeNum));
          if (
            sub.forecasts &&
            sub.forecasts.length > 0 &&
            enabledChallengeIds.has(challengeId)
          ) {
            completed.add(challengeId);
          }
        });

        setCompletedChallenges(completed);
      } catch (err) {
        console.error("Failed to load completed challenges:", err);
      }
    };

    loadCompletedChallenges();
  }, [
    participantId,
    enabledChallengeIds,
    challengeIdByNumber,
    tournamentConfig,
  ]);

  const latestObservationValue = useMemo(() => {
    if (!challenge || !scenarioData[challenge.number]) return 1000;

    const locationData = scenarioData[challenge.number];
    const groundTruthDates = locationData.ground_truth?.dates || [];
    const groundTruthValues =
      locationData.ground_truth?.[challenge.target] || [];

    // Get value at or before forecast date
    const forecastTimestamp = new Date(challenge.forecastDate).getTime();
    let lastValue = null;

    for (let i = groundTruthDates.length - 1; i >= 0; i--) {
      const entryTime = new Date(groundTruthDates[i]).getTime();
      if (
        entryTime <= forecastTimestamp &&
        Number.isFinite(groundTruthValues[i])
      ) {
        lastValue = groundTruthValues[i];
        break;
      }
    }

    return lastValue !== null ? lastValue : 1000;
  }, [challenge, scenarioData]);

  const initialInputs = useMemo(
    () =>
      initialiseForecastInputs(
        challenge?.horizons || [1, 2, 3],
        latestObservationValue,
      ),
    [challenge?.horizons, latestObservationValue],
  );

  const [forecastEntries, setForecastEntries] = useState(initialInputs);

  // Reset when moving to next challenge
  useEffect(() => {
    if (!challenge) return;

    setForecastEntries(initialInputs);
    setSubmissionErrors({});
    setScores(null);
    setInputMode("median");
    setVisibleRankings(0);
    setError(null);
  }, [currentChallengeIndex, initialInputs, challenge]);

  const groundTruthSeries = useMemo(() => {
    if (!challenge || !scenarioData[challenge.number]) return [];

    const locationData = scenarioData[challenge.number];
    const groundTruthDates = locationData.ground_truth?.dates || [];
    const groundTruthValues =
      locationData.ground_truth?.[challenge.target] || [];

    const forecastTimestamp = new Date(challenge.forecastDate).getTime();

    return groundTruthDates
      .map((date, idx) => ({
        date,
        value: groundTruthValues[idx],
      }))
      .filter((entry) => {
        const entryTime = new Date(entry.date).getTime();
        return Number.isFinite(entry.value) && entryTime <= forecastTimestamp;
      })
      .slice(-20); // Last 20 points
  }, [challenge, scenarioData]);

  const horizonDates = useMemo(() => {
    if (!challenge) return [];
    return challenge.horizons.map((horizon) =>
      addWeeksToDate(challenge.forecastDate, horizon),
    );
  }, [challenge]);

  const maxValue = useMemo(() => {
    const latestValue = Number.isFinite(latestObservationValue)
      ? latestObservationValue
      : 0;
    const baseMax = latestValue > 0 ? latestValue * 5 : 1;
    const userMaxCandidate = Math.max(
      ...forecastEntries.map(
        (entry) => (entry.median ?? 0) + (entry.width95 ?? 0),
      ),
      0,
    );
    return Math.max(
      baseMax,
      userMaxCandidate * 1.1 || 0,
      latestObservationValue,
      1,
    );
  }, [latestObservationValue, forecastEntries]);

  const handleAdjust = (index, field, value) => {
    setForecastEntries((prevEntries) =>
      prevEntries.map((entry, idx) => {
        if (idx !== index) return entry;

        const nextEntry = { ...entry };

        if (field === "median") {
          const oldMedian = entry.median;
          const newMedian = Math.max(0, value);
          const medianShift = newMedian - oldMedian;

          nextEntry.median = newMedian;

          if (entry.lower95 !== undefined && entry.upper95 !== undefined) {
            nextEntry.lower95 = Math.max(0, entry.lower95 + medianShift);
            nextEntry.upper95 = entry.upper95 + medianShift;
          }
          if (entry.lower50 !== undefined && entry.upper50 !== undefined) {
            nextEntry.lower50 = Math.max(0, entry.lower50 + medianShift);
            nextEntry.upper50 = entry.upper50 + medianShift;
          }
        } else if (field === "interval95") {
          const [lower, upper] = value;
          nextEntry.lower95 = Math.max(0, lower);
          nextEntry.upper95 = Math.max(lower, upper);
          if (nextEntry.lower50 < nextEntry.lower95)
            nextEntry.lower50 = nextEntry.lower95;
          if (nextEntry.upper50 > nextEntry.upper95)
            nextEntry.upper50 = nextEntry.upper95;
          nextEntry.width95 = Math.max(
            nextEntry.upper95 - entry.median,
            entry.median - nextEntry.lower95,
          );
        } else if (field === "interval50") {
          const [lower, upper] = value;
          nextEntry.lower50 = Math.max(nextEntry.lower95 || 0, lower);
          nextEntry.upper50 = Math.min(
            nextEntry.upper95 || 99999,
            Math.max(lower, upper),
          );
          nextEntry.width50 = Math.max(
            nextEntry.upper50 - entry.median,
            entry.median - nextEntry.lower50,
          );
        }

        return nextEntry;
      }),
    );
    setSubmissionErrors({});
  };

  const handleSubmit = async () => {
    const intervalsForValidation = convertToIntervals(forecastEntries);
    const { valid, errors } = validateForecastSubmission(
      intervalsForValidation,
    );
    setSubmissionErrors(errors);

    if (!valid) {
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      // Submit to backend
      await submitForecast(
        participantId,
        challenge.number,
        forecastEntries,
        tournamentConfig,
      );

      // Calculate scores
      const gtData = groundTruthData[challenge.number];
      if (!gtData) {
        throw new Error("Ground truth data not available");
      }

      const userScore = scoreUserForecast(forecastEntries, gtData.values);

      const locationData = scenarioData[challenge.number];
      const modelScores = scoreModels(
        locationData.forecasts?.[challenge.forecastDate]?.[challenge.target] ||
          {},
        challenge.horizons,
        gtData.values,
      );

      setScores({
        user: userScore,
        models: modelScores,
        groundTruth: gtData.values,
        horizonDates: gtData.dates,
      });

      // Load leaderboard to compare with other participants
      const leaderboard = await getLeaderboard(tournamentConfig);
      setLeaderboardData(leaderboard);

      // Mark as completed
      setCompletedChallenges((prev) => new Set([...prev, challenge.id]));

      // Move to scoring view
      setInputMode("scoring");
    } catch (err) {
      setError(err.message || "Submission failed");
    } finally {
      setSubmitting(false);
    }
  };

  // Animate rankings reveal
  useEffect(() => {
    if (
      inputMode === "scoring" &&
      scores &&
      visibleRankings < scores.models.length + 1
    ) {
      const timer = setTimeout(() => {
        setVisibleRankings((prev) => prev + 1);
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [inputMode, scores, visibleRankings]);

  if (loading) {
    return (
      <Center style={{ minHeight: "400px" }}>
        <Stack align="center" spacing="md">
          <Loader size="lg" />
          <Text>Loading tournament...</Text>
        </Stack>
      </Center>
    );
  }

  if (!challenge) {
    return (
      <Alert color="red" title="Error">
        Challenge not found
      </Alert>
    );
  }

  return (
    <Container size="xl" py="md">
      <Stack spacing="lg">
        {/* Header */}
        <Paper shadow="sm" p="md" withBorder>
          <Group position="apart">
            <div>
              <Title order={2}>{tournamentConfig.name}</Title>
              <Text size="sm" c="dimmed">
                {participantName}
              </Text>
            </div>
            <Badge size="xl" variant="filled">
              Challenge {currentChallengeIndex + 1}/
              {tournamentConfig.numChallenges}
            </Badge>
          </Group>
        </Paper>

        {/* Progress Stepper */}
        <Stepper active={currentChallengeIndex} size="sm">
          {tournamentConfig.challenges.map((ch, idx) => (
            <Stepper.Step
              key={ch.id}
              label={`Challenge ${idx + 1}`}
              description={`${ch.displayName} - ${ch.dataset.toUpperCase()}`}
              icon={
                completedChallenges.has(ch.id) ? (
                  <IconCheck size={18} />
                ) : undefined
              }
              color={completedChallenges.has(ch.id) ? "green" : undefined}
            />
          ))}
        </Stepper>

        {/* Challenge Content */}
        {inputMode !== "scoring" ? (
          <Paper shadow="sm" p="lg" withBorder>
            <Stack spacing="md">
              <div>
                <Title order={3}>{challenge.title}</Title>
                <Text size="sm" c="dimmed">
                  {challenge.description}
                </Text>
                <Text size="xs" c="dimmed" mt="xs">
                  Forecast date: {challenge.forecastDate} •{" "}
                  {challenge.displayName}
                </Text>
              </div>

              {error && (
                <Alert
                  icon={<IconAlertTriangle size={16} />}
                  title="Error"
                  color="red"
                >
                  {error}
                </Alert>
              )}

              {/* Input Mode Stepper */}
              <Stepper active={inputMode === "median" ? 0 : 1} size="sm">
                <Stepper.Step label="Median" description="Point forecasts" />
                <Stepper.Step label="Intervals" description="Uncertainty" />
              </Stepper>

              <Grid gutter="lg">
                {/* Left Panel - Chart */}
                <Grid.Col span={{ base: 12, lg: 7 }}>
                  <Stack gap="md">
                    <Group justify="space-between">
                      <Title order={5}>Interactive Chart</Title>
                      <Switch
                        label="Show More History"
                        checked={!zoomedView}
                        onChange={(e) =>
                          setZoomedView(!e.currentTarget.checked)
                        }
                        color="red"
                        size="md"
                      />
                    </Group>

                    <Box style={{ width: "100%", height: 380 }}>
                      <ForecastleChartCanvas
                        groundTruthSeries={groundTruthSeries}
                        horizonDates={horizonDates}
                        entries={forecastEntries}
                        maxValue={maxValue}
                        onAdjust={handleAdjust}
                        height={380}
                        showIntervals={inputMode === "intervals"}
                        zoomedView={zoomedView}
                      />
                    </Box>

                    <Text size="sm" c="dimmed">
                      {inputMode === "median"
                        ? "Drag the handles to set your median forecast for each week ahead."
                        : "Drag the handles to adjust interval bounds, or use the sliders for precise control."}
                    </Text>
                  </Stack>
                </Grid.Col>

                {/* Right Panel - Controls */}
                <Grid.Col span={{ base: 12, lg: 5 }}>
                  <Stack gap="md" h="100%">
                    <Title order={5}>
                      {inputMode === "median"
                        ? "Median Forecasts"
                        : "Uncertainty Intervals"}
                    </Title>

                    <ForecastleInputControls
                      entries={forecastEntries}
                      onChange={setForecastEntries}
                      maxValue={maxValue}
                      mode={inputMode}
                    />

                    {Object.keys(submissionErrors).length > 0 && (
                      <Alert
                        icon={<IconAlertTriangle size={16} />}
                        color="yellow"
                      >
                        {Object.values(submissionErrors).join(". ")}
                      </Alert>
                    )}

                    {/* Navigation */}
                    <Box mt="auto">
                      <Group justify="space-between">
                        {inputMode === "intervals" && (
                          <Button
                            variant="subtle"
                            onClick={() => setInputMode("median")}
                          >
                            ← Back
                          </Button>
                        )}
                        <div style={{ marginLeft: "auto" }}>
                          {inputMode === "median" ? (
                            <Button
                              onClick={() => setInputMode("intervals")}
                              rightSection="→"
                            >
                              Next: Set Intervals
                            </Button>
                          ) : (
                            <Button
                              onClick={handleSubmit}
                              loading={submitting}
                              color="green"
                              leftSection={<IconTarget size={16} />}
                            >
                              Submit Forecast
                            </Button>
                          )}
                        </div>
                      </Group>
                    </Box>
                  </Stack>
                </Grid.Col>
              </Grid>
            </Stack>
          </Paper>
        ) : (
          <ScoreDisplay
            scores={scores}
            challenge={challenge}
            participantId={participantId}
            participantName={participantName}
            leaderboardData={leaderboardData}
            visibleRankings={visibleRankings}
            onNextChallenge={() => {
              if (
                currentChallengeIndex <
                tournamentConfig.challenges.length - 1
              ) {
                setCurrentChallengeIndex((prev) => prev + 1);
              }
            }}
            allCompleted={allChallengesCompleted}
            onAllCompleted={onAllCompleted}
          />
        )}
      </Stack>
    </Container>
  );
};

const ScoreDisplay = ({
  scores,
  challenge,
  participantId,
  participantName,
  leaderboardData,
  visibleRankings,
  onNextChallenge,
  allCompleted,
  onAllCompleted,
}) => {
  if (!scores) return null;

  const { ensemble: ensembleKey, baseline: baselineKey } = getOfficialModels(
    challenge.datasetKey,
  );

  // Create unified ranking
  const allRanked = [
    { name: participantName, wis: scores.user.wis, isUser: true, type: "user" },
    ...scores.models.map((m) => ({
      name: m.modelName,
      wis: m.wis,
      isUser: false,
      type: "model",
    })),
  ].sort((a, b) => a.wis - b.wis);

  // Add other participants if available
  if (leaderboardData) {
    leaderboardData.forEach((p) => {
      const submission = getSubmissionForecasts(p.submissions, challenge);
      // Skip current participant by checking participantId
      if (
        submission &&
        submission.length > 0 &&
        p.participantId !== participantId
      ) {
        const forecastEntries = submission.map((f) => ({
          horizon: f.horizon,
          median: f.median,
          lower50: f.q25,
          upper50: f.q75,
          lower95: f.q025,
          upper95: f.q975,
        }));

        const pScore = scoreUserForecast(forecastEntries, scores.groundTruth);
        allRanked.push({
          name: p.name,
          wis: pScore.wis,
          isUser: false,
          type: "participant",
        });
      }
    });

    // Re-sort after adding participants
    allRanked.sort((a, b) => a.wis - b.wis);
  }

  // Calculate user rank AFTER adding all participants and final sort
  const userRank = allRanked.findIndex((e) => e.isUser) + 1;

  return (
    <Paper shadow="sm" p="lg" withBorder>
      <Stack spacing="lg">
        <div>
          <Group spacing="xs" mb="xs">
            <IconTrophy size={28} color="gold" />
            <Title order={2}>Results</Title>
          </Group>
          <Text size="sm" c="dimmed">
            {challenge.title}
          </Text>
        </div>

        {/* Action Buttons at Top */}
        <Divider />
        {!allCompleted ? (
          <Button
            size="lg"
            onClick={onNextChallenge}
            rightSection="→"
            fullWidth
          >
            Next Challenge
          </Button>
        ) : (
          <Button
            size="lg"
            onClick={onAllCompleted}
            leftSection={<IconTrophy size={16} />}
            color="green"
            fullWidth
          >
            View Leaderboard
          </Button>
        )}

        <Divider />

        {/* User Score - Simple */}
        <Paper p="lg" withBorder style={{ backgroundColor: "#e7f5ff" }}>
          <Stack spacing="sm" align="center">
            <Text weight={700} size="lg">
              Your Forecast
            </Text>
            <Badge size="xl" color="blue">
              Rank #{userRank}
            </Badge>
            {userRank / allRanked.length <= 0.1 && (
              <Text size="md" weight={600} c="green">
                Bravo! 😊
              </Text>
            )}
          </Stack>
        </Paper>

        {/* Rankings */}
        <div>
          <Text weight={600} mb="sm">
            Leaderboard
          </Text>
          <Stack spacing="xs">
            {(() => {
              const topN = 15;
              const visibleEntries = allRanked.slice(
                0,
                Math.min(visibleRankings, allRanked.length),
              );
              const displayEntries = visibleEntries.slice(0, topN);

              // Add ellipsis if there are more entries
              const hasMore = allRanked.length > topN;
              const userBelowCutoff =
                hasMore && allRanked.findIndex((e) => e.isUser) >= topN;

              return (
                <>
                  {displayEntries.map((entry, idx) => (
                    <Paper
                      key={`${entry.type}-${entry.name}`}
                      p="sm"
                      withBorder
                      style={{
                        backgroundColor: entry.isUser ? "#e7f5ff" : undefined,
                        opacity: visibleRankings >= idx + 1 ? 1 : 0,
                        transition: "opacity 0.3s ease-in",
                      }}
                    >
                      <Group position="apart">
                        <Group spacing="xs">
                          <Text weight={entry.isUser ? 700 : 500}>
                            #{idx + 1}
                          </Text>
                          <Text weight={entry.isUser ? 700 : 400}>
                            {entry.name}
                          </Text>
                          {entry.type === "participant" && (
                            <Badge size="sm">Participant</Badge>
                          )}
                          {entry.name === ensembleKey && (
                            <Badge size="sm" color="teal">
                              Ensemble
                            </Badge>
                          )}
                          {entry.name === baselineKey && (
                            <Badge size="sm" color="gray">
                              Baseline
                            </Badge>
                          )}
                          {entry.isUser && (
                            <Badge size="sm" color="blue">
                              You
                            </Badge>
                          )}
                        </Group>
                        <Text size="sm">{entry.wis.toFixed(1)}</Text>
                      </Group>
                    </Paper>
                  ))}

                  {hasMore && visibleRankings >= topN && (
                    <>
                      <Text size="sm" c="dimmed" ta="center" py="xs">
                        ⋮
                      </Text>
                      {userBelowCutoff && (
                        <Paper
                          p="sm"
                          withBorder
                          style={{ backgroundColor: "#e7f5ff" }}
                        >
                          <Group position="apart">
                            <Group spacing="xs">
                              <Text weight={700}>
                                #{allRanked.findIndex((e) => e.isUser) + 1}
                              </Text>
                              <Text weight={700}>{participantName}</Text>
                              <Badge size="sm" color="blue">
                                You
                              </Badge>
                            </Group>
                            <Text size="sm">
                              {allRanked.find((e) => e.isUser).wis.toFixed(1)}
                            </Text>
                          </Group>
                        </Paper>
                      )}
                      <Text size="sm" c="dimmed" ta="center">
                        {displayEntries.length} of {allRanked.length} entries
                        shown
                      </Text>
                    </>
                  )}
                </>
              );
            })()}
          </Stack>
        </div>
      </Stack>
    </Paper>
  );
};

export default TournamentGame;
