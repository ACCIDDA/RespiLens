import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Table,
  Text,
  Badge,
  Stack,
  Title,
  Alert,
  Loader,
  Anchor,
  Group,
} from "@mantine/core";
import { IconTrophy, IconAlertCircle } from "@tabler/icons-react";
import { getLeaderboard } from "../../utils/tournamentAPI";
import { TOURNAMENT_CONFIG } from "../../config";
import { scoreUserForecast } from "../../utils/forecastleScoring";

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

const TournamentLeaderboard = ({
  tournamentConfig = TOURNAMENT_CONFIG,
  participantId,
}) => {
  const [leaderboard, setLeaderboard] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [groundTruthData, setGroundTruthData] = useState({});

  // Load ground truth data for all challenges
  useEffect(() => {
    const loadGroundTruth = async () => {
      const gtData = {};

      for (const challenge of tournamentConfig.challenges) {
        try {
          const filePath = `/processed_data/${challenge.dataPath}/${challenge.location}_${challenge.fileSuffix}`;
          const response = await fetch(filePath);
          if (!response.ok) continue;

          const locationData = await response.json();
          const groundTruthDates = locationData.ground_truth?.dates || [];
          const groundTruthValues =
            locationData.ground_truth?.[challenge.target] || [];

          // Extract ground truth for each horizon
          const horizonDates = challenge.horizons.map((horizon) =>
            addWeeksToDate(challenge.forecastDate, horizon),
          );

          const groundTruthForHorizons = horizonDates.map((horizonDate) => {
            const index = groundTruthDates.indexOf(horizonDate);
            if (index >= 0 && Number.isFinite(groundTruthValues[index])) {
              return groundTruthValues[index];
            }
            return null;
          });

          gtData[challenge.id] = groundTruthForHorizons;
        } catch (error) {
          console.error(
            `Failed to load ground truth for challenge ${challenge.number}:`,
            error,
          );
        }
      }

      setGroundTruthData(gtData);
    };

    loadGroundTruth();
  }, [tournamentConfig]);

  // Load leaderboard and calculate scores
  useEffect(() => {
    const loadAndScoreLeaderboard = async () => {
      try {
        const data = await getLeaderboard(tournamentConfig);

        // Calculate WIS for each participant
        const scoredLeaderboard = data.map((participant) => {
          const challengeScores = {};
          let totalWIS = 0;
          let totalDispersion = 0;
          let totalUnderprediction = 0;
          let totalOverprediction = 0;
          let validChallenges = 0;
          const activeCompletedChallenges = tournamentConfig.challenges.reduce(
            (count, challenge) => {
              const forecasts = getSubmissionForecasts(
                participant.submissions,
                challenge,
              );
              return forecasts && forecasts.length > 0 ? count + 1 : count;
            },
            0,
          );

          // Score each active challenge
          tournamentConfig.challenges.forEach((challenge) => {
            const forecasts = getSubmissionForecasts(
              participant.submissions,
              challenge,
            );
            const groundTruth = groundTruthData[challenge.id];

            if (!groundTruth || !forecasts || forecasts.length === 0) return;

            // Convert forecasts to the format expected by scoreUserForecast
            const forecastEntries = forecasts.map((f) => ({
              horizon: f.horizon,
              median: f.median,
              lower50: f.q25,
              upper50: f.q75,
              lower95: f.q025,
              upper95: f.q975,
            }));

            // Calculate WIS with components
            const scoreResult = scoreUserForecast(forecastEntries, groundTruth);
            if (scoreResult.wis !== null) {
              challengeScores[challenge.id] = {
                wis: scoreResult.wis,
                dispersion: scoreResult.dispersion,
                underprediction: scoreResult.underprediction,
                overprediction: scoreResult.overprediction,
              };
              totalWIS += scoreResult.wis;
              totalDispersion += scoreResult.dispersion;
              totalUnderprediction += scoreResult.underprediction;
              totalOverprediction += scoreResult.overprediction;
              validChallenges++;
            }
          });

          const avgWIS =
            validChallenges > 0 ? totalWIS / validChallenges : null;
          const avgDispersion =
            validChallenges > 0 ? totalDispersion / validChallenges : null;
          const avgUnderprediction =
            validChallenges > 0 ? totalUnderprediction / validChallenges : null;
          const avgOverprediction =
            validChallenges > 0 ? totalOverprediction / validChallenges : null;

          return {
            ...participant,
            totalWIS,
            avgWIS,
            avgDispersion,
            avgUnderprediction,
            avgOverprediction,
            validChallenges,
            activeCompletedChallenges,
            challengeScores,
          };
        });

        // Sort participants: completed first (by avgWIS), then incomplete (by completed count)
        scoredLeaderboard.sort((a, b) => {
          const aCompleted =
            a.validChallenges === tournamentConfig.numChallenges;
          const bCompleted =
            b.validChallenges === tournamentConfig.numChallenges;

          // Both completed: sort by avgWIS (lower is better)
          if (aCompleted && bCompleted) {
            if (a.avgWIS === null) return 1;
            if (b.avgWIS === null) return -1;
            return a.avgWIS - b.avgWIS;
          }

          // Completed participants always come first
          if (aCompleted) return -1;
          if (bCompleted) return 1;

          // Both incomplete: sort by number of challenges completed (descending)
          return b.activeCompletedChallenges - a.activeCompletedChallenges;
        });

        setLeaderboard(scoredLeaderboard);
        setError(null);
      } catch (err) {
        setError(err.message || "Failed to load leaderboard");
      } finally {
        setLoading(false);
      }
    };

    // Only load leaderboard after ground truth is loaded
    if (Object.keys(groundTruthData).length > 0) {
      loadAndScoreLeaderboard();

      // Poll for updates
      const interval = setInterval(
        loadAndScoreLeaderboard,
        tournamentConfig.leaderboard.updateFrequency,
      );
      return () => clearInterval(interval);
    }
  }, [groundTruthData, tournamentConfig]);

  const getMedalEmoji = (rank) => {
    return tournamentConfig.ui.medals[rank] || "";
  };

  if (loading) {
    return (
      <Stack align="center" spacing="md">
        <Loader />
        <Text>Loading leaderboard...</Text>
      </Stack>
    );
  }

  if (error) {
    return (
      <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red">
        {error}
      </Alert>
    );
  }

  if (leaderboard.length === 0) {
    return (
      <Stack align="center" spacing="md" py="xl">
        <IconTrophy size={48} color="gray" />
        <Text color="dimmed" size="lg">
          No participants yet.
        </Text>
        <Text color="dimmed" size="sm">
          Complete challenges to appear on the leaderboard!
        </Text>
      </Stack>
    );
  }

  return (
    <Stack spacing="md">
      <Title order={3}>
        <IconTrophy
          size={24}
          style={{ verticalAlign: "middle", marginRight: 8 }}
        />
        Leaderboard
      </Title>

      <Text size="sm" color="dimmed">
        Ranked by average WIS (lower is better). Completed participants ranked
        first.
      </Text>

      <Stack spacing="sm">
        <Table striped highlightOnHover>
          <thead>
            <tr>
              <th style={{ width: 60 }}>Rank</th>
              <th>Participant</th>
              <th style={{ textAlign: "right", width: 90 }}>Avg WIS</th>
              <th style={{ textAlign: "right", width: 90 }}>Total WIS</th>
              {tournamentConfig.challenges.map((ch) => (
                <th key={ch.number} style={{ textAlign: "center", width: 80 }}>
                  Ch {ch.number}
                </th>
              ))}
              <th style={{ width: 200 }}>Calibration</th>
              <th style={{ textAlign: "center", width: 90 }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {(() => {
              const userIndex = leaderboard.findIndex(
                (e) => e.participantId === participantId,
              );
              const topN = 10;
              const bottomN = 3;
              const contextRadius = 2;

              // Build display list with ellipsis indicators
              const toDisplay = [];

              for (let i = 0; i < leaderboard.length; i++) {
                const entry = leaderboard[i];
                const rank = i + 1;
                const isUser = entry.participantId === participantId;

                // Always show: top N, around user (±contextRadius), bottom N
                const showTop = i < topN;
                const showAroundUser =
                  userIndex >= 0 && Math.abs(i - userIndex) <= contextRadius;
                const showBottom = i >= leaderboard.length - bottomN;

                if (showTop || showAroundUser || showBottom) {
                  // Check if we need ellipsis before this entry
                  if (toDisplay.length > 0) {
                    const lastDisplayedIndex =
                      toDisplay[toDisplay.length - 1].index;
                    if (i - lastDisplayedIndex > 1) {
                      toDisplay.push({
                        type: "ellipsis",
                        count: i - lastDisplayedIndex - 1,
                      });
                    }
                  }

                  toDisplay.push({
                    type: "entry",
                    entry,
                    rank,
                    isUser,
                    index: i,
                  });
                }
              }

              return toDisplay.map((item, idx) => {
                if (item.type === "ellipsis") {
                  return (
                    <tr key={`ellipsis-${idx}`}>
                      <td
                        colSpan={tournamentConfig.challenges.length + 6}
                        style={{ textAlign: "center", padding: "8px" }}
                      >
                        <Text size="sm" c="dimmed">
                          ⋮ {item.count} participant{item.count > 1 ? "s" : ""}{" "}
                          hidden ⋮
                        </Text>
                      </td>
                    </tr>
                  );
                }

                const { entry, rank, isUser } = item;
                const medal = getMedalEmoji(rank);

                return (
                  <tr
                    key={entry.participantId}
                    style={{
                      backgroundColor: isUser ? "#e7f5ff" : undefined,
                      fontWeight: isUser ? 600 : undefined,
                    }}
                  >
                    <td>
                      <Text weight={isUser ? 700 : 500} size="sm">
                        {medal && (
                          <span style={{ marginRight: 4 }}>{medal}</span>
                        )}
                        {rank}
                      </Text>
                    </td>
                    <td>
                      <Text weight={isUser ? 700 : 400} size="sm">
                        {entry.name}
                        {isUser && (
                          <Badge ml="xs" size="xs" color="blue">
                            You
                          </Badge>
                        )}
                      </Text>
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <Text size="sm" weight={isUser ? 700 : 400}>
                        {entry.avgWIS !== null ? entry.avgWIS.toFixed(1) : "—"}
                      </Text>
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <Text size="sm">
                        {entry.totalWIS !== null
                          ? entry.totalWIS.toFixed(1)
                          : "—"}
                      </Text>
                    </td>
                    {tournamentConfig.challenges.map((ch) => (
                      <td key={ch.number} style={{ textAlign: "center" }}>
                        <Text
                          size="xs"
                          c={
                            entry.challengeScores[ch.id] ? undefined : "dimmed"
                          }
                        >
                          {entry.challengeScores[ch.id]?.wis?.toFixed(1) || "—"}
                        </Text>
                      </td>
                    ))}
                    <td>
                      {entry.avgWIS !== null ? (
                        <div style={{ padding: "4px 0" }}>
                          <div
                            style={{
                              display: "flex",
                              height: 20,
                              borderRadius: 3,
                              overflow: "hidden",
                              border: "1px solid #dee2e6",
                            }}
                          >
                            {(() => {
                              const total =
                                entry.avgDispersion +
                                entry.avgUnderprediction +
                                entry.avgOverprediction;
                              const dispPct =
                                (entry.avgDispersion / total) * 100;
                              const underPct =
                                (entry.avgUnderprediction / total) * 100;
                              const overPct =
                                (entry.avgOverprediction / total) * 100;

                              return (
                                <>
                                  {dispPct > 0 && (
                                    <div
                                      style={{
                                        width: `${dispPct}%`,
                                        backgroundColor: "#228be6",
                                      }}
                                      title={`Dispersion: ${entry.avgDispersion.toFixed(1)} (${dispPct.toFixed(0)}%)`}
                                    />
                                  )}
                                  {underPct > 0 && (
                                    <div
                                      style={{
                                        width: `${underPct}%`,
                                        backgroundColor: "#fa5252",
                                      }}
                                      title={`Underprediction: ${entry.avgUnderprediction.toFixed(1)} (${underPct.toFixed(0)}%)`}
                                    />
                                  )}
                                  {overPct > 0 && (
                                    <div
                                      style={{
                                        width: `${overPct}%`,
                                        backgroundColor: "#fd7e14",
                                      }}
                                      title={`Overprediction: ${entry.avgOverprediction.toFixed(1)} (${overPct.toFixed(0)}%)`}
                                    />
                                  )}
                                </>
                              );
                            })()}
                          </div>
                        </div>
                      ) : (
                        <Text size="xs" c="dimmed">
                          —
                        </Text>
                      )}
                    </td>
                    <td style={{ textAlign: "center" }}>
                      <Badge
                        size="xs"
                        color={
                          entry.activeCompletedChallenges ===
                          tournamentConfig.numChallenges
                            ? "green"
                            : "gray"
                        }
                      >
                        {entry.activeCompletedChallenges}/
                        {tournamentConfig.numChallenges}
                      </Badge>
                    </td>
                  </tr>
                );
              });
            })()}
          </tbody>
        </Table>

        {/* Legend */}
        <Group spacing="lg" style={{ justifyContent: "center" }}>
          <Group spacing={6}>
            <div
              style={{
                width: 16,
                height: 16,
                backgroundColor: "#228be6",
                borderRadius: 3,
              }}
            />
            <Text size="xs">Dispersion</Text>
          </Group>
          <Group spacing={6}>
            <div
              style={{
                width: 16,
                height: 16,
                backgroundColor: "#fa5252",
                borderRadius: 3,
              }}
            />
            <Text size="xs">Underprediction</Text>
          </Group>
          <Group spacing={6}>
            <div
              style={{
                width: 16,
                height: 16,
                backgroundColor: "#fd7e14",
                borderRadius: 3,
              }}
            />
            <Text size="xs">Overprediction</Text>
          </Group>
        </Group>
      </Stack>

      <Text size="xs" color="dimmed" style={{ textAlign: "center" }}>
        Updates every 30 seconds
      </Text>

      <Text size="sm" style={{ textAlign: "center" }} mt="md">
        If you liked this, try{" "}
        <Anchor component={Link} to="/forecastle" weight={600}>
          Forecastle
        </Anchor>{" "}
        😊
      </Text>
    </Stack>
  );
};

export default TournamentLeaderboard;
