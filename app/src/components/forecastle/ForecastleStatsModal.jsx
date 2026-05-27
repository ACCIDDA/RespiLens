import { useMemo, useState, useEffect } from "react";
import {
  Modal,
  Stack,
  Text,
  Grid,
  Paper,
  Badge,
  Table,
  Group,
  ThemeIcon,
  Button,
  Alert,
  Divider,
  ScrollArea,
  Tooltip,
} from "@mantine/core";
import {
  IconChartBar,
  IconTarget,
  IconFlame,
  IconTrophy,
  IconDownload,
  IconTrash,
  IconAlertCircle,
  IconCopy,
  IconCheck as IconCheckCircle,
} from "@tabler/icons-react";
import { useRespilensStats } from "../../hooks/useRespilensStats";
import {
  clearForecastleGames,
  exportForecastleData,
} from "../../utils/respilensStorage";

const StatCard = ({ icon, label, value, color = "blue" }) => (
  <Paper p="md" withBorder>
    <Stack gap="xs">
      <Group gap="xs">
        <ThemeIcon color={color} variant="light" size="lg">
          {icon}
        </ThemeIcon>
        <Text size="sm" c="dimmed">
          {label}
        </Text>
      </Group>
      <Text size="xl" fw={700}>
        {value}
      </Text>
    </Stack>
  </Paper>
);

const ForecastleStatsModal = ({ opened, onClose }) => {
  const [refreshKey, setRefreshKey] = useState(0);
  const stats = useRespilensStats(refreshKey);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [exportError, setExportError] = useState(null);
  const [copied, setCopied] = useState(false);

  // Refresh stats when modal opens
  useEffect(() => {
    if (opened) {
      setRefreshKey((prev) => prev + 1);
      setExportError(null);
      setShowClearConfirm(false);
    }
  }, [opened]);

  const handleExport = () => {
    try {
      setExportError(null);
      const data = exportForecastleData();
      const blob = new Blob([data], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `respilens-forecastle-history-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
      setExportError("Failed to export data. Please try again.");
    }
  };

  const handleClear = () => {
    clearForecastleGames();
    setRefreshKey((prev) => prev + 1);
    setShowClearConfirm(false);
  };

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  // Get dataset label
  const getDatasetLabel = (key) => {
    const labels = {
      flusight: "Flu",
      rsv: "RSV",
      covid19: "COVID-19",
    };
    return labels[key] || key;
  };

  // Sort game history by date (most recent first)
  const sortedHistory = useMemo(() => {
    return [...stats.gameHistory].sort(
      (a, b) => new Date(b.challengeDate) - new Date(a.challengeDate),
    );
  }, [stats.gameHistory]);

  // Group games by pathogen and calculate stats
  const pathogenStats = useMemo(() => {
    const groups = {};

    stats.gameHistory.forEach((game) => {
      const pathogen = game.dataset;
      if (!groups[pathogen]) {
        groups[pathogen] = {
          games: [],
          totalWIS: 0,
          totalDispersion: 0,
          totalUnderprediction: 0,
          totalOverprediction: 0,
          totalEnsembleWIS: 0,
          totalEnsembleDispersion: 0,
          totalEnsembleUnderprediction: 0,
          totalEnsembleOverprediction: 0,
          totalBaselineWIS: 0,
          ensembleCount: 0,
          baselineCount: 0,
          beatEnsembleCount: 0,
          beatBaselineCount: 0,
          totalRankDiff: 0,
          rankDiffCount: 0,
          totalRankPercentile: 0,
          rankPercentileCount: 0,
          count: 0,
          coverage95Count: 0,
          coverage95Total: 0,
          coverage50Count: 0,
          coverage50Total: 0,
        };
      }

      if (Number.isFinite(game.wis)) {
        groups[pathogen].games.push(game);
        groups[pathogen].totalWIS += game.wis;
        groups[pathogen].totalDispersion += game.dispersion || 0;
        groups[pathogen].totalUnderprediction += game.underprediction || 0;
        groups[pathogen].totalOverprediction += game.overprediction || 0;
        groups[pathogen].count += 1;

        // Track coverage per pathogen
        if (game.groundTruth && game.horizonDates && game.userForecasts) {
          game.horizonDates.forEach((date, idx) => {
            const truthValue = game.groundTruth[idx];
            const forecast = game.userForecasts[idx];

            if (Number.isFinite(truthValue) && forecast) {
              // Check 95% coverage
              if (
                Number.isFinite(forecast.lower95) &&
                Number.isFinite(forecast.upper95)
              ) {
                groups[pathogen].coverage95Total += 1;
                if (
                  truthValue >= forecast.lower95 &&
                  truthValue <= forecast.upper95
                ) {
                  groups[pathogen].coverage95Count += 1;
                }
              }
              // Check 50% coverage
              if (
                Number.isFinite(forecast.lower50) &&
                Number.isFinite(forecast.upper50)
              ) {
                groups[pathogen].coverage50Total += 1;
                if (
                  truthValue >= forecast.lower50 &&
                  truthValue <= forecast.upper50
                ) {
                  groups[pathogen].coverage50Count += 1;
                }
              }
            }
          });
        }

        // Track ensemble and baseline scores
        if (Number.isFinite(game.ensembleWIS)) {
          groups[pathogen].totalEnsembleWIS += game.ensembleWIS;
          groups[pathogen].totalEnsembleDispersion +=
            game.ensembleDispersion || 0;
          groups[pathogen].totalEnsembleUnderprediction +=
            game.ensembleUnderprediction || 0;
          groups[pathogen].totalEnsembleOverprediction +=
            game.ensembleOverprediction || 0;
          groups[pathogen].ensembleCount += 1;

          // Count if user beat ensemble
          if (game.wis < game.ensembleWIS) {
            groups[pathogen].beatEnsembleCount += 1;
          }
        }
        if (Number.isFinite(game.baselineWIS)) {
          groups[pathogen].totalBaselineWIS += game.baselineWIS;
          groups[pathogen].baselineCount += 1;

          // Count if user beat baseline
          if (game.wis < game.baselineWIS) {
            groups[pathogen].beatBaselineCount += 1;
          }
        }

        // Track rank difference from ensemble
        if (
          Number.isFinite(game.userRank) &&
          Number.isFinite(game.ensembleRank)
        ) {
          const rankDiff = game.ensembleRank - game.userRank; // Positive if user is better
          groups[pathogen].totalRankDiff += rankDiff;
          groups[pathogen].rankDiffCount += 1;
        }

        // Track rank percentile (what % of models the user beat)
        if (
          Number.isFinite(game.userRank) &&
          Number.isFinite(game.totalModels) &&
          game.totalModels > 0
        ) {
          // Calculate percentile: (total - rank + 1) / (total + 1) * 100
          // This gives the % of the field the user beat (including themselves)
          const percentile =
            ((game.totalModels - game.userRank + 1) / (game.totalModels + 1)) *
            100;
          groups[pathogen].totalRankPercentile += percentile;
          groups[pathogen].rankPercentileCount += 1;
        }
      }
    });

    // Calculate averages
    const results = Object.entries(groups).map(([pathogen, data]) => ({
      pathogen,
      count: data.count,
      averageWIS: data.count > 0 ? data.totalWIS / data.count : null,
      averageDispersion:
        data.count > 0 ? data.totalDispersion / data.count : null,
      averageUnderprediction:
        data.count > 0 ? data.totalUnderprediction / data.count : null,
      averageOverprediction:
        data.count > 0 ? data.totalOverprediction / data.count : null,
      averageEnsembleWIS:
        data.ensembleCount > 0
          ? data.totalEnsembleWIS / data.ensembleCount
          : null,
      averageEnsembleDispersion:
        data.ensembleCount > 0
          ? data.totalEnsembleDispersion / data.ensembleCount
          : null,
      averageEnsembleUnderprediction:
        data.ensembleCount > 0
          ? data.totalEnsembleUnderprediction / data.ensembleCount
          : null,
      averageEnsembleOverprediction:
        data.ensembleCount > 0
          ? data.totalEnsembleOverprediction / data.ensembleCount
          : null,
      averageBaselineWIS:
        data.baselineCount > 0
          ? data.totalBaselineWIS / data.baselineCount
          : null,
      beatEnsembleCount: data.beatEnsembleCount,
      beatBaselineCount: data.beatBaselineCount,
      ensembleGamesCount: data.ensembleCount,
      baselineGamesCount: data.baselineCount,
      meanRankDiff:
        data.rankDiffCount > 0 ? data.totalRankDiff / data.rankDiffCount : null,
      averageRankPercentile:
        data.rankPercentileCount > 0
          ? data.totalRankPercentile / data.rankPercentileCount
          : null,
      coverage95Percent:
        data.coverage95Total > 0
          ? (data.coverage95Count / data.coverage95Total) * 100
          : null,
      coverage50Percent:
        data.coverage50Total > 0
          ? (data.coverage50Count / data.coverage50Total) * 100
          : null,
      coverage95Count: data.coverage95Count,
      coverage95Total: data.coverage95Total,
      coverage50Count: data.coverage50Count,
      coverage50Total: data.coverage50Total,
    }));

    // Sort by average WIS (best first)
    return results.sort(
      (a, b) => (a.averageWIS || Infinity) - (b.averageWIS || Infinity),
    );
  }, [stats.gameHistory]);

  // Calculate coverage percentages with color coding
  const getCoverageColor = (percent, expectedCoverage) => {
    if (percent === null) return "gray";
    const diff = Math.abs(percent - expectedCoverage);
    if (diff < 5) return "green"; // Well-calibrated
    if (diff < 15) return "yellow"; // Somewhat calibrated
    return "red"; // Poorly calibrated
  };

  // Generate Wordle-style shareable summary
  const generateShareSummary = () => {
    const lines = ["RespiLens.com/forecastle Stats 📊\n"];

    pathogenStats.forEach((stat) => {
      const pathogenLabel = getDatasetLabel(stat.pathogen);

      // Coverage with emojis and numbers
      const coverage95 =
        stat.coverage95Percent !== null
          ? Math.round(stat.coverage95Percent)
          : null;
      const coverage50 =
        stat.coverage50Percent !== null
          ? Math.round(stat.coverage50Percent)
          : null;

      let coverage95Emoji = "⚪";
      if (coverage95 !== null) {
        const diff95 = Math.abs(coverage95 - 95);
        if (diff95 < 5)
          coverage95Emoji = "🟢"; // Excellent
        else if (diff95 < 15)
          coverage95Emoji = "🟡"; // Good
        else coverage95Emoji = "🔴"; // Poor
      }

      let coverage50Emoji = "⚪";
      if (coverage50 !== null) {
        const diff50 = Math.abs(coverage50 - 50);
        if (diff50 < 5)
          coverage50Emoji = "🟢"; // Excellent
        else if (diff50 < 15)
          coverage50Emoji = "🟡"; // Good
        else coverage50Emoji = "🔴"; // Poor
      }

      lines.push(`\n${pathogenLabel} (${stat.count} games)`);

      // Coverage - only show if not N/A
      if (coverage95 !== null || coverage50 !== null) {
        const coverage95Text =
          coverage95 !== null
            ? `${coverage95Emoji} ${coverage95}% (${stat.coverage95Count}/${stat.coverage95Total})`
            : null;
        const coverage50Text =
          coverage50 !== null
            ? `${coverage50Emoji} ${coverage50}% (${stat.coverage50Count}/${stat.coverage50Total})`
            : null;

        const coverageParts = [];
        if (coverage95Text) coverageParts.push(`95%: ${coverage95Text}`);
        if (coverage50Text) coverageParts.push(`50%: ${coverage50Text}`);

        if (coverageParts.length > 0) {
          lines.push(`Coverage: ${coverageParts.join(" | ")}`);
        }
      }

      // Beat ensemble with emojis
      const beatPercent =
        stat.ensembleGamesCount > 0
          ? Math.round((stat.beatEnsembleCount / stat.ensembleGamesCount) * 100)
          : null;

      let ensembleEmoji = "😐";
      if (beatPercent !== null) {
        if (beatPercent >= 75)
          ensembleEmoji = "😎"; // Crushing it
        else if (beatPercent >= 50)
          ensembleEmoji = "😊"; // Beating ensemble
        else if (beatPercent >= 25)
          ensembleEmoji = "🙂"; // Competitive
        else ensembleEmoji = "😅"; // Room to grow
      }

      lines.push(
        `Beat Ensemble: ${ensembleEmoji} ${stat.beatEnsembleCount}/${stat.ensembleGamesCount}`,
      );
    });

    return lines.join("\n");
  };

  const handleCopyStats = async () => {
    try {
      const summary = generateShareSummary();
      await navigator.clipboard.writeText(summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const coverage95Color = getCoverageColor(stats.coverage95Percent, 95);
  const coverage50Color = getCoverageColor(stats.coverage50Percent, 50);

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={
        <Group
          gap="xs"
          justify="space-between"
          style={{ width: "100%", paddingRight: "40px" }}
        >
          <Group gap="xs">
            <IconChartBar size={24} />
            <Text size="lg" fw={700}>
              Your Forecastle Statistics
            </Text>
          </Group>
          <Button
            variant="light"
            size="sm"
            onClick={handleCopyStats}
            leftSection={
              copied ? <IconCheckCircle size={16} /> : <IconCopy size={16} />
            }
            color={copied ? "teal" : "blue"}
          >
            {copied ? "Copied!" : "Share"}
          </Button>
        </Group>
      }
      size="xl"
      scrollAreaComponent={ScrollArea.Autosize}
    >
      <Stack gap="lg">
        {stats.gamesPlayed === 0 ? (
          <Alert icon={<IconAlertCircle size={16} />} color="blue">
            No games played yet. Complete your first Forecastle challenge to see
            your statistics!
          </Alert>
        ) : (
          <>
            {/* Overview Stats */}
            <Grid>
              <Grid.Col span={{ base: 12, sm: 6, md: 4 }}>
                <StatCard
                  icon={<IconTarget size={20} />}
                  label="Games Played"
                  value={stats.gamesPlayed}
                  color="blue"
                />
              </Grid.Col>
              <Grid.Col span={{ base: 12, sm: 6, md: 4 }}>
                <StatCard
                  icon={<IconTrophy size={20} />}
                  label="Avg Rank vs Ensemble"
                  value={
                    stats.averageRankVsEnsemble !== null
                      ? `${stats.averageRankVsEnsemble > 0 ? "+" : ""}${stats.averageRankVsEnsemble.toFixed(1)}`
                      : "N/A"
                  }
                  color={
                    stats.averageRankVsEnsemble !== null &&
                    stats.averageRankVsEnsemble > 0
                      ? "green"
                      : "cyan"
                  }
                />
              </Grid.Col>
              <Grid.Col span={{ base: 12, sm: 6, md: 4 }}>
                <StatCard
                  icon={<IconFlame size={20} />}
                  label="Current Streak"
                  value={`${stats.currentStreak} day${stats.currentStreak !== 1 ? "s" : ""}`}
                  color="orange"
                />
              </Grid.Col>
            </Grid>

            {/* Interval Coverage */}
            <Paper p="md" withBorder>
              <Stack gap="md">
                <Text size="sm" fw={600}>
                  Interval Coverage (Forecast Calibration)
                </Text>
                <Text size="xs" c="dimmed">
                  Shows how often the true value fell within your prediction
                  intervals. Well-calibrated forecasts should have ~95% coverage
                  for 95% intervals and ~50% for 50% intervals.
                </Text>
                <Group grow>
                  <Paper
                    p="sm"
                    withBorder
                    bg={
                      coverage95Color === "green"
                        ? "green.0"
                        : coverage95Color === "yellow"
                          ? "yellow.0"
                          : "red.0"
                    }
                  >
                    <Stack gap="xs" align="center">
                      <Text size="xs" c="dimmed">
                        95% Interval Coverage
                      </Text>
                      <Group gap="xs">
                        <Text size="xl" fw={700}>
                          {stats.coverage95Percent !== null
                            ? `${stats.coverage95Percent.toFixed(1)}%`
                            : "N/A"}
                        </Text>
                        <Badge color={coverage95Color} size="xs">
                          {stats.coverage95Percent !== null
                            ? Math.abs(stats.coverage95Percent - 95) < 5
                              ? "Excellent"
                              : Math.abs(stats.coverage95Percent - 95) < 15
                                ? "Good"
                                : stats.coverage95Percent < 95
                                  ? "Too narrow"
                                  : "Too wide"
                            : "N/A"}
                        </Badge>
                      </Group>
                    </Stack>
                  </Paper>
                  <Paper
                    p="sm"
                    withBorder
                    bg={
                      coverage50Color === "green"
                        ? "green.0"
                        : coverage50Color === "yellow"
                          ? "yellow.0"
                          : "red.0"
                    }
                  >
                    <Stack gap="xs" align="center">
                      <Text size="xs" c="dimmed">
                        50% Interval Coverage
                      </Text>
                      <Group gap="xs">
                        <Text size="xl" fw={700}>
                          {stats.coverage50Percent !== null
                            ? `${stats.coverage50Percent.toFixed(1)}%`
                            : "N/A"}
                        </Text>
                        <Badge color={coverage50Color} size="xs">
                          {stats.coverage50Percent !== null
                            ? Math.abs(stats.coverage50Percent - 50) < 5
                              ? "Excellent"
                              : Math.abs(stats.coverage50Percent - 50) < 15
                                ? "Good"
                                : stats.coverage50Percent < 50
                                  ? "Too narrow"
                                  : "Too wide"
                            : "N/A"}
                        </Badge>
                      </Group>
                    </Stack>
                  </Paper>
                </Group>
              </Stack>
            </Paper>

            {/* Pathogen-based Performance */}
            {pathogenStats.length > 0 && (
              <Paper p="md" withBorder>
                <Stack gap="md">
                  <Text size="sm" fw={600}>
                    Performance by Pathogen
                  </Text>
                  <Text size="xs" c="dimmed">
                    Average relative WIS scores grouped by disease type. Compare
                    your performance against the hub ensemble and baseline.
                    Lower scores indicate better forecasting.
                  </Text>

                  {/* Summary stats */}
                  {pathogenStats.map((stat) => {
                    return (
                      <Paper key={stat.pathogen} p="sm" withBorder bg="gray.0">
                        <Stack gap="xs">
                          <Group justify="space-between">
                            <Badge size="md" variant="light">
                              {getDatasetLabel(stat.pathogen)}
                            </Badge>
                            <Text size="xs" c="dimmed">
                              {stat.count} games
                            </Text>
                          </Group>

                          <Group gap="lg" wrap="wrap">
                            <div>
                              <Text size="xs" c="dimmed">
                                95% Coverage
                              </Text>
                              <Text
                                size="md"
                                fw={600}
                                c={
                                  stat.coverage95Percent !== null &&
                                  Math.abs(stat.coverage95Percent - 95) < 5
                                    ? "green"
                                    : stat.coverage95Percent !== null &&
                                        Math.abs(stat.coverage95Percent - 95) <
                                          15
                                      ? "yellow"
                                      : "red"
                                }
                              >
                                {stat.coverage95Percent !== null
                                  ? `${stat.coverage95Percent.toFixed(1)}%`
                                  : "N/A"}
                              </Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">
                                50% Coverage
                              </Text>
                              <Text
                                size="md"
                                fw={600}
                                c={
                                  stat.coverage50Percent !== null &&
                                  Math.abs(stat.coverage50Percent - 50) < 5
                                    ? "green"
                                    : stat.coverage50Percent !== null &&
                                        Math.abs(stat.coverage50Percent - 50) <
                                          15
                                      ? "yellow"
                                      : "red"
                                }
                              >
                                {stat.coverage50Percent !== null
                                  ? `${stat.coverage50Percent.toFixed(1)}%`
                                  : "N/A"}
                              </Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">
                                Beat Ensemble
                              </Text>
                              <Text
                                size="md"
                                fw={600}
                                c={
                                  stat.beatEnsembleCount >
                                  stat.ensembleGamesCount / 2
                                    ? "green"
                                    : "red"
                                }
                              >
                                {stat.beatEnsembleCount}/
                                {stat.ensembleGamesCount} times
                              </Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">
                                Beat Baseline
                              </Text>
                              <Text
                                size="md"
                                fw={600}
                                c={
                                  stat.beatBaselineCount >
                                  stat.baselineGamesCount / 2
                                    ? "green"
                                    : "red"
                                }
                              >
                                {stat.beatBaselineCount}/
                                {stat.baselineGamesCount} times
                              </Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">
                                Mean Rank vs Ensemble
                              </Text>
                              <Text
                                size="md"
                                fw={600}
                                c={
                                  stat.meanRankDiff !== null &&
                                  stat.meanRankDiff > 0
                                    ? "green"
                                    : "red"
                                }
                              >
                                {stat.meanRankDiff !== null
                                  ? `${stat.meanRankDiff > 0 ? "+" : ""}${stat.meanRankDiff.toFixed(1)} spots`
                                  : "N/A"}
                              </Text>
                            </div>
                            <div>
                              <Text size="xs" c="dimmed">
                                Average Rank Percentile
                              </Text>
                              <Text
                                size="md"
                                fw={600}
                                c={
                                  stat.averageRankPercentile !== null &&
                                  stat.averageRankPercentile >= 50
                                    ? "green"
                                    : "orange"
                                }
                              >
                                {stat.averageRankPercentile !== null
                                  ? `Top ${(100 - stat.averageRankPercentile).toFixed(1)}%`
                                  : "N/A"}
                              </Text>
                            </div>
                          </Group>

                          {/* WIS Components Stacked Bar */}
                          <div>
                            <Text size="xs" c="dimmed" mb={4}>
                              WIS Components Comparison
                            </Text>
                            <Stack gap={4}>
                              {/* User bar */}
                              <div>
                                {(() => {
                                  const userComponentTotal =
                                    (stat.averageUnderprediction || 0) +
                                    (stat.averageOverprediction || 0) +
                                    (stat.averageDispersion || 0);
                                  return (
                                    <>
                                      <Text size="xs" fw={500} mb={2}>
                                        You
                                      </Text>
                                      <Group gap={0} style={{ height: 24 }}>
                                        {stat.averageUnderprediction !== null &&
                                          stat.averageUnderprediction > 0 && (
                                            <div
                                              style={{
                                                width: `${(stat.averageUnderprediction / (userComponentTotal || 1)) * 100}%`,
                                                height: "100%",
                                                backgroundColor: "#4c6ef5",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                              }}
                                              title={`Underprediction: ${stat.averageUnderprediction.toFixed(3)}`}
                                            >
                                              {stat.averageUnderprediction >
                                                5 && (
                                                <Text size="xs" c="white">
                                                  {stat.averageUnderprediction.toFixed(
                                                    1,
                                                  )}
                                                </Text>
                                              )}
                                            </div>
                                          )}
                                        {stat.averageOverprediction !== null &&
                                          stat.averageOverprediction > 0 && (
                                            <div
                                              style={{
                                                width: `${(stat.averageOverprediction / (userComponentTotal || 1)) * 100}%`,
                                                height: "100%",
                                                backgroundColor: "#f03e3e",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                              }}
                                              title={`Overprediction: ${stat.averageOverprediction.toFixed(3)}`}
                                            >
                                              {stat.averageOverprediction >
                                                5 && (
                                                <Text size="xs" c="white">
                                                  {stat.averageOverprediction.toFixed(
                                                    1,
                                                  )}
                                                </Text>
                                              )}
                                            </div>
                                          )}
                                        {stat.averageDispersion !== null && (
                                          <div
                                            style={{
                                              width: `${(stat.averageDispersion / (userComponentTotal || 1)) * 100}%`,
                                              height: "100%",
                                              backgroundColor: "#adb5bd",
                                              display: "flex",
                                              alignItems: "center",
                                              justifyContent: "center",
                                            }}
                                            title={`Dispersion: ${stat.averageDispersion.toFixed(3)}`}
                                          >
                                            {stat.averageDispersion > 5 && (
                                              <Text size="xs" c="white">
                                                {stat.averageDispersion.toFixed(
                                                  1,
                                                )}
                                              </Text>
                                            )}
                                          </div>
                                        )}
                                      </Group>
                                    </>
                                  );
                                })()}
                              </div>

                              {/* Ensemble bar */}
                              {stat.averageEnsembleWIS !== null && (
                                <div>
                                  {(() => {
                                    const ensembleComponentTotal =
                                      (stat.averageEnsembleUnderprediction ||
                                        0) +
                                      (stat.averageEnsembleOverprediction ||
                                        0) +
                                      (stat.averageEnsembleDispersion || 0);
                                    return (
                                      <>
                                        <Text size="xs" fw={500} mb={2}>
                                          Ensemble
                                        </Text>
                                        <Group gap={0} style={{ height: 24 }}>
                                          {stat.averageEnsembleUnderprediction !==
                                            null &&
                                            stat.averageEnsembleUnderprediction >
                                              0 && (
                                              <div
                                                style={{
                                                  width: `${(stat.averageEnsembleUnderprediction / (ensembleComponentTotal || 1)) * 100}%`,
                                                  height: "100%",
                                                  backgroundColor: "#4c6ef5",
                                                  display: "flex",
                                                  alignItems: "center",
                                                  justifyContent: "center",
                                                }}
                                                title={`Underprediction: ${stat.averageEnsembleUnderprediction.toFixed(3)}`}
                                              >
                                                {stat.averageEnsembleUnderprediction >
                                                  5 && (
                                                  <Text size="xs" c="white">
                                                    {stat.averageEnsembleUnderprediction.toFixed(
                                                      1,
                                                    )}
                                                  </Text>
                                                )}
                                              </div>
                                            )}
                                          {stat.averageEnsembleOverprediction !==
                                            null &&
                                            stat.averageEnsembleOverprediction >
                                              0 && (
                                              <div
                                                style={{
                                                  width: `${(stat.averageEnsembleOverprediction / (ensembleComponentTotal || 1)) * 100}%`,
                                                  height: "100%",
                                                  backgroundColor: "#f03e3e",
                                                  display: "flex",
                                                  alignItems: "center",
                                                  justifyContent: "center",
                                                }}
                                                title={`Overprediction: ${stat.averageEnsembleOverprediction.toFixed(3)}`}
                                              >
                                                {stat.averageEnsembleOverprediction >
                                                  5 && (
                                                  <Text size="xs" c="white">
                                                    {stat.averageEnsembleOverprediction.toFixed(
                                                      1,
                                                    )}
                                                  </Text>
                                                )}
                                              </div>
                                            )}
                                          {stat.averageEnsembleDispersion !==
                                            null && (
                                            <div
                                              style={{
                                                width: `${(stat.averageEnsembleDispersion / (ensembleComponentTotal || 1)) * 100}%`,
                                                height: "100%",
                                                backgroundColor: "#adb5bd",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                              }}
                                              title={`Dispersion: ${stat.averageEnsembleDispersion.toFixed(3)}`}
                                            >
                                              {stat.averageEnsembleDispersion >
                                                5 && (
                                                <Text size="xs" c="white">
                                                  {stat.averageEnsembleDispersion.toFixed(
                                                    1,
                                                  )}
                                                </Text>
                                              )}
                                            </div>
                                          )}
                                        </Group>
                                      </>
                                    );
                                  })()}
                                </div>
                              )}
                            </Stack>
                            <Group gap="md" mt={4}>
                              <Group gap={4}>
                                <div
                                  style={{
                                    width: 12,
                                    height: 12,
                                    backgroundColor: "#adb5bd",
                                  }}
                                />
                                <Text size="xs" c="dimmed">
                                  Dispersion
                                </Text>
                              </Group>
                              <Group gap={4}>
                                <div
                                  style={{
                                    width: 12,
                                    height: 12,
                                    backgroundColor: "#4c6ef5",
                                  }}
                                />
                                <Text size="xs" c="dimmed">
                                  Underprediction
                                </Text>
                              </Group>
                              <Group gap={4}>
                                <div
                                  style={{
                                    width: 12,
                                    height: 12,
                                    backgroundColor: "#f03e3e",
                                  }}
                                />
                                <Text size="xs" c="dimmed">
                                  Overprediction
                                </Text>
                              </Group>
                            </Group>
                          </div>
                        </Stack>
                      </Paper>
                    );
                  })}
                </Stack>
              </Paper>
            )}

            {/* Streaks */}
            {stats.maxStreak > 1 && (
              <Paper p="md" withBorder>
                <Group justify="space-between">
                  <Group gap="xs">
                    <ThemeIcon color="orange" variant="light">
                      <IconFlame size={20} />
                    </ThemeIcon>
                    <div>
                      <Text size="sm" fw={600}>
                        Max Streak
                      </Text>
                      <Text size="xs" c="dimmed">
                        Longest consecutive days played
                      </Text>
                    </div>
                  </Group>
                  <Text size="xl" fw={700}>
                    {stats.maxStreak} days
                  </Text>
                </Group>
              </Paper>
            )}

            <Divider />

            {/* Game History */}
            <Stack gap="sm">
              <Text size="sm" fw={600}>
                Game History ({sortedHistory.length} games)
              </Text>
              <ScrollArea h={300}>
                <Table striped highlightOnHover>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Date</Table.Th>
                      <Table.Th>Dataset</Table.Th>
                      <Table.Th>Location</Table.Th>
                      <Table.Th>rWIS</Table.Th>
                      <Table.Th>
                        <Tooltip label="Your rank vs all models">
                          <Text size="sm" style={{ cursor: "help" }}>
                            Rank
                          </Text>
                        </Tooltip>
                      </Table.Th>
                      <Table.Th>
                        <Tooltip label="Your position relative to the hub ensemble">
                          <Text size="sm" style={{ cursor: "help" }}>
                            vs Ensemble
                          </Text>
                        </Tooltip>
                      </Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {sortedHistory.map((game) => {
                      const spotsDiffEnsemble =
                        game.userRank && game.ensembleRank
                          ? game.ensembleRank - game.userRank
                          : null;

                      return (
                        <Table.Tr key={game.id}>
                          <Table.Td>
                            <Text size="sm">{formatDate(game.playedAt)}</Text>
                          </Table.Td>
                          <Table.Td>
                            <Badge size="sm" variant="light">
                              {getDatasetLabel(game.dataset)}
                            </Badge>
                          </Table.Td>
                          <Table.Td>
                            <Text size="sm">{game.location}</Text>
                          </Table.Td>
                          <Table.Td>
                            <Text size="sm" fw={500}>
                              {game.wis !== null ? game.wis.toFixed(3) : "N/A"}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            <Text size="sm" c="dimmed">
                              {game.userRank && game.totalModels
                                ? `#${game.userRank}/${game.totalModels}`
                                : "N/A"}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            {spotsDiffEnsemble !== null ? (
                              <Badge
                                size="sm"
                                color={
                                  spotsDiffEnsemble > 0
                                    ? "green"
                                    : spotsDiffEnsemble < 0
                                      ? "red"
                                      : "gray"
                                }
                                variant="light"
                              >
                                {spotsDiffEnsemble > 0
                                  ? `+${spotsDiffEnsemble}`
                                  : spotsDiffEnsemble < 0
                                    ? spotsDiffEnsemble
                                    : "0"}
                              </Badge>
                            ) : (
                              <Text size="sm" c="dimmed">
                                N/A
                              </Text>
                            )}
                          </Table.Td>
                        </Table.Tr>
                      );
                    })}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Stack>

            <Divider />

            {/* Export Error */}
            {exportError && (
              <Alert
                icon={<IconAlertCircle size={16} />}
                color="red"
                onClose={() => setExportError(null)}
                withCloseButton
              >
                {exportError}
              </Alert>
            )}

            {/* Actions */}
            <Group justify="space-between">
              <Button
                leftSection={<IconDownload size={16} />}
                variant="light"
                onClick={handleExport}
              >
                Export Data
              </Button>
              {!showClearConfirm ? (
                <Button
                  leftSection={<IconTrash size={16} />}
                  variant="light"
                  color="red"
                  onClick={() => setShowClearConfirm(true)}
                >
                  Clear History
                </Button>
              ) : (
                <Group gap="xs">
                  <Text size="sm" c="dimmed">
                    Are you sure?
                  </Text>
                  <Button size="xs" color="red" onClick={handleClear}>
                    Yes, Clear All
                  </Button>
                  <Button
                    size="xs"
                    variant="light"
                    onClick={() => setShowClearConfirm(false)}
                  >
                    Cancel
                  </Button>
                </Group>
              )}
            </Group>
          </>
        )}
      </Stack>
    </Modal>
  );
};

export default ForecastleStatsModal;
