import { useState, useEffect, useMemo } from "react";
import {
  Card,
  Title,
  Text,
  Button,
  Badge,
  Stack,
  Group,
  Modal,
  Alert,
  Loader,
  Stepper,
  Box,
  Switch,
} from "@mantine/core";
import {
  IconCheck,
  IconLock,
  IconEdit,
  IconAlertCircle,
} from "@tabler/icons-react";
import { useForecastData } from "../../hooks/useForecastData";
import { submitForecast, getParticipant } from "../../utils/tournamentAPI";
import { TOURNAMENT_CONFIG } from "../../config";
import {
  initialiseForecastInputs,
  convertToIntervals,
} from "../../utils/forecastleInputs";
import { validateForecastSubmission } from "../../utils/forecastleValidation";
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

const TournamentChallengeCard = ({
  tournamentConfig = TOURNAMENT_CONFIG,
  challenge,
  participantId,
  isCompleted,
  onSubmissionComplete,
}) => {
  const [modalOpened, setModalOpened] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [inputMode, setInputMode] = useState("median"); // 'median' or 'intervals'
  const [existingSubmission, setExistingSubmission] = useState(null);
  const [zoomedView, setZoomedView] = useState(true); // Start with zoomed view

  // Map dataset to viewType
  const getViewType = (dataset) => {
    const map = {
      flu: "flu_projs",
      covid: "covid_projs",
      rsv: "rsv_projs",
    };
    return map[dataset] || "flu_projs";
  };

  // Load forecast data for the challenge (for displaying chart)
  const { data, loading: dataLoading } = useForecastData(
    challenge.location,
    getViewType(challenge.dataset),
  );

  // Get ground truth data for the chart
  const groundTruthSeries = useMemo(() => {
    return data?.ground_truth?.dates && data?.ground_truth?.[challenge.target]
      ? data.ground_truth.dates
          .map((date, idx) => ({
            date,
            value: data.ground_truth[challenge.target][idx],
          }))
          .filter((entry) => entry.value !== null)
          .slice(-20) // Last 20 points
      : [];
  }, [data, challenge.target]);

  // Calculate latest observation value for initializing forecasts
  const latestObservationValue = useMemo(() => {
    if (!groundTruthSeries || groundTruthSeries.length === 0) return 1000;
    const lastValue = groundTruthSeries[groundTruthSeries.length - 1]?.value;
    return Number.isFinite(lastValue) ? lastValue : 1000;
  }, [groundTruthSeries]);

  // Forecast state - array of forecasts for each horizon (like Forecastle)
  const initialForecasts = useMemo(
    () =>
      initialiseForecastInputs(
        challenge.horizons || [1, 2, 3],
        latestObservationValue,
      ),
    [challenge.horizons, latestObservationValue],
  );

  const [forecastEntries, setForecastEntries] = useState(initialForecasts);

  // Reset forecast entries when latestObservationValue changes
  useEffect(() => {
    setForecastEntries(
      initialiseForecastInputs(
        challenge.horizons || [1, 2, 3],
        latestObservationValue,
      ),
    );
  }, [challenge.horizons, latestObservationValue]);

  // Load existing submission if challenge is completed
  useEffect(() => {
    const loadExistingSubmission = async () => {
      if (isCompleted && participantId) {
        try {
          const participantData = await getParticipant(
            participantId,
            tournamentConfig,
          );
          const submission = participantData.submissions.find(
            (sub) =>
              sub.challengeId === challenge.id ||
              Number(sub.challengeNum) === Number(challenge.number),
          );
          if (submission && submission.forecasts) {
            setExistingSubmission(submission);
            // Pre-fill forecast with existing submission (supports multiple horizons)
            const restoredForecasts = submission.forecasts.map((f) => ({
              horizon: f.horizon,
              median: f.median,
              lower50: f.q25,
              upper50: f.q75,
              lower95: f.q025,
              upper95: f.q975,
              width50: Math.max(f.q75 - f.median, f.median - f.q25),
              width95: Math.max(f.q975 - f.median, f.median - f.q025),
            }));
            setForecastEntries(restoredForecasts);
          }
        } catch (err) {
          console.error("Failed to load existing submission:", err);
        }
      }
    };

    loadExistingSubmission();
  }, [
    isCompleted,
    participantId,
    challenge.id,
    challenge.number,
    tournamentConfig,
  ]);

  // Use the challenge's forecast date (historical date)
  const forecastDate = challenge.forecastDate;

  // Horizon dates
  const horizonDates = (challenge.horizons || [1, 2, 3]).map((horizon) =>
    addWeeksToDate(forecastDate, horizon),
  );

  // Calculate max value for chart
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
  const yAxisMax = Math.max(
    baseMax,
    userMaxCandidate * 1.1 || 0,
    latestObservationValue,
    1,
  );

  // Handle forecast adjustments
  const handleMedianAdjust = (index, field, value) => {
    setForecastEntries((prevEntries) =>
      prevEntries.map((entry, idx) => {
        if (idx !== index) return entry;

        const nextEntry = { ...entry };

        if (field === "median") {
          const oldMedian = entry.median;
          const newMedian = Math.max(0, value);
          const medianShift = newMedian - oldMedian;

          nextEntry.median = newMedian;

          // Shift intervals to maintain their widths relative to new median
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
          // Ensure 50% interval stays within 95% bounds
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
    setError(null);
  };

  // Handle submission with WIS scoring
  const handleSubmit = async () => {
    setError(null);

    // Validate that forecastEntries is properly populated
    if (!forecastEntries || forecastEntries.length === 0) {
      setError("No forecast entries to submit");
      return;
    }

    // Check if all entries have valid median values
    const hasInvalidEntries = forecastEntries.some(
      (entry) =>
        !entry ||
        entry.median === null ||
        entry.median === undefined ||
        !Number.isFinite(entry.median),
    );

    if (hasInvalidEntries) {
      setError("Some forecast entries have invalid median values");
      return;
    }

    // Convert to intervals for validation
    const intervalsForValidation = convertToIntervals(forecastEntries);
    const { valid, errors } = validateForecastSubmission(
      intervalsForValidation,
    );

    if (!valid) {
      setError(Object.values(errors).join(". "));
      return;
    }

    setSubmitting(true);

    try {
      // Submit forecasts for all horizons (scoring will be done on frontend)
      await submitForecast(
        participantId,
        challenge.number,
        forecastEntries,
        tournamentConfig,
      );

      setModalOpened(false);
      setInputMode("median");
      if (onSubmissionComplete) {
        onSubmissionComplete();
      }
    } catch (err) {
      setError(err.message || "Submission failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const getDatasetColor = (dataset) => {
    const colors = {
      flu: "#4c6ef5",
      covid: "#f03e3e",
      rsv: "#f59f00",
    };
    return colors[dataset] || "#868e96";
  };

  return (
    <>
      <Card
        shadow="sm"
        p="lg"
        radius="md"
        withBorder
        style={{
          borderLeft: `4px solid ${getDatasetColor(challenge.dataset)}`,
          cursor: "pointer",
          transition: "transform 0.2s",
        }}
        onClick={() => setModalOpened(true)}
      >
        <Stack spacing="sm">
          <Group position="apart">
            <Badge color={getDatasetColor(challenge.dataset)} variant="filled">
              Challenge {challenge.number}
            </Badge>
            {isCompleted ? (
              <IconCheck size={24} color="green" />
            ) : (
              <IconLock size={20} color="#adb5bd" />
            )}
          </Group>

          <Title order={4}>{challenge.title}</Title>
          <Text size="sm" color="dimmed">
            {challenge.description}
          </Text>

          <Group spacing="xs" mt="xs">
            <Badge size="sm" variant="outline">
              {challenge.dataset.toUpperCase()}
            </Badge>
            <Badge size="sm" variant="outline">
              {challenge.location}
            </Badge>
            <Badge size="sm" variant="outline">
              {challenge.horizons?.length || 3} horizons
            </Badge>
          </Group>

          {isCompleted &&
            existingSubmission &&
            existingSubmission.forecasts && (
              <Alert
                color="green"
                variant="light"
                icon={<IconCheck size={16} />}
              >
                Submitted {existingSubmission.forecasts.length} forecast
                {existingSubmission.forecasts.length !== 1 ? "s" : ""}
              </Alert>
            )}

          <Button
            variant={isCompleted ? "light" : "filled"}
            leftSection={isCompleted ? <IconEdit size={16} /> : null}
            onClick={() => setModalOpened(true)}
            fullWidth
          >
            {isCompleted ? "Update Forecast" : "Start Challenge"}
          </Button>
        </Stack>
      </Card>

      {/* Challenge Modal */}
      <Modal
        opened={modalOpened}
        onClose={() => {
          setModalOpened(false);
          setInputMode("median");
          setError(null);
        }}
        title={<Title order={3}>{challenge.title}</Title>}
        size="xl"
      >
        <Stack spacing="md">
          <Text color="dimmed">{challenge.description}</Text>

          {error && (
            <Alert
              icon={<IconAlertCircle size={16} />}
              title="Error"
              color="red"
            >
              {error}
            </Alert>
          )}

          {/* Step indicator */}
          <Stepper
            active={inputMode === "median" ? 0 : 1}
            onStepClick={(step) => {
              if (step === 0) setInputMode("median");
              else if (step === 1) setInputMode("intervals");
            }}
            size="sm"
          >
            <Stepper.Step
              label="Median"
              description="Point forecasts"
              completedIcon={<IconCheck size={16} />}
            />
            <Stepper.Step
              label="Intervals"
              description="Uncertainty"
              completedIcon={<IconCheck size={16} />}
            />
          </Stepper>

          {dataLoading ? (
            <Stack align="center" py="xl">
              <Loader />
              <Text>Loading data...</Text>
            </Stack>
          ) : (
            <>
              {/* Chart with zoom toggle */}
              <Group justify="space-between" mb="xs">
                <Title order={5}>Interactive Chart</Title>
                <Switch
                  label="Show More History"
                  checked={!zoomedView}
                  onChange={(event) =>
                    setZoomedView(!event.currentTarget.checked)
                  }
                  color="red"
                  size="md"
                />
              </Group>
              <Box style={{ width: "100%", height: 400 }}>
                <ForecastleChartCanvas
                  groundTruthSeries={groundTruthSeries}
                  horizonDates={horizonDates}
                  entries={forecastEntries}
                  maxValue={yAxisMax}
                  onAdjust={handleMedianAdjust}
                  height={400}
                  showIntervals={inputMode === "intervals"}
                  zoomedView={zoomedView}
                />
              </Box>

              <Text size="sm" c="dimmed" mt="xs">
                {inputMode === "median"
                  ? "Drag the handles to set your median forecast for each week ahead."
                  : "Drag the handles to adjust interval bounds, or use the sliders for precise control."}
              </Text>

              {/* Input Controls */}
              <ForecastleInputControls
                entries={forecastEntries}
                onChange={setForecastEntries}
                maxValue={yAxisMax}
                mode={inputMode}
              />
            </>
          )}

          <Group position="apart">
            {inputMode === "intervals" && (
              <Button variant="subtle" onClick={() => setInputMode("median")}>
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
                >
                  Submit Forecast
                </Button>
              )}
            </div>
          </Group>
        </Stack>
      </Modal>
    </>
  );
};

export default TournamentChallengeCard;
