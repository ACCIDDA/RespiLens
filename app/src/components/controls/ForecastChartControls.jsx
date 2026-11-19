import {
  Stack,
  Text,
  SegmentedControl,
  Checkbox,
  Group,
  Switch,
  ActionIcon,
  Tooltip,
} from "@mantine/core";
import { IconArrowsShuffle } from "@tabler/icons-react";

const INTERVAL_OPTIONS = [
  { value: "median", label: "Median" },
  { value: "ci50", label: "50% interval" },
  { value: "ci95", label: "95% interval" },
];

const COLOR_OPTIONS = [
  { value: "standard", label: "Standard" },
  { value: "distinct", label: "Distinct" },
];

const SCALE_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "log10", label: "Log10" },
  { value: "log2", label: "Log2" },
  { value: "sqrt", label: "Sqrt" },
];

const ForecastChartControls = ({
  chartScale,
  setChartScale,
  intervalVisibility,
  setIntervalVisibility,
  showLegend,
  setShowLegend,
  showNowcasts = false,
  setShowNowcasts = null,
  showSampleTrajectories = false,
  setShowSampleTrajectories = null,
  showOtherGroundTruthSeasons = false,
  setShowOtherGroundTruthSeasons = null,
  disableOtherGroundTruthSeasons = false,
  showIntervals = true,
  intervalOptions = INTERVAL_OPTIONS,
  showHistogram = false,
  setShowHistogram = null,
  colorSeed = null,
  setColorSeed = null,
}) => {
  const selectedIntervals = intervalOptions
    .filter((option) => intervalVisibility?.[option.value])
    .map((option) => option.value);

  const handleIntervalChange = (values) => {
    const nextVisibility = {};
    intervalOptions.forEach((option) => {
      nextVisibility[option.value] = values.includes(option.value);
    });
    setIntervalVisibility(nextVisibility);
  };

  return (
    <Stack gap="xs">
      <Group align="center" gap="md">
        <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
          Y-scale
        </Text>
        <SegmentedControl
          data={SCALE_OPTIONS}
          value={chartScale}
          onChange={setChartScale}
          size="xs"
        />
      </Group>
      {showIntervals && (
        <Group align="center" gap="md">
          <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
            Intervals
          </Text>
          <Checkbox.Group
            value={selectedIntervals}
            onChange={handleIntervalChange}
          >
            <Group gap="sm">
              {intervalOptions.map((option) => (
                <Checkbox
                  key={option.value}
                  value={option.value}
                  label={option.label}
                  size="xs"
                />
              ))}
            </Group>
          </Checkbox.Group>
        </Group>
      )}
      <Group align="center" gap="md">
        <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
          Legend
        </Text>
        <Switch
          checked={showLegend}
          onChange={(event) => setShowLegend(event.currentTarget.checked)}
          size="sm"
          onLabel="On"
          offLabel="Off"
        />
      </Group>
      {typeof setColorSeed === "function" && (
        <Group align="center" gap="md">
          <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
            Model colors
          </Text>
          <SegmentedControl
            data={COLOR_OPTIONS}
            value={colorSeed === null ? "standard" : "distinct"}
            onChange={(value) => setColorSeed(value === "distinct" ? 0 : null)}
            size="xs"
            aria-label="Model colors"
          />
          {colorSeed !== null && (
            <Tooltip label="Shuffle the colors" openDelay={300}>
              <ActionIcon
                variant="subtle"
                color="gray"
                onClick={() => setColorSeed(colorSeed + 1)}
                aria-label="Shuffle the colors"
              >
                <IconArrowsShuffle size={16} />
              </ActionIcon>
            </Tooltip>
          )}
        </Group>
      )}
      {typeof setShowNowcasts === "function" && (
        <Group align="center" gap="md">
          <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
            Allow negative horizons
          </Text>
          <Switch
            aria-label="Allow negative horizons"
            checked={showNowcasts}
            onChange={(event) => setShowNowcasts(event.currentTarget.checked)}
            size="sm"
            onLabel="On"
            offLabel="Off"
          />
        </Group>
      )}
      {typeof setShowSampleTrajectories === "function" && (
        <Group align="center" gap="md">
          <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
            Sample trajectories
          </Text>
          <Switch
            aria-label="Sample trajectories"
            checked={showSampleTrajectories}
            onChange={(event) =>
              setShowSampleTrajectories(event.currentTarget.checked)
            }
            size="sm"
            onLabel="On"
            offLabel="Off"
          />
        </Group>
      )}
      {typeof setShowHistogram === "function" && (
        <Group align="center" gap="md">
          <Text size="xs" c="dimmed" style={{ minWidth: 90 }}>
            Categorical forecasts
          </Text>
          <Switch
            aria-label="Show categorical forecasts"
            checked={showHistogram}
            onChange={(event) => setShowHistogram(event.currentTarget.checked)}
            size="sm"
            onLabel="On"
            offLabel="Off"
          />
        </Group>
      )}
      {typeof setShowOtherGroundTruthSeasons === "function" && (
        <Group align="center" gap="md">
          <Text
            size="xs"
            c={disableOtherGroundTruthSeasons ? "gray.6" : "dimmed"}
            style={{ minWidth: 90 }}
          >
            Historical seasons
          </Text>
          <Switch
            checked={showOtherGroundTruthSeasons}
            disabled={disableOtherGroundTruthSeasons}
            onChange={(event) =>
              setShowOtherGroundTruthSeasons(event.currentTarget.checked)
            }
            size="sm"
            onLabel="On"
            offLabel="Off"
          />
        </Group>
      )}
    </Stack>
  );
};

export default ForecastChartControls;
