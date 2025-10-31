import { useMemo } from 'react';
import { Group, NumberInput, RangeSlider, Stack, Text } from '@mantine/core';

const formatHorizonLabel = (horizon) => {
  if (horizon === 1) return '1 week ahead';
  return `${horizon} weeks ahead`;
};

const ForecastleInputControls = ({ entries, onChange, maxValue, mode = 'intervals', disabled = false }) => {
  const sliderMax = useMemo(() => Math.max(maxValue, 1), [maxValue]);

  const updateEntry = (index, field, value) => {
    const nextEntries = entries.map((entry, idx) => {
      if (idx !== index) return entry;

      const nextEntry = { ...entry };

      if (field === 'median') {
        nextEntry.median = Math.max(0, value);
      } else if (field === 'interval95') {
        // Two-point slider for 95% interval
        const [lower, upper] = value;
        nextEntry.lower95 = Math.max(0, lower);
        nextEntry.upper95 = Math.max(lower, upper);
        // Ensure 50% interval stays within 95% bounds
        if (nextEntry.lower50 < nextEntry.lower95) nextEntry.lower50 = nextEntry.lower95;
        if (nextEntry.upper50 > nextEntry.upper95) nextEntry.upper50 = nextEntry.upper95;
        // Update widths for backward compatibility
        nextEntry.width95 = Math.max(nextEntry.upper95 - entry.median, entry.median - nextEntry.lower95);
      } else if (field === 'interval50') {
        // Two-point slider for 50% interval
        const [lower, upper] = value;
        nextEntry.lower50 = Math.max(nextEntry.lower95 || 0, lower);
        nextEntry.upper50 = Math.min(nextEntry.upper95 || sliderMax, Math.max(lower, upper));
        // Update widths for backward compatibility
        nextEntry.width50 = Math.max(nextEntry.upper50 - entry.median, entry.median - nextEntry.lower50);
      } else if (field === 'width95') {
        // Legacy symmetric width support
        nextEntry.width95 = Math.max(0, value);
        nextEntry.lower95 = Math.max(0, entry.median - value);
        nextEntry.upper95 = entry.median + value;
        if (nextEntry.width50 > nextEntry.width95) {
          nextEntry.width50 = nextEntry.width95;
          nextEntry.lower50 = Math.max(0, entry.median - nextEntry.width50);
          nextEntry.upper50 = entry.median + nextEntry.width50;
        }
      } else if (field === 'width50') {
        // Legacy symmetric width support
        nextEntry.width50 = Math.min(Math.max(0, value), entry.width95);
        nextEntry.lower50 = Math.max(0, entry.median - nextEntry.width50);
        nextEntry.upper50 = entry.median + nextEntry.width50;
      }

      return nextEntry;
    });

    onChange(nextEntries);
  };

  // In median mode, show only median controls
  if (mode === 'median') {
    return (
      <Group align="flex-start" gap="lg" wrap="wrap">
        {entries.map((entry, index) => (
          <Stack key={entry.horizon} gap="xs" style={{ minWidth: 160 }}>
            <Text size="sm" fw={600} ta="center">
              {formatHorizonLabel(entry.horizon)}
            </Text>

            {/* Median */}
            <Stack gap={4}>
              <Text size="xs" c="dimmed" fw={500}>Median Forecast</Text>
              <NumberInput
                value={entry.median}
                onChange={(val) => updateEntry(index, 'median', val)}
                min={0}
                max={sliderMax}
                step={10}
                size="sm"
                disabled={disabled}
              />
            </Stack>
          </Stack>
        ))}
      </Group>
    );
  }

  // In intervals mode, show two-point range sliders
  return (
    <Group align="flex-start" gap="lg" wrap="wrap">
      {entries.map((entry, index) => (
        <Stack key={entry.horizon} gap="xs" style={{ minWidth: 200 }}>
          <Text size="sm" fw={600} ta="center">
            {formatHorizonLabel(entry.horizon)}
          </Text>

          <Text size="xs" c="dimmed" ta="center">
            Median: <Text component="span" fw={600}>{Math.round(entry.median)}</Text>
          </Text>

          {/* 95% Interval - Two-point range slider */}
          <Stack gap={4}>
            <Group justify="space-between">
              <Text size="xs" c="dimmed">95% Interval</Text>
              <Text size="xs" fw={500}>
                [{Math.round(entry.lower95)}, {Math.round(entry.upper95)}]
              </Text>
            </Group>
            <RangeSlider
              value={[entry.lower95, entry.upper95]}
              onChange={(val) => updateEntry(index, 'interval95', val)}
              min={0}
              max={sliderMax}
              step={1}
              color="red"
              size="sm"
              minRange={0}
              disabled={disabled}
              marks={[
                { value: 0, label: '0' },
                { value: entry.median, label: `${Math.round(entry.median)}` },
                { value: sliderMax, label: `${Math.round(sliderMax)}` },
              ]}
            />
            <Text size="xs" c="dimmed" ta="center">
              Range: {Math.round(entry.upper95 - entry.lower95)}
            </Text>
          </Stack>

          {/* 50% Interval - Two-point range slider */}
          <Stack gap={4}>
            <Group justify="space-between">
              <Text size="xs" c="dimmed">50% Interval</Text>
              <Text size="xs" fw={500}>
                [{Math.round(entry.lower50)}, {Math.round(entry.upper50)}]
              </Text>
            </Group>
            <RangeSlider
              value={[entry.lower50, entry.upper50]}
              onChange={(val) => updateEntry(index, 'interval50', val)}
              min={entry.lower95}
              max={entry.upper95}
              step={1}
              color="pink"
              size="sm"
              minRange={0}
              disabled={disabled}
            />
            <Text size="xs" c="dimmed" ta="center">
              Range: {Math.round(entry.upper50 - entry.lower50)}
            </Text>
          </Stack>
        </Stack>
      ))}
    </Group>
  );
};

export default ForecastleInputControls;
