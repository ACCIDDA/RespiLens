import React, { useMemo } from "react";
import {
  Alert,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { IconInfoCircle } from "@tabler/icons-react";

const formatValue = (value) => {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
};

const NSSPView = ({ location, data, metadata }) => {
  const summary = useMemo(() => {
    const series = data?.series;
    const dates = series?.dates || [];
    const metricNames = Object.keys(series || {}).filter(
      (key) => key !== "dates",
    );
    const latestDate = dates.length > 0 ? dates[dates.length - 1] : null;

    const latestMetrics = metricNames.map((metricName) => ({
      metricName,
      latestValue: series?.[metricName]?.[dates.length - 1] ?? null,
    }));

    return {
      locationName: data?.metadata?.location_name || location,
      datasetName: metadata?.dataset || data?.metadata?.dataset || "NSSP",
      lastUpdated: metadata?.last_updated || "Unknown",
      dateCount: dates.length,
      firstDate: dates[0] || "N/A",
      latestDate: latestDate || "N/A",
      metricCount: metricNames.length,
      latestMetrics,
    };
  }, [data, location, metadata]);

  if (!data?.series?.dates) {
    return (
      <Alert icon={<IconInfoCircle size={16} />} color="yellow" variant="light">
        NSSP data loaded, but the expected time series structure was not found.
      </Alert>
    );
  }

  return (
    <Stack gap="lg">
      <Stack gap={4}>
        <Title order={2}>NSSP Surveillance Data</Title>
        <Text c="dimmed">
          Frontend summary check for the processed NSSP payload delivered
          through the standard RespiLens data hook.
        </Text>
      </Stack>

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
        <Paper withBorder radius="md" p="md">
          <Text size="sm" c="dimmed">
            Dataset
          </Text>
          <Text fw={600}>{summary.datasetName}</Text>
        </Paper>
        <Paper withBorder radius="md" p="md">
          <Text size="sm" c="dimmed">
            Location
          </Text>
          <Text fw={600}>{summary.locationName}</Text>
        </Paper>
        <Paper withBorder radius="md" p="md">
          <Text size="sm" c="dimmed">
            Date Range
          </Text>
          <Text fw={600}>
            {summary.firstDate} to {summary.latestDate}
          </Text>
        </Paper>
        <Paper withBorder radius="md" p="md">
          <Text size="sm" c="dimmed">
            Observations
          </Text>
          <Text fw={600}>{summary.dateCount}</Text>
        </Paper>
      </SimpleGrid>

      <Paper withBorder radius="md" p="lg">
        <Stack gap="md">
          <Group justify="space-between" align="flex-start">
            <div>
              <Title order={4}>Latest Metric Values</Title>
              <Text size="sm" c="dimmed">
                Last updated: {summary.lastUpdated}
              </Text>
            </div>
            <Text size="sm" c="dimmed">
              {summary.metricCount} metrics
            </Text>
          </Group>

          <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
            {summary.latestMetrics.map(({ metricName, latestValue }) => (
              <Paper key={metricName} withBorder radius="md" p="md">
                <Text size="sm" c="dimmed">
                  {metricName}
                </Text>
                <Text fw={700} size="lg">
                  {formatValue(latestValue)}
                </Text>
                <Text size="xs" c="dimmed">
                  Latest date: {summary.latestDate}
                </Text>
              </Paper>
            ))}
          </SimpleGrid>
        </Stack>
      </Paper>
    </Stack>
  );
};

export default NSSPView;
