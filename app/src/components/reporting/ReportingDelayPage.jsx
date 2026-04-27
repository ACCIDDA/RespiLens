import { useEffect, useMemo, useRef, useState } from "react";
import {
  Anchor,
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Container,
  Divider,
  Group,
  List,
  Modal,
  NumberInput,
  Paper,
  RangeSlider,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Table,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import {
  IconArrowsMaximize,
  IconArrowsMinimize,
  IconArrowRight,
  IconClock,
  IconDownload,
  IconFileUpload,
} from "@tabler/icons-react";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Chart } from "react-chartjs-2";
import Plot from "react-plotly.js";
import { driver } from "driver.js";
import "driver.js/dist/driver.css";
import Seo from "../Seo";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
);

const SAMPLE_CSV = `reference_date,report_date,value
2024-02-01,2024-02-02,31
2024-02-01,2024-02-03,56
2024-02-01,2024-02-04,71
2024-02-01,2024-02-05,79
2024-02-02,2024-02-03,28
2024-02-02,2024-02-04,52
2024-02-02,2024-02-05,68
2024-02-02,2024-02-06,74
2024-02-03,2024-02-04,26
2024-02-03,2024-02-05,45
2024-02-03,2024-02-06,62
2024-02-03,2024-02-07,70
2024-02-04,2024-02-05,24
2024-02-04,2024-02-06,39
2024-02-04,2024-02-07,55
2024-02-04,2024-02-08,63
2024-02-05,2024-02-06,23
2024-02-05,2024-02-07,35
2024-02-05,2024-02-08,49
2024-02-05,2024-02-09,58`;

const parseCsv = (text) => {
  const rows = text.trim().split(/\r?\n/).filter(Boolean);
  if (rows.length < 2) {
    throw new Error("CSV appears to be empty.");
  }

  const headers = rows[0].split(",").map((header) => header.trim());
  const normalizedHeaders = headers.map((header) => header.toLowerCase());
  const dataRows = rows.slice(1).map((row) => row.split(","));

  const records = dataRows.map((parts, index) => {
    const entry = {};
    headers.forEach((header, headerIndex) => {
      entry[header] = parts[headerIndex]?.trim() ?? "";
    });
    entry._rowIndex = index + 2;
    return entry;
  });

  return { headers, normalizedHeaders, records };
};

const formatDateLabel = (value) =>
  new Date(value).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

const getDefaultReferenceRange = (dates) => {
  if (!dates.length) return [0, 0];
  return [0, dates.length - 1];
};

const getFrequencyUnit = (dates) => {
  if (dates.length < 2) {
    return { unit: "day", unitDays: 1 };
  }
  const diffs = dates
    .slice(1)
    .map((date, index) => new Date(date) - new Date(dates[index]))
    .map((diff) => Math.max(1, Math.round(diff / (1000 * 60 * 60 * 24))))
    .sort((a, b) => a - b);
  const median = diffs[Math.floor(diffs.length / 2)];
  if (median >= 28) {
    return { unit: "month", unitDays: 30 };
  }
  if (median >= 6) {
    return { unit: "week", unitDays: 7 };
  }
  return { unit: "day", unitDays: 1 };
};

const buildTriangle = (records, { referenceDates, maxLagDays } = {}) => {
  const allReferenceDates = Array.from(
    new Set(records.map((record) => record.referenceDate)),
  ).sort();
  const allReportDates = Array.from(
    new Set(records.map((record) => record.reportDate)),
  ).sort();
  const activeReferenceDates = referenceDates?.length
    ? referenceDates
    : allReferenceDates;
  const activeReportDates = maxLagDays
    ? allReportDates.filter((date) => {
        return activeReferenceDates.some((referenceDate) => {
          const diff = Math.round(
            (new Date(date) - new Date(referenceDate)) / (1000 * 60 * 60 * 24),
          );
          return diff >= 0 && diff <= maxLagDays;
        });
      })
    : allReportDates;
  const reportDatesWithDiagonal = Array.from(
    new Set([...activeReportDates, ...activeReferenceDates]),
  ).sort();
  const allowedReferenceDates = new Set(activeReferenceDates);
  const allowedReportDates = new Set(reportDatesWithDiagonal);
  const filteredRecords = records.filter(
    (record) =>
      allowedReferenceDates.has(record.referenceDate) &&
      allowedReportDates.has(record.reportDate),
  );
  const lagFilteredRecords = maxLagDays
    ? filteredRecords.filter((record) => {
        const delay = Math.round(
          (new Date(record.reportDate) - new Date(record.referenceDate)) /
            (1000 * 60 * 60 * 24),
        );
        return delay >= 0 && delay <= maxLagDays;
      })
    : filteredRecords;
  const valueMap = new Map(
    lagFilteredRecords.map((record) => [
      `${record.referenceDate}|${record.reportDate}`,
      record.value,
    ]),
  );

  return {
    referenceDates: activeReferenceDates,
    reportDates: reportDatesWithDiagonal,
    valueMap,
    filteredRecords: lagFilteredRecords,
  };
};

const buildDelayDistribution = (records) => {
  const delayMap = new Map();
  records.forEach((record) => {
    const delayDays = Math.max(
      0,
      Math.round(
        (new Date(record.reportDate) - new Date(record.referenceDate)) /
          (1000 * 60 * 60 * 24),
      ),
    );
    delayMap.set(delayDays, (delayMap.get(delayDays) || 0) + record.value);
  });

  const delays = Array.from(delayMap.keys()).sort((a, b) => a - b);
  const values = delays.map((delay) => delayMap.get(delay) || 0);
  return { delays, values };
};

const calculateQuantiles = (delays, values) => {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (total === 0) {
    return { medianDelay: 0, delay95: 0, total: 0 };
  }

  let cumulative = 0;
  let medianDelay = delays[0];
  let delay95 = delays[0];

  delays.forEach((delay, index) => {
    cumulative += values[index];
    const proportion = cumulative / total;
    if (proportion >= 0.5 && medianDelay === delays[0]) {
      medianDelay = delay;
    }
    if (proportion >= 0.95 && delay95 === delays[0]) {
      delay95 = delay;
    }
  });

  return { medianDelay, delay95, total };
};

const buildRecordsFromMapping = (rows, mapping) => {
  const { referenceDate, reportDate, value } = mapping;
  if (!referenceDate || !reportDate || !value) {
    return [];
  }

  return rows
    .map((row) => {
      const referenceValue = row[referenceDate];
      const reportValue = row[reportDate];
      const numericValue = Number(row[value]);
      if (!referenceValue || !reportValue || Number.isNaN(numericValue)) {
        return null;
      }
      return {
        referenceDate: referenceValue,
        reportDate: reportValue,
        value: numericValue,
      };
    })
    .filter(Boolean);
};

const INITIAL_PARSED = parseCsv(SAMPLE_CSV);
const INITIAL_MAPPING = {
  referenceDate: "",
  reportDate: "",
  value: "",
};
const SAMPLE_MAPPING = {
  referenceDate:
    INITIAL_PARSED.headers[
      INITIAL_PARSED.normalizedHeaders.indexOf("reference_date")
    ] ?? "",
  reportDate:
    INITIAL_PARSED.headers[
      INITIAL_PARSED.normalizedHeaders.indexOf("report_date")
    ] ?? "",
  value:
    INITIAL_PARSED.headers[INITIAL_PARSED.normalizedHeaders.indexOf("value")] ??
    "",
};

const ReportingDelayPage = () => {
  const inputRef = useRef(null);
  const [csvRows, setCsvRows] = useState(() => INITIAL_PARSED.records);
  const [csvHeaders, setCsvHeaders] = useState(() => INITIAL_PARSED.headers);
  const [fileName, setFileName] = useState("sample-epinowcast.csv");
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [columnMapping, setColumnMapping] = useState(SAMPLE_MAPPING);
  const [columnFilters, setColumnFilters] = useState({});
  const [isTriangleFullscreen, setIsTriangleFullscreen] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [analysisStarted, setAnalysisStarted] = useState(true);

  const headerOptions = useMemo(
    () => csvHeaders.map((header) => ({ value: header, label: header })),
    [csvHeaders],
  );

  const extraColumns = useMemo(() => {
    const mappedColumns = new Set(Object.values(columnMapping).filter(Boolean));
    return csvHeaders.filter((header) => !mappedColumns.has(header));
  }, [csvHeaders, columnMapping]);

  const extraColumnOptions = useMemo(() => {
    return extraColumns.map((column) => ({
      column,
      options: Array.from(
        new Set(csvRows.map((row) => row[column]).filter(Boolean)),
      )
        .sort()
        .map((value) => ({
          value,
          label: value,
        })),
    }));
  }, [csvRows, extraColumns]);

  const filteredRows = useMemo(() => {
    return csvRows.filter((row) => {
      return Object.entries(columnFilters).every(([column, value]) => {
        if (!value) return true;
        return row[column] === value;
      });
    });
  }, [csvRows, columnFilters]);

  const records = useMemo(
    () => buildRecordsFromMapping(filteredRows, columnMapping),
    [filteredRows, columnMapping],
  );

  const mappingComplete = Boolean(
    columnMapping.referenceDate &&
    columnMapping.reportDate &&
    columnMapping.value,
  );

  const allReferenceDates = useMemo(
    () =>
      Array.from(new Set(records.map((record) => record.referenceDate))).sort(),
    [records],
  );
  const [referenceRange, setReferenceRange] = useState(() =>
    getDefaultReferenceRange(allReferenceDates),
  );
  const { unit, unitDays } = useMemo(
    () => getFrequencyUnit(allReferenceDates),
    [allReferenceDates],
  );
  const maxLagDays = useMemo(() => {
    if (!records.length) return 0;
    return Math.max(
      ...records.map((record) =>
        Math.round(
          (new Date(record.reportDate) - new Date(record.referenceDate)) /
            (1000 * 60 * 60 * 24),
        ),
      ),
    );
  }, [records]);
  const [maxLagUnits, setMaxLagUnits] = useState(() =>
    Math.max(0, Math.ceil(maxLagDays / unitDays)),
  );

  const activeReferenceDates = useMemo(() => {
    const [start, end] = referenceRange;
    return allReferenceDates.slice(start, end + 1);
  }, [allReferenceDates, referenceRange]);

  const triangle = useMemo(
    () =>
      buildTriangle(records, {
        referenceDates: activeReferenceDates,
        maxLagDays: maxLagUnits * unitDays,
      }),
    [records, activeReferenceDates, maxLagUnits, unitDays],
  );
  const distribution = useMemo(
    () => buildDelayDistribution(triangle.filteredRecords),
    [triangle.filteredRecords],
  );
  const summary = useMemo(
    () => calculateQuantiles(distribution.delays, distribution.values),
    [distribution.delays, distribution.values],
  );

  const hasLongDelay = summary.delay95 >= 7;
  const hasShortDelay = summary.delay95 <= 3;

  const recommendation = hasLongDelay
    ? "Nowcasting is recommended to account for substantial reporting delays."
    : hasShortDelay
      ? "Nowcasting may be optional; delays resolve quickly."
      : "Consider nowcasting when you need near-real-time situational awareness.";

  const handleFile = async (file) => {
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
      setCsvRows(parsed.records);
      setCsvHeaders(parsed.headers);
      setFileName(file.name);
      setError(null);
      setColumnMapping(INITIAL_MAPPING);
      setColumnFilters({});
      setAnalysisStarted(false);
      setReferenceRange([0, 0]);
      setMaxLagUnits(0);
    } catch (err) {
      setError(err.message);
    }
  };

  const sampleDataUri = useMemo(() => {
    const encoded = encodeURIComponent(SAMPLE_CSV);
    return `data:text/csv;charset=utf-8,${encoded}`;
  }, []);

  const startTour = () => {
    setShowHeatmap(false);
    const tour = driver({
      showProgress: true,
      steps: [
        {
          element: "#reporting-triangle-upload",
          popover: {
            title: "Import CSV",
            description: "Drop your file or download the sample CSV to start.",
          },
        },
        {
          element: "#reporting-triangle-mapping",
          popover: {
            title: "Map columns",
            description:
              "Confirm which columns hold reference dates and reports.",
          },
        },
        {
          element: "#reporting-triangle-trajectory",
          popover: {
            title: "Trajectories",
            description: "Watch how reports revise over time.",
          },
        },
        {
          element: "#reporting-triangle-distribution",
          popover: {
            title: "Delay distribution",
            description: "Inspect how long delays typically are.",
          },
        },
        {
          element: "#reporting-triangle-axis-cell",
          popover: {
            title: "Axes",
            description:
              "Rows are reference dates and columns are report dates.",
          },
        },
        {
          element: "#reporting-triangle-diagonal-cell",
          popover: {
            title: "Diagonal",
            description:
              "Delay = 0 reports arrive on the same day as the reference.",
          },
        },
      ],
    });
    tour.drive();
  };

  useEffect(() => {
    const maxIndex = Math.max(0, allReferenceDates.length - 1);
    setReferenceRange((prev) => {
      const nextStart = Math.min(prev[0], maxIndex);
      const nextEnd = Math.min(prev[1], maxIndex);
      if (nextStart === 0 && nextEnd === maxIndex) {
        return getDefaultReferenceRange(allReferenceDates);
      }
      return [nextStart, nextEnd];
    });
  }, [allReferenceDates]);

  useEffect(() => {
    if (!mappingComplete) {
      setAnalysisStarted(false);
    }
  }, [mappingComplete]);

  useEffect(() => {
    setMaxLagUnits(Math.max(0, Math.ceil(maxLagDays / unitDays)));
  }, [maxLagDays, unitDays]);

  const sliderMarks = useMemo(() => {
    if (allReferenceDates.length <= 1) {
      return [
        {
          value: 0,
          label: allReferenceDates[0]
            ? formatDateLabel(allReferenceDates[0])
            : "",
        },
      ];
    }
    return [
      { value: 0, label: formatDateLabel(allReferenceDates[0]) },
      {
        value: allReferenceDates.length - 1,
        label: formatDateLabel(allReferenceDates[allReferenceDates.length - 1]),
      },
    ];
  }, [allReferenceDates]);

  const maxDisplayRows = 80;
  const maxDisplayCols = 80;
  const displayReferenceDates = triangle.referenceDates.slice(-maxDisplayRows);
  const displayReportDates = triangle.reportDates.slice(-maxDisplayCols);
  const isTriangleTruncated =
    displayReferenceDates.length < triangle.referenceDates.length ||
    displayReportDates.length < triangle.reportDates.length;
  const diagonalDates = useMemo(() => {
    const reportSet = new Set(displayReportDates);
    return displayReferenceDates.filter((date) => reportSet.has(date));
  }, [displayReferenceDates, displayReportDates]);
  const activeRangeLabel = activeReferenceDates.length
    ? `${formatDateLabel(activeReferenceDates[0])}–${formatDateLabel(activeReferenceDates.at(-1))}`
    : "No data selected";
  const diagonalHighlightDate = diagonalDates[0] ?? null;
  const triangleRows = displayReferenceDates.map((referenceDate) => (
    <Table.Tr key={referenceDate}>
      <Table.Td fw={600}>{formatDateLabel(referenceDate)}</Table.Td>
      {displayReportDates.map((reportDate) => {
        const value = triangle.valueMap.get(`${referenceDate}|${reportDate}`);
        const intensity = value ? Math.min(1, value / 80) : 0;
        const isDiagonal = referenceDate === reportDate;
        const highlightDiagonal =
          diagonalHighlightDate &&
          referenceDate === diagonalHighlightDate &&
          reportDate === diagonalHighlightDate;
        return (
          <Table.Td
            key={`${referenceDate}-${reportDate}`}
            id={
              highlightDiagonal ? "reporting-triangle-diagonal-cell" : undefined
            }
            ta="center"
            style={{
              backgroundColor: value
                ? `rgba(34, 139, 230, ${0.08 + intensity * 0.45})`
                : "transparent",
              borderRadius: 6,
              border: isDiagonal
                ? "2px solid var(--mantine-color-dark-6)"
                : undefined,
            }}
          >
            {value ?? "—"}
          </Table.Td>
        );
      })}
    </Table.Tr>
  ));

  const heatmapData = useMemo(() => {
    return [
      {
        z: displayReferenceDates.map((referenceDate) =>
          displayReportDates.map(
            (reportDate) =>
              triangle.valueMap.get(`${referenceDate}|${reportDate}`) ?? null,
          ),
        ),
        x: displayReportDates.map(formatDateLabel),
        y: displayReferenceDates.map(formatDateLabel),
        type: "heatmap",
        colorscale: "Blues",
        showscale: false,
        hovertemplate:
          "Reference %{y}<br>Report %{x}<br>Value %{z}<extra></extra>",
      },
    ];
  }, [displayReferenceDates, displayReportDates, triangle.valueMap]);

  const heatmapLayout = useMemo(() => {
    const diagonalLine =
      diagonalDates.length > 1
        ? [
            {
              type: "line",
              x0: formatDateLabel(diagonalDates[0]),
              y0: formatDateLabel(diagonalDates[0]),
              x1: formatDateLabel(diagonalDates[diagonalDates.length - 1]),
              y1: formatDateLabel(diagonalDates[diagonalDates.length - 1]),
              line: { color: "#1a1b1e", width: 2 },
            },
          ]
        : [];
    return {
      margin: { l: 80, r: 20, t: 20, b: 60 },
      xaxis: { title: "Report date", type: "category" },
      yaxis: {
        title: "Reference date",
        type: "category",
        autorange: "reversed",
      },
      shapes: diagonalLine,
      height: 520,
    };
  }, [diagonalDates]);

  const canAnalyze = mappingComplete;
  const showFilters = extraColumnOptions.length > 0;

  const revisionChartData = useMemo(() => {
    const referenceSeries = triangle.referenceDates.slice(0, 3);
    const labels = triangle.reportDates.map(formatDateLabel);
    return {
      labels,
      datasets: referenceSeries.map((referenceDate, index) => ({
        label: formatDateLabel(referenceDate),
        data: triangle.reportDates.map(
          (reportDate) =>
            triangle.valueMap.get(`${referenceDate}|${reportDate}`) ?? null,
        ),
        borderColor: ["#1c7ed6", "#2f9e44", "#f59f00"][index % 3],
        backgroundColor: "transparent",
        tension: 0.3,
      })),
    };
  }, [triangle.referenceDates, triangle.reportDates, triangle.valueMap]);

  const delayChartData = useMemo(() => {
    return {
      labels: distribution.delays.map((delay) => `${delay}d`),
      datasets: [
        {
          label: "Total reports",
          data: distribution.values,
          backgroundColor: "#4dabf7",
        },
      ],
    };
  }, [distribution.delays, distribution.values]);

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "bottom" },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      y: { beginAtZero: true },
    },
  };

  return (
    <>
      <Seo
        title="RespiLens | Reporting Delay Explorer"
        description="Upload reporting data to explore delay distributions, build reporting triangles, and assess nowcasting needs in RespiLens."
        canonicalPath="/reporting-triangle"
      />
      <Container size="xl" py="xl">
        <Stack gap="xl">
          <Stack gap="xs">
            <Badge
              variant="light"
              leftSection={<IconClock size={14} />}
              w="fit-content"
            >
              Reporting triangle explorer
            </Badge>
            <Title order={1}>
              Do you need to nowcast? What is your reporting delay distribution?
            </Title>
            <Text c="dimmed" size="lg">
              Analyze your reporting delay distribution with RespiLens and
              epinowcast. Everything runs locally in your browser.
            </Text>
          </Stack>

          <Card withBorder radius="md" padding="lg">
            <Stack gap="sm">
              <Title order={3}>Introduction</Title>
              <Text c="dimmed">
                Do you need nowcasting ? What does your reporting delay
                distrubution look like ? Let's dive into that using this little
                app. Nothing leave your computer (say how to check). So upload a
                data with some columns indicating the refrence date of an event,
                the report date when it was reported, and the value reported.
                Optionally you can have other columns like location, age group,
                or target type to filter the data. You'll see your reporting
                distrubtion and the so called reporting triangle introduced by
                (probably Kaitlyn Johnson et al. but really i need to check
                this). This For any deeper dive open the link below to
                epinowcast on which this work is based.
              </Text>
              <Group gap="xs">
                <Anchor
                  href="https://package.epinowcast.org"
                  target="_blank"
                  rel="noreferrer"
                  size="sm"
                >
                  EpiNowcast
                </Anchor>
                <Anchor
                  href="https://baselinenowcast.epinowcast.org"
                  target="_blank"
                  rel="noreferrer"
                  size="sm"
                >
                  Baselinenowcast
                </Anchor>
              </Group>
              <Button size="sm" w="fit-content" onClick={startTour}>
                Start guided tour
              </Button>
            </Stack>
          </Card>

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
            <Paper
              withBorder
              radius="md"
              p="lg"
              id="reporting-triangle-upload"
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                handleFile(event.dataTransfer.files?.[0]);
              }}
              style={{
                borderStyle: "dashed",
                borderColor: isDragging
                  ? "var(--mantine-color-blue-6)"
                  : undefined,
                background: isDragging
                  ? "var(--mantine-color-blue-0)"
                  : undefined,
              }}
            >
              <Stack align="center" gap="sm">
                <ThemeIcon size={52} radius="xl" variant="light">
                  <IconFileUpload size={28} />
                </ThemeIcon>
                <Text fw={600}>Drag & drop a CSV file here</Text>
                <Text size="sm" c="dimmed" ta="center">
                  Expected columns: <code>reference_date</code>,{" "}
                  <code>report_date</code>, <code>value</code>.
                </Text>
                <Text size="sm" c="dimmed" ta="center">
                  Each row should be a cumulative total for one reference date
                  as reported on a later report date.
                </Text>
                <Text size="sm" c="dimmed" ta="center">
                  Optional columns like location, age, or target are supported
                  and can be filtered after upload.
                </Text>
                <Group>
                  <Button onClick={() => inputRef.current?.click()}>
                    Choose file
                  </Button>
                  <Button
                    component="a"
                    href={sampleDataUri}
                    download="sample-epinowcast.csv"
                    variant="subtle"
                    leftSection={<IconDownload size={16} />}
                  >
                    Download sample
                  </Button>
                </Group>
                <Text size="sm" c="dimmed">
                  Current dataset: <strong>{fileName}</strong>
                </Text>
                {error && (
                  <Text size="sm" c="red">
                    {error} Showing the last valid dataset.
                  </Text>
                )}
                {!mappingComplete && (
                  <Text size="sm" c="orange">
                    Please map the required columns to continue.
                  </Text>
                )}
                <input
                  ref={inputRef}
                  type="file"
                  accept=".csv,text/csv"
                  hidden
                  onChange={(event) => handleFile(event.target.files?.[0])}
                />
              </Stack>
            </Paper>

            <Card
              withBorder
              radius="md"
              padding="lg"
              id="reporting-triangle-mapping"
            >
              <Stack gap="sm">
                <Group justify="space-between">
                  <Title order={3}>Column mapping</Title>
                  <Badge variant="outline">
                    {csvHeaders.length} columns detected
                  </Badge>
                </Group>
                <Text size="sm" c="dimmed">
                  Map your CSV columns to the required fields. Each upload
                  resets the mapping.
                </Text>
                <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
                  <Select
                    label="Reference date (rows)"
                    placeholder="Choose column"
                    data={headerOptions}
                    value={columnMapping.referenceDate}
                    onChange={(value) => {
                      setColumnMapping((prev) => ({
                        ...prev,
                        referenceDate: value ?? "",
                      }));
                      setAnalysisStarted(false);
                    }}
                    size="sm"
                  />
                  <Select
                    label="Report date (columns)"
                    placeholder="Choose column"
                    data={headerOptions}
                    value={columnMapping.reportDate}
                    onChange={(value) => {
                      setColumnMapping((prev) => ({
                        ...prev,
                        reportDate: value ?? "",
                      }));
                      setAnalysisStarted(false);
                    }}
                    size="sm"
                  />
                  <Select
                    label="Value"
                    placeholder="Choose column"
                    data={headerOptions}
                    value={columnMapping.value}
                    onChange={(value) => {
                      setColumnMapping((prev) => ({
                        ...prev,
                        value: value ?? "",
                      }));
                      setAnalysisStarted(false);
                    }}
                    size="sm"
                  />
                </SimpleGrid>
                {showFilters && (
                  <>
                    <Divider />
                    <Stack gap="xs">
                      <Group justify="space-between">
                        <Text fw={600}>Filter optional columns</Text>
                        <Badge variant="outline">Applied on start</Badge>
                      </Group>
                      <Text size="sm" c="dimmed">
                        Narrow by location, age, target, or other metadata
                        before starting the analysis.
                      </Text>
                      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
                        {extraColumnOptions.map(({ column, options }) => (
                          <Select
                            key={column}
                            label={column}
                            placeholder="All values"
                            data={options}
                            value={columnFilters[column] ?? null}
                            onChange={(value) => {
                              setColumnFilters((prev) => ({
                                ...prev,
                                [column]: value,
                              }));
                              setAnalysisStarted(false);
                            }}
                            clearable
                            searchable
                            size="sm"
                          />
                        ))}
                      </SimpleGrid>
                    </Stack>
                  </>
                )}
                {!mappingComplete && (
                  <Text size="sm" c="orange">
                    Please map reference date, report date, and value to
                    continue.
                  </Text>
                )}
                <Button
                  size="sm"
                  disabled={!canAnalyze}
                  onClick={() => {
                    setAnalysisStarted(true);
                    setReferenceRange(
                      getDefaultReferenceRange(allReferenceDates),
                    );
                    setMaxLagUnits(
                      Math.max(0, Math.ceil(maxLagDays / unitDays)),
                    );
                  }}
                >
                  Start analysis
                </Button>
              </Stack>
            </Card>
          </SimpleGrid>

          {analysisStarted && (
            <>
              <Card withBorder radius="md" padding="lg">
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Title order={3}>Window & cutoff</Title>
                    <Badge variant="outline">
                      {triangle.referenceDates.length} reference dates
                    </Badge>
                  </Group>
                  <Text size="sm" c="dimmed">
                    Use the slider to focus on a subset of reference dates
                    (rows), and set how far after reference dates to include
                    reports (columns).
                  </Text>
                  <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
                    <Stack gap="xs">
                      <Text size="sm" fw={500}>
                        Reference-date window
                      </Text>
                      <RangeSlider
                        value={referenceRange}
                        onChange={setReferenceRange}
                        min={0}
                        max={Math.max(0, allReferenceDates.length - 1)}
                        step={1}
                        marks={sliderMarks}
                        label={(value) =>
                          formatDateLabel(allReferenceDates[value])
                        }
                      />
                      <Text size="xs" c="dimmed">
                        Showing {activeRangeLabel}
                      </Text>
                    </Stack>
                    <Stack gap={4}>
                      <NumberInput
                        label={`Report cutoff (${unit}s after reference)`}
                        value={maxLagUnits}
                        onChange={(value) => setMaxLagUnits(Number(value) || 0)}
                        min={0}
                        max={Math.max(0, Math.ceil(maxLagDays / unitDays))}
                        clampBehavior="strict"
                        size="sm"
                      />
                      <Text size="xs" c="dimmed">
                        Latest observed delay: {maxLagDays} days (~
                        {Math.ceil(maxLagDays / unitDays)} {unit}s)
                      </Text>
                    </Stack>
                  </SimpleGrid>
                </Stack>
              </Card>

              <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
                <Card
                  withBorder
                  radius="md"
                  padding="lg"
                  id="reporting-triangle-trajectory"
                >
                  <Stack gap="sm">
                    <Group justify="space-between">
                      <Title order={3}>Revision trajectories</Title>
                      <Badge variant="outline">first 3 reference dates</Badge>
                    </Group>
                    <Text size="sm" c="dimmed">
                      View how reported totals evolve over successive reports.
                    </Text>
                    <Chart
                      type="line"
                      data={revisionChartData}
                      options={chartOptions}
                    />
                  </Stack>
                </Card>

                <Card
                  withBorder
                  radius="md"
                  padding="lg"
                  id="reporting-triangle-distribution"
                >
                  <Stack gap="sm">
                    <Group justify="space-between">
                      <Title order={3}>Delay distribution</Title>
                      <Badge variant="outline">
                        {summary.total} total reports
                      </Badge>
                    </Group>
                    <Text size="sm" c="dimmed">
                      How long it takes for reports to arrive after the
                      reference date.
                    </Text>
                    <Chart
                      type="bar"
                      data={delayChartData}
                      options={chartOptions}
                    />
                  </Stack>
                </Card>
              </SimpleGrid>

              <Card withBorder radius="md" padding="lg">
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Title order={3}>Reporting triangle</Title>
                    <Group gap="xs">
                      <Badge variant="outline">
                        {triangle.referenceDates.length} reference dates
                      </Badge>
                      <ActionIcon
                        variant="light"
                        aria-label={
                          isTriangleFullscreen
                            ? "Exit full screen"
                            : "Expand to full screen"
                        }
                        onClick={() => setIsTriangleFullscreen((prev) => !prev)}
                      >
                        {isTriangleFullscreen ? (
                          <IconArrowsMinimize size={16} />
                        ) : (
                          <IconArrowsMaximize size={16} />
                        )}
                      </ActionIcon>
                    </Group>
                  </Group>
                  <Group justify="space-between" align="flex-start">
                    <Stack gap={2}>
                      <Text size="sm" c="dimmed">
                        Rows = <strong>reference date</strong>, columns ={" "}
                        <strong>report date</strong>.
                      </Text>
                      <Text size="sm" c="dimmed">
                        Diagonal cells (delay = 0) represent reports received on
                        the same day as the reference date.
                      </Text>
                    </Stack>
                    <Switch
                      label="Display as heatmap (best for large tables)"
                      checked={showHeatmap}
                      onChange={(event) =>
                        setShowHeatmap(event.currentTarget.checked)
                      }
                      size="sm"
                    />
                  </Group>
                  <Text size="sm" c="dimmed">
                    Darker cells are larger cumulative counts.
                    {isTriangleTruncated &&
                      " Showing a recent subset to keep the table responsive."}
                  </Text>
                  {showHeatmap ? (
                    <Plot
                      data={heatmapData}
                      layout={heatmapLayout}
                      config={{ displayModeBar: false }}
                      style={{ width: "100%" }}
                    />
                  ) : (
                    <ScrollArea>
                      <Table withTableBorder striped highlightOnHover>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th id="reporting-triangle-axis-cell">
                              Reference date
                            </Table.Th>
                            {displayReportDates.map((date) => (
                              <Table.Th key={date} ta="center">
                                {formatDateLabel(date)}
                              </Table.Th>
                            ))}
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>{triangleRows}</Table.Tbody>
                      </Table>
                    </ScrollArea>
                  )}
                </Stack>
              </Card>

              <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
                <Card withBorder radius="md" padding="lg">
                  <Stack gap="sm">
                    <Title order={3}>Nota Bene</Title>
                    <List spacing="xs" size="sm">
                      <List.Item>
                        Late reports can be structurally different (e.g., lab
                        corrections, backfills).
                      </List.Item>
                      <List.Item>
                        Holiday effects and reporting interruptions bias delay
                        estimates.
                      </List.Item>
                      <List.Item>
                        Changes in case definitions or data pipelines break
                        comparability over time.
                      </List.Item>
                    </List>
                  </Stack>
                </Card>

                <Card withBorder radius="md" padding="lg">
                  <Stack gap="md">
                    <Title order={3}>Summary</Title>
                    <Paper withBorder radius="md" p="md">
                      <Stack gap={4}>
                        <Text fw={600}>Based on this dataset:</Text>
                        <Text>
                          The reported quantity is{" "}
                          <strong>
                            95% complete after {summary.delay95} days
                          </strong>
                          , with a
                          <strong>
                            {" "}
                            median delay of {summary.medianDelay} days
                          </strong>
                          .
                        </Text>
                        <Text c="dimmed" size="sm">
                          {recommendation}
                        </Text>
                      </Stack>
                    </Paper>

                    <Divider />

                    <Stack gap="xs">
                      <Text fw={600}>Decision tree for nowcasting</Text>
                      <List
                        spacing="xs"
                        size="sm"
                        icon={
                          <ThemeIcon size={20} radius="xl" variant="light">
                            <IconArrowRight size={14} />
                          </ThemeIcon>
                        }
                      >
                        <List.Item>Simple: use baselinenowcast</List.Item>
                        <List.Item>
                          Better but more complex: use epinowcast (tood link)
                        </List.Item>
                        <List.Item>(both use the same data format)</List.Item>
                        <List.Item>
                          For more information about nowcasting (epinowcast, the
                          forum)
                        </List.Item>
                      </List>
                    </Stack>
                  </Stack>
                </Card>
              </SimpleGrid>

              <Card withBorder radius="md" padding="lg">
                <Stack gap="sm">
                  <Title order={3}>Useful links</Title>
                  <List spacing="xs" size="sm">
                    <List.Item>
                      <Anchor
                        href="https://baselinenowcast.epinowcast.org"
                        target="_blank"
                        rel="noreferrer"
                      >
                        Baseline nowcast toolkit
                      </Anchor>
                    </List.Item>
                    <List.Item>
                      <Anchor
                        href="https://epinowcast.org"
                        target="_blank"
                        rel="noreferrer"
                      >
                        EpiNowcast documentation
                      </Anchor>
                    </List.Item>
                  </List>
                  <Box>
                    <Text size="sm" c="dimmed">
                      Want help operationalizing? Start with the baseline
                      nowcast decision tree and upgrade to EpiNowcast when you
                      need probabilistic delay distributions.
                    </Text>
                  </Box>
                </Stack>
              </Card>
            </>
          )}
        </Stack>
        <Modal
          opened={isTriangleFullscreen}
          onClose={() => setIsTriangleFullscreen(false)}
          fullScreen
          title="Reporting triangle"
          padding="md"
        >
          <Stack gap="sm">
            <Group justify="space-between">
              <Text size="sm" c="dimmed">
                Showing {activeRangeLabel} · Cutoff {maxLagUnits} {unit}(s)
              </Text>
              <Switch
                label="Display as heatmap"
                checked={showHeatmap}
                onChange={(event) =>
                  setShowHeatmap(event.currentTarget.checked)
                }
                size="sm"
              />
            </Group>
            {showHeatmap ? (
              <Plot
                data={heatmapData}
                layout={heatmapLayout}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            ) : (
              <ScrollArea>
                <Table withTableBorder striped highlightOnHover>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Reference date</Table.Th>
                      {displayReportDates.map((date) => (
                        <Table.Th key={`modal-${date}`} ta="center">
                          {formatDateLabel(date)}
                        </Table.Th>
                      ))}
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>{triangleRows}</Table.Tbody>
                </Table>
              </ScrollArea>
            )}
          </Stack>
        </Modal>
      </Container>
    </>
  );
};

export default ReportingDelayPage;
