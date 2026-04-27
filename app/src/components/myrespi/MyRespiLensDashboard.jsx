import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  Container,
  Title,
  Text,
  Group,
  Stack,
  ThemeIcon,
  Paper,
  Center,
  Loader,
  useMantineColorScheme,
  Button,
  Modal,
  Anchor,
  Select,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useSearchParams } from "react-router-dom";
import {
  IconUpload,
  IconFileText,
  IconArrowLeft,
  IconInfoCircle,
  IconDashboard,
} from "@tabler/icons-react";
import Plot from "react-plotly.js";
import Plotly from "plotly.js/dist/plotly";
import ModelSelector from "../ModelSelector";
import DateSelector from "../DateSelector";
import { MODEL_COLORS } from "../../config/datasets";
import Seo from "../Seo";

const formatTargetNameForTitle = (name) => {
  if (!name) return "Value";
  return name
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const MyRespiLensDashboard = () => {
  const [, setSearchParams] = useSearchParams();
  useEffect(() => {
    setSearchParams({}, { replace: true });
  }, [setSearchParams]);

  const [yAxisRange, setYAxisRange] = useState(null);
  const [xAxisRange, setXAxisRange] = useState(null);
  const plotRef = useRef(null);
  const isResettingRef = useRef(false);

  const [opened, { open, close }] = useDisclosure(false);
  const { colorScheme } = useMantineColorScheme();

  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const [modelsByTarget, setModelsByTarget] = useState({});
  const [selectedModels, setSelectedModels] = useState([]);
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDates, setSelectedDates] = useState([]);
  const [activeDate, setActiveDate] = useState(null);

  const [availableTargets, setAvailableTargets] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState(null);

  const modelsForView = useMemo(() => {
    if (selectedTarget && modelsByTarget[selectedTarget]) {
      return modelsByTarget[selectedTarget];
    }
    return [];
  }, [selectedTarget, modelsByTarget]);

  useEffect(() => {
    if (modelsForView.length === 0) {
      setSelectedModels([]);
    } else {
      setSelectedModels([modelsForView[0]]);
    }
  }, [modelsForView]);

  useEffect(() => {
    if (!uploadedFile) return;

    setIsProcessing(true);
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const content = event.target?.result;
        const data = JSON.parse(content);
        if (!data.forecasts || !data.ground_truth) {
          throw new Error("Missing required RespiLens fields");
        }
        setFileData(data);

        const forecastDates = Object.keys(data.forecasts || {}).sort(
          (a, b) => new Date(a) - new Date(b),
        );

        const allModelsSet = new Set();
        const modelsByTargetMap = new Map();

        if (data.forecasts) {
          Object.values(data.forecasts).forEach((dateData) => {
            Object.entries(dateData).forEach(([target, targetData]) => {
              if (!modelsByTargetMap.has(target)) {
                modelsByTargetMap.set(target, new Set());
              }
              const modelSetForTarget = modelsByTargetMap.get(target);
              Object.keys(targetData).forEach((model) => {
                allModelsSet.add(model);
                modelSetForTarget.add(model);
              });
            });
          });
        }

        const allModels = Array.from(allModelsSet).sort();
        const modelsByTargetState = {};
        for (const [target, modelSet] of modelsByTargetMap.entries()) {
          modelsByTargetState[target] = Array.from(modelSet).sort();
        }

        setModelsByTarget(modelsByTargetState);
        setAvailableDates(forecastDates);

        const targets = Object.keys(data.ground_truth || {}).filter(
          (key) => key !== "dates",
        );
        setAvailableTargets(targets);

        let defaultTarget = null;
        if (targets.length > 0) {
          defaultTarget = targets[0];
          setSelectedTarget(defaultTarget);
        } else {
          setSelectedTarget(null);
        }

        const modelsForDefaultTarget = modelsByTargetState[defaultTarget] || [];
        if (modelsForDefaultTarget.length > 0) {
          setSelectedModels([modelsForDefaultTarget[0]]);
        } else if (allModels.length > 0) {
          setSelectedModels([allModels[0]]);
        } else {
          setSelectedModels([]);
        }

        if (forecastDates.length > 0) {
          const latestDate = forecastDates[forecastDates.length - 1];
          setSelectedDates([latestDate]);
          setActiveDate(latestDate);
        } else {
          setSelectedDates([]);
          setActiveDate(null);
        }
      } catch (error) {
        console.error("Error parsing JSON file:", error);
        alert(
          "Could not read the file. Please ensure it is a valid RespiLens projections JSON file.",
        );
        setUploadedFile(null);
        setFileData(null);
      } finally {
        setIsProcessing(false);
      }
    };

    reader.readAsText(uploadedFile);
  }, [uploadedFile]);

  const handleDragEnter = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
  }, []);

  const processFile = useCallback((file) => {
    if (file && file.name.endsWith(".json")) {
      setFileData(null);
      setUploadedFile(file);
    } else {
      alert("Please upload a .json file");
    }
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      event.stopPropagation();
      setDragActive(false);
      if (event.dataTransfer?.files?.[0]) {
        processFile(event.dataTransfer.files[0]);
      }
    },
    [processFile],
  );

  const handleFileSelect = useCallback(
    (event) => {
      if (event.target.files?.[0]) {
        processFile(event.target.files[0]);
      }
    },
    [processFile],
  );

  const handleReset = useCallback(() => {
    setUploadedFile(null);
    setFileData(null);
    setSelectedModels([]);
    setAvailableDates([]);
    setSelectedDates([]);
    setActiveDate(null);
    setAvailableTargets([]);
    setSelectedTarget(null);
    setXAxisRange(null);
    setYAxisRange(null);
  }, []);

  const getModelColor = useCallback(
    (model) => {
      const index = selectedModels.indexOf(model);
      return MODEL_COLORS[index % MODEL_COLORS.length];
    },
    [selectedModels],
  );

  const groundTruthTrace = useMemo(() => {
    if (!fileData) return {};
    return {
      x: fileData.ground_truth?.dates || [],
      y: selectedTarget ? fileData.ground_truth?.[selectedTarget] || [] : [],
      type: "scatter",
      mode: "lines+markers",
      name: "Observed",
      line: { color: "black", width: 2, dash: "dash" },
      marker: { size: 4, color: "black" },
      hoverinfo: "text",
      hovertemplate:
        `<b>Observed Data</b><br>` +
        `Value: %{y:.1f}<br>` +
        `Date: %{x}<br>` +
        `<extra></extra>`,
    };
  }, [fileData, selectedTarget]);

  const modelTraces = useMemo(() => {
    if (!fileData) return [];
    return selectedModels.flatMap((model) => {
      const modelColor = getModelColor(model);
      return selectedDates.flatMap((forecastDate, dateIndex) => {
        const forecastData =
          fileData.forecasts?.[forecastDate]?.[selectedTarget]?.[model];
        if (
          !forecastData ||
          forecastData.type !== "quantile" ||
          !forecastData.predictions
        ) {
          return [];
        }

        const isFirstDate = dateIndex === 0;
        const predictions = Object.values(forecastData.predictions || {}).sort(
          (a, b) => new Date(a.date) - new Date(b.date),
        );
        const forecastDates = predictions.map((pred) => pred.date);

        const getQuantile = (q) =>
          predictions.map((pred) => {
            if (!pred.quantiles || !pred.values) return 0;
            const index = pred.quantiles.indexOf(q);
            return index !== -1 ? (pred.values[index] ?? 0) : 0;
          });

        const availableQuantiles = predictions[0]?.quantiles || [];

        const quantilePairs = availableQuantiles
          .filter((q) => q < 0.5)
          .map((q) => {
            const upper = parseFloat((1 - q).toFixed(3));
            return availableQuantiles.includes(upper)
              ? { lower: q, upper, spread: upper - q }
              : null;
          })
          .filter((pair) => pair !== null)
          .sort((a, b) => b.spread - a.spread);

        const areaTraces = quantilePairs.map((pair) => {
          const confidenceLevel = Math.round(pair.spread * 100);
          const opacityBase = Math.floor(40 - pair.spread * 30);
          const opacityHex = Math.max(10, opacityBase).toString();

          const upperValues = getQuantile(pair.upper);
          const lowerValues = getQuantile(pair.lower);

          return {
            x: [...forecastDates, ...[...forecastDates].reverse()],
            y: [...upperValues, ...[...lowerValues].reverse()],
            customdata: [...lowerValues, ...[...upperValues].reverse()],
            fill: "toself",
            fillcolor: `${modelColor}${opacityHex}`,
            line: { color: "transparent" },
            showlegend: false,
            type: "scatter",
            name: `${model} ${confidenceLevel}% CI`,
            legendgroup: model,
            hoverinfo: "text",
            hovertemplate:
              `<b>${model}</b> - ${confidenceLevel}% CI<br>` +
              `pred. <b>${forecastDate}</b><br>` +
              `Upper: %{y:.1f}<br>` +
              `Lower: %{customdata:.1f}<br>` +
              `<extra></extra>`,
          };
        });

        const medianTrace = {
          x: forecastDates,
          y: getQuantile(0.5),
          name: model,
          type: "scatter",
          mode: "lines+markers",
          line: { color: modelColor, width: 2 },
          marker: { size: 6, color: modelColor },
          showlegend: isFirstDate,
          legendgroup: model,
          hoverinfo: "text",
          hovertemplate:
            `<b>${model}</b><br>` +
            `pred. <b>${forecastDate}</b><br>` +
            `Median Forecast: %{y:.1f}<br>` +
            `<extra></extra>`,
        };

        return [...areaTraces, medianTrace];
      });
    });
  }, [fileData, selectedModels, selectedDates, selectedTarget, getModelColor]);

  const traces = useMemo(() => {
    if (!fileData) return [];
    return [groundTruthTrace, ...modelTraces];
  }, [groundTruthTrace, modelTraces, fileData]);

  const activeModels = useMemo(() => {
    const activeModelSet = new Set();
    if (
      !fileData ||
      !fileData.forecasts ||
      !selectedTarget ||
      !selectedDates.length
    )
      return activeModelSet;

    selectedDates.forEach((date) => {
      const targetData = fileData.forecasts[date]?.[selectedTarget];
      if (targetData)
        Object.keys(targetData).forEach((model) => activeModelSet.add(model));
    });

    return activeModelSet;
  }, [fileData, selectedDates, selectedTarget]);

  const calculateYRange = useCallback(
    (data, xRange) => {
      if (!data || !xRange || data.length === 0 || !selectedTarget) return null;
      let minY = Infinity,
        maxY = -Infinity;
      const [startX, endX] = [new Date(xRange[0]), new Date(xRange[1])];

      data.forEach((trace) => {
        if (!trace.x || !trace.y) return;
        for (let i = 0; i < trace.x.length; i++) {
          const pointDate = new Date(trace.x[i]);
          if (pointDate >= startX && pointDate <= endX) {
            const val = Number(trace.y[i]);
            if (!isNaN(val)) {
              minY = Math.min(minY, val);
              maxY = Math.max(maxY, val);
            }
          }
        }
      });
      if (minY !== Infinity) {
        const padding = maxY * 0.1;
        return [Math.max(0, minY - padding), maxY + padding];
      }
      return null;
    },
    [selectedTarget],
  );

  const getDefaultRange = useCallback(
    (isFullRange = false) => {
      if (isFullRange) {
        if (!traces.length) return [null, null];
        let minDate = "9999-12-31",
          maxDate = "1000-01-01";
        traces.forEach((t) => {
          if (t.x?.length) {
            if (t.x[0] < minDate) minDate = t.x[0];
            if (t.x[t.x.length - 1] > maxDate) maxDate = t.x[t.x.length - 1];
          }
        });
        return minDate === "9999-12-31" ? [null, null] : [minDate, maxDate];
      }
      if (!selectedDates.length) return [null, null];
      const start = new Date(selectedDates[0]);
      const end = new Date(selectedDates[selectedDates.length - 1]);
      start.setDate(start.getDate() - 35);
      end.setDate(end.getDate() + 35);
      return [
        start.toISOString().split("T")[0],
        end.toISOString().split("T")[0],
      ];
    },
    [traces, selectedDates],
  );

  const defaultRange = useMemo(() => getDefaultRange(), [getDefaultRange]);

  useEffect(() => {
    setXAxisRange(null);
  }, [selectedTarget]);

  useEffect(() => {
    const currentX = xAxisRange || defaultRange;
    if (traces.length && currentX?.[0]) {
      setYAxisRange(calculateYRange(traces, currentX));
    } else {
      setYAxisRange(null);
    }
  }, [traces, xAxisRange, defaultRange, calculateYRange]);

  const handlePlotUpdate = useCallback(
    (figure) => {
      if (isResettingRef.current) {
        isResettingRef.current = false;
        return;
      }
      if (figure?.["xaxis.range"]) {
        const newX = figure["xaxis.range"];
        if (JSON.stringify(newX) !== JSON.stringify(xAxisRange))
          setXAxisRange(newX);
      }
    },
    [xAxisRange],
  );

  const layout = useMemo(
    () => ({
      template: colorScheme === "dark" ? "plotly_dark" : "plotly_white",
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: colorScheme === "dark" ? "#c1c2c5" : "#000000" },
      height: 600,
      margin: { l: 60, r: 30, t: 50, b: 80 },
      showlegend: selectedModels.length < 15,
      legend: {
        x: 0,
        y: 1,
        bgcolor:
          colorScheme === "dark"
            ? "rgba(26,27,30,0.8)"
            : "rgba(255,255,255,0.8)",
        font: { size: 10 },
      },
      hovermode: "x unified",
      dragmode: false,
      xaxis: {
        rangeslider: { range: getDefaultRange(true) },
        range: xAxisRange || defaultRange,
      },
      yaxis: {
        title: formatTargetNameForTitle(selectedTarget),
        range: yAxisRange,
        autorange: yAxisRange === null,
      },
      shapes: selectedDates.map((date) => ({
        type: "line",
        x0: date,
        x1: date,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "red", width: 1, dash: "dash" },
      })),
    }),
    [
      colorScheme,
      selectedModels.length,
      selectedDates,
      selectedTarget,
      yAxisRange,
      xAxisRange,
      defaultRange,
      getDefaultRange,
    ],
  );

  const config = useMemo(
    () => ({
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["resetScale2d", "select2d", "lasso2d"],
      modeBarButtonsToAdd: [
        {
          name: "Reset view",
          icon: Plotly.Icons.home,
          click: function (gd) {
            const r = getDefaultRange();
            const y = traces.length ? calculateYRange(traces, r) : null;
            isResettingRef.current = true;
            setXAxisRange(null);
            setYAxisRange(y);
            Plotly.relayout(gd, {
              "xaxis.range": r,
              "yaxis.range": y,
              "yaxis.autorange": y === null,
            });
          },
        },
      ],
    }),
    [getDefaultRange, traces, calculateYRange],
  );

  if (isProcessing)
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: "70vh" }}>
          <Loader size="lg" />
        </Center>
      </Container>
    );

  if (fileData) {
    return (
      <Container size="xl" py="xl" style={{ maxWidth: "1400px" }}>
        <Group justify="flex-start" mb="md">
          <Button
            variant="light"
            leftSection={<IconArrowLeft size={16} />}
            onClick={handleReset}
          >
            Upload a different file
          </Button>
        </Group>
        <Title order={2} mb="xl" ta="center">
          Forecasts for{" "}
          {fileData.metadata?.location_name || "Selected Location"}
        </Title>
        <Paper shadow="sm" p="lg" radius="md" withBorder>
          <Stack gap="md" style={{ minHeight: "70vh" }}>
            <Group justify="center">
              <DateSelector
                availableDates={availableDates}
                selectedDates={selectedDates}
                setSelectedDates={setSelectedDates}
                activeDate={activeDate}
                setActiveDate={setActiveDate}
              />
              <Select
                label="Select Target"
                data={availableTargets}
                value={selectedTarget}
                onChange={setSelectedTarget}
                disabled={availableTargets.length <= 1}
                style={{ minWidth: 250 }}
                allowDeselect={false}
              />
            </Group>
            <div style={{ flex: 1, minHeight: 0 }}>
              <Plot
                ref={plotRef}
                data={traces}
                layout={layout}
                style={{ width: "100%" }}
                config={config}
                onRelayout={handlePlotUpdate}
              />
            </div>
            <ModelSelector
              models={modelsForView}
              selectedModels={selectedModels}
              setSelectedModels={setSelectedModels}
              activeModels={activeModels}
              getModelColor={getModelColor}
            />
          </Stack>
        </Paper>
      </Container>
    );
  }

  return (
    <>
      <Seo
        title="RespiLens | MyRespiLens"
        description="Upload and visualize your own RespiLens-compatible respiratory disease forecasts and observed data in a private browser-based dashboard."
        canonicalPath="/myrespilens"
      />
      <Container size="xl" py="xl" style={{ maxWidth: "800px" }}>
        <Center style={{ minHeight: "70vh" }}>
          <Stack
            align="center"
            gap="xl"
            style={{ width: "100%", maxWidth: "600px" }}
          >
            <Modal
              opened={opened}
              onClose={close}
              title={
                <Group gap="xs">
                  <IconDashboard color="var(--mantine-color-blue-6)" />
                  <Text fw={700} size="lg">
                    MyRespiLens
                  </Text>
                </Group>
              }
              centered
            >
              <Stack>
                <Title order={4}>About</Title>
                <Text>
                  MyRespiLens allows epidemiologists to visualize their own
                  public health projections directly in their browser. All the
                  processing happens locally, meaning your data is never
                  uploaded nor shared on any server.
                </Text>
                <Title order={4}>Data Structure</Title>
                <Text>
                  MyRespiLens expects uploaded data to be valid JSON and in
                  RespiLens projections format.
                </Text>
                <Text fw={700}>
                  RespiLens projections format is the internally-defined JSON
                  style for forecast data. Documentation for this JSON format
                  can be found on the MyRespiLens
                  <Anchor
                    href="https://staging.respilens.com/documentation"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {" "}
                    documentation
                  </Anchor>{" "}
                  page.
                </Text>
                <Text>
                  {" "}
                  If you have questions or concerns, don't hesitate to{" "}
                  <Anchor
                    href="https://github.com/ACCIDDA/RespiLens/issues/new"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    contact the RespiLens Team.
                  </Anchor>
                </Text>
              </Stack>
            </Modal>

            <Group justify="center" style={{ width: "100%" }}>
              <Button
                variant="light"
                size="xs"
                color="red"
                onClick={open}
                leftSection={<IconInfoCircle size={16} />}
              >
                What is MyRespiLens?
              </Button>
              <Button
                component="a"
                href="/documentation"
                target="_blank"
                variant="light"
                size="xs"
                color="blue"
                leftSection={<IconFileText size={16} />}
              >
                View Documentation
              </Button>
            </Group>

            <Paper
              shadow="sm"
              p="xl"
              radius="lg"
              withBorder
              style={{
                width: "100%",
                cursor: "pointer",
                transition: "all 0.2s ease",
                border: dragActive
                  ? "2px dashed var(--mantine-primary-color-filled)"
                  : "2px dashed var(--mantine-color-gray-4)",
                backgroundColor: dragActive
                  ? "var(--mantine-primary-color-light)"
                  : "transparent",
              }}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <Stack align="center" gap="xl">
                <ThemeIcon
                  size={80}
                  variant="light"
                  color={dragActive ? "blue" : "gray"}
                  style={{ transition: "all 0.2s ease" }}
                >
                  <IconUpload size={40} />
                </ThemeIcon>

                <div style={{ textAlign: "center" }}>
                  <Title order={2} mb="md" c={dragActive ? "blue" : "dark"}>
                    Drop your RespiLens .json file here
                  </Title>
                  <Text size="sm" c="dimmed">
                    Upload your RespiLens data file to view your personalized
                    RespiLens dashboard
                  </Text>
                </div>

                <Group gap="sm">
                  <ThemeIcon size="sm" variant="light" color="blue">
                    <IconFileText size={14} />
                  </ThemeIcon>
                  <Text size="sm" fw={500} c="blue">
                    RespiLens projections-style .json files only
                  </Text>
                </Group>
              </Stack>
              <input
                id="file-input"
                type="file"
                accept=".json"
                style={{ display: "none" }}
                onChange={handleFileSelect}
              />
            </Paper>
          </Stack>
        </Center>
      </Container>
    </>
  );
};

export default MyRespiLensDashboard;
