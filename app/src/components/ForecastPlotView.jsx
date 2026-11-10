import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { usePersistentXRange } from "../hooks/usePersistentXRange";
import { useMantineColorScheme, Stack, Text, Box, Center } from "@mantine/core";
import Plot from "react-plotly.js";
import ModelSelector from "./ModelSelector";
import {
  CHART_CONSTANTS,
  FORECAST_LINE_OFFSET_DAYS,
  GROUND_TRUTH_LINE_WIDTH,
  GROUND_TRUTH_MARKER_SIZE,
  PLOT_CONFIG,
  RANGESLIDER_STYLE,
  getBaseChartLayout,
  getChartInk,
  getForecastDateMarks,
  getRangeSelector,
} from "../constants/chart";
import { targetDisplayNameMap, targetYAxisLabelMap } from "../utils/mapUtils";
import useQuantileForecastTraces from "../hooks/useQuantileForecastTraces";
import useForecastDateDrag from "../hooks/useForecastDateDrag";
import {
  getScaleYAxis,
  normalizeChartScale,
  transformValueForScale,
} from "../utils/scaleUtils";
import { useView } from "../hooks/useView";
import { useChartReset } from "../hooks/useChartReset";
import { getOfficialModels } from "../utils/forecastleScoring";
import {
  getEarliestGroundTruthSeasonStartDate,
  getSeasonDateRange,
  getSeasonStartYear,
} from "../utils/forecastSeasons";
import { copyRange, getRelayoutXRange } from "../utils/plotRange";
import ChartCaption from "./ChartCaption";

const FORECAST_DATASET_KEYS_BY_VIEW = {
  fludetailed: "flusight",
  flu_forecasts: "flusight",
  rsv_forecasts: "rsv",
  covid_forecasts: "covid19",
};

const ForecastPlotView = ({
  data,
  selectedDates,
  selectedModels,
  models,
  setSelectedModels,
  getDefaultRange,
  selectedTarget,
  forecastTarget = null,
  displayTarget = null,
  requireTarget = true,
  activeModels: activeModelsOverride = null,
  extraTraces = null,
  layoutOverrides = null,
  groundTruthValueFormat = "%{y}",
}) => {
  const [yAxisRange, setYAxisRange] = useState(null);
  const plotRef = useRef(null);
  const { colorScheme } = useMantineColorScheme();
  const {
    chartScale,
    intervalVisibility,
    showLegend,
    showNowcasts,
    showOtherGroundTruthSeasons,
    viewType,
  } = useView();

  const groundTruth = data?.ground_truth;
  const forecasts = data?.forecasts;

  const resolvedForecastTarget = forecastTarget || selectedTarget;
  // Kept across locations (the view remounts while one loads); each target
  // has its own window
  const [xAxisRange, setXAxisRange] = usePersistentXRange(
    `${viewType}:${resolvedForecastTarget}`,
  );
  const resolvedDisplayTarget =
    displayTarget || selectedTarget || resolvedForecastTarget;
  const showMedian = intervalVisibility?.median ?? true;
  const show50 = intervalVisibility?.ci50 ?? true;
  const show95 = intervalVisibility?.ci95 ?? true;
  const baselineModelName =
    getOfficialModels(FORECAST_DATASET_KEYS_BY_VIEW[viewType]).baseline || null;
  const normalizedChartScale = normalizeChartScale(chartScale);

  const manualScaleTransform = useMemo(() => {
    if (normalizedChartScale !== "sqrt" && normalizedChartScale !== "log2") {
      return null;
    }
    return (value) => transformValueForScale(value, normalizedChartScale);
  }, [normalizedChartScale]);

  const calculateYRange = useCallback(
    (chartData, xRange) => {
      if (
        !chartData ||
        !xRange ||
        !Array.isArray(chartData) ||
        chartData.length === 0 ||
        !resolvedForecastTarget
      )
        return null;
      let minY = Infinity;
      let maxY = -Infinity;
      const [startX, endX] = xRange;
      const startDate = new Date(startX);
      const endDate = new Date(endX);

      chartData.forEach((trace) => {
        if (!trace.x || !trace.y) return;

        for (let i = 0; i < trace.x.length; i++) {
          const pointDate = new Date(trace.x[i]);
          if (pointDate >= startDate && pointDate <= endDate) {
            const value = Number(trace.y[i]);
            if (!isNaN(value)) {
              minY = Math.min(minY, value);
              maxY = Math.max(maxY, value);
            }
          }
        }
      });
      if (minY !== Infinity && maxY !== -Infinity) {
        const padding = maxY * (CHART_CONSTANTS.Y_AXIS_PADDING_PERCENT / 100);
        const rangeMin = Math.max(0, minY - padding);
        return [rangeMin, maxY + padding];
      }
      return null;
    },
    [resolvedForecastTarget],
  );

  const {
    traces: projectionsData,
    rawYRange,
    hasForecastTraces,
  } = useQuantileForecastTraces({
    groundTruth,
    forecasts,
    selectedDates,
    selectedModels,
    target: resolvedForecastTarget,
    groundTruthLabel: "Observed",
    groundTruthValueFormat,
    valueSuffix: "",
    modelLineWidth: 2,
    modelMarkerSize: 6,
    groundTruthLineWidth: GROUND_TRUTH_LINE_WIDTH,
    groundTruthMarkerSize: GROUND_TRUTH_MARKER_SIZE,
    groundTruthColor: getChartInk(colorScheme).text,
    showLegendForFirstDate: showLegend,
    fillMissingQuantiles: false,
    showNowcasts,
    showMedian,
    show50,
    show95,
    showOtherGroundTruthSeasons,
    viewType,
    transformY: manualScaleTransform,
    baselineModelName,
    groundTruthHoverFormatter: manualScaleTransform
      ? (value) =>
          groundTruthValueFormat.includes(":.2f")
            ? Number(value).toFixed(2)
            : Number(value).toLocaleString(undefined, {
                maximumFractionDigits: 2,
              })
      : null,
  });

  const appendedTraces = useMemo(() => {
    if (!extraTraces) return [];
    if (typeof extraTraces === "function") {
      return extraTraces({ baseTraces: projectionsData }) || [];
    }
    return Array.isArray(extraTraces) ? extraTraces : [];
  }, [extraTraces, projectionsData]);

  const finalTraces = useMemo(() => {
    if (!appendedTraces.length) return projectionsData;
    return [...projectionsData, ...appendedTraces];
  }, [projectionsData, appendedTraces]);

  const activeModels = useMemo(() => {
    if (activeModelsOverride) {
      return activeModelsOverride;
    }
    const activeModelSet = new Set();
    if (!forecasts || !resolvedForecastTarget || !selectedDates.length) {
      return activeModelSet;
    }

    selectedDates.forEach((date) => {
      const forecastsForDate = forecasts[date];
      if (!forecastsForDate) return;

      const targetData = forecastsForDate[resolvedForecastTarget];
      if (!targetData) return;

      Object.keys(targetData).forEach((model) => {
        activeModelSet.add(model);
      });
    });

    return activeModelSet;
  }, [activeModelsOverride, forecasts, selectedDates, resolvedForecastTarget]);

  const defaultRange = useMemo(() => getDefaultRange(), [getDefaultRange]);

  // Click the chart to add a forecast date, drag a date line to move it.
  // Pin the visible window first so the chart does not re-centre on the
  // new dates under the pointer.
  const { containerRef, displayDates, draggingDate } = useForecastDateDrag({
    selectedDates,
    lineOffsetDays: FORECAST_LINE_OFFSET_DAYS,
    onBeforeCommit: (gd) => {
      const range = gd?._fullLayout?.xaxis?.range;
      if (!xAxisRange && range) setXAxisRange([...range]);
    },
  });

  useChartReset(() => setXAxisRange(null));

  useEffect(() => {
    const currentXRange = xAxisRange || defaultRange;
    if (projectionsData.length > 0 && currentXRange) {
      const initialYRange = calculateYRange(projectionsData, currentXRange);
      setYAxisRange(initialYRange);
    } else {
      setYAxisRange(null);
    }
  }, [projectionsData, xAxisRange, defaultRange, calculateYRange]);

  const handlePlotUpdate = useCallback(
    (figure) => {
      const newXRange = getRelayoutXRange(figure, projectionsData);
      if (
        newXRange &&
        JSON.stringify(newXRange) !== JSON.stringify(xAxisRange)
      ) {
        setXAxisRange(newXRange);
      }
    },
    [xAxisRange, setXAxisRange, projectionsData],
  );

  const seasonDividerShapes = useMemo(() => {
    if (!showOtherGroundTruthSeasons || !groundTruth?.dates?.length) {
      return [];
    }

    const hubSeasonStartDate = getEarliestGroundTruthSeasonStartDate({
      groundTruth,
      target: resolvedForecastTarget,
    });
    const dividerYears = Array.from(
      new Set(
        groundTruth.dates
          .filter((dateString) =>
            hubSeasonStartDate ? dateString >= hubSeasonStartDate : true,
          )
          .map((dateString) => getSeasonStartYear(dateString)),
      ),
    )
      .sort((a, b) => a - b)
      .slice(1);

    return dividerYears.map((seasonStartYear) => {
      const { start } = getSeasonDateRange(seasonStartYear);
      return {
        type: "line",
        x0: start,
        x1: start,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: {
          color: colorScheme === "dark" ? "#495057" : "#ced4da",
          width: 1,
          dash: "dot",
        },
        layer: "below",
      };
    });
  }, [
    colorScheme,
    groundTruth,
    resolvedForecastTarget,
    showOtherGroundTruthSeasons,
  ]);

  const layout = useMemo(() => {
    const dateMarks = getForecastDateMarks(
      colorScheme,
      displayDates,
      draggingDate,
    );
    const base = getBaseChartLayout(colorScheme);
    const longName = targetDisplayNameMap[resolvedDisplayTarget];
    const baseLayout = {
      ...base,
      showlegend: showLegend,
      hovermode: "closest",
      dragmode: false,
      margin: { l: 64, r: 30, t: 36, b: 30 },
      xaxis: {
        ...base.xaxis,
        domain: [0, 1],
        rangeslider: { ...RANGESLIDER_STYLE, range: getDefaultRange(true) },
        rangeselector: getRangeSelector(colorScheme),
        range: copyRange(xAxisRange || defaultRange),
      },
      yaxis: {
        ...base.yaxis,
        ...getScaleYAxis({
          scale: normalizedChartScale,
          rawRange: rawYRange,
          range: yAxisRange,
          title:
            targetYAxisLabelMap[longName] ||
            longName ||
            resolvedDisplayTarget ||
            "Value",
        }),
      },
      shapes: [...seasonDividerShapes, ...dateMarks.shapes],
      annotations: dateMarks.annotations,
    };

    if (layoutOverrides) {
      return layoutOverrides(baseLayout);
    }

    return baseLayout;
  }, [
    colorScheme,
    defaultRange,
    resolvedDisplayTarget,
    displayDates,
    draggingDate,
    yAxisRange,
    xAxisRange,
    getDefaultRange,
    layoutOverrides,
    normalizedChartScale,
    rawYRange,
    showLegend,
    seasonDividerShapes,
  ]);

  const hasForecasts = hasForecastTraces;
  if (requireTarget && !selectedTarget) {
    return (
      <Stack align="center" justify="center" style={{ height: "300px" }}>
        <Text>Please select a target to view data.</Text>
      </Stack>
    );
  }

  return (
    <Stack>
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "min(880px, 60vh)",
          minHeight: 340,
          position: "relative", // Ensure the container is relative for absolute positioning
        }}
      >
        {!hasForecasts && (
          <Box
            style={{
              position: "absolute",
              top: 80, // Adjusted slightly for the larger main view
              left: 0,
              right: 0,
              zIndex: 1,
              pointerEvents: "none",
            }}
          >
            <Center>
              <Text size="sm" c="dimmed" fs="italic">
                No forecast data available for the current selection
              </Text>
            </Center>
          </Box>
        )}

        <Plot
          ref={plotRef}
          useResizeHandler
          style={{
            width: "100%",
            height: "100%",
            opacity: hasForecasts ? 1 : 0.6, // Dim the plot if no forecasts exist
          }}
          data={finalTraces}
          layout={layout}
          config={PLOT_CONFIG}
          onRelayout={(figure) => handlePlotUpdate(figure)}
        />
      </div>
      <Stack gap={2}>
        <ChartCaption forecastNote />
        <ModelSelector
          models={models}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          activeModels={activeModels}
          selectedDates={selectedDates}
          keyboardShortcut
        />
      </Stack>
    </Stack>
  );
};

export default ForecastPlotView;
