import { useMemo } from "react";
import {
  Center,
  Loader,
  Text,
  Box,
  useMantineColorScheme,
} from "@mantine/core";
import Plot from "react-plotly.js";
import useQuantileForecastTraces from "../../hooks/useQuantileForecastTraces";
import { useAsyncData } from "../../hooks/useAsyncData";
import { fetchJson, getDataPath } from "../../utils/paths";
import {
  getModelStyle,
  getSeriesColor,
  NSSP_COLUMN_LABELS,
} from "../../config/datasets";
import { hexToRgba } from "../../utils/modelColorUtils";
import {
  COMPACT_GROUND_TRUTH_LINE_WIDTH,
  getBaseChartLayout,
  getChartInk,
  getForecastDateLineStyle,
} from "../../constants/chart";
import { nhsnSlugToNameMap } from "../../utils/mapUtils";
import {
  buildLog2Ticks,
  buildSqrtTicks,
  getYRangeFromTraces,
  isPlotlyLogScale,
  normalizeChartScale,
  transformValueForScale,
} from "../../utils/scaleUtils";
import {
  FLU_PEAK_GROUND_TRUTH_START,
  FLU_PEAK_NORMALIZED_X_RANGE,
  getNormalizedPeakDate,
} from "../../utils/forecastSeasons";

const MiniPlot = ({ plot, onMetadataLoad, plotHeight = 210 }) => {
  const { colorScheme } = useMantineColorScheme();

  const isNHSN = plot.viewType === "nhsnall";
  const isNSSP = plot.viewType === "nsspall";
  const isFluPeak = plot.viewType === "flu_peak";
  const isSeriesView = isNHSN || isNSSP;
  const normalizedScale = normalizeChartScale(plot.settings.scale);

  const { data, loading, error } = useAsyncData(async () => {
    const json = await fetchJson(
      getDataPath(plot.fullDataPath),
      "Data not found",
    );
    onMetadataLoad?.(json?.metadata || null);
    return json;
  }, [plot.fullDataPath, onMetadataLoad]);

  const { traces: forecastTraces } = useQuantileForecastTraces({
    groundTruth: isSeriesView || isFluPeak ? null : data?.ground_truth,
    forecasts: isSeriesView || isFluPeak ? null : data?.forecasts,
    selectedDates: plot.settings.dates || [],
    selectedModels: plot.settings.models || [],
    target: plot.settings.target,
    showMedian: plot.settings.intervals?.includes("median") ?? true,
    show50: plot.settings.intervals?.includes("ci50") ?? true,
    show95: plot.settings.intervals?.includes("ci95") ?? true,
    showLegendForFirstDate: false,
    showNowcasts: plot.settings.showNowcasts ?? false,
    modelLineWidth: 1.5,
    modelMarkerSize: 4,
    // Observed data heavier than model lines, as on the dashboard
    groundTruthLineWidth: COMPACT_GROUND_TRUTH_LINE_WIDTH,
    groundTruthMarkerSize: 3,
    groundTruthColor: getChartInk(colorScheme).text,
    transformY:
      normalizedScale === "sqrt" || normalizedScale === "log2"
        ? (value) => transformValueForScale(value, normalizedScale)
        : null,
  });

  const nhsnTraces = useMemo(() => {
    if (!isNHSN || !data?.series) return [];

    const dateAxis = data.series.dates;
    return (plot.settings.columns || [])
      .flatMap((slug) => {
        const longformName = nhsnSlugToNameMap[slug] || slug;
        const officialRawY = data.series[longformName] || [];
        const officialYValues = officialRawY.map((value) =>
          transformValueForScale(value, normalizedScale),
        );
        const traces = [];

        if (officialYValues.length > 0) {
          traces.push({
            x: dateAxis,
            y: officialYValues,
            name: longformName,
            type: "scatter",
            mode: "lines",
            line: {
              color: getSeriesColor(slug, plot.settings.columns || []),
              width: 2,
            },
          });
        }

        const preliminaryRawY = data.preliminary_series?.[longformName] || [];
        const preliminaryDates = data.preliminary_series?.dates || [];
        const preliminaryYValues = preliminaryRawY.map((value) =>
          transformValueForScale(value, normalizedScale),
        );

        if (preliminaryYValues.length > 0 && preliminaryDates.length > 0) {
          traces.push({
            x: preliminaryDates,
            y: preliminaryYValues,
            name: `${longformName} (preliminary)`,
            type: "scatter",
            mode: "lines",
            line: {
              color: getSeriesColor(slug, plot.settings.columns || []),
              width: 2,
              dash: "dash",
            },
          });
        }

        return traces;
      })
      .filter((trace) => trace.y.length > 0);
  }, [isNHSN, data, plot.settings, normalizedScale]);

  const nsspTraces = useMemo(() => {
    if (!isNSSP || !data?.series) return [];

    const dateAxis = data.series.dates || [];
    return (plot.settings.columns || [])
      .map((column) => {
        const rawY = data.series[column] || [];
        const yValues = rawY.map((value) =>
          transformValueForScale(value, normalizedScale),
        );

        return {
          x: dateAxis,
          y: yValues,
          name: NSSP_COLUMN_LABELS[column] || column,
          type: "scatter",
          mode: "lines+markers",
          line: {
            color: getSeriesColor(column, plot.settings.columns || []),
            width: 2,
          },
          marker: { size: 4 },
          customdata: rawY,
          hovertemplate:
            "%{x}<br>%{fullData.name}: %{customdata:.2f}%<extra></extra>",
        };
      })
      .filter((trace) => trace.y.length > 0);
  }, [isNSSP, data, plot.settings, normalizedScale]);

  const fluPeakTraces = useMemo(() => {
    if (!isFluPeak || !data) return [];

    const traces = [];
    const selectedDates = (plot.settings.dates || []).slice().sort();
    const selectedModels = plot.settings.models || [];
    const peaks = data.peaks || {};
    const groundTruth = data.ground_truth;
    const showMedian = plot.settings.intervals?.includes("median") ?? true;
    const show50 = plot.settings.intervals?.includes("ci50") ?? true;
    const show95 = plot.settings.intervals?.includes("ci95") ?? true;
    const transformY = (value) => {
      if (value === null || value === undefined) return value;
      return transformValueForScale(value, normalizedScale);
    };

    if (groundTruth?.["wk inc flu hosp"] && groundTruth?.dates) {
      const currentSeason = groundTruth.dates.reduce(
        (accumulator, date, index) => {
          if (date >= FLU_PEAK_GROUND_TRUTH_START) {
            const rawValue = groundTruth["wk inc flu hosp"][index];
            accumulator.x.push(getNormalizedPeakDate(date));
            accumulator.y.push(transformY(rawValue));
            accumulator.rawY.push(rawValue);
          }
          return accumulator;
        },
        { x: [], y: [], rawY: [] },
      );

      if (currentSeason.x.length > 0) {
        traces.push({
          x: currentSeason.x,
          y: currentSeason.y,
          customdata: currentSeason.rawY,
          name: "Current season",
          type: "scatter",
          mode: "lines+markers",
          line: { color: "black", width: 2, dash: "dash" },
          marker: { size: 4, color: "black" },
          hovertemplate:
            "<b>Current season</b><br>%{x|%b %d}: %{customdata:.0f}<extra></extra>",
        });
      }
    }

    selectedModels.forEach((model) => {
      const { color: baseColor, marker: modelMarker } = getModelStyle(model);

      selectedDates.forEach((referenceDate, dateIndex) => {
        const dateData = peaks?.[referenceDate];
        const intensityData = dateData?.["peak inc flu hosp"]?.[model];
        const timingData = dateData?.["peak week inc flu hosp"]?.[model];

        if (!intensityData?.predictions || !timingData?.predictions) {
          return;
        }

        const intensityPredictions = intensityData.predictions;
        const getQuantileValue = (quantile) => {
          const quantileIndex =
            intensityPredictions.quantiles?.indexOf(quantile);
          return quantileIndex !== -1
            ? intensityPredictions.values?.[quantileIndex]
            : null;
        };

        const medianY = getQuantileValue(0.5);
        const low95 = getQuantileValue(0.025);
        const high95 = getQuantileValue(0.975);
        const low50 = getQuantileValue(0.25);
        const high50 = getQuantileValue(0.75);

        if (medianY === null) {
          return;
        }

        const timingPredictions = timingData.predictions;
        const dateArray =
          timingPredictions["peak week"] || timingPredictions.values || [];
        const probabilityArray = timingPredictions.probabilities || [];
        if (dateArray.length === 0) {
          return;
        }

        let cumulativeProbability = 0;
        let medianIndex = -1;
        probabilityArray.forEach((probability, probabilityIndex) => {
          cumulativeProbability += probability;
          if (medianIndex === -1 && cumulativeProbability >= 0.5) {
            medianIndex = probabilityIndex;
          }
        });

        if (medianIndex === -1) {
          medianIndex = Math.floor(dateArray.length / 2);
        }

        const normalizedDate = getNormalizedPeakDate(dateArray[medianIndex]);
        const opacity =
          selectedDates.length <= 1
            ? 1
            : 0.4 + (dateIndex / (selectedDates.length - 1)) * 0.6;
        const traceColor = hexToRgba(baseColor, opacity);

        if (show95 && low95 !== null && high95 !== null) {
          traces.push({
            x: [normalizedDate, normalizedDate],
            y: [transformY(low95), transformY(high95)],
            mode: "lines+markers",
            line: { color: traceColor, width: 1, dash: "dash" },
            marker: {
              symbol: "line-ew",
              color: traceColor,
              size: 8,
              line: { width: 1, color: traceColor },
            },
            hoverinfo: "skip",
            showlegend: false,
          });
        }

        if (show50 && low50 !== null && high50 !== null) {
          traces.push({
            x: [normalizedDate, normalizedDate],
            y: [transformY(low50), transformY(high50)],
            mode: "lines",
            line: { color: traceColor, width: 4 },
            hoverinfo: "skip",
            showlegend: false,
          });
        }

        if (showMedian) {
          traces.push({
            x: [normalizedDate],
            y: [transformY(medianY)],
            customdata: [[referenceDate, medianY]],
            name: model,
            type: "scatter",
            mode: "markers",
            marker: {
              color: traceColor,
              size: 8,
              symbol: modelMarker,
              line: { color: baseColor, width: 1 },
            },
            hovertemplate:
              "<b>%{fullData.name}</b><br>Reference date: %{customdata[0]}<br>Peak: %{customdata[1]:.0f}<br>Week: %{x|%b %d}<extra></extra>",
          });
        }
      });
    });

    return traces;
  }, [isFluPeak, data, plot.settings, normalizedScale]);

  let finalTraces = forecastTraces;
  if (isNHSN) {
    finalTraces = nhsnTraces;
  } else if (isNSSP) {
    finalTraces = nsspTraces;
  } else if (isFluPeak) {
    finalTraces = fluPeakTraces;
  }

  const layout = useMemo(() => {
    let xRange;
    let yRange;

    if (data) {
      if (isSeriesView && data.series?.dates?.length > 0) {
        const lastDate = new Date(
          data.series.dates[data.series.dates.length - 1],
        );
        const startDate = new Date(lastDate);
        startDate.setMonth(startDate.getMonth() - (isNSSP ? 6 : 3));
        const endDate = new Date(lastDate);
        if (isNSSP) {
          endDate.setDate(endDate.getDate() + 14);
        }
        xRange = [
          startDate.toISOString().split("T")[0],
          isNSSP
            ? endDate.toISOString().split("T")[0]
            : data.series.dates[data.series.dates.length - 1],
        ];
      } else if (isFluPeak) {
        xRange = FLU_PEAK_NORMALIZED_X_RANGE;
      } else if (plot.settings.dates?.length > 0) {
        const sortedDates = [...plot.settings.dates].sort();
        const earliestDate = new Date(sortedDates[0]);
        const latestDate = new Date(sortedDates[sortedDates.length - 1]);
        const startDate = new Date(earliestDate);
        startDate.setMonth(startDate.getMonth() - 3);
        const endDate = new Date(latestDate);
        endDate.setDate(endDate.getDate() + 42);

        xRange = [
          startDate.toISOString().split("T")[0],
          endDate.toISOString().split("T")[0],
        ];
      }

      if (finalTraces?.length > 0) {
        const traceRange = getYRangeFromTraces(finalTraces);
        if (traceRange) {
          const maxY = traceRange[1];
          const padding = maxY === 0 ? 1 : maxY * 0.2;
          yRange = [0, maxY + padding];
        }
      }
    }

    const usesPercentSuffix =
      isNSSP ||
      plot.settings.target?.includes("%") ||
      plot.settings.target?.includes("pct") ||
      plot.settings.target?.includes("Percent") ||
      plot.settings.target?.includes("percent");

    const sqrtTicks =
      normalizedScale === "sqrt" && yRange
        ? buildSqrtTicks({
            rawRange: [0, yRange[1] ** 2],
            tickCount: 4,
            formatValue: (value) =>
              value.toLocaleString(undefined, {
                maximumFractionDigits: 0,
              }),
          })
        : null;

    const log2Ticks =
      normalizedScale === "log2" && yRange
        ? buildLog2Ticks({
            rawRange: [1, 2 ** yRange[1]],
            maxTickCount: 4,
            formatValue: (value) =>
              value.toLocaleString(undefined, {
                maximumFractionDigits: usesPercentSuffix ? 2 : 0,
              }),
          })
        : null;

    // Same base as the front-page cards (compact size)
    const shared = getBaseChartLayout(colorScheme, { compact: true });
    return {
      ...shared,
      height: plotHeight,
      margin: { l: 44, r: 8, t: 8, b: 30 },
      showlegend: false,
      dragmode: "pan",
      xaxis: {
        ...shared.xaxis,
        showgrid: false,
        fixedrange: false,
        range: xRange,
        tickformat: isFluPeak ? "%b" : undefined,
      },
      yaxis: {
        ...shared.yaxis,
        fixedrange: true,
        type: isPlotlyLogScale(normalizedScale) ? "log" : "linear",
        range: isPlotlyLogScale(normalizedScale) ? undefined : yRange,
        nticks: 5,
        ticksuffix:
          normalizedScale === "sqrt" || normalizedScale === "log2"
            ? undefined
            : usesPercentSuffix
              ? "%"
              : "",
        tickvals: sqrtTicks?.tickvals ?? log2Ticks?.tickvals,
        ticktext: sqrtTicks?.ticktext ?? log2Ticks?.ticktext,
      },
      shapes:
        !isNHSN && !isNSSP && !isFluPeak
          ? (plot.settings.dates || []).map((date) => ({
              type: "line",
              x0: date,
              x1: date,
              y0: 0,
              y1: 1,
              yref: "paper",
              line: getForecastDateLineStyle(colorScheme),
            }))
          : [],
    };
  }, [
    colorScheme,
    plot.settings,
    normalizedScale,
    isNHSN,
    isNSSP,
    isFluPeak,
    isSeriesView,
    data,
    finalTraces,
    plotHeight,
  ]);

  if (loading) {
    return (
      <Center h={plotHeight}>
        <Loader size="sm" variant="dots" />
      </Center>
    );
  }

  if (error) {
    return (
      <Center h={plotHeight}>
        <Text size="xs" c="red">
          Error loading chart
        </Text>
      </Center>
    );
  }

  return (
    <Box h={plotHeight} style={{ overflow: "hidden", cursor: "grab" }}>
      <Plot
        data={finalTraces}
        layout={layout}
        config={{
          displayModeBar: false,
          staticPlot: false,
          scrollZoom: true,
          responsive: true,
        }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </Box>
  );
};

export default MiniPlot;
