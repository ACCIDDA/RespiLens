import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  BarElement,
  BarController,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  ScatterController,
  Tooltip,
} from "chart.js";
import { Chart } from "react-chartjs-2";
import { CHART_CONFIG } from "../../config";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  LineController,
  ScatterController,
  BarElement,
  BarController,
  Tooltip,
  Legend,
  Filler,
);

const INTERVAL95_COLOR = CHART_CONFIG.forecastleColors.interval95;
const INTERVAL50_COLOR = CHART_CONFIG.forecastleColors.interval50;
const MEDIAN_COLOR = "#000000";
const HANDLE_MEDIAN = "#dc143c"; // Crimson
const HANDLE_OUTLINE = "#000000";

const buildLabels = (groundTruthSeries, horizonDates) => {
  const observedLabels = groundTruthSeries.map((entry) => entry.date);
  const seen = new Set(observedLabels);
  horizonDates.forEach((date) => {
    if (!seen.has(date)) {
      observedLabels.push(date);
      seen.add(date);
    }
  });
  return observedLabels;
};

const ForecastleChartCanvasInner = ({
  groundTruthSeries,
  horizonDates,
  entries,
  maxValue,
  onAdjust,
  height = 380,
  showIntervals = true,
  zoomedView = false,
  scores = null,
  showScoring = false,
  // fullGroundTruthSeries = null, // remove unused var
  modelForecasts = {},
  horizons = [],
  dateLabelFormatter = (label) => label,
}) => {
  const chartRef = useRef(null);
  const [dragState, setDragState] = useState(null);

  // For zoomed view, only show last 3 observed data points
  const visibleGroundTruth = useMemo(() => {
    if (!zoomedView) return groundTruthSeries;
    return groundTruthSeries.slice(-3);
  }, [groundTruthSeries, zoomedView]);

  const labels = useMemo(
    () => buildLabels(visibleGroundTruth, horizonDates),
    [visibleGroundTruth, horizonDates],
  );

  const observedDataset = useMemo(() => {
    const valueMap = new Map(
      visibleGroundTruth.map((entry) => [entry.date, entry.value]),
    );
    return labels.map((label) => {
      if (valueMap.has(label)) {
        return { x: label, y: valueMap.get(label) ?? null };
      }
      return { x: label, y: null };
    });
  }, [visibleGroundTruth, labels]);

  // Calculate interval bounds as separate upper/lower datasets for fill
  // Support both symmetric (width-based) and asymmetric (lower/upper) intervals
  const interval95Upper = useMemo(
    () =>
      horizonDates.map((date, idx) => ({
        x: date,
        y:
          entries[idx]?.upper95 ??
          (entries[idx]?.median ?? 0) + (entries[idx]?.width95 ?? 0),
      })),
    [entries, horizonDates],
  );

  const interval95Lower = useMemo(
    () =>
      horizonDates.map((date, idx) => ({
        x: date,
        y:
          entries[idx]?.lower95 ??
          Math.max(
            0,
            (entries[idx]?.median ?? 0) - (entries[idx]?.width95 ?? 0),
          ),
      })),
    [entries, horizonDates],
  );

  const interval50Upper = useMemo(
    () =>
      horizonDates.map((date, idx) => ({
        x: date,
        y:
          entries[idx]?.upper50 ??
          (entries[idx]?.median ?? 0) + (entries[idx]?.width50 ?? 0),
      })),
    [entries, horizonDates],
  );

  const interval50Lower = useMemo(
    () =>
      horizonDates.map((date, idx) => ({
        x: date,
        y:
          entries[idx]?.lower50 ??
          Math.max(
            0,
            (entries[idx]?.median ?? 0) - (entries[idx]?.width50 ?? 0),
          ),
      })),
    [entries, horizonDates],
  );

  // Median line data - start from last observed point
  const medianData = useMemo(() => {
    const data = [];
    // Add the last observed point to connect the forecast line
    if (groundTruthSeries.length > 0) {
      const lastObserved = groundTruthSeries[groundTruthSeries.length - 1];
      data.push({
        x: lastObserved.date,
        y: lastObserved.value,
      });
    }
    // Add the forecast points
    horizonDates.forEach((date, idx) => {
      data.push({
        x: date,
        y: entries[idx]?.median ?? 0,
      });
    });
    return data;
  }, [entries, horizonDates, groundTruthSeries]);

  // Draggable handles for median and interval bounds
  const medianHandles = useMemo(() => {
    return horizonDates.map((date, idx) => ({
      x: date,
      y: entries[idx]?.median ?? 0,
      meta: { index: idx, type: "median" },
      radius: 8,
    }));
  }, [entries, horizonDates]);

  // Draggable handles for interval bounds (only in interval mode)
  const intervalHandles = useMemo(() => {
    if (!showIntervals) return [];
    const handles = [];
    horizonDates.forEach((date, idx) => {
      // 95% upper bound
      handles.push({
        x: date,
        y:
          entries[idx]?.upper95 ??
          (entries[idx]?.median ?? 0) + (entries[idx]?.width95 ?? 0),
        meta: { index: idx, type: "upper95" },
        radius: 6,
      });
      // 95% lower bound
      handles.push({
        x: date,
        y:
          entries[idx]?.lower95 ??
          Math.max(
            0,
            (entries[idx]?.median ?? 0) - (entries[idx]?.width95 ?? 0),
          ),
        meta: { index: idx, type: "lower95" },
        radius: 6,
      });
      // 50% upper bound
      handles.push({
        x: date,
        y:
          entries[idx]?.upper50 ??
          (entries[idx]?.median ?? 0) + (entries[idx]?.width50 ?? 0),
        meta: { index: idx, type: "upper50" },
        radius: 5,
      });
      // 50% lower bound
      handles.push({
        x: date,
        y:
          entries[idx]?.lower50 ??
          Math.max(
            0,
            (entries[idx]?.median ?? 0) - (entries[idx]?.width50 ?? 0),
          ),
        meta: { index: idx, type: "lower50" },
        radius: 5,
      });
    });
    return handles;
  }, [entries, horizonDates, showIntervals]);

  const handleIndicesRef = useRef([]);
  useEffect(() => {
    // Combine median and interval handles metadata
    const allHandles = [...medianHandles, ...intervalHandles];
    handleIndicesRef.current = allHandles.map((point) => point.meta);
  }, [medianHandles, intervalHandles]);

  const dynamicMax = useMemo(() => {
    if (zoomedView) {
      // In zoomed view, calculate 50% higher than the last observed data point
      const lastObserved = groundTruthSeries[groundTruthSeries.length - 1];
      const lastValue = lastObserved?.value ?? 0;

      // Also consider forecast values and ground truth in scoring mode
      const forecastMax = Math.max(
        ...entries.map((entry) => (entry.median ?? 0) + (entry.width95 ?? 0)),
      );

      let groundTruthMax = 0;
      if (showScoring && scores?.groundTruth) {
        groundTruthMax = Math.max(
          ...scores.groundTruth.filter((v) => v !== null),
        );
      }

      // Get the maximum of all visible values
      const visibleMax = Math.max(lastValue, forecastMax, groundTruthMax);

      return visibleMax > 0 ? visibleMax * 1.5 : 1;
    }

    // In full view, use existing logic
    const entryMax = Math.max(
      maxValue,
      ...entries.map((entry) => (entry.median ?? 0) + (entry.width95 ?? 0)),
    );
    return entryMax > 0 ? entryMax * 1.1 : 1;
  }, [entries, maxValue, zoomedView, groundTruthSeries, showScoring, scores]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return undefined;

    const canvas = chart.canvas;
    if (!canvas) return undefined;

    const getHandleMeta = (activeElement) => {
      if (!activeElement) return null;
      const chart = chartRef.current;
      const dataset = chart.data.datasets[activeElement.datasetIndex];
      if (!dataset) return null;

      // Get the correct metadata based on which dataset was clicked
      if (dataset.label === "Median Handles") {
        return medianHandles[activeElement.index]?.meta || null;
      } else if (dataset.label === "Interval Handles") {
        return intervalHandles[activeElement.index]?.meta || null;
      }

      return null;
    };

    let animationFrame = null;
    const pointerMove = (event) => {
      if (!dragState) return;
      // Prevent scrolling while dragging on mobile
      event.preventDefault();
      if (animationFrame) cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => {
        const yScale = chart.scales.y;
        const bounds = canvas.getBoundingClientRect();
        const offsetY = event.clientY - bounds.top;
        const nextValue = Math.max(0, yScale.getValueForPixel(offsetY));
        const roundedValue = Math.round(nextValue);

        // Handle different types of adjustments
        if (dragState.type === "median") {
          onAdjust(dragState.index, "median", roundedValue);
        } else if (dragState.type === "upper95") {
          const entry = entries[dragState.index];
          onAdjust(dragState.index, "interval95", [
            entry.lower95,
            roundedValue,
          ]);
        } else if (dragState.type === "lower95") {
          const entry = entries[dragState.index];
          onAdjust(dragState.index, "interval95", [
            roundedValue,
            entry.upper95,
          ]);
        } else if (dragState.type === "upper50") {
          const entry = entries[dragState.index];
          onAdjust(dragState.index, "interval50", [
            entry.lower50,
            roundedValue,
          ]);
        } else if (dragState.type === "lower50") {
          const entry = entries[dragState.index];
          onAdjust(dragState.index, "interval50", [
            roundedValue,
            entry.upper50,
          ]);
        }
      });
    };

    const pointerUp = () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      setDragState(null);
    };

    const pointerDown = (event) => {
      const elements = chart.getElementsAtEventForMode(
        event,
        "nearest",
        { intersect: true },
        false,
      );
      if (!elements?.length) {
        setDragState(null);
        return;
      }
      const meta = getHandleMeta(elements[0]);
      if (!meta) {
        setDragState(null);
        return;
      }
      // Prevent default immediately to stop scrolling and text selection on mobile
      event.preventDefault();
      event.stopPropagation();
      setDragState(meta);
    };

    canvas.addEventListener("pointerdown", pointerDown);
    window.addEventListener("pointermove", pointerMove);
    window.addEventListener("pointerup", pointerUp);
    window.addEventListener("pointercancel", pointerUp);

    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      window.removeEventListener("pointermove", pointerMove);
      window.removeEventListener("pointerup", pointerUp);
      window.removeEventListener("pointercancel", pointerUp);
    };
  }, [dragState, onAdjust, entries, medianHandles, intervalHandles]);

  // Ground truth for forecast horizons (scoring mode only)
  const groundTruthForecastPoints = useMemo(() => {
    if (!showScoring || !scores?.groundTruth) return [];
    return horizonDates
      .map((date, idx) => ({
        x: date,
        y: scores.groundTruth[idx],
      }))
      .filter((point) => point.y !== null);
  }, [showScoring, scores, horizonDates]);

  // Top model forecasts (scoring mode only)
  const topModelForecasts = useMemo(() => {
    if (!showScoring || !scores?.models || !modelForecasts) return [];

    // Show top 3 models
    return scores.models
      .slice(0, 3)
      .map((model, idx) => {
        const isHub =
          model.modelName.toLowerCase().includes("hub") ||
          model.modelName.toLowerCase().includes("ensemble");

        // Extract medians from model predictions
        const modelData = modelForecasts[model.modelName];
        if (!modelData?.predictions) {
          return null;
        }

        const modelMedians = horizons.map((horizon) => {
          const horizonPrediction = modelData.predictions[String(horizon)];
          if (!horizonPrediction) return null;

          const quantiles = horizonPrediction.quantiles;
          const values = horizonPrediction.values;

          if (!quantiles || !values || quantiles.length !== values.length)
            return null;

          // Find the 0.5 quantile (median)
          const medianIndex = quantiles.findIndex(
            (q) => Math.abs(q - 0.5) < 0.001,
          );
          if (medianIndex === -1) return null;

          return values[medianIndex];
        });

        return {
          modelName: model.modelName,
          data: horizonDates
            .map((date, horizonIdx) => ({
              x: date,
              y: modelMedians[horizonIdx],
            }))
            .filter((point) => point.y !== null && Number.isFinite(point.y)),
          // Green for ensemble/hub, otherwise use other colors
          color: isHub
            ? "rgba(34, 139, 34, 0.8)" // Forest green for ensemble
            : idx === 0
              ? "rgba(30, 144, 255, 0.8)" // Blue for best
              : idx === 1
                ? "rgba(255, 99, 71, 0.6)" // Tomato for 2nd
                : "rgba(255, 165, 0, 0.6)", // Orange for 3rd
        };
      })
      .filter((model) => model !== null);
  }, [showScoring, scores, horizonDates, modelForecasts, horizons]);

  const chartData = useMemo(() => {
    const datasets = [
      {
        type: "line",
        label: "Observed",
        data: observedDataset,
        parsing: false,
        tension: 0, // No smoothing - straight lines between points
        spanGaps: true,
        borderColor: MEDIAN_COLOR,
        backgroundColor: MEDIAN_COLOR,
        borderWidth: 2,
        pointRadius: 4, // Show dots
        pointHoverRadius: 6,
        pointBackgroundColor: MEDIAN_COLOR,
        pointBorderColor: "#ffffff",
        pointBorderWidth: 1,
      },
    ];

    // Only show intervals when in interval mode - use filled areas
    if (showIntervals) {
      datasets.push(
        // 95% interval lower bound (invisible line, used for fill)
        {
          type: "line",
          label: "95% lower",
          data: interval95Lower,
          parsing: false,
          tension: 0,
          borderColor: "transparent",
          backgroundColor: "transparent",
          pointRadius: 0,
          fill: false,
        },
        // 95% interval upper bound (fills down to previous dataset)
        {
          type: "line",
          label: "95% interval",
          data: interval95Upper,
          parsing: false,
          tension: 0,
          borderColor: "transparent", // No border line
          backgroundColor: INTERVAL95_COLOR,
          borderWidth: 0,
          pointRadius: 0,
          fill: "-1", // Fill to previous dataset (95% lower)
        },
        // 50% interval lower bound (invisible line)
        {
          type: "line",
          label: "50% lower",
          data: interval50Lower,
          parsing: false,
          tension: 0,
          borderColor: "transparent",
          backgroundColor: "transparent",
          pointRadius: 0,
          fill: false,
        },
        // 50% interval upper bound (fills down to previous dataset)
        {
          type: "line",
          label: "50% interval",
          data: interval50Upper,
          parsing: false,
          tension: 0,
          borderColor: "transparent", // No border line
          backgroundColor: INTERVAL50_COLOR,
          borderWidth: 0,
          pointRadius: 0,
          fill: "-1", // Fill to previous dataset (50% lower)
        },
      );
    }

    // In scoring mode, show ground truth as a connected line (like observed data)
    if (showScoring && groundTruthForecastPoints.length > 0) {
      // Combine observed data with ground truth for forecast period
      const fullGroundTruthLine = [
        ...observedDataset.filter((d) => d.y !== null),
        ...groundTruthForecastPoints,
      ];

      datasets.push({
        type: "line",
        label: "Ground Truth",
        data: fullGroundTruthLine,
        parsing: false,
        tension: 0,
        spanGaps: false,
        borderColor: MEDIAN_COLOR,
        backgroundColor: MEDIAN_COLOR,
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: MEDIAN_COLOR,
        pointBorderColor: "#ffffff",
        pointBorderWidth: 1,
      });
    }

    // User's forecast
    datasets.push({
      type: "line",
      label: showScoring ? "Your Forecast" : "Median",
      data: medianData,
      parsing: false,
      tension: 0, // No smoothing - straight lines
      borderColor: showScoring ? "#dc143c" : MEDIAN_COLOR,
      backgroundColor: showScoring ? "#dc143c" : MEDIAN_COLOR,
      borderWidth: showScoring ? 3 : 3,
      pointRadius: showScoring ? 5 : 0,
      pointBackgroundColor: showScoring ? "#dc143c" : HANDLE_MEDIAN,
      pointBorderColor: showScoring ? "#ffffff" : "#000000",
      pointBorderWidth: showScoring ? 1 : 2,
      borderDash: [5, 5], // Always dashed for forecasts
    });

    // Top model forecasts (scoring mode only)
    if (showScoring) {
      topModelForecasts.forEach((model) => {
        datasets.push({
          type: "line",
          label: model.modelName,
          data: model.data,
          parsing: false,
          tension: 0,
          borderColor: model.color,
          backgroundColor: model.color,
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: model.color,
          pointBorderColor: "#ffffff",
          pointBorderWidth: 1,
          borderDash: [5, 5], // Dashed for all forecasts
        });
      });
    }

    // Only show draggable handles when not in scoring mode
    if (!showScoring) {
      // Median handles (always visible when not scoring)
      datasets.push({
        type: "scatter",
        label: "Median Handles",
        data: medianHandles,
        parsing: false,
        pointBackgroundColor: HANDLE_MEDIAN,
        pointBorderColor: HANDLE_OUTLINE,
        pointBorderWidth: 2,
        pointHoverRadius: 10,
        pointRadius: 8,
        showLine: false,
        hitRadius: 15,
      });

      // Interval bound handles (only in interval mode)
      if (showIntervals && intervalHandles.length > 0) {
        datasets.push({
          type: "scatter",
          label: "Interval Handles",
          data: intervalHandles,
          parsing: false,
          pointBackgroundColor: (context) => {
            const meta = intervalHandles[context.dataIndex]?.meta;
            if (!meta) return "rgba(220, 20, 60, 0.8)";
            // Color-code by interval type - lighter for 95%
            if (meta.type === "upper95" || meta.type === "lower95") {
              return "rgba(255, 182, 193, 0.9)"; // Light pink for 95%
            }
            return "rgba(220, 20, 60, 0.9)"; // Crimson for 50%
          },
          pointBorderColor: HANDLE_OUTLINE,
          pointBorderWidth: 2,
          pointHoverRadius: 8,
          pointRadius: (context) => {
            return intervalHandles[context.dataIndex]?.radius ?? 6;
          },
          showLine: false,
          hitRadius: 12,
        });
      }
    }

    return { datasets, labels };
  }, [
    medianHandles,
    intervalHandles,
    interval50Upper,
    interval50Lower,
    interval95Upper,
    interval95Lower,
    medianData,
    labels,
    observedDataset,
    showIntervals,
    showScoring,
    groundTruthForecastPoints,
    topModelForecasts,
  ]);

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: true,
      },
      plugins: {
        legend: {
          display: showScoring,
          position: "bottom",
          labels: {
            filter: (legendItem) => {
              // Hide helper datasets from legend
              return (
                !legendItem.text.includes("lower") &&
                !legendItem.text.includes("Handles")
              );
            },
            usePointStyle: true,
            padding: 12,
            font: {
              size: 11,
            },
          },
        },
        tooltip: {
          callbacks: {
            title: (items) => {
              const item = items?.[0];
              const rawDate =
                item?.raw?.x ??
                item?.dataset?.data?.[item?.dataIndex ?? -1]?.x ??
                item?.label;
              return rawDate ? dateLabelFormatter(rawDate) : "";
            },
            label: (context) => {
              const datasetLabel = context.dataset.label || "";
              if (datasetLabel === "Observed") {
                return `${datasetLabel}: ${context.parsed.y?.toLocaleString("en-US") ?? "—"}`;
              }
              if (datasetLabel.includes("interval")) {
                if (Array.isArray(context.raw?.y)) {
                  const [lower, upper] = context.raw.y;
                  return `${datasetLabel}: ${Math.round(lower)} – ${Math.round(upper)}`;
                }
                return datasetLabel;
              }
              if (
                datasetLabel === "Median" ||
                datasetLabel === "Median Handles"
              ) {
                return `Median: ${Math.round(context.parsed.y)}`;
              }
              return datasetLabel;
            },
          },
        },
      },
      scales: {
        x: {
          type: "category",
          ticks: {
            color: "#000000",
            autoSkip: true,
            maxRotation: 45,
            minRotation: 45,
            callback: (value) => dateLabelFormatter(labels[value] ?? value),
          },
          grid: {
            display: false,
          },
        },
        y: {
          beginAtZero: true,
          suggestedMax: dynamicMax,
          ticks: {
            color: "#000000",
            callback: (value) => Math.round(value).toLocaleString("en-US"),
          },
          grid: {
            color: "rgba(0, 0, 0, 0.08)",
          },
        },
      },
      layout: {
        padding: {
          right: 12,
          left: 12,
        },
      },
    }),
    [dateLabelFormatter, dynamicMax, labels, showScoring],
  );

  return (
    <div
      style={{
        width: "100%",
        height,
        touchAction: "none", // Prevent default touch gestures (scrolling, zooming)
        userSelect: "none", // Prevent text selection on long press
        WebkitUserSelect: "none", // Safari
        MozUserSelect: "none", // Firefox
        msUserSelect: "none", // IE/Edge
      }}
    >
      <Chart ref={chartRef} type="bar" data={chartData} options={options} />
    </div>
  );
};

const ForecastleChartCanvas = memo(ForecastleChartCanvasInner);
ForecastleChartCanvas.displayName = "ForecastleChartCanvas";

export default ForecastleChartCanvas;
