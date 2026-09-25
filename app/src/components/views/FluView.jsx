import { useMemo, useCallback, useRef } from "react";
import ForecastPlotView from "../ForecastPlotView";
import FluPeak from "../FluPeak";
import { getModelColor } from "../../config/datasets";
import { useMantineColorScheme } from "@mantine/core";
import {
  RATE_CHANGE_CATEGORIES,
  getChartAxisStyle,
  getChartInk,
} from "../../constants/chart";
import { useView } from "../../hooks/useView";
import { extendStableModelOrder } from "../../utils/modelColorUtils";

const FluView = ({
  data,
  metadata,
  selectedDates,
  selectedModels,
  models,
  setSelectedModels,
  viewType,
  windowSize,
  getDefaultRange,
  selectedTarget,
  peaks,
  availablePeakDates,
  availablePeakModels,
}) => {
  const {
    chartScale,
    intervalVisibility,
    showLegend,
    showOtherGroundTruthSeasons,
    showHistogram,
  } = useView();
  const histogramVisible =
    viewType === "fludetailed" ||
    (viewType === "flu_forecasts" && showHistogram);
  const forecasts = data?.forecasts;
  const stableModelOrderRef = useRef([]);
  const stableModelOrder = useMemo(() => {
    const nextOrder = extendStableModelOrder(
      stableModelOrderRef.current,
      selectedModels,
    );
    stableModelOrderRef.current = nextOrder;
    return nextOrder;
  }, [selectedModels]);

  const lastSelectedDate = useMemo(() => {
    if (selectedDates.length === 0) return null;
    return selectedDates.slice().sort().pop();
  }, [selectedDates]);

  const rateChangeData = useMemo(() => {
    if (!forecasts || selectedDates.length === 0) return [];
    const categoryOrder = RATE_CHANGE_CATEGORIES;
    return selectedModels
      .map((model) => {
        const forecast =
          forecasts[lastSelectedDate]?.["wk flu hosp rate change"]?.[model];
        if (!forecast) return null;
        const horizon0 = forecast.predictions["0"];
        if (!horizon0) return null;
        const modelColor = getModelColor(model, stableModelOrder);
        const orderedData = categoryOrder.map((cat) => ({
          category: cat.replace("_", "<br>"),
          value:
            (horizon0.probabilities[horizon0.categories.indexOf(cat)] || 0) *
            100,
        }));
        return {
          name: `${model} (${lastSelectedDate})`,
          y: orderedData.map((d) => d.category),
          x: orderedData.map((d) => d.value),
          type: "bar",
          orientation: "h",
          marker: { color: modelColor },
          showlegend: true,
          legendgroup: "histogram",
          xaxis: "x2",
          yaxis: "y2",
          hovertemplate:
            "<b>%{fullData.name}</b><br>%{y}: %{x:.1f}%<extra></extra>",
        };
      })
      .filter(Boolean);
  }, [
    forecasts,
    selectedDates,
    selectedModels,
    lastSelectedDate,
    stableModelOrder,
  ]);

  const extraTraces = useMemo(() => {
    if (!histogramVisible) return [];
    return rateChangeData.map((trace) => ({
      ...trace,
      orientation: "h",
      xaxis: "x2",
      yaxis: "y2",
    }));
  }, [rateChangeData, histogramVisible]);

  const activeModels = useMemo(() => {
    const activeModelSet = new Set();
    if (viewType === "flu_peak" || !forecasts || !selectedDates.length) {
      return activeModelSet;
    }

    const targetForProjections =
      viewType === "flu" || viewType === "flu_forecasts"
        ? selectedTarget
        : "wk inc flu hosp";

    if (
      (viewType === "flu" || viewType === "flu_forecasts") &&
      !targetForProjections
    )
      return activeModelSet;

    selectedDates.forEach((date) => {
      const forecastsForDate = forecasts[date];
      if (!forecastsForDate) return;

      if (targetForProjections) {
        const targetData = forecastsForDate[targetForProjections];
        if (targetData) {
          Object.keys(targetData).forEach((model) => activeModelSet.add(model));
        }
      }

      if (histogramVisible) {
        const rateChangeSet = forecastsForDate["wk flu hosp rate change"];
        if (rateChangeSet) {
          Object.keys(rateChangeSet).forEach((model) =>
            activeModelSet.add(model),
          );
        }
      }
    });

    return activeModelSet;
  }, [forecasts, selectedDates, selectedTarget, viewType, histogramVisible]);

  const forecastTarget =
    viewType === "flu" || viewType === "flu_forecasts"
      ? selectedTarget
      : "wk inc flu hosp";

  const displayTarget = selectedTarget || forecastTarget;
  const requireTarget = viewType === "flu";

  const { colorScheme } = useMantineColorScheme();

  const layoutOverrides = useCallback(
    (baseLayout) => {
      const baseXAxis = {
        ...baseLayout.xaxis,
        domain: histogramVisible ? [0, 0.8] : baseLayout.xaxis.domain,
      };

      const nextLayout = {
        ...baseLayout,
        hoverlabel: { namelength: -1 },
        xaxis: baseXAxis,
      };

      if (!histogramVisible) {
        return nextLayout;
      }

      return {
        ...nextLayout,
        grid: {
          columns: 1,
          rows: 1,
          pattern: "independent",
          subplots: [["xy"], ["x2y2"]],
          xgap: 0.15,
        },
        xaxis2: {
          ...getChartAxisStyle(colorScheme),
          title: {
            text: `displaying date ${lastSelectedDate || "N/A"}`,
            font: { size: 12, color: getChartInk(colorScheme).soft },
            standoff: 10,
          },
          domain: [0.85, 1],
          anchor: "y2",
          showgrid: false,
        },
        yaxis2: {
          ...getChartAxisStyle(colorScheme),
          title: "",
          showticklabels: true,
          type: "category",
          // Spine along the bars' left edge, labels outside it (in the gap
          // between the two charts), like every other axis
          anchor: "x2",
          side: "left",
          // Category names run along the bars, so they never wrap or overlap
          tickangle: -90,
          tickfont: {
            ...getChartAxisStyle(colorScheme).tickfont,
            size: 13,
            color: getChartInk(colorScheme).text,
          },
          showgrid: false,
        },
      };
    },
    [histogramVisible, lastSelectedDate, colorScheme],
  );

  if (viewType === "flu_peak") {
    return (
      <>
        <FluPeak
          data={data}
          peaks={peaks}
          peakDates={availablePeakDates}
          peakModels={availablePeakModels}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          selectedDates={selectedDates}
          windowSize={windowSize}
          chartScale={chartScale}
          intervalVisibility={intervalVisibility}
          showLegend={showLegend}
          showOtherGroundTruthSeasons={showOtherGroundTruthSeasons}
        />
      </>
    );
  }

  return (
    <ForecastPlotView
      data={data}
      metadata={metadata}
      selectedDates={selectedDates}
      selectedModels={selectedModels}
      models={models}
      setSelectedModels={setSelectedModels}
      windowSize={windowSize}
      getDefaultRange={getDefaultRange}
      selectedTarget={selectedTarget}
      forecastTarget={forecastTarget}
      displayTarget={displayTarget}
      requireTarget={requireTarget}
      activeModels={activeModels}
      extraTraces={extraTraces}
      layoutOverrides={layoutOverrides}
      groundTruthValueFormat="%{y}"
    />
  );
};

export default FluView;
