import {
  useState,
  useEffect,
  useLayoutEffect,
  useMemo,
  useCallback,
  useRef,
} from "react";
import { usePersistentXRange } from "../../hooks/usePersistentXRange";
import {
  useMantineColorScheme,
  Stack,
  Text,
  Center,
  Box,
  UnstyledButton,
} from "@mantine/core";
import Plot from "react-plotly.js";
import ModelSelector from "../ModelSelector";
import { DATASETS } from "../../config";
import { useView } from "../../hooks/useView";
import { useChartReset } from "../../hooks/useChartReset";
import {
  CHART_CONSTANTS,
  GROUND_TRUTH_LINE_WIDTH,
  GROUND_TRUTH_MARKER_SIZE,
  PLOT_CONFIG,
  RANGESLIDER_STYLE,
  COMPACT_GROUND_TRUTH_LINE_WIDTH,
  getBaseChartLayout,
  getChartInk,
  getForecastDateMarks,
  getRangeSelector,
} from "../../constants/chart";
import {
  targetDisplayNameMap,
  targetYAxisLabelMap,
} from "../../utils/mapUtils";
import { fetchJson, getDataPath } from "../../utils/paths";
import { useAsyncData } from "../../hooks/useAsyncData";
import useQuantileForecastTraces from "../../hooks/useQuantileForecastTraces";
import useForecastDateDrag from "../../hooks/useForecastDateDrag";
import {
  getScaleYAxis,
  normalizeChartScale,
  transformValueForScale,
} from "../../utils/scaleUtils";
import { copyRange, getRelayoutXRange } from "../../utils/plotRange";
import ChartCaption from "../ChartCaption";

const METRO_STATE_MAP = {
  California: "CA",
  Colorado: "CO",
  Georgia: "GA",
  Indiana: "IN",
  Maine: "ME",
  Maryland: "MD",
  Massachusetts: "MA",
  Minnesota: "MN",
  "New York": "NY",
  "South Carolina": "SC",
  Texas: "TX",
  Utah: "UT",
  Virginia: "VA",
  "North Carolina": "NC",
  Oregon: "OR",
};

// The "4 weeks" span shows horizons 0-3; long-term models go further
const SHORT_TERM_MAX_HORIZON = 3;

const MODEL_TAGS = {
  short: { label: "ST", title: "Short-term forecast (4 weeks)" },
  long: {
    label: "LT",
    title: "Long-term forecast through the end of the season",
  },
  both: {
    label: "ST, LT",
    title: "Short-term and long-term forecast through the end of the season",
  },
};

// Same height as the other forecast charts
const MAIN_PLOT_HEIGHT = "min(880px, 60vh)";
const TILE_PLOT_HEIGHT = 150;

// State view: the state's chart (and its model list) takes the left two
// thirds; the city tiles fill the column beside it, two per row, then wrap
// into full rows below. The main block spans as many tile rows as its
// height needs, measured as it resizes.
const MetroStateLayout = ({ main, tiles }) => {
  const mainRef = useRef(null);
  const tilesRef = useRef(null);
  const [rows, setRows] = useState(1);

  useLayoutEffect(() => {
    const mainEl = mainRef.current;
    const layoutEl = tilesRef.current;
    if (!mainEl || !layoutEl) return undefined;
    const measure = () => {
      const tile = layoutEl.querySelector(".respilens-metro-tile-cell");
      if (!tile) return;
      const gap = parseFloat(getComputedStyle(layoutEl).rowGap) || 0;
      const tileHeight = tile.getBoundingClientRect().height;
      const mainHeight = mainEl.getBoundingClientRect().height;
      setRows(Math.max(1, Math.round((mainHeight + gap) / (tileHeight + gap))));
    };
    const observer = new ResizeObserver(measure);
    observer.observe(mainEl);
    measure();
    return () => observer.disconnect();
  }, [tiles.length]);

  return (
    <div
      ref={tilesRef}
      className="respilens-metro-layout"
      style={{ "--metro-main-rows": rows }}
    >
      <div ref={mainRef} className="respilens-metro-main">
        {main}
      </div>
      {tiles}
    </div>
  );
};

const MetroPlotCard = ({
  locationData,
  title,
  isSmall = false,
  colorScheme,
  selectedTarget,
  selectedModels,
  selectedDates,
  getDefaultRange,
  xAxisRange,
  setXAxisRange,
  chartScale,
  intervalVisibility,
  showLegend = true,
  showOtherGroundTruthSeasons = false,
  span = "4wk",
}) => {
  const groundTruth = locationData?.ground_truth;
  const allForecasts = locationData?.forecasts;

  // "4 weeks" cuts every model to horizons 0-3; "Full season" keeps the
  // long-term models' later horizons
  const forecasts = useMemo(() => {
    if (!allForecasts || span !== "4wk") return allForecasts;
    const trimmed = {};
    selectedDates.forEach((date) => {
      const targetData = allForecasts[date]?.[selectedTarget];
      if (!targetData) return;
      const models = {};
      Object.entries(targetData).forEach(([model, forecast]) => {
        const predictions = Object.fromEntries(
          Object.entries(forecast.predictions || {}).filter(
            ([horizon]) => Number(horizon) <= SHORT_TERM_MAX_HORIZON,
          ),
        );
        models[model] = { ...forecast, predictions };
      });
      trimmed[date] = { [selectedTarget]: models };
    });
    return trimmed;
  }, [allForecasts, span, selectedDates, selectedTarget]);

  // Last plotted forecast date, so the full season window reaches it
  const lastForecastDate = useMemo(() => {
    if (span !== "season" || !forecasts) return null;
    let last = null;
    selectedDates.forEach((date) => {
      const targetData = forecasts[date]?.[selectedTarget];
      selectedModels.forEach((model) => {
        Object.values(targetData?.[model]?.predictions || {}).forEach(
          (prediction) => {
            if (!last || prediction.date > last) last = prediction.date;
          },
        );
      });
    });
    return last;
  }, [span, forecasts, selectedDates, selectedTarget, selectedModels]);

  const getRange = useCallback(
    (forRangeslider = false) => {
      const range = getDefaultRange(forRangeslider);
      if (!range || !lastForecastDate) return range;
      const end = new Date(lastForecastDate);
      end.setDate(end.getDate() + 7);
      const endDate = end.toISOString().split("T")[0];
      return [range[0], endDate > range[1] ? endDate : range[1]];
    },
    [getDefaultRange, lastForecastDate],
  );

  const calculateYRange = useCallback(
    (plotData, xRange) => {
      if (!plotData?.length || !xRange || !selectedTarget) return null;
      let minY = Infinity,
        maxY = -Infinity;
      const [startX, endX] = xRange;
      const start = new Date(startX),
        end = new Date(endX);

      plotData.forEach((trace) => {
        if (!trace.x || !trace.y) return;
        for (let i = 0; i < trace.x.length; i++) {
          const d = new Date(trace.x[i]);
          if (d >= start && d <= end) {
            const v = Number(trace.y[i]);
            if (!isNaN(v)) {
              minY = Math.min(minY, v);
              maxY = Math.max(maxY, v);
            }
          }
        }
      });
      if (minY === Infinity) return null;
      const pad = maxY * (CHART_CONSTANTS.Y_AXIS_PADDING_PERCENT / 100);
      return [Math.max(0, minY - pad), maxY + pad];
    },
    [selectedTarget],
  );

  const showMedian = intervalVisibility?.median ?? true;
  const show50 = intervalVisibility?.ci50 ?? true;
  const show95 = intervalVisibility?.ci95 ?? true;
  const normalizedChartScale = normalizeChartScale(chartScale);
  // ED visit targets are percentages; hospital admissions are counts
  const isPercentTarget = selectedTarget?.endsWith("pct") ?? true;
  const valueSuffix = isPercentTarget ? "%" : "";
  const valueDecimals = isPercentTarget ? 2 : 0;

  const manualScaleTransform = useMemo(() => {
    if (normalizedChartScale !== "sqrt" && normalizedChartScale !== "log2") {
      return null;
    }
    return (value) => transformValueForScale(value, normalizedChartScale);
  }, [normalizedChartScale]);

  const {
    traces: projectionsData,
    rawYRange,
    hasForecastTraces,
  } = useQuantileForecastTraces({
    groundTruth,
    forecasts,
    selectedDates,
    selectedModels,
    target: selectedTarget,
    groundTruthLabel: "Ground Truth Data",
    groundTruthValueFormat: `%{y:.${valueDecimals}f}`,
    valueSuffix,
    formatValue: (value) => value.toFixed(valueDecimals),
    modelLineWidth: isSmall ? 1.5 : 2,
    modelMarkerSize: isSmall ? 3 : 6,
    groundTruthLineWidth: isSmall
      ? COMPACT_GROUND_TRUTH_LINE_WIDTH
      : GROUND_TRUTH_LINE_WIDTH,
    groundTruthMarkerSize: isSmall ? 2 : GROUND_TRUTH_MARKER_SIZE,
    groundTruthColor: getChartInk(colorScheme).text,
    showLegendForFirstDate: showLegend && !isSmall,
    fillMissingQuantiles: true,
    showMedian,
    show50,
    show95,
    showOtherGroundTruthSeasons,
    viewType: "metrocast_forecasts",
    transformY: manualScaleTransform,
    groundTruthHoverFormatter: manualScaleTransform
      ? (value) => Number(value).toFixed(valueDecimals)
      : null,
  });

  const defRange = useMemo(() => getRange(), [getRange]);

  // On the main chart: click to add a forecast date, drag a line to move it
  const { containerRef, displayDates, draggingDate } = useForecastDateDrag({
    selectedDates,
    enabled: !isSmall,
    onBeforeCommit: (gd) => {
      const range = gd?._fullLayout?.xaxis?.range;
      if (!xAxisRange && range) setXAxisRange([...range]);
    },
  });

  const yAxisRange = useMemo(() => {
    const range = xAxisRange || defRange;
    return calculateYRange(projectionsData, range);
  }, [projectionsData, xAxisRange, defRange, calculateYRange]);

  const hasForecasts = hasForecastTraces;
  const base = getBaseChartLayout(colorScheme, { compact: isSmall });
  const isTransformedScale =
    normalizedChartScale === "sqrt" || normalizedChartScale === "log2";
  const longName = targetDisplayNameMap[selectedTarget];

  const PlotContent = (
    <>
      {title && (
        <Text
          fw={600}
          size="sm"
          mb={4}
          ta="left"
          className="respilens-metro-tile-title"
        >
          {title}
        </Text>
      )}

      {!hasForecasts && (
        <Box
          style={{
            position: "absolute",
            top: 40,
            left: 0,
            right: 0,
            zIndex: 1,
            pointerEvents: "none",
          }}
        >
          <Center>
            <Text size="xs" c="dimmed" fs="italic">
              No forecast data for selection
            </Text>
          </Center>
        </Box>
      )}

      <Plot
        style={{
          width: "100%",
          height: isSmall ? TILE_PLOT_HEIGHT : MAIN_PLOT_HEIGHT,
          minHeight: isSmall ? undefined : 340,
          opacity: hasForecasts ? 1 : 0.6,
        }}
        data={projectionsData}
        layout={{
          ...base,
          // Full size: same frame as the other forecast charts
          margin: isSmall
            ? { l: 45, r: 12, t: 10, b: 30 }
            : { l: 64, r: 30, t: 36, b: 30 },
          showlegend: showLegend && !isSmall,
          dragmode: false,
          xaxis: {
            ...base.xaxis,
            range: copyRange(xAxisRange || defRange),
            ...(isSmall && { nticks: 4 }),
            ...(!isSmall && { rangeselector: getRangeSelector(colorScheme) }),
            rangeslider: {
              ...RANGESLIDER_STYLE,
              visible: !isSmall,
              range: getRange(true),
            },
          },
          yaxis: {
            ...base.yaxis,
            ...getScaleYAxis({
              scale: normalizedChartScale,
              rawRange: rawYRange,
              range: yAxisRange,
              title: isSmall
                ? undefined
                : targetYAxisLabelMap[longName] ||
                  longName ||
                  selectedTarget ||
                  "Value",
              formatValue: (value) =>
                `${value.toFixed(valueDecimals)}${valueSuffix}`,
            }),
            tickformat: isTransformedScale ? undefined : `.${valueDecimals}f`,
            ticksuffix: isTransformedScale ? undefined : valueSuffix,
          },
          hovermode: isSmall ? false : "closest",
          hoverlabel: {
            namelength: -1,
          },
          ...getForecastDateMarks(colorScheme, displayDates, draggingDate, 0),
        }}
        config={{ ...PLOT_CONFIG, staticPlot: isSmall }}
        onRelayout={(e) => {
          const newRange = getRelayoutXRange(e, projectionsData);
          if (
            newRange &&
            JSON.stringify(xAxisRange) !== JSON.stringify(newRange)
          ) {
            setXAxisRange(newRange);
          }
        }}
      />
    </>
  );

  // Small city charts are tiles like the front page's: no card, a hairline
  // along the top, the title turning blue on hover (the whole tile opens the
  // city).
  return isSmall ? (
    <Box className="respilens-tile respilens-metro-tile">{PlotContent}</Box>
  ) : (
    <Box ref={containerRef} style={{ position: "relative" }}>
      {PlotContent}
    </Box>
  );
};

const MetroCastView = ({
  data,
  metadata,
  selectedDates,
  selectedModels,
  models,
  setSelectedModels,
  windowSize,
  getDefaultRange,
  selectedTarget,
}) => {
  const { colorScheme } = useMantineColorScheme();
  const {
    handleLocationSelect,
    chartScale,
    intervalVisibility,
    showLegend,
    showOtherGroundTruthSeasons,
    metrocastSpan,
  } = useView();
  // Kept across locations (the view remounts while one loads); each target
  // has its own window
  const [xAxisRange, setXAxisRange] = usePersistentXRange(
    `metrocast:${selectedTarget}`,
  );
  useChartReset(() => setXAxisRange(null));
  // A new span starts from its default window
  const lastSpanRef = useRef(metrocastSpan);
  useEffect(() => {
    if (lastSpanRef.current === metrocastSpan) return;
    lastSpanRef.current = metrocastSpan;
    setXAxisRange(null);
  }, [metrocastSpan, setXAxisRange]);

  const stateName = data?.metadata?.location_name;
  const stateCode = METRO_STATE_MAP[stateName];
  const forecasts = data?.forecasts;

  const activeModels = useMemo(() => {
    const activeModelSet = new Set();
    if (!forecasts || !selectedTarget || !selectedDates.length)
      return activeModelSet;
    selectedDates.forEach((date) => {
      const targetData = forecasts[date]?.[selectedTarget];
      if (targetData)
        Object.keys(targetData).forEach((m) => activeModelSet.add(m));
    });
    return activeModelSet;
  }, [forecasts, selectedDates, selectedTarget]);

  // Which spans each model covers on the selected dates. The long-term
  // ensemble also has weeks 0-3, but the short-term ensemble is the one for them.
  const modelTags = useMemo(() => {
    const spans = new Map();
    if (!forecasts || !selectedTarget) return spans;
    selectedDates.forEach((date) => {
      Object.entries(forecasts[date]?.[selectedTarget] || {}).forEach(
        ([model, forecast]) => {
          const horizons = Object.keys(forecast.predictions || {}).map(Number);
          const seen = spans.get(model) || {};
          spans.set(model, {
            short:
              seen.short || horizons.some((h) => h <= SHORT_TERM_MAX_HORIZON),
            long: seen.long || horizons.some((h) => h > SHORT_TERM_MAX_HORIZON),
          });
        },
      );
    });
    return new Map(
      [...spans].map(([model, { short, long }]) => {
        const longOnly =
          model === DATASETS.metrocast.longTermDefaultModel || !short;
        const tag = longOnly
          ? MODEL_TAGS.long
          : long
            ? MODEL_TAGS.both
            : MODEL_TAGS.short;
        return [model, tag];
      }),
    );
  }, [forecasts, selectedDates, selectedTarget]);

  const { data: childData, loading: loadingChildren } = useAsyncData(
    stateCode && metadata?.locations
      ? async () => {
          const cityList = metadata.locations.filter((l) =>
            l.location_name.includes(`, ${stateCode}`),
          );
          const entries = await Promise.all(
            cityList.map((city) =>
              fetchJson(
                getDataPath(
                  `flumetrocast/${city.abbreviation}_flu_metrocast.json`,
                ),
              ).then(
                (json) => [city.abbreviation, json],
                (e) => {
                  console.error(e);
                  return null;
                },
              ),
            ),
          );
          return Object.fromEntries(entries.filter(Boolean));
        }
      : null,
    [stateCode, metadata],
  );

  if (!selectedTarget)
    return (
      <Center h={300}>
        <Text>Please select a target.</Text>
      </Center>
    );

  const main = (
    <Stack gap="xl">
      <MetroPlotCard
        locationData={data}
        title={null}
        colorScheme={colorScheme}
        windowSize={windowSize}
        selectedTarget={selectedTarget}
        selectedModels={selectedModels}
        selectedDates={selectedDates}
        getDefaultRange={getDefaultRange}
        xAxisRange={xAxisRange}
        setXAxisRange={setXAxisRange}
        isSmall={false}
        chartScale={chartScale}
        intervalVisibility={intervalVisibility}
        showLegend={showLegend}
        showOtherGroundTruthSeasons={showOtherGroundTruthSeasons}
        span={metrocastSpan}
      />
      <Stack gap={2}>
        <ChartCaption forecastNote />
        <ModelSelector
          models={models}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          activeModels={activeModels}
          modelTags={modelTags}
          selectedDates={selectedDates}
          keyboardShortcut
        />
      </Stack>
    </Stack>
  );

  // A city: just its chart, like the other forecast pages
  if (!stateCode) return main;

  const tiles = loadingChildren
    ? []
    : Object.entries(childData ?? {}).map(([abbr, cityData]) => (
        <UnstyledButton
          key={abbr}
          className="respilens-metro-tile-cell"
          onClick={() => handleLocationSelect(abbr)}
          style={{ width: "100%", display: "block", minWidth: 0 }}
        >
          <MetroPlotCard
            locationData={cityData}
            title={cityData.metadata?.location_name}
            isSmall={true}
            colorScheme={colorScheme}
            windowSize={windowSize}
            selectedTarget={selectedTarget}
            selectedModels={selectedModels}
            selectedDates={selectedDates}
            getDefaultRange={getDefaultRange}
            xAxisRange={xAxisRange}
            setXAxisRange={setXAxisRange}
            chartScale={chartScale}
            intervalVisibility={intervalVisibility}
            showLegend={showLegend}
            showOtherGroundTruthSeasons={showOtherGroundTruthSeasons}
            span={metrocastSpan}
          />
        </UnstyledButton>
      ));

  return <MetroStateLayout main={main} tiles={tiles} />;
};

export default MetroCastView;
