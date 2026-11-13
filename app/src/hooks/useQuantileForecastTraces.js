import { useMemo } from "react";
import { getVisiblePredictions } from "../utils/forecastPredictions";
import { getModelStyle } from "../config/datasets";
import { calculateRelativeWIS, calculateWIS } from "../utils/forecastleScoring";
import { buildHistoricalGroundTruthTraces } from "../utils/forecastSeasons";

const defaultFormatValue = (value) =>
  value.toLocaleString(undefined, { maximumFractionDigits: 2 });

const DEFAULT_INTERVAL_DEFINITIONS = [
  {
    key: "ci50",
    label: "50% interval",
    lowerQuantile: 0.25,
    upperQuantile: 0.75,
  },
  {
    key: "ci95",
    label: "95% interval",
    lowerQuantile: 0.025,
    upperQuantile: 0.975,
  },
];

const buildDefaultModelHoverText = ({
  model,
  pointDate,
  formattedMedian,
  formattedIntervals,
  issuedDate,
  formattedRelativeWis,
  valueSuffix,
  showMedian,
}) => {
  const rows = [`<b>${model}</b><br>Date: ${pointDate}<br>`];

  if (showMedian && formattedMedian !== null) {
    rows.push(`Median: <b>${formattedMedian}${valueSuffix}</b><br>`);
  }

  formattedIntervals.forEach((interval) => {
    rows.push(
      `${interval.label}: [${interval.formattedRange}${valueSuffix}]<br>`,
    );
  });

  if (formattedRelativeWis !== null) {
    rows.push(`rWIS: <b>${formattedRelativeWis}</b><br>`);
  }

  rows.push(
    `<span style="color: rgba(255,255,255,0.8); font-size: 0.8em">predicted as of ${issuedDate}</span>` +
      `<extra></extra>`,
  );

  return rows.join("");
};

const findQuantileValue = (quantiles, values, requestedQuantile) => {
  const index = quantiles.findIndex(
    (quantile) => Number(quantile) === requestedQuantile,
  );
  return index !== -1 ? values[index] : null;
};

const extractPredictionSummary = (prediction, fillMissingQuantiles = false) => {
  if (!prediction) return null;

  const { quantiles = [], values = [] } = prediction;
  const normalizedQuantiles = quantiles.map((quantile) => Number(quantile));
  const median = findQuantileValue(normalizedQuantiles, values, 0.5);

  if (!Number.isFinite(median)) {
    return null;
  }

  const lower50 =
    findQuantileValue(normalizedQuantiles, values, 0.25) ??
    (fillMissingQuantiles ? median : null);
  const upper50 =
    findQuantileValue(normalizedQuantiles, values, 0.75) ??
    (fillMissingQuantiles ? median : null);
  const lower95 =
    findQuantileValue(normalizedQuantiles, values, 0.025) ??
    (fillMissingQuantiles ? median : null);
  const upper95 =
    findQuantileValue(normalizedQuantiles, values, 0.975) ??
    (fillMissingQuantiles ? median : null);

  // The 50% / 95% bounds are only needed for scoring (hasScoringBounds); a
  // prediction with a median and other quantiles still plots without them.
  return {
    quantiles: normalizedQuantiles,
    values,
    median,
    lower50,
    upper50,
    lower95,
    upper95,
  };
};

const hasScoringBounds = (summary) =>
  Number.isFinite(summary?.lower50) &&
  Number.isFinite(summary?.upper50) &&
  Number.isFinite(summary?.lower95) &&
  Number.isFinite(summary?.upper95);

const buildIntervalFillColor = (modelColor, intervalIndex, intervalCount) => {
  const alphaStart = 0.1;
  const alphaEnd = 0.34;
  const denominator = Math.max(1, intervalCount - 1);
  const alpha =
    intervalCount === 1
      ? alphaEnd
      : alphaStart + ((alphaEnd - alphaStart) * intervalIndex) / denominator;
  const alphaHex = Math.round(alpha * 255)
    .toString(16)
    .padStart(2, "0");
  return `${modelColor}${alphaHex}`;
};

const useQuantileForecastTraces = ({
  groundTruth,
  forecasts,
  selectedDates,
  selectedModels,
  target,
  groundTruthLabel = "Observed",
  groundTruthValueFormat = "%{y}",
  valueSuffix = "",
  formatValue = defaultFormatValue,
  modelHoverBuilder = null,
  modelColorFn = null,
  modelLineWidth = 2,
  modelMarkerSize = 6,
  groundTruthLineWidth = 1.5,
  groundTruthMarkerSize = 4,
  groundTruthColor = "black",
  showLegendForFirstDate = true,
  fillMissingQuantiles = false,
  showNowcasts = false,
  showMedian = true,
  show50 = true,
  show95 = true,
  intervalDefinitions = null,
  intervalVisibility = null,
  transformY = null,
  groundTruthHoverFormatter = null,
  baselineModelName = null,
  showOtherGroundTruthSeasons = false,
}) => {
  return useMemo(() => {
    if (!groundTruth || !forecasts || selectedDates.length === 0 || !target) {
      return { traces: [], rawYRange: null, hasForecastTraces: false };
    }

    const groundTruthValues = groundTruth[target];
    if (!groundTruthValues) {
      console.warn(`Ground truth data not found for target: ${target}`);
      return { traces: [], rawYRange: null, hasForecastTraces: false };
    }

    let rawMin = Infinity;
    let rawMax = -Infinity;
    const updateRange = (value) => {
      if (value === null || value === undefined) return;
      const numeric = Number(value);
      if (Number.isNaN(numeric)) return;
      rawMin = Math.min(rawMin, numeric);
      rawMax = Math.max(rawMax, numeric);
    };

    groundTruthValues.forEach((value) => updateRange(value));
    const observedValueByDate = new Map(
      (groundTruth.dates || []).map((date, index) => [
        date,
        groundTruthValues[index],
      ]),
    );

    const groundTruthY = transformY
      ? groundTruthValues.map((value) => transformY(value))
      : groundTruthValues;

    const groundTruthTrace = {
      x: groundTruth.dates || [],
      y: groundTruthY,
      name: groundTruthLabel,
      type: "scatter",
      mode: "lines+markers",
      line: {
        color: groundTruthColor,
        width: groundTruthLineWidth,
        dash: "solid",
      },
      marker: { size: groundTruthMarkerSize, color: groundTruthColor },
    };

    if (groundTruthHoverFormatter) {
      groundTruthTrace.text = groundTruthValues.map((value) =>
        groundTruthHoverFormatter(value),
      );
      groundTruthTrace.hovertemplate = `<b>${groundTruthLabel}</b><br>Date: %{x}<br>Value: <b>%{text}${valueSuffix}</b><extra></extra>`;
    } else {
      groundTruthTrace.hovertemplate = `<b>${groundTruthLabel}</b><br>Date: %{x}<br>Value: <b>${groundTruthValueFormat}${valueSuffix}</b><extra></extra>`;
    }

    const modelTraces = selectedModels.flatMap((model) =>
      selectedDates.flatMap((date, dateIndex) => {
        const forecastsForDate = forecasts[date] || {};
        const forecast = forecastsForDate[target]?.[model];
        if (!forecast || forecast.type !== "quantile") return [];
        const baselineForecast =
          baselineModelName && forecastsForDate[target]
            ? forecastsForDate[target][baselineModelName]
            : null;
        const baselinePredictionsByDate = new Map(
          Object.values(baselineForecast?.predictions || {}).map(
            (prediction) => [prediction.date, prediction],
          ),
        );

        const forecastDates = [];
        const medianValues = [];
        const hoverTexts = [];
        const resolvedIntervalDefinitions =
          intervalDefinitions ??
          DEFAULT_INTERVAL_DEFINITIONS.filter((definition) => {
            if (definition.key === "ci50") return show50;
            if (definition.key === "ci95") return show95;
            return true;
          });
        const activeIntervalDefinitions = resolvedIntervalDefinitions.filter(
          (definition) =>
            intervalVisibility?.[definition.key] ??
            (definition.key === "ci50"
              ? show50
              : definition.key === "ci95"
                ? show95
                : true),
        );
        const intervalSeries = Object.fromEntries(
          activeIntervalDefinitions.map((definition) => [
            definition.key,
            {
              definition,
              dates: [],
              lower: [],
              upper: [],
            },
          ]),
        );

        const sortedPredictions = getVisiblePredictions(
          forecast.predictions,
          date,
          showNowcasts,
        ).sort((a, b) => new Date(a.date) - new Date(b.date));

        sortedPredictions.forEach((pred) => {
          const pointDate = pred.date;
          const predictionSummary = extractPredictionSummary(
            pred,
            fillMissingQuantiles,
          );
          const normalizedQuantiles = predictionSummary?.quantiles ?? [];
          const values = predictionSummary?.values ?? [];
          const resolvedMedian = predictionSummary?.median ?? null;
          const formattedIntervals = [];

          activeIntervalDefinitions.forEach((definition) => {
            const lowerValue = findQuantileValue(
              normalizedQuantiles,
              values,
              definition.lowerQuantile,
            );
            const upperValue = findQuantileValue(
              normalizedQuantiles,
              values,
              definition.upperQuantile,
            );
            const resolvedLower =
              lowerValue ?? (fillMissingQuantiles ? resolvedMedian : null);
            const resolvedUpper =
              upperValue ?? (fillMissingQuantiles ? resolvedMedian : null);

            if (
              resolvedLower === null ||
              resolvedUpper === null ||
              resolvedLower === undefined ||
              resolvedUpper === undefined
            ) {
              return;
            }

            intervalSeries[definition.key].dates.push(pointDate);
            intervalSeries[definition.key].lower.push(
              transformY ? transformY(resolvedLower) : resolvedLower,
            );
            intervalSeries[definition.key].upper.push(
              transformY ? transformY(resolvedUpper) : resolvedUpper,
            );
            updateRange(resolvedLower);
            updateRange(resolvedUpper);
            formattedIntervals.push({
              label: definition.label,
              formattedRange: `${formatValue(resolvedLower)} - ${formatValue(resolvedUpper)}`,
            });
          });

          if (resolvedMedian === null && formattedIntervals.length === 0) {
            return;
          }

          if (resolvedMedian !== null) {
            forecastDates.push(pointDate);
            if (showMedian) {
              medianValues.push(
                transformY ? transformY(resolvedMedian) : resolvedMedian,
              );
              updateRange(resolvedMedian);
            }
          }

          const formattedMedian =
            resolvedMedian === null ? null : formatValue(resolvedMedian);
          const observedValue = observedValueByDate.get(pointDate);
          let formattedRelativeWis = null;

          if (
            hasScoringBounds(predictionSummary) &&
            Number.isFinite(observedValue)
          ) {
            const pointWis = calculateWIS(
              observedValue,
              predictionSummary.median,
              predictionSummary.lower50,
              predictionSummary.upper50,
              predictionSummary.lower95,
              predictionSummary.upper95,
            );
            const baselinePredictionSummary = extractPredictionSummary(
              baselinePredictionsByDate.get(pointDate),
              fillMissingQuantiles,
            );
            const baselineWis = hasScoringBounds(baselinePredictionSummary)
              ? calculateWIS(
                  observedValue,
                  baselinePredictionSummary.median,
                  baselinePredictionSummary.lower50,
                  baselinePredictionSummary.upper50,
                  baselinePredictionSummary.lower95,
                  baselinePredictionSummary.upper95,
                )
              : null;
            const relativeWis = calculateRelativeWIS(
              pointWis?.wis ?? null,
              baselineWis?.wis ?? null,
            );

            if (Number.isFinite(relativeWis)) {
              formattedRelativeWis = relativeWis.toFixed(3);
            }
          }

          const hoverText = modelHoverBuilder
            ? modelHoverBuilder({
                model,
                pointDate,
                formattedMedian,
                formattedIntervals,
                issuedDate: date,
                formattedRelativeWis,
                valueSuffix,
              })
            : buildDefaultModelHoverText({
                model,
                pointDate,
                formattedMedian,
                formattedIntervals,
                issuedDate: date,
                formattedRelativeWis,
                valueSuffix,
                showMedian,
              });

          if (resolvedMedian !== null) {
            hoverTexts.push(hoverText);
          }
        });

        const hasIntervalSeries = Object.values(intervalSeries).some(
          (series) => series.dates.length > 0,
        );
        if (forecastDates.length === 0 && !hasIntervalSeries) return [];

        const modelStyle = getModelStyle(model);
        const modelColor = modelColorFn
          ? modelColorFn(model)
          : modelStyle.color;
        const modelDash = modelStyle.dashed ? "dash" : "solid";
        const isFirstDate = dateIndex === 0;

        const traces = [];

        activeIntervalDefinitions.forEach((definition, intervalIndex) => {
          const series = intervalSeries[definition.key];
          if (!series || series.dates.length === 0) {
            return;
          }
          traces.push({
            x: [...series.dates, ...series.dates.slice().reverse()],
            y: [...series.upper, ...series.lower.slice().reverse()],
            fill: "toself",
            fillcolor: buildIntervalFillColor(
              modelColor,
              intervalIndex,
              activeIntervalDefinitions.length,
            ),
            line: { color: "transparent" },
            showlegend: false,
            type: "scatter",
            name: `${model} ${definition.label}`,
            hoverinfo: "none",
            legendgroup: model,
          });
        });

        if (showMedian && forecastDates.length > 0) {
          traces.push({
            x: forecastDates,
            y: medianValues,
            name: model,
            type: "scatter",
            mode: "lines+markers",
            line: { color: modelColor, width: modelLineWidth, dash: modelDash },
            marker: {
              size: modelMarkerSize,
              color: modelColor,
              symbol: modelStyle.marker,
            },
            showlegend: showLegendForFirstDate ? isFirstDate : false,
            legendgroup: model,
            text: hoverTexts,
            hovertemplate: "%{text}",
            hoverlabel: {
              bgcolor: modelColor,
              font: { color: "#ffffff" },
              bordercolor: "#ffffff",
            },
          });
        }

        if (
          (showMedian && forecastDates.length === 0 && hasIntervalSeries) ||
          (!showMedian && hasIntervalSeries)
        ) {
          traces.push({
            x: [null],
            y: [null],
            name: model,
            type: "scatter",
            mode: "lines",
            line: { color: modelColor, width: modelLineWidth, dash: modelDash },
            showlegend: showLegendForFirstDate ? isFirstDate : false,
            legendgroup: model,
            hoverinfo: "skip",
          });
        }

        return traces;
      }),
    );

    const historicalGroundTruthTraces = showOtherGroundTruthSeasons
      ? buildHistoricalGroundTruthTraces({
          groundTruth,
          target,
          transformY,
          groundTruthLineWidth,
          groundTruthHoverFormatter,
          valueSuffix,
        })
      : [];

    const rawYRange =
      rawMin === Infinity || rawMax === -Infinity ? null : [rawMin, rawMax];

    return {
      traces: [
        groundTruthTrace,
        ...historicalGroundTruthTraces,
        ...modelTraces,
      ],
      rawYRange,
      hasForecastTraces: modelTraces.length > 0,
    };
  }, [
    groundTruth,
    forecasts,
    selectedDates,
    selectedModels,
    target,
    groundTruthLabel,
    groundTruthValueFormat,
    valueSuffix,
    formatValue,
    modelHoverBuilder,
    modelColorFn,
    modelLineWidth,
    modelMarkerSize,
    groundTruthLineWidth,
    groundTruthMarkerSize,
    groundTruthColor,
    showLegendForFirstDate,
    fillMissingQuantiles,
    showNowcasts,
    showMedian,
    show50,
    show95,
    intervalDefinitions,
    intervalVisibility,
    transformY,
    groundTruthHoverFormatter,
    baselineModelName,
    showOtherGroundTruthSeasons,
  ]);
};

export default useQuantileForecastTraces;
