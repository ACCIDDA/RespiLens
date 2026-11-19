import { getModelStyle } from "../config/datasets";
import { hexToRgba } from "./modelColorUtils";

/**
 * Peak forecasts (FluSight's "peak inc flu hosp" and "peak week inc flu
 * hosp") as Plotly traces: per model and forecast date, a marker at the
 * median peak size in the median peak week, with whiskers for the 50% and 95%
 * intervals of both. Used by the peak view and the state summary, which adds
 * them to its admissions chart (`nameSuffix` keeps them apart in its legend).
 * MetroCast passes its own targets (peaks computed from the long-term models'
 * sample seasons), puts the peaks in each model's own legend entry
 * (`showLegendEntry: false`), and draws them smaller on its city tiles.
 */
export const buildPeakForecastTraces = ({
  peaks,
  selectedModels,
  selectedDates,
  peakDates = [],
  colorSeed = null,
  showMedian = true,
  show50 = true,
  show95 = true,
  nameSuffix = "",
  sizeTarget = "peak inc flu hosp",
  weekTarget = "peak week inc flu hosp",
  sizeLabel = "Peak hospitalization",
  formatValue = (value) => Math.round(value).toLocaleString(),
  modelColorFn = null,
  showLegendEntry = true,
  compact = false,
}) => {
  const traces = [];
  if (!peaks || !selectedModels?.length) return traces;

  const rawDates =
    selectedDates && selectedDates.length > 0 ? selectedDates : peakDates || [];
  const datesToCheck = [...rawDates].sort(); // Sort chronological

  selectedModels.forEach((model) => {
    const xValues = [];
    const yValues = [];
    const hoverTexts = [];
    const pointColors = [];

    // Base color for this model (Solid, used for Legend)
    const style = getModelStyle(model, {
      shownModels: selectedModels,
      colorSeed,
    });
    const baseColorHex = modelColorFn ? modelColorFn(model) : style.color;
    const modelMarker = style.marker;
    const capSize = compact ? 6 : 10;
    const width50 = compact ? 2.5 : 4;

    datesToCheck.forEach((refDate, index) => {
      const dateData = peaks[refDate];
      if (!dateData) return;

      const intensityData = dateData[sizeTarget]?.[model];
      if (!intensityData || !intensityData.predictions) return;

      // extract confidence intervals
      const iPreds = intensityData.predictions;
      const getVal = (q) => {
        const idx = iPreds.quantiles.indexOf(q);
        return idx !== -1 ? iPreds.values[idx] : null;
      };

      const medianVal = getVal(0.5);
      const low95 = getVal(0.025);
      const high95 = getVal(0.975);
      const low50 = getVal(0.25);
      const high50 = getVal(0.75);

      if (medianVal === null) return;

      const timingData = dateData[weekTarget]?.[model];
      if (!timingData || !timingData.predictions) return;

      const tPreds = timingData.predictions;
      const dateArray = tPreds["peak week"] || tPreds["values"];
      const probArray = tPreds["probabilities"];

      let bestDateStr = null;
      let lowDate95 = null,
        highDate95 = null;
      let lowDate50 = null,
        highDate50 = null;

      if (dateArray && probArray) {
        let cumulativeProb = 0;
        let medianIdx = -1,
          q025Idx = -1,
          q975Idx = -1,
          q25Idx = -1,
          q75Idx = -1;
        for (let i = 0; i < probArray.length; i++) {
          cumulativeProb += probArray[i];

          if (q025Idx === -1 && cumulativeProb >= 0.025) q025Idx = i;
          if (q25Idx === -1 && cumulativeProb >= 0.25) q25Idx = i;
          if (medianIdx === -1 && cumulativeProb >= 0.5) medianIdx = i;
          if (q75Idx === -1 && cumulativeProb >= 0.75) q75Idx = i;
          if (q975Idx === -1 && cumulativeProb >= 0.975) q975Idx = i;
        }
        if (medianIdx === -1) medianIdx = probArray.length - 1;
        if (q975Idx === -1) q975Idx = probArray.length - 1;
        if (q75Idx === -1) q75Idx = probArray.length - 1;

        bestDateStr = dateArray[medianIdx];
        lowDate95 = dateArray[q025Idx !== -1 ? q025Idx : 0];
        highDate95 = dateArray[q975Idx];
        lowDate50 = dateArray[q25Idx !== -1 ? q25Idx : 0];
        highDate50 = dateArray[q75Idx];
      } else if (dateArray && dateArray.length > 0) {
        bestDateStr = dateArray[Math.floor(dateArray.length / 2)];
      }
      if (!bestDateStr) return;

      const normalizedDate = bestDateStr;
      // Gradient Opacity Calculation
      const minOpacity = 0.4;
      const alpha =
        datesToCheck.length === 1
          ? 1.0
          : minOpacity + (index / (datesToCheck.length - 1)) * (1 - minOpacity);

      const dynamicColor = hexToRgba(baseColorHex, alpha);

      if (show50 || show95) {
        // 95% vertical whisker (hosp)
        if (show95 && low95 !== null && high95 !== null) {
          traces.push({
            x: [normalizedDate, normalizedDate],
            y: [low95, high95],
            mode: "lines+markers",
            line: {
              color: dynamicColor,
              width: 1,
              dash: "dash",
            },
            marker: {
              symbol: "line-ew",
              color: dynamicColor,
              size: capSize,
              line: {
                width: 1,
                color: dynamicColor,
              },
            },
            legendgroup: model + nameSuffix,
            showlegend: false,
            hoverinfo: "skip",
          });
        }

        // 50% vertical whisker (hosp)
        if (show50 && low50 !== null && high50 !== null) {
          traces.push({
            x: [normalizedDate, normalizedDate],
            y: [low50, high50],
            mode: "lines",
            line: {
              color: dynamicColor,
              width: width50,
              dash: "6px, 3px",
            },
            legendgroup: model + nameSuffix,
            showlegend: false,
            hoverinfo: "skip",
          });
        }

        // 95% horizontal whisker (dates)
        if (show95 && lowDate95 && highDate95) {
          traces.push({
            x: [lowDate95, highDate95],
            y: [medianVal, medianVal],
            mode: "lines+markers",
            line: {
              color: dynamicColor,
              width: 1,
              dash: "dash",
            },
            marker: {
              symbol: "line-ns",
              color: dynamicColor,
              size: capSize,
              line: { width: 1, color: dynamicColor },
            },
            legendgroup: model + nameSuffix,
            showlegend: false,
            hoverinfo: "skip",
          });
        }

        // 50% horizontal whisker (dates)
        if (show50 && lowDate50 && highDate50) {
          traces.push({
            x: [lowDate50, highDate50],
            y: [medianVal, medianVal],
            mode: "lines",
            line: {
              color: dynamicColor,
              width: width50,
              dash: "6px, 3px",
            },
            legendgroup: model + nameSuffix,
            showlegend: false,
            hoverinfo: "skip",
          });
        }
      }
      if (showMedian) {
        xValues.push(bestDateStr);
        yValues.push(medianVal);
        pointColors.push(dynamicColor);
      }

      const timing50 = `${lowDate50} - ${highDate50}`;
      const timing95 = `${lowDate95} - ${highDate95}`;
      const formattedMedian = formatValue(medianVal);
      const formatted50 = `${formatValue(low50)} - ${formatValue(high50)}`;
      const formatted95 = `${formatValue(low95)} - ${formatValue(high95)}`;

      const timing50Row = show50 ? `50% CI: [${timing50}]<br>` : "";
      const timing95Row = show95 ? `95% CI: [${timing95}]<br>` : "";
      const burden50Row = show50 ? `50% CI: [${formatted50}]<br>` : "";
      const burden95Row = show95 ? `95% CI: [${formatted95}]<br>` : "";

      hoverTexts.push(
        `${model}<br>` +
          `<b>Peak timing:</b><br>` +
          `Median Week: <b>${bestDateStr}</b><br>` +
          timing50Row +
          timing95Row +
          `<span style="border-bottom: 1px solid #ccc; display: block; margin: 5px 0;"></span>` +
          `<b>${sizeLabel}:</b><br>` +
          `Median: <b>${formattedMedian}</b><br>` +
          burden50Row +
          burden95Row +
          `<span style="color: #ffffff; font-size: 0.8em">predicted as of ${refDate}</span>`,
      );
    });

    // actual trace
    if (showMedian && xValues.length > 0) {
      traces.push({
        x: xValues,
        y: yValues,
        name: model + nameSuffix,
        type: "scatter",
        mode: "markers",
        marker: {
          color: pointColors,
          size: compact ? 7 : 12,
          symbol: modelMarker,
          line: { width: 1, color: "white" },
        },
        hoverlabel: {
          font: { color: "#ffffff" },
          bordercolor: "#ffffff", // maakes border white
        },
        hovertemplate: "%{text}<extra></extra>",
        text: hoverTexts,
        showlegend: false,
        legendgroup: model + nameSuffix,
      });

      // dummy legend
      if (!showLegendEntry) return;
      traces.push({
        x: [null],
        y: [null],
        name: model + nameSuffix,
        type: "scatter",
        mode: "markers",
        marker: {
          color: baseColorHex,
          size: 12,
          symbol: modelMarker,
          line: { width: 1, color: "white" },
        },
        showlegend: true,
        legendgroup: model + nameSuffix,
      });
    }
  });
  return traces;
};
