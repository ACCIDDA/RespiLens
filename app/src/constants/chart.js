// Chart and visualization constants
import { CHART_CONFIG } from "../config";
import { Chart as ChartJS } from "chart.js";
import { SYSTEM_SANS } from "../theme/mantine";

export const CHART_CONSTANTS = {
  // Plot dimensions (from centralized config)
  MAX_WIDTH: CHART_CONFIG.maxWidth,
  MAX_HEIGHT: CHART_CONFIG.maxHeight,
  WIDTH_RATIO: 0.8,
  HEIGHT_RATIO: 0.6,

  // Color opacity values for confidence intervals (from centralized config)
  CI_95_OPACITY: CHART_CONFIG.opacity.ci95,
  CI_50_OPACITY: CHART_CONFIG.opacity.ci50,

  // Date calculation offsets in weeks (from centralized config)
  DEFAULT_WEEKS_BEFORE: CHART_CONFIG.timeRange.weeksBefore,
  DEFAULT_WEEKS_AFTER: CHART_CONFIG.timeRange.weeksAfter,
  RANGESLIDER_WEEKS_AFTER: CHART_CONFIG.timeRange.rangeSliderWeeksAfter,

  // Grid layout for RSV subplots
  RSV_GRID: {
    ROWS: 3,
    COLUMNS: 2,
    ROW_HEIGHTS: [0.6, 0.2, 0.2],
    COLUMN_WIDTHS: [0.5, 0.5],
  },

  // Default margins
  MARGINS: {
    LEFT: 60,
    RIGHT: 30,
    TOP: 30,
    BOTTOM: 30,
  },

  // Y-axis padding percentage (from centralized config)
  Y_AXIS_PADDING_PERCENT: CHART_CONFIG.yAxisPaddingPercent,
};

// Rate change category order
export const RATE_CHANGE_CATEGORIES = [
  "large_decrease",
  "decrease",
  "stable",
  "increase",
  "large_increase",
];

// Plot chrome shared by every chart. Charts inherit the background of the
// surface holding them rather than painting their own white rectangle, and
// plotly's own furniture is tinted to match instead of falling back to its
// opaque grey defaults.
export const PLOT_CHROME = {
  // Fully transparent: the chart sits on the page rather than on a panel of
  // its own.
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  modebar: {
    bgcolor: "rgba(255,255,255,0.85)",
    color: "#9ca3af",
    activecolor: "#316896",
  },
};

// Slim minimap: enough to see the full series and drag a window, without
// eating a sixth of the chart's height (Plotly's default thickness is 0.15).
export const RANGESLIDER_STYLE = {
  thickness: 0.06,
  // Always show the whole series: by default the minimap reuses the main
  // chart's y-range, so zooming in would clip it too
  yaxis: { rangemode: "auto" },
  bgcolor: "rgba(0,0,0,0)",
  bordercolor: "rgba(15,23,42,0.10)",
  borderwidth: 1,
};

// Chart typography and furniture, matched to the app (same font stack, ink
// colours and hairlines as global.css) so charts read as part of the page.
const CHART_INK = {
  light: {
    text: "#23262b",
    soft: "#5d6470",
    spine: "#5d6470",
    grid: "rgba(15, 23, 42, 0.07)",
    legendBg: "rgba(255, 255, 255, 0.85)",
    marker: "rgba(15, 23, 42, 0.35)",
  },
  dark: {
    text: "#c1c2c5",
    soft: "#909296",
    spine: "#909296",
    grid: "rgba(255, 255, 255, 0.08)",
    legendBg: "rgba(26, 27, 30, 0.85)",
    marker: "rgba(255, 255, 255, 0.35)",
  },
};

export const getChartInk = (colorScheme) =>
  colorScheme === "dark" ? CHART_INK.dark : CHART_INK.light;

export const getChartFont = (colorScheme) => ({
  family: SYSTEM_SANS,
  size: 13,
  color: getChartInk(colorScheme).text,
});

// Both axes get a spine and outside ticks; the grid stays a faint hairline.
export const getChartAxisStyle = (colorScheme) => {
  const ink = getChartInk(colorScheme);
  return {
    showline: true,
    linewidth: 1.5,
    linecolor: ink.spine,
    ticks: "outside",
    ticklen: 5,
    tickwidth: 1.5,
    tickcolor: ink.spine,
    tickfont: { size: 12, color: ink.soft },
    gridcolor: ink.grid,
    zeroline: false,
    // Grow the margin to fit the larger tick labels instead of overlapping
    automargin: true,
  };
};

// Borderless, on a translucent wash of the page colour
export const getChartLegendStyle = (colorScheme) => {
  const ink = getChartInk(colorScheme);
  return {
    bgcolor: ink.legendBg,
    borderwidth: 0,
    font: { size: 12, color: ink.text },
    itemwidth: 30,
  };
};

// 1m / 6m / all: quiet text buttons instead of grey tiles
export const getRangeSelectorStyle = (colorScheme) => {
  const ink = getChartInk(colorScheme);
  return {
    bgcolor: "rgba(0,0,0,0)",
    activecolor: colorScheme === "dark" ? "#1c3a55" : "#e3eef8",
    font: { size: 12, color: ink.soft },
  };
};

// Range buttons for the time-series charts: 1m / 6m (/ 1y) / All
export const getRangeSelector = (
  colorScheme,
  { includeYear = false } = {},
) => ({
  ...getRangeSelectorStyle(colorScheme),
  buttons: [
    { count: 1, label: "1m", step: "month", stepmode: "backward" },
    { count: 6, label: "6m", step: "month", stepmode: "backward" },
    ...(includeYear
      ? [{ count: 1, label: "1y", step: "year", stepmode: "backward" }]
      : []),
    { step: "all", label: "All" },
  ],
});

// Plotly config for the full-size charts. The toolbar is hidden: download
// lives in the chart header, zoom in the minimap and range buttons, and the
// header's "Reset axes" button resets the view. Double-click reset is off: on
// mobile it fired when tapping legend items (e.g. toggling ground truth).
export const PLOT_CONFIG = {
  responsive: true,
  displayModeBar: false,
  displaylogo: false,
  scrollZoom: false,
  doubleClick: false,
};

// Vertical marker for a forecast date: a thin neutral rule, not a red dash
export const getForecastDateLineStyle = (colorScheme) => ({
  color: getChartInk(colorScheme).marker,
  width: 1.25,
});

// Forecast date lines are drawn a few days before the reference date
export const FORECAST_LINE_OFFSET_DAYS = -3;

export const shiftDateStringByDays = (dateString, days) => {
  const [year, month, day] = dateString.split("-").map(Number);
  const shiftedDate = new Date(Date.UTC(year, month - 1, day + days));
  return shiftedDate.toISOString().slice(0, 10);
};

// Layout shapes and annotations marking forecast dates (see
// useForecastDateDrag): a rule per date, the one being dragged dashed and
// labelled with the date it will land on
export const getForecastDateMarks = (
  colorScheme,
  dates,
  draggingDate = null,
  offsetDays = FORECAST_LINE_OFFSET_DAYS,
) => {
  const lineStyle = getForecastDateLineStyle(colorScheme);
  const shapes = dates.map((date) => {
    const x = shiftDateStringByDays(date, offsetDays);
    return {
      type: "line",
      x0: x,
      x1: x,
      y0: 0,
      y1: 1,
      yref: "paper",
      line:
        date === draggingDate
          ? { ...lineStyle, width: 2.5, dash: "dash" }
          : lineStyle,
    };
  });
  const annotations = draggingDate
    ? [
        {
          x: shiftDateStringByDays(draggingDate, offsetDays),
          y: 1,
          yref: "paper",
          yanchor: "bottom",
          text: draggingDate,
          showarrow: false,
          font: { ...getChartFont(colorScheme), size: 12 },
          bgcolor: colorScheme === "dark" ? "#25262b" : "#ffffff",
          borderpad: 2,
        },
      ]
    : [];
  return { shapes, annotations };
};

// Preliminary NHSN releases are drawn dashed in each pathogen's colour and
// share their pathogen's legend entry; this plain-text legend title explains
// the dashes.
export const getPreliminaryLegendTitle = (colorScheme) => ({
  text: "Dashed: preliminary release",
  side: "top",
  font: { size: 11, color: getChartInk(colorScheme).soft },
});

// Observed data is the reference series: drawn heavier than model lines
export const GROUND_TRUTH_LINE_WIDTH = 2.5;
export const GROUND_TRUTH_MARKER_SIZE = 5;

// The layout every Plotly chart in the app starts from: app font, spined
// axes, borderless legend, transparent background. `compact` is for small
// charts (front-page cards, My Plots tiles). Callers spread it and add their
// own ranges, titles and shapes.
export const getBaseChartLayout = (colorScheme, { compact = false } = {}) => {
  const axis = getChartAxisStyle(colorScheme);
  const tickfont = { ...axis.tickfont, size: compact ? 10 : 12 };
  return {
    autosize: true,
    template: colorScheme === "dark" ? "plotly_dark" : "plotly_white",
    ...PLOT_CHROME,
    font: { ...getChartFont(colorScheme), size: compact ? 11 : 13 },
    legend: {
      ...getChartLegendStyle(colorScheme),
      x: 0.01,
      y: 0.99,
      xanchor: "left",
      yanchor: "top",
    },
    xaxis: { ...axis, tickfont },
    yaxis: { ...axis, tickfont },
  };
};

// Observed line on small charts: still the heaviest line, scaled down
export const COMPACT_GROUND_TRUTH_LINE_WIDTH = 2;

// Global Chart.js defaults matching the Plotly charts: app font, muted tick
// text, hairline grid
export const applyChartJsDefaults = () => {
  const ink = CHART_INK.light;
  ChartJS.defaults.font.family = SYSTEM_SANS;
  ChartJS.defaults.font.size = 12;
  ChartJS.defaults.color = ink.soft;
  ChartJS.defaults.borderColor = ink.grid;
};
