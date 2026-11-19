import { useMemo, useState } from "react";
import { normalizeCountyBasename } from "../../utils/nsspGeo";
import Plot from "react-plotly.js";
import {
  Button,
  Group,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Text,
  useMantineColorScheme,
} from "@mantine/core";
import NSSPGeoMap from "../NSSPGeoMap";
import { NSSP_MAP_COLORS, NSSP_MAP_HEIGHTS } from "../../utils/nsspMap";
import {
  PLOT_CONFIG,
  getBaseChartLayout,
  getChartInk,
  shiftDateStringByDays,
} from "../../constants/chart";

// A snapshot row is [latest reported value, ensemble median at horizons
// 0-3]; the latest reported week is the one before the forecast date, so
// horizons 1-3 are the 2-4 weeks after it. Every label says which it is.
const WHEN = [
  { index: 0, weeks: 0, label: "Reported" },
  { index: 2, weeks: 2, label: "Forecast" },
  { index: 3, weeks: 3, label: "Forecast" },
  { index: 4, weeks: 4, label: "Forecast" },
];
const PALETTE = [
  "#eff3ff",
  "#c6dbef",
  "#9ecae1",
  "#6baed6",
  "#3182bd",
  "#08519c",
];
const FOCUS_COLOR = "#e03131";
const SELECTED_COLOR = "#f08c00";

const shortDate = (date) =>
  new Date(`${date}T00:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });

// How far a forecast week is past the latest reported week
const ahead = (weeks, short = false) =>
  `${weeks} ${short ? "wk" : weeks === 1 ? "week" : "weeks"} ahead`;

const ordinal = (n) => {
  const suffixes = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${suffixes[(v - 20) % 10] || suffixes[v] || suffixes[0]}`;
};

/**
 * Where one location sits among its peers (states, or a state's districts):
 * a map colored by the latest reported value or the forecast 2-4 weeks
 * later, a sentence on the clicked location's rank, and every peer's change
 * from reported to forecast 4 weeks later.
 *
 * rows: {key: snapshot row}; names: {key: name}; focus: the summary's key
 * (highlighted); geo: features for the map; featureRow(feature) -> key;
 * reference: a row shown for context but not ranked (the US).
 */
const SummaryComparison = ({
  rows,
  names,
  focus,
  reference,
  date,
  geo,
  featureRow,
  projection,
  format,
  suffix = "",
  peers,
  // States: a compact strip first, the full ranking on demand. Districts
  // are few, so they always show the ranking.
  collapsible = false,
}) => {
  const { colorScheme } = useMantineColorScheme();
  const [when, setWhen] = useState("0");
  const [selected, setSelected] = useState(null);
  const [expanded, setExpanded] = useState(!collapsible);
  const { index, weeks } = WHEN.find((w) => String(w.index) === when);
  const observedWeek = shiftDateStringByDays(date, -7);
  const weekLabel = (n) =>
    shortDate(shiftDateStringByDays(observedWeek, 7 * n));

  // Ranked by the shown week's value, highest first
  const ranked = useMemo(
    () =>
      Object.keys(rows)
        .filter((key) => rows[key][index] != null)
        .sort((a, b) => rows[b][index] - rows[a][index]),
    [rows, index],
  );
  const max = useMemo(
    () =>
      Math.max(
        ...Object.values(rows).flatMap((r) => r.filter((v) => v != null)),
        0,
      ),
    [rows],
  );
  const binOf = (value) =>
    Math.min(
      PALETTE.length - 1,
      Math.floor((value / (max || 1)) * PALETTE.length),
    );
  const fill = (feature) => {
    const value = rows[featureRow(feature)]?.[index];
    return value == null ? NSSP_MAP_COLORS.unavailable : PALETTE[binOf(value)];
  };

  const describe = (key) => {
    const value = rows[key]?.[index];
    const week = `the week ending ${weekLabel(weeks)}`;
    if (value == null)
      return `${names[key]}: no ${weeks ? "forecast" : "reported value"} for ${week}.`;
    const rank = `${ordinal(ranked.indexOf(key) + 1)} highest of ${ranked.length} ${peers}`;
    if (weeks)
      return `${names[key]}: forecast ${format(value)} for ${week} (${ahead(weeks)}), ${rank}.`;
    const reported = rows[key][0];
    const forecast = rows[key][4];
    let text = `${names[key]}: reported ${format(value)} for ${week}, ${rank}.`;
    if (forecast != null)
      text += ` Forecast ${format(forecast)} for the week ending ${weekLabel(4)} (${ahead(4)}, ${forecast >= reported ? "+" : ""}${Math.round((forecast / (reported || 1) - 1) * 100)}%).`;
    return text;
  };

  // Reported → forecast 4 weeks later, every peer, lowest at the bottom
  const chart = useMemo(() => {
    const order = Object.keys(rows)
      .filter((key) => rows[key][0] != null)
      .sort((a, b) => rows[a][0] - rows[b][0]);
    const label = (key) =>
      key === focus ? `<b>${names[key]}</b>` : names[key];
    const color = (key, base) =>
      key === focus ? FOCUS_COLOR : key === selected ? SELECTED_COLOR : base;
    const ink = getChartInk(colorScheme);
    const y = order.map(label);
    // Strip: one dot per peer on two rows (reported, forecast), names on
    // hover. A fixed small jitter keeps neighbours apart; the highlighted
    // ones sit on the row's line, drawn last so they stay on top.
    const highlighted = [selected, focus].filter((k) => k && rows[k]);
    const stripOrder = [
      ...order.filter((k) => !highlighted.includes(k)),
      ...highlighted,
    ];
    const jitter = (key) => {
      const hash = [...key].reduce(
        (h, c) => (h * 31 + c.charCodeAt(0)) >>> 0,
        7,
      );
      return highlighted.includes(key) ? 0 : ((hash % 100) / 100 - 0.5) * 0.4;
    };
    const stripRow = (col, row, name, symbol, base) => ({
      x: stripOrder.map((k) => rows[k][col]),
      y: stripOrder.map((k) => row + jitter(k)),
      customdata: stripOrder,
      text: stripOrder.map((k) => names[k]),
      mode: "markers",
      marker: {
        symbol,
        size: stripOrder.map((k) => (highlighted.includes(k) ? 12 : 8)),
        color: stripOrder.map((k) => color(k, base)),
        opacity: stripOrder.map((k) => (highlighted.includes(k) ? 1 : 0.65)),
        line: { color: "#fff", width: 1 },
      },
      hovertemplate: `<b>%{text}</b><br>${name}: %{x:.2f}${suffix}<extra></extra>`,
    });
    const strip = [
      // The highlighted peers' move from reported to forecast
      ...highlighted.map((k) => ({
        x: [rows[k][0], rows[k][4]],
        y: [1, 0],
        mode: "lines",
        line: { color: color(k), width: 1.5, dash: "dot" },
        hoverinfo: "skip",
      })),
      stripRow(
        0,
        1,
        `Reported, week ending ${weekLabel(0)}`,
        "circle",
        "#adb5bd",
      ),
      stripRow(
        4,
        0,
        `Forecast, week ending ${weekLabel(4)} (${ahead(4)})`,
        "diamond",
        PALETTE[4],
      ),
    ];
    return {
      strip,
      // Compact rows: every label shown, small markers
      height: 56 + 12 * order.length,
      labels: y,
      data: [
        {
          x: order.flatMap((k) => [rows[k][0], rows[k][4], null]),
          y: order.flatMap((k) => [label(k), label(k), null]),
          mode: "lines",
          line: { color: ink.grid, width: 2 },
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x: order.map((k) => rows[k][0]),
          y,
          customdata: order,
          mode: "markers",
          name: `Reported, week ending ${weekLabel(0)}`,
          marker: { size: 6, color: order.map((k) => color(k, "#adb5bd")) },
          hovertemplate: `%{y}<br>Reported, week ending ${weekLabel(0)}: %{x:.2f}${suffix}<extra></extra>`,
        },
        {
          x: order.map((k) => rows[k][4]),
          y,
          customdata: order,
          mode: "markers",
          name: `Forecast, week ending ${weekLabel(4)} (${ahead(4)}, ensemble median)`,
          marker: {
            size: 7,
            symbol: "diamond",
            color: order.map((k) => color(k, PALETTE[4])),
          },
          hovertemplate: `%{y}<br>Forecast, week ending ${weekLabel(4)} (${ahead(4)}): %{x:.2f}${suffix}<extra></extra>`,
        },
      ],
    };
  }, [rows, names, focus, selected, colorScheme, suffix]); // eslint-disable-line react-hooks/exhaustive-deps

  const base = getBaseChartLayout(colorScheme, { compact: true });
  // Every peer ranked, reported → forecast: beside the map for districts,
  // full width under it for states once expanded
  const rankingChart = (
    <Plot
      style={{ width: "100%", height: chart.height }}
      data={chart.data}
      useResizeHandler
      layout={{
        ...base,
        margin: { l: 10, r: 10, t: 28, b: 24 },
        hovermode: "closest",
        showlegend: true,
        legend: {
          ...base.legend,
          orientation: "h",
          x: 0,
          y: 1.01,
          yanchor: "bottom",
          font: { size: 11 },
        },
        xaxis: {
          ...base.xaxis,
          rangemode: "tozero",
          ticksuffix: suffix,
          automargin: true,
        },
        yaxis: {
          ...base.yaxis,
          automargin: true,
          type: "category",
          // All rows labelled, in a small font
          tickmode: "array",
          tickvals: chart.labels,
          ticktext: chart.labels,
          tickfont: { ...base.yaxis.tickfont, size: 9 },
          ticklen: 3,
        },
      }}
      config={{ ...PLOT_CONFIG }}
      onClick={(event) => {
        const key = event.points?.[0]?.customdata;
        if (key) setSelected(key);
      }}
    />
  );
  const legendEdges = PALETTE.map((_, i) => format((max * i) / PALETTE.length));

  return (
    <Stack gap="xs">
      <Group justify="space-between" wrap="wrap">
        <SegmentedControl
          size="xs"
          value={when}
          onChange={setWhen}
          data={WHEN.map((w) => ({
            value: String(w.index),
            label: w.weeks
              ? `${w.label} ${weekLabel(w.weeks)} (${ahead(w.weeks, true)})`
              : `${w.label} ${weekLabel(w.weeks)}`,
          }))}
          aria-label="Week shown on the map (week ending)"
        />
        <Group gap={2} wrap="nowrap" aria-label="Map color scale">
          {PALETTE.map((c, i) => (
            <Stack key={c} gap={0} align="center">
              <div
                style={{
                  width: 34,
                  height: 10,
                  background: c,
                  border: "1px solid #ccc",
                }}
              />
              <Text size="10px" c="dimmed">
                {legendEdges[i]}
              </Text>
            </Stack>
          ))}
        </Group>
      </Group>
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        <div>
          <NSSPGeoMap
            featureCollection={geo}
            height={NSSP_MAP_HEIGHTS[projection]}
            projectionKind={projection}
            getFeatureKey={(f) => f.properties.GEOID}
            getFeatureLabel={(f) =>
              names[featureRow(f)] ? describe(featureRow(f)) : f.properties.NAME
            }
            getFeatureFill={fill}
            isFeatureClickable={(f) => Boolean(rows[featureRow(f)])}
            onFeatureClick={(f) => setSelected(featureRow(f))}
            getFeatureCallout={(f) =>
              projection === "usa" && [focus, selected].includes(featureRow(f))
                ? {
                    label: featureRow(f),
                    offset: [0, 0],
                    // Same color as the state's dot in the text and charts
                    color:
                      featureRow(f) === focus ? FOCUS_COLOR : SELECTED_COLOR,
                  }
                : null
            }
          />
        </div>
        <Stack gap={4}>
          {focus && (
            <Text size="sm">
              <Text span c={FOCUS_COLOR} fw={600}>
                ●
              </Text>{" "}
              {describe(focus)}
            </Text>
          )}
          {selected && selected !== focus && (
            <Text size="sm">
              <Text span c={SELECTED_COLOR} fw={600}>
                ●
              </Text>{" "}
              {describe(selected)}
            </Text>
          )}
          {reference && (
            <Text size="xs" c="dimmed">
              {reference.name}:{" "}
              {reference.row[index] == null
                ? `no ${weeks ? "forecast" : "reported value"}`
                : `${weeks ? "forecast" : "reported"} ${format(reference.row[index])}`}
            </Text>
          )}
          <Text size="xs" c="dimmed">
            Highlight another {peers === "states" ? "state" : "district"} by
            clicking it on the map.
          </Text>
          {!collapsible && rankingChart}
          {!expanded && (
            <Plot
              style={{ width: "100%", height: 150 }}
              data={chart.strip}
              useResizeHandler
              layout={{
                ...base,
                margin: { l: 10, r: 10, t: 8, b: 24 },
                hovermode: "closest",
                showlegend: false,
                xaxis: {
                  ...base.xaxis,
                  rangemode: "tozero",
                  ticksuffix: suffix,
                  automargin: true,
                },
                yaxis: {
                  ...base.yaxis,
                  automargin: true,
                  range: [-0.5, 1.5],
                  tickvals: [0, 1],
                  ticktext: [
                    `Forecast, ${weekLabel(4)} (${ahead(4, true)})`,
                    `Reported, ${weekLabel(0)}`,
                  ],
                  showgrid: false,
                  zeroline: false,
                },
              }}
              config={{ ...PLOT_CONFIG }}
              onClick={(event) => {
                const key = event.points?.[0]?.customdata;
                if (key) setSelected(key);
              }}
            />
          )}
          {collapsible && (
            <Button
              variant="subtle"
              size="compact-sm"
              w="fit-content"
              onClick={() => setExpanded((e) => !e)}
            >
              {expanded ? "Show less" : `Show the ranking of all ${peers}`}
            </Button>
          )}
        </Stack>
      </SimpleGrid>
      {collapsible && expanded && rankingChart}
    </Stack>
  );
};

const METRICS = {
  ed: { label: "ED visits (%)", format: (v) => `${v.toFixed(2)}%` },
  hosp: {
    label: "Hospital admissions per 100k",
    format: (v) => v.toFixed(1),
  },
};

// The state among all states, by ED visits or admissions per 100,000
export const StateComparison = ({ day, snapshot, geo, stateCode, date }) => {
  const [metric, setMetric] = useState("ed");
  const { US, ...rows } = day?.[metric] || {};
  const names = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(snapshot.states).map(([k, s]) => [k, s.name]),
      ),
    [snapshot],
  );
  return (
    <Stack gap="xs">
      <SegmentedControl
        size="xs"
        w="fit-content"
        value={metric}
        onChange={setMetric}
        data={Object.entries(METRICS).map(([value, m]) => ({
          value,
          label: m.label,
        }))}
        aria-label="Measure compared"
      />
      {rows[stateCode] ? (
        <SummaryComparison
          rows={rows}
          names={names}
          focus={stateCode}
          reference={US && { name: "United States", row: US }}
          date={date}
          geo={geo}
          featureRow={(f) => f.properties.STUSAB}
          projection="usa"
          format={METRICS[metric].format}
          suffix={metric === "ed" ? "%" : ""}
          peers="states"
          collapsible
        />
      ) : (
        <Text size="sm" c="dimmed" fs="italic">
          No comparison for this date.
        </Text>
      )}
    </Stack>
  );
};

// The state's MetroCast districts against each other (flu ED visits); the
// map colors each county by its district
export const DistrictComparison = ({ day, snapshot, geo, stateCode, date }) => {
  const { rows, names, countyDistrict } = useMemo(() => {
    const subs = Object.entries(snapshot.subLocations).filter(
      ([, s]) => s.state === stateCode,
    );
    return {
      rows: Object.fromEntries(
        subs.filter(([k]) => day?.edSub?.[k]).map(([k]) => [k, day.edSub[k]]),
      ),
      names: Object.fromEntries(subs.map(([k, s]) => [k, s.name])),
      countyDistrict: Object.fromEntries(
        subs.flatMap(([k, s]) =>
          s.counties.map((c) => [normalizeCountyBasename(c), k]),
        ),
      ),
    };
  }, [snapshot, day, stateCode]);
  if (!Object.keys(rows).length) return null;
  const state = day?.ed?.[stateCode];
  return (
    <SummaryComparison
      rows={rows}
      names={names}
      reference={
        state && { name: snapshot.states[stateCode]?.name, row: state }
      }
      date={date}
      geo={geo}
      featureRow={(f) =>
        countyDistrict[normalizeCountyBasename(f.properties.BASENAME)]
      }
      projection="state"
      format={METRICS.ed.format}
      suffix="%"
      peers="districts"
    />
  );
};

export default SummaryComparison;
