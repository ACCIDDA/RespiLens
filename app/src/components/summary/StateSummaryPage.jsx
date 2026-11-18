import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Accordion,
  ActionIcon,
  Badge,
  Box,
  Center,
  Container,
  Group,
  Loader,
  Paper,
  Popover,
  Select,
  SimpleGrid,
  Stack,
  Tabs,
  Text,
  Title,
  Tooltip,
  useMantineColorScheme,
} from "@mantine/core";
import {
  IconAdjustmentsHorizontal,
  IconChevronLeft,
  IconChevronRight,
} from "@tabler/icons-react";
import Seo from "../Seo";
import NotFoundPage from "../NotFoundPage";
import { MetroPlotCard } from "../views/MetroCastView";
import ForecastChartControls from "../controls/ForecastChartControls";
import { useView } from "../../hooks/useView";
import { useAsyncData } from "../../hooks/useAsyncData";
import { fetchJson, getDataPath } from "../../utils/paths";
import { buildPeakForecastTraces } from "../../utils/peakForecastTraces";
import {
  fetchNsspCountiesGeoJson,
  fetchNsspStatesGeoJson,
} from "../../utils/nsspGeo";
import { DistrictComparison, StateComparison } from "./SummaryComparison";
import { getModelColor } from "../../config";
import { shiftDateStringByDays as shift } from "../../constants/chart";
import {
  describeSeries,
  peakSentence,
  seriesSentence,
} from "./summaryNarrative";

// Proof of concept: one state. Each entry says where its data lives.
const SUMMARY_STATES = {
  UT: {
    name: "Utah",
    flusight: "flusight/UT_flu.json",
    metrocast: "flumetrocast/utah_flu_metrocast.json",
    covid: "covid19forecasthub/UT_covid19.json",
    rsv: "rsvforecasthub/UT_rsv.json",
  },
};

const FLUSIGHT = "FluSight-ensemble";
// MetroCast's two ensembles share flu's ED chart: the 4-week one (named
// _short, or _mean last season) and the one through the end of the season
const MC_4WK = "MetroCast ensemble (4 weeks)";
const MC_SEASON = "MetroCast ensemble (full season)";
const METROCAST_ENSEMBLES = {
  [MC_4WK]: ["epiENGAGE-ensemble_short", "epiENGAGE-ensemble_mean"],
  [MC_SEASON]: ["epiENGAGE-ensemble_long"],
};
const COLORS = {
  [FLUSIGHT]: "#0d8ae6",
  [MC_4WK]: "#3f51b5",
  [MC_SEASON]: "#f08c00",
};
const colorOf = (model) => COLORS[model] ?? getModelColor(model);

const FLU_ED = "Flu ED visits pct";
// Same NSSP signal as MetroCast's, as a proportion instead of a percent
const FLU_ED_FLUSIGHT = "wk inc flu prop ed visits";
const FLU_HOSP = "wk inc flu hosp";

// COVID-19 and RSV: one hub each, same two targets as FluSight
const OTHER_HUBS = [
  {
    key: "covid",
    label: "COVID-19",
    hubName: "COVID-19 Forecast Hub",
    ensemble: "CovidHub-ensemble",
    edTarget: "wk inc covid prop ed visits",
    edKey: "COVID ED visits pct",
    hospTarget: "wk inc covid hosp",
  },
  {
    key: "rsv",
    label: "RSV",
    hubName: "RSV Forecast Hub",
    ensemble: "RSVHub-ensemble",
    edTarget: "wk inc rsv prop ed visits",
    edKey: "RSV ED visits pct",
    hospTarget: "wk inc rsv hosp",
  },
];

const RATE_CHANGE_LABELS = {
  large_decrease: "Large decrease",
  decrease: "Decrease",
  stable: "Stable",
  increase: "Increase",
  large_increase: "Large increase",
};

const WEEKS_BEFORE = 20;
const WEEKS_AFTER = 5;
// ED visits are percents, admissions counts; below `floor` changes are noise
const PERCENT = { format: (v) => `${v.toFixed(2)}%`, floor: 0.2, step: 0.01 };
const COUNT = {
  format: (v) => Math.round(v).toLocaleString(),
  floor: 10,
  step: 1,
};

const formatDate = (date) =>
  new Date(`${date}T00:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

// Both MetroCast ensembles a date has, under their display names
const metrocastEnsembles = (targetData) =>
  Object.fromEntries(
    Object.entries(METROCAST_ENSEMBLES).map(([label, names]) => [
      label,
      targetData?.[names.find((m) => targetData?.[m])],
    ]),
  );

const scaleForecast = (forecast, factor) =>
  forecast && {
    ...forecast,
    predictions: Object.fromEntries(
      Object.entries(forecast.predictions).map(([h, p]) => [
        h,
        { ...p, values: p.values.map((v) => v * factor) },
      ]),
    ),
  };

// One ground-truth series as a percent, under a new key
const percentGroundTruth = (groundTruth, target, key) => ({
  dates: groundTruth?.dates || [],
  [key]: (groundTruth?.[target] || []).map((v) => (v == null ? v : v * 100)),
});

// The last observation before the reference date: what forecasters had
// (the reference week itself is not reported yet when they submit)
const latestObserved = (groundTruth, target, date) => {
  const dates = groundTruth?.dates || [];
  for (let i = dates.length - 1; i >= 0; i -= 1) {
    const v = groundTruth[target]?.[i];
    if (dates[i] < date && v != null) return { date: dates[i], value: v };
  }
  return null;
};

const quantile = (prediction, q) => {
  const i = (prediction?.quantiles || []).findIndex((x) => Number(x) === q);
  return i === -1 ? null : prediction.values[i];
};

// A box for a model: next week's median against the latest observed value,
// and its 95% interval
const forecastTile = (label, model, forecast, latest, unit) => {
  const p = forecast?.predictions?.["1"];
  const tile = { label: `${label} · next week`, color: colorOf(model) };
  if (!p) return { ...tile, detail: "No forecast" };
  const median = quantile(p, 0.5);
  const diff = latest ? median - latest.value : null;
  return {
    ...tile,
    value: unit.format(median),
    delta: latest && {
      up: diff > 0,
      flat: Math.abs(diff) < unit.step,
      text:
        latest.value > 0
          ? `${Math.round(Math.abs(diff / latest.value) * 100)}%`
          : unit.format(Math.abs(diff)),
    },
    detail: `95%: ${unit.format(quantile(p, 0.025))}–${unit.format(quantile(p, 0.975))} · wk ending ${formatDate(p.date)}`,
  };
};

const rateChangeTile = (forecast) => {
  const label = "Rate change · next week";
  const p = forecast?.predictions?.["1"];
  if (!p) return { label, detail: "No forecast" };
  const best = p.probabilities.indexOf(Math.max(...p.probabilities));
  return {
    label,
    value: RATE_CHANGE_LABELS[p.categories[best]],
    detail: `${Math.round(p.probabilities[best] * 100)}% probability (FluSight ensemble)`,
  };
};

/**
 * One statewide chart: an observed series, the forecasts drawn on it (in
 * drawing order) and its boxes. `tiles` names the models that get one.
 */
const buildChart = ({
  groundTruth,
  target,
  unit,
  date,
  models,
  tiles,
  extraTiles = [],
  ...rest
}) => {
  const latest = latestObserved(groundTruth, target, date);
  const present = Object.fromEntries(
    Object.entries(models).filter(([, forecast]) => forecast),
  );
  return {
    ...rest,
    target,
    data: {
      ground_truth: groundTruth,
      forecasts: { [date]: { [target]: present } },
    },
    models: Object.keys(present),
    facts: describeSeries(groundTruth, target, date, unit.floor),
    tiles: [
      {
        label: "Latest observed",
        value: latest && unit.format(latest.value),
        detail: latest && `wk ending ${formatDate(latest.date)}`,
      },
      ...tiles.map(([model, label]) =>
        forecastTile(label, model, models[model], latest, unit),
      ),
      ...extraTiles,
    ],
  };
};

// Flu: FluSight and MetroCast at state level, FluSight's peak forecast on the
// admissions chart (drawn as in the peak view), and MetroCast's counties
const fluSection = (data, date, stateCode, intervals) => {
  const fs = data.flusight?.forecasts?.[date] || {};
  const mc = metrocastEnsembles(data.metrocast?.forecasts?.[date]?.[FLU_ED]);
  const peak = data.flusight?.peaks?.[date];
  const peakTraces = buildPeakForecastTraces({
    peaks: data.flusight?.peaks,
    selectedModels: [FLUSIGHT],
    selectedDates: [date],
    showMedian: intervals?.median ?? true,
    show50: intervals?.ci50 ?? true,
    show95: intervals?.ci95 ?? true,
    nameSuffix: " peak",
  });
  // With a peak forecast the admissions chart spans the season, like the
  // peak view, through the end of the peak's whiskers
  const peakEnd = peakTraces
    .flatMap((t) => t.x)
    .filter(Boolean)
    .sort()
    .at(-1);
  const [year, month] = date.split("-").map(Number);

  const ed = buildChart({
    title: "Emergency department visits",
    subtitle:
      "Percent of ED visits due to influenza (NSSP), statewide. FluSight and MetroCast forecast the same target.",
    groundTruth:
      data.metrocast?.ground_truth ??
      percentGroundTruth(data.flusight?.ground_truth, FLU_ED_FLUSIGHT, FLU_ED),
    target: FLU_ED,
    unit: PERCENT,
    date,
    // In drawing order: the wide season band underneath
    models: {
      [MC_SEASON]: mc[MC_SEASON],
      [MC_4WK]: mc[MC_4WK],
      [FLUSIGHT]: scaleForecast(fs[FLU_ED_FLUSIGHT]?.[FLUSIGHT], 100),
    },
    tiles: [
      [FLUSIGHT, "FluSight ensemble"],
      [MC_4WK, "MetroCast ensemble"],
    ],
  });
  const hosp = buildChart({
    title: "Hospital admissions",
    subtitle:
      "Weekly confirmed influenza hospital admissions (NHSN), statewide, with the FluSight ensemble's forecast of the season peak (size and week).",
    groundTruth: data.flusight?.ground_truth,
    target: FLU_HOSP,
    unit: COUNT,
    date,
    models: { [FLUSIGHT]: fs[FLU_HOSP]?.[FLUSIGHT] },
    tiles: [[FLUSIGHT, "FluSight ensemble"]],
    extraTiles: [rateChangeTile(fs["wk flu hosp rate change"]?.[FLUSIGHT])],
    extraTraces: peakTraces,
    bounds: peakEnd && {
      start: `${month >= 8 ? year : year - 1}-08-01`,
      end: shift(peakEnd, 14),
    },
  });

  return {
    key: "flu",
    label: "Flu",
    brief: [
      seriesSentence("Flu ED visits were at", ed.facts, PERCENT.format),
      seriesSentence(
        "Weekly flu hospital admissions were",
        hosp.facts,
        COUNT.format,
      ),
      peakSentence(
        peak?.["peak inc flu hosp"]?.[FLUSIGHT]?.predictions,
        peak?.["peak week inc flu hosp"]?.[FLUSIGHT]?.predictions,
        hosp.facts?.peak,
        COUNT.format,
      ),
    ],
    charts: [ed, hosp],
    subLocations: data.subLocations.map((l) => {
      const ens = metrocastEnsembles(l.data.forecasts?.[date]?.[FLU_ED]);
      return {
        key: l.abbreviation,
        title: l.location_name.replace(`, ${stateCode}`, ""),
        models: [MC_SEASON, MC_4WK].filter((m) => ens[m]),
        data: {
          ground_truth: l.data.ground_truth,
          forecasts: { [date]: { [FLU_ED]: ens } },
        },
      };
    }),
  };
};

const hubSection = (hub, hubData, date) => {
  if (!hubData) return { ...hub, missing: `No ${hub.hubName} data.` };
  const fc = hubData.forecasts?.[date] || {};
  const gt = hubData.ground_truth;
  const tiles = [[hub.ensemble, hub.ensemble]];
  const ed = buildChart({
    title: "Emergency department visits",
    subtitle: `Percent of ED visits due to ${hub.label} (NSSP), statewide.`,
    groundTruth: percentGroundTruth(gt, hub.edTarget, hub.edKey),
    target: hub.edKey,
    unit: PERCENT,
    date,
    models: {
      [hub.ensemble]: scaleForecast(fc[hub.edTarget]?.[hub.ensemble], 100),
    },
    tiles,
  });
  const hosp = buildChart({
    title: "Hospital admissions",
    subtitle: `Weekly confirmed ${hub.label} hospital admissions (NHSN), statewide.`,
    groundTruth: gt,
    target: hub.hospTarget,
    unit: COUNT,
    date,
    models: { [hub.ensemble]: fc[hub.hospTarget]?.[hub.ensemble] },
    tiles,
  });
  return {
    ...hub,
    brief: [
      seriesSentence(
        `${hub.label} ED visits were at`,
        ed.facts,
        PERCENT.format,
      ),
      seriesSentence(
        `Weekly ${hub.label} hospital admissions were`,
        hosp.facts,
        COUNT.format,
      ),
    ],
    charts: [ed, hosp],
  };
};

const StatTile = ({ label, value, detail, color, delta }) => (
  <Paper withBorder radius="md" p="sm">
    <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
      {label}
    </Text>
    <Group gap="xs" align="baseline">
      <Text size="xl" fw={700} c={color}>
        {value ?? "—"}
      </Text>
      {delta && (
        <Text
          size="sm"
          fw={600}
          c={delta.flat ? "dimmed" : delta.up ? "red.7" : "teal.7"}
        >
          {delta.flat ? "≈ no change" : `${delta.up ? "▲" : "▼"} ${delta.text}`}{" "}
          <Text span size="xs" c="dimmed" fw={400}>
            vs. latest observed
          </Text>
        </Text>
      )}
    </Group>
    {detail && (
      <Text size="xs" c="dimmed">
        {detail}
      </Text>
    )}
  </Paper>
);

// One target or comparison in a disease tab, collapsible (see DiseasePanel)
const Section = ({ title, subtitle, children }) => (
  <Accordion.Item value={title}>
    <Accordion.Control>
      <Title order={3}>{title}</Title>
    </Accordion.Control>
    <Accordion.Panel>
      <Stack gap="xs">
        <Text size="sm" c="dimmed">
          {subtitle}
        </Text>
        {children}
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

// A chart with its own zoom, back to the default window on a new date: a
// few months before the date to a few weeks after, or `bounds`
const SummaryPlot = ({ bounds, ...props }) => {
  const date = props.selectedDates[0];
  const firstDate = props.locationData.ground_truth?.dates?.[0];
  const [range, setRange] = useState(null);
  useEffect(() => setRange(null), [date]);
  const getDefaultRange = useCallback(
    (forRangeslider = false) => {
      const after = shift(date, 7 * WEEKS_AFTER);
      const end = bounds?.end > after ? bounds.end : after;
      if (forRangeslider) return [firstDate ?? shift(date, -365), end];
      return [bounds?.start ?? shift(date, -7 * WEEKS_BEFORE), end];
    },
    [date, firstDate, bounds?.start, bounds?.end],
  );
  return (
    <MetroPlotCard
      {...props}
      getDefaultRange={getDefaultRange}
      xAxisRange={range}
      setXAxisRange={setRange}
    />
  );
};

// `compare`: the all-locations snapshot for this disease and date, with the
// map shapes, for the comparisons with other states and between districts
const DiseasePanel = ({ section, chartProps, compare }) => {
  if (section.missing)
    return (
      <Text size="sm" c="dimmed" fs="italic">
        {section.missing}
      </Text>
    );
  const sentences = section.brief.filter(Boolean);
  const subs = section.subLocations;
  const compareTitle = compare?.stateName
    ? `How ${compare.stateName} compares with other states`
    : null;
  const subsTitle = "Counties and health districts";
  // Every section starts open
  const titles = [
    ...section.charts.map((chart) => chart.title),
    compare?.day && compareTitle,
    subs && subsTitle,
  ].filter(Boolean);
  return (
    <Stack gap="md">
      {sentences.length > 0 && (
        <Paper
          withBorder
          radius="md"
          p="md"
          bg="var(--mantine-color-default-hover)"
        >
          <Text size="xs" c="dimmed" tt="uppercase" fw={600} mb={4}>
            Compared with last season
          </Text>
          <Stack gap={6}>
            {sentences.map((s) => (
              <Text key={s} size="sm">
                {s}
              </Text>
            ))}
          </Stack>
        </Paper>
      )}
      <Accordion
        multiple
        defaultValue={titles}
        variant="separated"
        chevronPosition="left"
      >
        {section.charts.map((chart) => (
          <Section
            key={chart.title}
            title={chart.title}
            subtitle={chart.subtitle}
          >
            <SimpleGrid cols={{ base: 1, xs: chart.tiles.length }} spacing="sm">
              {chart.tiles.map((tile) => (
                <StatTile key={tile.label} {...tile} />
              ))}
            </SimpleGrid>
            <SummaryPlot
              {...chartProps}
              locationData={chart.data}
              selectedTarget={chart.target}
              selectedModels={chart.models}
              extraTraces={chart.extraTraces}
              bounds={chart.bounds}
            />
          </Section>
        ))}
        {compare?.day && (
          <Section
            title={compareTitle}
            subtitle="Latest week and the hub ensemble's forecast 2–4 weeks on. Hospital admissions are per 100,000 people."
          >
            <StateComparison {...compare} geo={compare.geo.states} />
          </Section>
        )}
        {subs && (
          <Section
            title={subsTitle}
            subtitle={
              subs.some((l) => l.models.length)
                ? "Flu ED visits (%), MetroCast ensembles (4 weeks and full season)."
                : "Flu ED visits (%). MetroCast has no sub-state forecasts for this date; observed data only."
            }
          >
            {compare?.day && (
              <DistrictComparison {...compare} geo={compare.geo.counties} />
            )}
            {/* At most 3 per row, taller than the MetroCast page's tiles */}
            <SimpleGrid cols={{ base: 1, xs: 2, md: 3 }} spacing="md">
              {subs.map((l) => (
                <Box key={l.key} style={{ minWidth: 0, position: "relative" }}>
                  <SummaryPlot
                    {...chartProps}
                    isSmall
                    tileHeight={220}
                    title={l.title}
                    locationData={l.data}
                    selectedTarget={FLU_ED}
                    selectedModels={l.models}
                  />
                </Box>
              ))}
            </SimpleGrid>
          </Section>
        )}
      </Accordion>
    </Stack>
  );
};

const StateSummaryPage = () => {
  const { location, date } = useParams();
  const navigate = useNavigate();
  const { colorScheme } = useMantineColorScheme();
  const {
    chartScale,
    setChartScale,
    intervalVisibility,
    setIntervalVisibility,
    showLegend,
    setShowLegend,
  } = useView();
  // Past seasons are drawn behind this one by default on this page; the
  // setting stays local so the forecast pages keep their own
  const [showOtherSeasons, setShowOtherSeasons] = useState(true);
  const [tab, setTab] = useState("flu");
  const stateCode = location?.toUpperCase();
  const config = SUMMARY_STATES[stateCode];

  const { data, loading, error } = useAsyncData(
    config
      ? async () => {
          // A file that fails to load leaves its section empty, not the page
          const load = (path) =>
            fetchJson(getDataPath(path)).catch((e) => {
              console.error(e);
              return null;
            });
          const [flusight, metrocast, meta, covid, rsv, snapshot] =
            await Promise.all(
              [
                config.flusight,
                config.metrocast,
                "flumetrocast/metadata.json",
                config.covid,
                config.rsv,
                "summary/snapshot.json",
              ].map(load),
            );
          const [states, counties] = await Promise.all([
            fetchNsspStatesGeoJson(),
            fetchNsspCountiesGeoJson(stateCode),
          ]).catch(() => []);
          const subLocations = (meta?.locations || [])
            .filter((l) => l.location_name.endsWith(`, ${stateCode}`))
            .sort((a, b) => (b.population || 0) - (a.population || 0));
          const subData = await Promise.all(
            subLocations.map((l) =>
              load(`flumetrocast/${l.abbreviation}_flu_metrocast.json`),
            ),
          );
          return {
            flusight,
            metrocast,
            hubs: { covid, rsv },
            snapshot,
            geo: { states, counties },
            subLocations: subLocations
              .map((l, i) => ({ ...l, data: subData[i] }))
              .filter((l) => l.data),
          };
        }
      : null,
    [stateCode],
  );

  // Every date any file has a forecast for
  const availableDates = useMemo(() => {
    if (!data) return [];
    const files = [
      data.flusight,
      data.metrocast,
      ...Object.values(data.hubs),
      ...data.subLocations.map((l) => l.data),
    ];
    const dates = files.flatMap((d) => Object.keys(d?.forecasts || {}));
    return [...new Set(dates)].sort();
  }, [data]);

  const latestDate = availableDates.at(-1);
  const selectedDate = availableDates.includes(date) ? date : null;
  const index = availableDates.indexOf(selectedDate);
  const goTo = (d) => d && navigate(`/summary/${stateCode}/${d}`);

  // No date, or one with no forecast: go to the latest one
  useEffect(() => {
    if (latestDate && !selectedDate)
      navigate(`/summary/${stateCode}/${latestDate}`, { replace: true });
  }, [latestDate, selectedDate, stateCode, navigate]);

  const sections = useMemo(
    () =>
      data && selectedDate
        ? [
            fluSection(data, selectedDate, stateCode, intervalVisibility),
            ...OTHER_HUBS.map((hub) =>
              hubSection(hub, data.hubs[hub.key], selectedDate),
            ),
          ]
        : null,
    [data, selectedDate, stateCode, intervalVisibility],
  );

  if (!config) {
    return (
      <NotFoundPage
        title="No summary for this location"
        detail={`State summaries are only available for ${Object.keys(SUMMARY_STATES).join(", ")} for now.`}
      />
    );
  }

  // Settings every chart on the page shares (the toolbox's)
  const chartProps = {
    colorScheme,
    selectedDates: selectedDate ? [selectedDate] : [],
    chartScale,
    intervalVisibility,
    showLegend,
    showOtherGroundTruthSeasons: showOtherSeasons,
    // The 4-week and full-season forecasts are drawn together
    span: "season",
    modelColorFn: colorOf,
  };
  const title = `${config.name} State Summary`;

  return (
    <>
      <Seo
        title={`RespiLens | ${title}`}
        description={`Flu, COVID-19 and RSV forecast summary for ${config.name}: ED visits, hospital admissions, peak and sub-state forecasts.`}
        canonicalPath={`/summary/${stateCode}`}
      />
      <Container size="xl" pt="md" pb="xl">
        <Stack gap="xl">
          <Group justify="space-between" align="flex-end" wrap="wrap">
            <div>
              <Group gap="xs">
                <Title order={1}>{title}</Title>
                <Badge variant="filled" color="red">
                  Beta
                </Badge>
              </Group>
              <Text c="dimmed">
                Ensemble forecasts from FluSight, MetroCast and the COVID-19 and
                RSV Forecast Hubs
              </Text>
            </div>
            <Group gap={4} wrap="nowrap">
              <ActionIcon
                variant="default"
                size="lg"
                aria-label="Previous forecast date"
                disabled={index <= 0}
                onClick={() => goTo(availableDates[index - 1])}
              >
                <IconChevronLeft size={18} />
              </ActionIcon>
              <Select
                aria-label="Forecast date"
                w={150}
                searchable
                allowDeselect={false}
                value={selectedDate}
                onChange={goTo}
                data={[...availableDates].reverse()}
              />
              <ActionIcon
                variant="default"
                size="lg"
                aria-label="Next forecast date"
                disabled={index < 0 || index >= availableDates.length - 1}
                onClick={() => goTo(availableDates[index + 1])}
              >
                <IconChevronRight size={18} />
              </ActionIcon>
              <Popover position="bottom-end" shadow="md" width={360}>
                <Popover.Target>
                  <Tooltip label="Display options" openDelay={300}>
                    <ActionIcon
                      variant="subtle"
                      color="gray"
                      size="lg"
                      aria-label="Display options"
                    >
                      <IconAdjustmentsHorizontal size={18} />
                    </ActionIcon>
                  </Tooltip>
                </Popover.Target>
                <Popover.Dropdown>
                  <ForecastChartControls
                    chartScale={chartScale}
                    setChartScale={setChartScale}
                    intervalVisibility={intervalVisibility}
                    setIntervalVisibility={setIntervalVisibility}
                    showLegend={showLegend}
                    setShowLegend={setShowLegend}
                    showOtherGroundTruthSeasons={showOtherSeasons}
                    setShowOtherGroundTruthSeasons={setShowOtherSeasons}
                  />
                </Popover.Dropdown>
              </Popover>
            </Group>
          </Group>

          {error && <Text c="red">Could not load the data: {error}</Text>}
          {!sections && !error && (
            <Center h={300}>{loading && <Loader />}</Center>
          )}

          {sections && (
            // One tab per disease. Only the open tab is mounted, so its
            // charts size themselves to the visible panel.
            <Tabs value={tab} onChange={setTab} keepMounted={false}>
              <Tabs.List mb="md">
                {sections.map((section) => (
                  <Tabs.Tab
                    key={section.key}
                    value={section.key}
                    fz="lg"
                    fw={600}
                  >
                    {section.label}
                  </Tabs.Tab>
                ))}
              </Tabs.List>
              {sections.map((section) => (
                <Tabs.Panel key={section.key} value={section.key}>
                  <DiseasePanel
                    section={section}
                    chartProps={chartProps}
                    compare={
                      data.snapshot && {
                        day: data.snapshot.dates[selectedDate]?.[section.key],
                        snapshot: data.snapshot,
                        geo: data.geo,
                        stateCode,
                        stateName: config.name,
                        date: selectedDate,
                      }
                    }
                  />
                </Tabs.Panel>
              ))}
            </Tabs>
          )}
        </Stack>
      </Container>
    </>
  );
};

export default StateSummaryPage;
