import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Alert,
  Badge,
  Button,
  Center,
  Group,
  Loader,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
  useMantineColorScheme,
} from "@mantine/core";
import { IconAlertTriangle, IconArrowLeft, IconMap } from "@tabler/icons-react";
import { geoAlbersUsa, geoMercator, geoPath } from "d3-geo";
import Plot from "react-plotly.js";
import Plotly from "plotly.js/dist/plotly";
import NSSPColumnSelector from "../NSSPColumnSelector";
import { MODEL_COLORS } from "../../config/datasets";
import { useView } from "../../hooks/useView";
import { buildPlotDownloadName } from "../../utils/plotDownloadName";
import { buildSqrtTicks, getYRangeFromTraces } from "../../utils/scaleUtils";
import {
  NSSP_STATE_ABBREVIATION_TO_INFO,
  fetchNsspCountiesGeoJson,
  fetchNsspCountyAssignments,
  fetchNsspStateCoverage,
  fetchNsspStatesGeoJson,
  getCountyDisplayLabel,
  getCountySelectionForFeature,
  getNsspStateAbbreviationFromLocation,
  isNsspStatewideLocation,
  isNsspUnitedStatesLocation,
  normalizeCountyBasename,
} from "../../utils/nsspGeo";

const MAP_WIDTH = 960;
const US_MAP_HEIGHT = 620;
const STATE_MAP_HEIGHT = 720;

const MAP_COLORS = {
  base: "#d9e4f5",
  active: "#8bb6ff",
  selected: "#245bdb",
  outline: "#355070",
  hover: "#5f8fda",
  fallback: "#edf3fb",
  unavailable: "#d7d7db",
};

const NSSP_COLUMN_LABELS = {
  percent_visits_covid: "COVID-19",
  percent_visits_influenza: "Influenza",
  percent_visits_rsv: "RSV",
};

const NSSP_DEFAULT_COLUMNS = Object.keys(NSSP_COLUMN_LABELS);

const formatPercentValue = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }

  return `${Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  })}%`;
};

const summarizeData = (location, data, metadata, countySelection) => {
  const series = data?.series;
  const dates = series?.dates || [];
  const metricNames = Object.keys(series || {}).filter(
    (key) => key !== "dates",
  );
  const latestDate = dates.length > 0 ? dates[dates.length - 1] : null;

  return {
    locationName:
      countySelection?.countyName || data?.metadata?.location_name || location,
    datasetName: metadata?.dataset || data?.metadata?.dataset || "NSSP",
    lastUpdated: metadata?.last_updated || "Unknown",
    dateCount: dates.length,
    firstDate: dates[0] || "N/A",
    latestDate: latestDate || "N/A",
    metricCount: metricNames.length,
    latestMetrics: metricNames.map((metricName) => ({
      metricName,
      latestValue: series?.[metricName]?.[dates.length - 1] ?? null,
    })),
  };
};

const GeoMap = ({
  featureCollection,
  height,
  projectionKind,
  onFeatureClick,
  isFeatureClickable,
  getFeatureKey,
  getFeatureLabel,
  getFeatureFill,
}) => {
  const pathGenerator = useMemo(() => {
    if (!featureCollection?.features?.length) {
      return null;
    }

    const projection =
      projectionKind === "usa" ? geoAlbersUsa() : geoMercator();
    projection.fitSize([MAP_WIDTH, height], featureCollection);

    return geoPath(projection);
  }, [featureCollection, height, projectionKind]);

  if (!featureCollection?.features?.length || !pathGenerator) {
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${MAP_WIDTH} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="Interactive geographic map"
    >
      {featureCollection.features.map((feature) => {
        const pathData = pathGenerator(feature);
        if (!pathData) {
          return null;
        }

        const label = getFeatureLabel(feature);
        const isClickable = isFeatureClickable
          ? isFeatureClickable(feature)
          : true;
        return (
          <path
            key={getFeatureKey(feature)}
            d={pathData}
            fill={getFeatureFill(feature)}
            stroke={MAP_COLORS.outline}
            strokeWidth={0.8}
            style={{
              cursor: isClickable ? "pointer" : "not-allowed",
              transition: "fill 150ms ease",
            }}
            onClick={() => {
              if (isClickable) {
                onFeatureClick(feature);
              }
            }}
            onMouseEnter={(event) => {
              if (isClickable) {
                event.currentTarget.style.fill = MAP_COLORS.hover;
              }
            }}
            onMouseLeave={(event) => {
              event.currentTarget.style.fill = getFeatureFill(feature);
            }}
          >
            <title>{label}</title>
          </path>
        );
      })}
    </svg>
  );
};

const NSSPView = ({ location, data, metadata }) => {
  const { handleLocationSelect, locationMessage, chartScale, showLegend } =
    useView();
  const { colorScheme } = useMantineColorScheme();
  const [searchParams, setSearchParams] = useSearchParams();
  const [usMapData, setUsMapData] = useState(null);
  const [stateMapData, setStateMapData] = useState(null);
  const [countyAssignmentData, setCountyAssignmentData] = useState(null);
  const [stateCoverage, setStateCoverage] = useState({});
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState(null);
  const [selectedCounty, setSelectedCounty] = useState(null);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [dataRevision, setDataRevision] = useState(0);
  const [plotRevision, setPlotRevision] = useState(0);
  const [xAxisRange, setXAxisRange] = useState(null);
  const [yAxisRange, setYAxisRange] = useState(null);

  const hasInteractedRef = useRef(false);
  const isResettingRef = useRef(false);
  const plotRef = useRef(null);

  const stateAbbreviation = getNsspStateAbbreviationFromLocation(location);
  const stateInfo = NSSP_STATE_ABBREVIATION_TO_INFO[stateAbbreviation];
  const isUnitedStates = isNsspUnitedStatesLocation(location);
  const isStatewide = isNsspStatewideLocation(location);
  const currentStateCoverage = stateCoverage[stateAbbreviation] || {
    hasAnyData: false,
    hasCountyData: false,
  };
  const availableColumns = useMemo(
    () =>
      Object.keys(data?.series || {}).filter((key) => key !== "dates" && key),
    [data],
  );

  const getProcessedYValues = useCallback(
    (rawValues) => {
      if (!rawValues) return [];

      return rawValues.map((value) => {
        if (value === null || value === undefined) {
          return value;
        }

        return chartScale === "sqrt" ? Math.sqrt(Math.max(0, value)) : value;
      });
    },
    [chartScale],
  );

  const getFullXRange = useCallback(() => {
    if (!data?.series?.dates?.length) return [null, null];

    const firstDate = data.series.dates[0];
    const lastDate = new Date(data.series.dates[data.series.dates.length - 1]);
    const twoWeeksAfter = new Date(lastDate);
    twoWeeksAfter.setDate(twoWeeksAfter.getDate() + 14);

    return [firstDate, twoWeeksAfter.toISOString().split("T")[0]];
  }, [data]);

  const getDefaultXRange = useCallback(() => {
    if (!data?.series?.dates?.length) return [null, null];

    const lastDate = new Date(data.series.dates[data.series.dates.length - 1]);
    const twoWeeksAfter = new Date(lastDate);
    twoWeeksAfter.setDate(twoWeeksAfter.getDate() + 14);

    const sixMonthsAgo = new Date(lastDate);
    sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

    return [
      sixMonthsAgo.toISOString().split("T")[0],
      twoWeeksAfter.toISOString().split("T")[0],
    ];
  }, [data]);

  const defaultRange = useMemo(() => getDefaultXRange(), [getDefaultXRange]);
  const fullRange = useMemo(() => getFullXRange(), [getFullXRange]);

  const calculateYRange = useCallback((traces, range) => {
    if (!traces?.length || !range?.[0]) {
      return null;
    }

    let maxY = -Infinity;
    const [startX, endX] = range;
    const startDate = new Date(startX);
    const endDate = new Date(endX);

    traces.forEach((trace) => {
      if (!trace.x || !trace.y) return;

      for (let index = 0; index < trace.x.length; index += 1) {
        const pointDate = new Date(trace.x[index]);
        if (pointDate < startDate || pointDate > endDate) {
          continue;
        }

        const value = Number(trace.y[index]);
        if (!Number.isNaN(value)) {
          maxY = Math.max(maxY, value);
        }
      }
    });

    if (maxY === -Infinity) {
      return null;
    }

    const padding = maxY < 1 ? 0.1 : maxY * 0.15;
    return [0, maxY + padding];
  }, []);

  useEffect(() => {
    if (isUnitedStates || isStatewide) {
      setSelectedCounty(null);
    }
  }, [isStatewide, isUnitedStates, location]);

  useEffect(() => {
    let isActive = true;

    const loadCoverage = async () => {
      try {
        const coverage = await fetchNsspStateCoverage();
        if (isActive) {
          setStateCoverage(coverage);
        }
      } catch (error) {
        if (isActive) {
          setMapError(error.message);
        }
      }
    };

    loadCoverage();
    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    let isActive = true;

    const loadUnitedStatesMap = async () => {
      if (!isUnitedStates) {
        return;
      }

      try {
        setMapLoading(true);
        setMapError(null);
        setUsMapData(null);
        const statesGeoJson = await fetchNsspStatesGeoJson();
        if (isActive) {
          setUsMapData(statesGeoJson);
        }
      } catch (error) {
        if (isActive) {
          setMapError(error.message);
        }
      } finally {
        if (isActive) {
          setMapLoading(false);
        }
      }
    };

    loadUnitedStatesMap();
    return () => {
      isActive = false;
    };
  }, [isUnitedStates]);

  useEffect(() => {
    let isActive = true;

    const loadStateMap = async () => {
      if (
        isUnitedStates ||
        !stateAbbreviation ||
        stateAbbreviation === "US" ||
        !currentStateCoverage.hasCountyData
      ) {
        setStateMapData(null);
        setCountyAssignmentData(null);
        return;
      }

      try {
        setMapLoading(true);
        setMapError(null);
        setStateMapData(null);
        setCountyAssignmentData(null);
        const [countiesGeoJson, assignments] = await Promise.all([
          fetchNsspCountiesGeoJson(stateAbbreviation),
          fetchNsspCountyAssignments(stateAbbreviation),
        ]);

        if (isActive) {
          setStateMapData(countiesGeoJson);
          setCountyAssignmentData(assignments);
        }
      } catch (error) {
        if (isActive) {
          setMapError(error.message);
        }
      } finally {
        if (isActive) {
          setMapLoading(false);
        }
      }
    };

    loadStateMap();
    return () => {
      isActive = false;
    };
  }, [currentStateCoverage.hasCountyData, isUnitedStates, stateAbbreviation]);

  useEffect(() => {
    if (!availableColumns.length) {
      setSelectedColumns([]);
      return;
    }

    const urlColumns = searchParams.getAll("nssp_cols");
    const isExplicitlyEmpty = urlColumns.includes("none");
    const validUrlColumns = urlColumns.filter((column) =>
      availableColumns.includes(column),
    );

    let nextColumns;
    if (validUrlColumns.length > 0) {
      nextColumns = validUrlColumns;
    } else if (isExplicitlyEmpty) {
      nextColumns = [];
    } else {
      nextColumns = NSSP_DEFAULT_COLUMNS.filter((column) =>
        availableColumns.includes(column),
      );
    }

    setSelectedColumns((currentColumns) => {
      const sortedCurrent = [...currentColumns].sort();
      const sortedNext = [...nextColumns].sort();
      if (JSON.stringify(sortedCurrent) === JSON.stringify(sortedNext)) {
        return currentColumns;
      }

      return nextColumns;
    });
  }, [availableColumns, searchParams]);

  useEffect(() => {
    const nextParams = new URLSearchParams(window.location.search);
    nextParams.delete("nssp_cols");

    const defaultColumns = NSSP_DEFAULT_COLUMNS.filter((column) =>
      availableColumns.includes(column),
    );
    const isDefaultSelection =
      JSON.stringify([...selectedColumns].sort()) ===
      JSON.stringify([...defaultColumns].sort());

    if (!isDefaultSelection) {
      if (selectedColumns.length > 0) {
        selectedColumns.forEach((column) => {
          nextParams.append("nssp_cols", column);
        });
      } else if (hasInteractedRef.current) {
        nextParams.set("nssp_cols", "none");
      }
    }

    const currentParams = new URLSearchParams(window.location.search);
    if (nextParams.toString() !== currentParams.toString()) {
      setSearchParams(nextParams, { replace: true });
    }
  }, [availableColumns, selectedColumns, setSearchParams]);

  useEffect(() => {
    setXAxisRange(null);
  }, [location]);

  useEffect(() => {
    if (data) {
      setPlotRevision((current) => current + 1);
    }
  }, [data]);

  useEffect(() => {
    if (data) {
      setDataRevision((current) => current + 1);
    }
  }, [data, selectedColumns]);

  useEffect(() => {
    if (!data || selectedColumns.length === 0) {
      setYAxisRange(null);
      return;
    }

    const currentTraces = selectedColumns.map((column) => ({
      x: data.series.dates,
      y: getProcessedYValues(data.series[column]),
    }));
    const activeRange = xAxisRange || defaultRange;

    if (!activeRange?.[0]) {
      setYAxisRange(null);
      return;
    }

    setYAxisRange(calculateYRange(currentTraces, activeRange));
  }, [
    calculateYRange,
    data,
    defaultRange,
    getProcessedYValues,
    selectedColumns,
    xAxisRange,
  ]);

  const handleRelayout = useCallback(
    (figure) => {
      if (isResettingRef.current) {
        isResettingRef.current = false;
        return;
      }

      if (figure && figure["xaxis.range"]) {
        const nextXRange = figure["xaxis.range"];
        if (JSON.stringify(nextXRange) !== JSON.stringify(xAxisRange)) {
          setXAxisRange(nextXRange);
        }
      }
    },
    [xAxisRange],
  );

  const summary = useMemo(
    () => summarizeData(location, data, metadata, selectedCounty),
    [location, data, metadata, selectedCounty],
  );

  const currentGroupLabel =
    selectedCounty?.groupLabel || data?.metadata?.location_name;
  const currentHsaId = selectedCounty?.hsaId || data?.metadata?.location;
  const hasReachedCountyDetail =
    !isUnitedStates && !isStatewide && currentStateCoverage.hasCountyData;
  const detailHeading =
    selectedCounty?.countyName ||
    data?.metadata?.location_name ||
    stateInfo?.name ||
    "Selected county";

  const handleUnitedStatesStateClick = (feature) => {
    const nextStateAbbreviation = feature?.properties?.STUSAB;
    if (
      !nextStateAbbreviation ||
      !stateCoverage[nextStateAbbreviation]?.hasAnyData
    ) {
      return;
    }
    setSelectedCounty(null);
    handleLocationSelect(`${nextStateAbbreviation}_All`);
  };

  const handleCountyClick = (feature) => {
    if (!countyAssignmentData) {
      return;
    }

    const selection = getCountySelectionForFeature(
      feature,
      countyAssignmentData,
    );
    if (!selection.hasData || !selection.locationId) {
      return;
    }
    setSelectedCounty(selection);
    handleLocationSelect(selection.locationId);
  };

  const getCountyFill = (feature) => {
    if (!countyAssignmentData) {
      return MAP_COLORS.base;
    }

    const selection = getCountySelectionForFeature(
      feature,
      countyAssignmentData,
    );
    if (!selection.hasData) {
      return MAP_COLORS.unavailable;
    }

    const isSelectedByLocation = selection.locationId === location;
    const isExplicitCountySelection =
      selectedCounty &&
      normalizeCountyBasename(selection.countyName) ===
        normalizeCountyBasename(selectedCounty.countyName);

    if (isExplicitCountySelection) {
      return MAP_COLORS.selected;
    }

    if (isSelectedByLocation) {
      return MAP_COLORS.active;
    }

    return selection.isStatewideFallback
      ? MAP_COLORS.fallback
      : MAP_COLORS.base;
  };

  const getStateFill = (feature) => {
    const featureStateAbbreviation = feature.properties?.STUSAB;
    const coverage = stateCoverage[featureStateAbbreviation];

    if (!coverage?.hasAnyData) {
      return MAP_COLORS.unavailable;
    }

    return featureStateAbbreviation === stateAbbreviation
      ? MAP_COLORS.selected
      : MAP_COLORS.base;
  };

  const isStateClickable = (feature) =>
    Boolean(stateCoverage[feature.properties?.STUSAB]?.hasAnyData);

  const isCountyClickable = (feature) =>
    Boolean(
      getCountySelectionForFeature(feature, countyAssignmentData).hasData,
    );

  const rawTraces = useMemo(
    () =>
      selectedColumns.map((column) => ({
        x: data?.series?.dates || [],
        y: getProcessedYValues(data?.series?.[column]),
      })),
    [data, getProcessedYValues, selectedColumns],
  );

  const rawYRange = useMemo(() => getYRangeFromTraces(rawTraces), [rawTraces]);
  const sqrtTicks = useMemo(() => {
    if (chartScale !== "sqrt") {
      return null;
    }

    return buildSqrtTicks({
      rawRange: rawYRange,
      formatValue: (value) =>
        `${value.toLocaleString(undefined, {
          maximumFractionDigits: 2,
        })}%`,
    });
  }, [chartScale, rawYRange]);

  const plotTraces = useMemo(() => {
    if (!data?.series?.dates?.length) {
      return [];
    }

    if (selectedColumns.length === 0) {
      return [
        {
          x: [data.series.dates[0]],
          y: [null],
          type: "scatter",
          mode: "lines",
          showlegend: false,
        },
      ];
    }

    return selectedColumns.map((column) => {
      const columnIndex = availableColumns.indexOf(column);

      return {
        x: data.series.dates,
        y: getProcessedYValues(data.series[column]),
        name: NSSP_COLUMN_LABELS[column] || column,
        type: "scatter",
        mode: "lines+markers",
        line: {
          color: MODEL_COLORS[columnIndex % MODEL_COLORS.length],
          width: 2.5,
        },
        marker: { size: 6 },
        hovertemplate:
          "%{x}<br>%{fullData.name}: %{customdata:.2f}%<extra></extra>",
        customdata: data.series[column],
      };
    });
  }, [availableColumns, data, getProcessedYValues, selectedColumns]);

  const plotLayout = useMemo(
    () => ({
      autosize: true,
      template: colorScheme === "dark" ? "plotly_dark" : "plotly_white",
      paper_bgcolor: colorScheme === "dark" ? "#1a1b1e" : "#ffffff",
      plot_bgcolor: colorScheme === "dark" ? "#1a1b1e" : "#ffffff",
      font: {
        color: colorScheme === "dark" ? "#c1c2c5" : "#000000",
      },
      title: {
        text: `${detailHeading} NSSP Percent of Visits`,
        x: 0,
        xanchor: "left",
      },
      xaxis: {
        title: "Date",
        rangeslider: {
          visible: true,
          range: fullRange,
        },
        rangeselector: {
          buttons: [
            { count: 1, label: "1m", step: "month", stepmode: "backward" },
            { count: 6, label: "6m", step: "month", stepmode: "backward" },
            { count: 1, label: "1y", step: "year", stepmode: "backward" },
            { step: "all", label: "All" },
          ],
          activecolor: colorScheme === "dark" ? "#4c6ef5" : "#228be6",
          bgcolor: colorScheme === "dark" ? "#2c2e33" : "#f1f3f5",
        },
        range: xAxisRange || defaultRange,
      },
      yaxis: {
        title: "Percent of visits",
        range: chartScale === "log" ? undefined : yAxisRange,
        autorange:
          chartScale === "log"
            ? true
            : yAxisRange === null || selectedColumns.length === 0,
        type: chartScale === "log" ? "log" : "linear",
        tickmode: chartScale === "sqrt" && sqrtTicks ? "array" : undefined,
        tickvals:
          chartScale === "sqrt" && sqrtTicks ? sqrtTicks.tickvals : undefined,
        ticktext:
          chartScale === "sqrt" && sqrtTicks ? sqrtTicks.ticktext : undefined,
        tickformat: chartScale === "sqrt" ? undefined : ".2f",
        ticksuffix: chartScale === "sqrt" ? undefined : "%",
      },
      showlegend: showLegend ?? true,
      legend: {
        x: 0,
        y: 1,
        xanchor: "left",
        yanchor: "top",
        bgcolor:
          colorScheme === "dark"
            ? "rgba(26, 27, 30, 0.8)"
            : "rgba(255, 255, 255, 0.8)",
        bordercolor: colorScheme === "dark" ? "#444" : "#ccc",
        borderwidth: 1,
        font: { size: 10 },
      },
      margin: { t: 56, r: 10, l: 72, b: 120 },
      uirevision: plotRevision,
      annotations:
        selectedColumns.length === 0
          ? [
              {
                text: "No series selected",
                xref: "paper",
                yref: "paper",
                showarrow: false,
                font: {
                  size: 20,
                  color: colorScheme === "dark" ? "#5c5f66" : "#adb5bd",
                },
              },
            ]
          : [],
    }),
    [
      chartScale,
      colorScheme,
      defaultRange,
      detailHeading,
      fullRange,
      plotRevision,
      selectedColumns.length,
      showLegend,
      sqrtTicks,
      xAxisRange,
      yAxisRange,
    ],
  );

  const plotConfig = useMemo(
    () => ({
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      showSendToCloud: false,
      plotlyServerURL: "",
      toImageButtonOptions: {
        format: "png",
        filename: buildPlotDownloadName("nssp-plot"),
      },
      modeBarButtonsToRemove: ["resetScale2d", "select2d", "lasso2d"],
      modeBarButtonsToAdd: [
        {
          name: "Reset view",
          icon: Plotly.Icons.home,
          click: function (graphDiv) {
            if (!data) return;

            const nextDefaultRange = getDefaultXRange();
            if (!nextDefaultRange?.[0]) return;

            const currentTraces = selectedColumns.map((column) => ({
              x: data.series.dates,
              y: getProcessedYValues(data.series[column]),
            }));
            const nextYRange = calculateYRange(currentTraces, nextDefaultRange);

            isResettingRef.current = true;
            setXAxisRange(null);
            setYAxisRange(nextYRange);

            Plotly.relayout(graphDiv, {
              "xaxis.range": nextDefaultRange,
              "yaxis.range": nextYRange,
              "yaxis.autorange": nextYRange === null,
            });
          },
        },
      ],
    }),
    [
      calculateYRange,
      data,
      getDefaultXRange,
      getProcessedYValues,
      selectedColumns,
    ],
  );

  const handleSetSelectedColumns = useCallback((nextColumns) => {
    hasInteractedRef.current = true;
    setSelectedColumns(nextColumns);
  }, []);

  if (!data?.series?.dates) {
    return (
      <Alert
        icon={<IconAlertTriangle size={16} />}
        color="yellow"
        variant="light"
      >
        NSSP data loaded, but the expected time series structure was not found.
      </Alert>
    );
  }

  return (
    <Stack gap="lg">
      <Stack gap={4}>
        <Title order={2}>NSSP Surveillance Data</Title>
        <Text c="dimmed">
          Explore the nationwide view, then drill into a state and click a
          county. Counties that share the same HSA grouping will open the same
          plotted view.
        </Text>
      </Stack>

      {locationMessage ? (
        <Alert
          icon={<IconAlertTriangle size={16} />}
          color="yellow"
          variant="light"
        >
          {locationMessage}
        </Alert>
      ) : null}

      <Group gap="sm">
        {!isUnitedStates && (
          <Button
            variant="light"
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => {
              setSelectedCounty(null);
              handleLocationSelect("US_All");
            }}
          >
            Back to United States
          </Button>
        )}
        {!isUnitedStates && !isStatewide && (
          <Button
            variant="subtle"
            onClick={() => {
              setSelectedCounty(null);
              handleLocationSelect(`${stateAbbreviation}_All`);
            }}
          >
            Back to {stateInfo?.name} counties
          </Button>
        )}
      </Group>

      <Paper withBorder radius="md" p="lg">
        <Stack gap="md">
          <Group gap="sm" align="center">
            <IconMap size={18} />
            <Title order={4}>
              {isUnitedStates
                ? "United States map"
                : hasReachedCountyDetail
                  ? `${getCountyDisplayLabel(detailHeading)} detail`
                  : `${stateInfo?.name || stateAbbreviation} county map`}
            </Title>
          </Group>
          <Text size="sm" c="dimmed">
            {isUnitedStates
              ? "Click a state to open county data when available. States without NSSP data are grayed out."
              : hasReachedCountyDetail
                ? "County-level NSSP data selected. The chart below reflects the county's HSA grouping."
                : "Click a county to load its NSSP time series. Shared HSA groupings will lead to the same plot."}
          </Text>

          {mapLoading ? (
            <Center py="xl">
              <Loader />
            </Center>
          ) : mapError ? (
            <Alert
              color="red"
              variant="light"
              icon={<IconAlertTriangle size={16} />}
            >
              {mapError}
            </Alert>
          ) : isUnitedStates ? (
            <GeoMap
              featureCollection={usMapData}
              height={US_MAP_HEIGHT}
              projectionKind="usa"
              onFeatureClick={handleUnitedStatesStateClick}
              isFeatureClickable={isStateClickable}
              getFeatureKey={(feature) => feature.properties?.GEOID}
              getFeatureLabel={(feature) => feature.properties?.NAME}
              getFeatureFill={getStateFill}
            />
          ) : hasReachedCountyDetail ? (
            <Stack gap="md">
              <Paper
                radius="md"
                p="md"
                style={{
                  background:
                    "linear-gradient(180deg, rgba(36,91,219,0.05), rgba(36,91,219,0.01))",
                  border: "1px solid var(--mantine-color-blue-2)",
                }}
              >
                <div
                  style={{
                    width: "100%",
                    height: "min(720px, 65vh)",
                    minHeight: 380,
                  }}
                >
                  <Plot
                    ref={plotRef}
                    useResizeHandler
                    data={plotTraces}
                    layout={plotLayout}
                    config={plotConfig}
                    style={{ width: "100%", height: "100%" }}
                    revision={dataRevision}
                    onRelayout={handleRelayout}
                  />
                </div>
              </Paper>

              <Paper withBorder radius="md" p="lg">
                <NSSPColumnSelector
                  availableColumns={availableColumns}
                  selectedColumns={selectedColumns}
                  setSelectedColumns={handleSetSelectedColumns}
                  columnLabelMap={NSSP_COLUMN_LABELS}
                />
              </Paper>
            </Stack>
          ) : currentStateCoverage.hasCountyData ? (
            <GeoMap
              featureCollection={stateMapData}
              height={STATE_MAP_HEIGHT}
              projectionKind="state"
              onFeatureClick={handleCountyClick}
              isFeatureClickable={isCountyClickable}
              getFeatureKey={(feature) => feature.properties?.GEOID}
              getFeatureLabel={(feature) => feature.properties?.NAME}
              getFeatureFill={getCountyFill}
            />
          ) : (
            <Alert color="blue" variant="light">
              County-level NSSP data is not available for {stateInfo?.name}.
              Showing the statewide summary below.
            </Alert>
          )}
        </Stack>
      </Paper>

      {!isUnitedStates && (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
          <Paper withBorder radius="md" p="md">
            <Text size="sm" c="dimmed">
              Dataset
            </Text>
            <Text fw={600}>{summary.datasetName}</Text>
          </Paper>
          <Paper withBorder radius="md" p="md">
            <Text size="sm" c="dimmed">
              State
            </Text>
            <Text fw={600}>{stateInfo?.name || stateAbbreviation}</Text>
          </Paper>
          <Paper withBorder radius="md" p="md">
            <Text size="sm" c="dimmed">
              Date range
            </Text>
            <Text fw={600}>
              {summary.firstDate} to {summary.latestDate}
            </Text>
          </Paper>
          <Paper withBorder radius="md" p="md">
            <Text size="sm" c="dimmed">
              Observations
            </Text>
            <Text fw={600}>{summary.dateCount}</Text>
          </Paper>
        </SimpleGrid>
      )}

      {!isUnitedStates &&
        (!isStatewide || !currentStateCoverage.hasCountyData) && (
          <Paper withBorder radius="md" p="lg">
            <Stack gap="md">
              <Group justify="space-between" align="flex-start">
                <div>
                  <Title order={4}>
                    {selectedCounty
                      ? getCountyDisplayLabel(selectedCounty.countyName)
                      : "Selected HSA summary"}
                  </Title>
                  <Text size="sm" c="dimmed">
                    Last updated: {summary.lastUpdated}
                  </Text>
                </div>
                <Badge variant="light">
                  {currentHsaId === "All"
                    ? "Statewide data"
                    : `HSA ${currentHsaId}`}
                </Badge>
              </Group>

              <Alert color="blue" variant="light">
                <Text size="sm">Data source grouping: {currentGroupLabel}</Text>
              </Alert>

              <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
                {summary.latestMetrics.map(({ metricName, latestValue }) => (
                  <Paper key={metricName} withBorder radius="md" p="md">
                    <Text size="sm" c="dimmed">
                      {NSSP_COLUMN_LABELS[metricName] || metricName}
                    </Text>
                    <Text fw={700} size="lg">
                      {formatPercentValue(latestValue)}
                    </Text>
                    <Text size="xs" c="dimmed">
                      Latest date: {summary.latestDate}
                    </Text>
                  </Paper>
                ))}
              </SimpleGrid>
            </Stack>
          </Paper>
        )}
    </Stack>
  );
};

export default NSSPView;
