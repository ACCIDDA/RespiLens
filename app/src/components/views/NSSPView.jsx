import { useEffect, useMemo, useState } from "react";
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
} from "@mantine/core";
import { IconAlertTriangle, IconArrowLeft, IconMap } from "@tabler/icons-react";
import { geoAlbersUsa, geoMercator, geoPath } from "d3-geo";
import { useView } from "../../hooks/useView";
import {
  NSSP_STATE_ABBREVIATION_TO_INFO,
  fetchNsspCountiesGeoJson,
  fetchNsspCountyAssignments,
  fetchNsspStatesGeoJson,
  fetchNsspTopLevelLocations,
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
};

const formatValue = (value) => {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
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
        return (
          <path
            key={getFeatureKey(feature)}
            d={pathData}
            fill={getFeatureFill(feature)}
            stroke={MAP_COLORS.outline}
            strokeWidth={0.8}
            style={{ cursor: "pointer", transition: "fill 150ms ease" }}
            onClick={() => onFeatureClick(feature)}
            onMouseEnter={(event) => {
              event.currentTarget.style.fill = MAP_COLORS.hover;
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
  const { handleLocationSelect } = useView();
  const [usMapData, setUsMapData] = useState(null);
  const [stateMapData, setStateMapData] = useState(null);
  const [countyAssignmentData, setCountyAssignmentData] = useState(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState(null);
  const [selectedCounty, setSelectedCounty] = useState(null);

  const stateAbbreviation = getNsspStateAbbreviationFromLocation(location);
  const stateInfo = NSSP_STATE_ABBREVIATION_TO_INFO[stateAbbreviation];
  const isUnitedStates = isNsspUnitedStatesLocation(location);
  const isStatewide = isNsspStatewideLocation(location);

  useEffect(() => {
    if (isUnitedStates || isStatewide) {
      setSelectedCounty(null);
    }
  }, [isStatewide, isUnitedStates, location]);

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
        const topLevelLocations = await fetchNsspTopLevelLocations();
        const allowedAbbreviations = topLevelLocations.map((entry) =>
          entry.abbreviation.replace("_All", ""),
        );
        const statesGeoJson =
          await fetchNsspStatesGeoJson(allowedAbbreviations);
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
      if (isUnitedStates || !stateAbbreviation || stateAbbreviation === "US") {
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
  }, [isUnitedStates, stateAbbreviation]);

  const summary = useMemo(
    () => summarizeData(location, data, metadata, selectedCounty),
    [location, data, metadata, selectedCounty],
  );

  const currentGroupLabel =
    selectedCounty?.groupLabel || data?.metadata?.location_name;
  const currentHsaId = selectedCounty?.hsaId || data?.metadata?.location;

  const handleUnitedStatesStateClick = (feature) => {
    const nextStateAbbreviation = feature?.properties?.STUSAB;
    if (!nextStateAbbreviation) {
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
          data summary.
        </Text>
      </Stack>

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
                : `${stateInfo?.name || stateAbbreviation} county map`}
            </Title>
          </Group>
          <Text size="sm" c="dimmed">
            {isUnitedStates
              ? "Click a state to open its county map."
              : "Click a county to load its NSSP summary. Shared HSA groupings will lead to the same summary."}
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
              getFeatureKey={(feature) => feature.properties?.GEOID}
              getFeatureLabel={(feature) => feature.properties?.NAME}
              getFeatureFill={(feature) =>
                feature.properties?.STUSAB === stateAbbreviation
                  ? MAP_COLORS.selected
                  : MAP_COLORS.base
              }
            />
          ) : (
            <GeoMap
              featureCollection={stateMapData}
              height={STATE_MAP_HEIGHT}
              projectionKind="state"
              onFeatureClick={handleCountyClick}
              getFeatureKey={(feature) => feature.properties?.GEOID}
              getFeatureLabel={(feature) => feature.properties?.NAME}
              getFeatureFill={getCountyFill}
            />
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

      {!isUnitedStates && !isStatewide && (
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
                    {metricName}
                  </Text>
                  <Text fw={700} size="lg">
                    {formatValue(latestValue)}
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
