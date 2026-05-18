import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Group,
  Loader,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { IconAlertTriangle, IconChevronRight } from "@tabler/icons-react";
import { useView } from "../hooks/useView";
import NSSPGeoMap from "./NSSPGeoMap";
import {
  NSSP_STATE_ABBREVIATION_TO_INFO,
  fetchNsspCountiesGeoJson,
  fetchNsspCountyAssignments,
  getCountySelectionForFeature,
  fetchNsspStateCoverage,
  fetchNsspStatesGeoJson,
  getNsspStateAbbreviationFromLocation,
  isNsspStatewideLocation,
  isNsspUnitedStatesLocation,
} from "../utils/nsspGeo";
import { NSSP_MAP_COLORS, NSSP_MAP_HEIGHTS } from "../utils/nsspMap";

const NSSPOverviewGraph = () => {
  const {
    selectedLocation,
    viewType: activeViewType,
    setViewAndLocation,
  } = useView();
  const [usMapData, setUsMapData] = useState(null);
  const [stateMapData, setStateMapData] = useState(null);
  const [countyAssignmentData, setCountyAssignmentData] = useState(null);
  const [stateCoverage, setStateCoverage] = useState({});
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);

  const resolvedNsspLocation = useMemo(() => {
    if (
      !selectedLocation ||
      selectedLocation === "US" ||
      selectedLocation === "US_All"
    ) {
      return "US_All";
    }

    const stateAbbreviation =
      getNsspStateAbbreviationFromLocation(selectedLocation);
    if (!NSSP_STATE_ABBREVIATION_TO_INFO[stateAbbreviation]) {
      return "US_All";
    }

    return selectedLocation.includes("_")
      ? selectedLocation
      : `${stateAbbreviation}_All`;
  }, [selectedLocation]);

  const selectedStateAbbreviation = useMemo(
    () => getNsspStateAbbreviationFromLocation(resolvedNsspLocation),
    [resolvedNsspLocation],
  );
  const stateInfo = NSSP_STATE_ABBREVIATION_TO_INFO[selectedStateAbbreviation];
  const isUnitedStates = isNsspUnitedStatesLocation(resolvedNsspLocation);
  const isStatewide = isNsspStatewideLocation(resolvedNsspLocation);
  const currentStateCoverage = stateCoverage[selectedStateAbbreviation] || {
    hasAnyData: false,
    hasCountyData: false,
  };
  const isActive = activeViewType === "nsspall";

  useEffect(() => {
    let isActiveRequest = true;

    const loadMap = async () => {
      try {
        setLoading(true);
        setError(null);

        const [coverage, statesGeoJson] = await Promise.all([
          fetchNsspStateCoverage(),
          fetchNsspStatesGeoJson(),
        ]);

        if (!isActiveRequest) {
          return;
        }

        setStateCoverage(coverage);
        setUsMapData(statesGeoJson);
      } catch (err) {
        console.error("Failed to load NSSP front page map", err);
        if (isActiveRequest) {
          setError(err.message);
          setUsMapData(null);
        }
      } finally {
        if (isActiveRequest) {
          setLoading(false);
        }
      }
    };

    loadMap();
    return () => {
      isActiveRequest = false;
    };
  }, []);

  useEffect(() => {
    let isActiveRequest = true;

    const loadStateMap = async () => {
      if (
        loading ||
        isUnitedStates ||
        !selectedStateAbbreviation ||
        selectedStateAbbreviation === "US" ||
        !currentStateCoverage.hasAnyData ||
        !currentStateCoverage.hasCountyData
      ) {
        setStateMapData(null);
        setCountyAssignmentData(null);
        setDetailLoading(false);
        return;
      }

      try {
        setDetailLoading(true);
        const [countiesGeoJson, assignments] = await Promise.all([
          fetchNsspCountiesGeoJson(selectedStateAbbreviation),
          fetchNsspCountyAssignments(selectedStateAbbreviation),
        ]);

        if (!isActiveRequest) {
          return;
        }

        setStateMapData(countiesGeoJson);
        setCountyAssignmentData(assignments);
      } catch (err) {
        console.error("Failed to load NSSP state map", err);
        if (isActiveRequest) {
          setStateMapData(null);
          setCountyAssignmentData(null);
        }
      } finally {
        if (isActiveRequest) {
          setDetailLoading(false);
        }
      }
    };

    loadStateMap();
    return () => {
      isActiveRequest = false;
    };
  }, [
    currentStateCoverage.hasAnyData,
    currentStateCoverage.hasCountyData,
    isUnitedStates,
    loading,
    selectedStateAbbreviation,
  ]);

  const isStateClickable = (feature) =>
    Boolean(stateCoverage[feature.properties?.STUSAB]?.hasAnyData);

  const getStateFill = (feature) => {
    const featureStateAbbreviation = feature.properties?.STUSAB;
    const coverage = stateCoverage[featureStateAbbreviation];

    if (!coverage?.hasAnyData) {
      return NSSP_MAP_COLORS.unavailable;
    }

    return featureStateAbbreviation === selectedStateAbbreviation
      ? NSSP_MAP_COLORS.selected
      : NSSP_MAP_COLORS.base;
  };

  const handleStateClick = (feature) => {
    const nextStateAbbreviation = feature?.properties?.STUSAB;
    if (!stateCoverage[nextStateAbbreviation]?.hasAnyData) {
      return;
    }

    setViewAndLocation("nsspall", `${nextStateAbbreviation}_All`);
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

    setViewAndLocation("nsspall", selection.locationId);
  };

  const isCountyClickable = (feature) =>
    Boolean(
      getCountySelectionForFeature(feature, countyAssignmentData).hasData,
    );

  const getCountyFill = (feature) => {
    if (!countyAssignmentData) {
      return NSSP_MAP_COLORS.base;
    }

    const selection = getCountySelectionForFeature(
      feature,
      countyAssignmentData,
    );

    if (!selection.hasData) {
      return NSSP_MAP_COLORS.unavailable;
    }

    const isSelectedByLocation = selection.locationId === resolvedNsspLocation;
    if (isSelectedByLocation) {
      return isStatewide ? NSSP_MAP_COLORS.active : NSSP_MAP_COLORS.selected;
    }

    return selection.isStatewideFallback
      ? NSSP_MAP_COLORS.fallback
      : NSSP_MAP_COLORS.base;
  };

  const hasUsMap = !loading && !error && usMapData?.features?.length;
  const hasStateMap = !detailLoading && stateMapData?.features?.length;
  const nsspViewTarget = currentStateCoverage.hasAnyData
    ? resolvedNsspLocation
    : "US_All";
  const locationLabel = isUnitedStates
    ? "U.S. entry map"
    : currentStateCoverage.hasAnyData
      ? `${stateInfo?.name || selectedStateAbbreviation} county map`
      : `No NSSP data for ${stateInfo?.name || selectedStateAbbreviation}`;

  return (
    <Card withBorder radius="md" padding="lg" shadow="xs">
      <Stack gap="sm">
        <Group justify="space-between" align="center">
          <Title order={5}>NSSP data</Title>
        </Group>

        {loading && (
          <Stack align="center" gap="xs" py="lg">
            <Loader size="sm" />
            <Text size="sm" c="dimmed">
              Loading NSSP map...
            </Text>
          </Stack>
        )}

        {!loading && error && (
          <Text size="sm" c="red">
            No NSSP map available
          </Text>
        )}

        {hasUsMap && isUnitedStates && (
          <Stack gap="xs">
            <div style={{ width: "100%", minHeight: 200 }}>
              <NSSPGeoMap
                featureCollection={usMapData}
                height={NSSP_MAP_HEIGHTS.usa}
                projectionKind="usa"
                onFeatureClick={handleStateClick}
                isFeatureClickable={isStateClickable}
                getFeatureKey={(feature) => feature.properties?.STUSAB}
                getFeatureLabel={(feature) => {
                  const stateAbbreviation = feature.properties?.STUSAB;
                  const stateName =
                    feature.properties?.NAME || stateAbbreviation || "State";
                  return stateCoverage[stateAbbreviation]?.hasAnyData
                    ? stateName
                    : `${stateName}: no NSSP data available`;
                }}
                getFeatureFill={getStateFill}
              />
            </div>
          </Stack>
        )}

        {!loading &&
          !error &&
          !isUnitedStates &&
          !currentStateCoverage.hasAnyData && (
            <Alert
              icon={<IconAlertTriangle size={16} />}
              color="yellow"
              variant="light"
            >
              No NSSP data is available for{" "}
              {stateInfo?.name || selectedStateAbbreviation}.
            </Alert>
          )}

        {!loading &&
          !error &&
          !isUnitedStates &&
          currentStateCoverage.hasAnyData &&
          !currentStateCoverage.hasCountyData && (
            <Alert
              icon={<IconAlertTriangle size={16} />}
              color="yellow"
              variant="light"
            >
              County-level NSSP data is not available for {stateInfo?.name}.
            </Alert>
          )}

        {!loading &&
          !error &&
          !isUnitedStates &&
          currentStateCoverage.hasAnyData &&
          currentStateCoverage.hasCountyData && (
            <Stack gap="xs">
              {detailLoading ? (
                <Stack align="center" gap="xs" py="lg">
                  <Loader size="sm" />
                  <Text size="sm" c="dimmed">
                    Loading {stateInfo?.name} NSSP map...
                  </Text>
                </Stack>
              ) : hasStateMap ? (
                <div style={{ width: "100%", minHeight: 200 }}>
                  <NSSPGeoMap
                    featureCollection={stateMapData}
                    height={NSSP_MAP_HEIGHTS.state}
                    projectionKind="state"
                    onFeatureClick={handleCountyClick}
                    isFeatureClickable={isCountyClickable}
                    getFeatureKey={(feature) => feature.properties?.GEOID}
                    getFeatureLabel={(feature) => {
                      const selection = getCountySelectionForFeature(
                        feature,
                        countyAssignmentData,
                      );
                      const countyName = feature.properties?.NAME || "County";

                      if (!selection.hasData) {
                        return `${countyName}: no NSSP data available`;
                      }

                      return selection.isStatewideFallback
                        ? `${countyName}: uses statewide NSSP data`
                        : countyName;
                    }}
                    getFeatureFill={getCountyFill}
                  />
                </div>
              ) : (
                <Alert
                  icon={<IconAlertTriangle size={16} />}
                  color="yellow"
                  variant="light"
                >
                  County-level NSSP data is not available for {stateInfo?.name}.
                </Alert>
              )}
            </Stack>
          )}

        <Group justify="space-between" align="center">
          <Button
            size="xs"
            variant={isActive ? "light" : "filled"}
            onClick={() => {
              setViewAndLocation("nsspall", nsspViewTarget);
            }}
            rightSection={<IconChevronRight size={14} />}
          >
            {isActive ? "Viewing" : "View NSSP data"}
          </Button>
          <Text size="xs" c="dimmed">
            {locationLabel}
          </Text>
        </Group>
      </Stack>
    </Card>
  );
};

export default NSSPOverviewGraph;
