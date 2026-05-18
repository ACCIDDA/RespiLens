import { useEffect, useMemo, useState } from "react";
import { Button, Card, Group, Loader, Stack, Text, Title } from "@mantine/core";
import { IconChevronRight } from "@tabler/icons-react";
import { useView } from "../hooks/useView";
import NSSPGeoMap from "./NSSPGeoMap";
import {
  fetchNsspStateCoverage,
  fetchNsspStatesGeoJson,
  getNsspStateAbbreviationFromLocation,
} from "../utils/nsspGeo";
import { NSSP_MAP_COLORS, NSSP_MAP_HEIGHTS } from "../utils/nsspMap";

const NSSPOverviewGraph = () => {
  const {
    selectedLocation,
    viewType: activeViewType,
    setViewAndLocation,
  } = useView();
  const [usMapData, setUsMapData] = useState(null);
  const [stateCoverage, setStateCoverage] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const selectedStateAbbreviation = useMemo(() => {
    if (!selectedLocation || selectedLocation === "US") {
      return null;
    }

    const stateAbbreviation =
      getNsspStateAbbreviationFromLocation(selectedLocation);
    return stateAbbreviation && stateAbbreviation !== "US"
      ? stateAbbreviation
      : null;
  }, [selectedLocation]);

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

  const hasMap = !loading && !error && usMapData?.features?.length;

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

        {hasMap && (
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

        <Group justify="space-between" align="center">
          <Button
            size="xs"
            variant={isActive ? "light" : "filled"}
            onClick={() => {
              setViewAndLocation("nsspall", "US_All");
            }}
            rightSection={<IconChevronRight size={14} />}
          >
            {isActive ? "Viewing" : "View NSSP data"}
          </Button>
          <Text size="xs" c="dimmed">
            U.S. entry map
          </Text>
        </Group>
      </Stack>
    </Card>
  );
};

export default NSSPOverviewGraph;
