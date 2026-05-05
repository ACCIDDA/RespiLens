import { useState, useEffect } from "react";
import {
  Stack,
  ScrollArea,
  Button,
  TextInput,
  Text,
  Divider,
  Loader,
  Center,
  Alert,
  Accordion,
} from "@mantine/core";
import {
  IconSearch,
  IconAlertTriangle,
  IconAdjustmentsHorizontal,
} from "@tabler/icons-react";
import { useView } from "../hooks/useView";
import ViewSelector from "./ViewSelector";
import TargetSelector from "./TargetSelector";
import ForecastChartControls from "./controls/ForecastChartControls";
import { getDataPath } from "../utils/paths";

const METRO_STATE_MAP = {
  Colorado: "CO",
  Georgia: "GA",
  Indiana: "IN",
  Maine: "ME",
  Maryland: "MD",
  Massachusetts: "MA",
  Minnesota: "MN",
  "South Carolina": "SC",
  Texas: "TX",
  Utah: "UT",
  Virginia: "VA",
  "North Carolina": "NC",
  Oregon: "OR",
};

const STATE_ABBREVIATIONS = {
  Alabama: "AL",
  Alaska: "AK",
  Arizona: "AZ",
  Arkansas: "AR",
  California: "CA",
  Colorado: "CO",
  Connecticut: "CT",
  Delaware: "DE",
  "District of Columbia": "DC",
  Florida: "FL",
  Georgia: "GA",
  Hawaii: "HI",
  Idaho: "ID",
  Illinois: "IL",
  Indiana: "IN",
  Iowa: "IA",
  Kansas: "KS",
  Kentucky: "KY",
  Louisiana: "LA",
  Maine: "ME",
  Maryland: "MD",
  Massachusetts: "MA",
  Michigan: "MI",
  Minnesota: "MN",
  Mississippi: "MS",
  Montana: "MT",
  Nebraska: "NE",
  Nevada: "NV",
  "New Hampshire": "NH",
  "New Jersey": "NJ",
  "New Mexico": "NM",
  "New York": "NY",
  "North Carolina": "NC",
  "North Dakota": "ND",
  Ohio: "OH",
  Oklahoma: "OK",
  Oregon: "OR",
  Pennsylvania: "PA",
  "Rhode Island": "RI",
  "South Carolina": "SC",
  "South Dakota": "SD",
  Tennessee: "TN",
  Texas: "TX",
  "United States": "US",
  Utah: "UT",
  Vermont: "VT",
  Virginia: "VA",
  Washington: "WA",
  "West Virginia": "WV",
  Wisconsin: "WI",
  Wyoming: "WY",
};

const normalizeNsspLocations = (locations = []) =>
  locations
    .map(([stateName, subLocation]) => {
      const stateCode = STATE_ABBREVIATIONS[stateName];
      if (!stateCode || !subLocation) {
        return null;
      }

      return {
        abbreviation: `${stateCode}_${subLocation}`,
        location_name:
          subLocation === "All" ? stateName : `${stateName} (${subLocation})`,
      };
    })
    .filter(Boolean);

const StateSelector = () => {
  const {
    selectedLocation,
    handleLocationSelect,
    viewType,
    currentDataset,
    chartScale,
    setChartScale,
    intervalVisibility,
    setIntervalVisibility,
    showLegend,
    setShowLegend,
  } = useView();

  const [states, setStates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  const [highlightedIndex, setHighlightedIndex] = useState(-1);

  useEffect(() => {
    const controller = new AbortController(); // controller prevents issues if you click away while locs are loading

    setStates([]);
    setLoading(true);

    const fetchStates = async () => {
      // different fetching/ordering if it is metrocast vs. other views
      try {
        const isMetro = viewType === "metrocast_forecasts";
        const directory = isMetro
          ? "flumetrocast"
          : currentDataset?.dataPath || "flusight";

        const manifestResponse = await fetch(
          getDataPath(`${directory}/metadata.json`),
          { signal: controller.signal },
        );

        if (!manifestResponse.ok)
          throw new Error(`Failed: ${manifestResponse.statusText}`);

        const metadata = await manifestResponse.json();
        let finalOrderedList = [];

        if (isMetro) {
          const locations = metadata.locations;
          const statesOnly = locations.filter(
            (l) => !l.location_name.includes(","),
          );
          const citiesOnly = locations.filter((l) =>
            l.location_name.includes(","),
          );
          statesOnly.sort((a, b) =>
            a.location_name.localeCompare(b.location_name),
          );

          statesOnly.forEach((stateObj) => {
            finalOrderedList.push(stateObj);
            const code = METRO_STATE_MAP[stateObj.location_name];

            const children = citiesOnly
              .filter((city) => city.location_name.endsWith(`, ${code}`))
              .sort((a, b) => a.location_name.localeCompare(b.location_name));

            finalOrderedList.push(...children);
          });

          const handledIds = finalOrderedList.map((l) => l.abbreviation);
          const leftovers = locations.filter(
            (l) => !handledIds.includes(l.abbreviation),
          );
          finalOrderedList.push(...leftovers);
        } else if (viewType === "nsspall") {
          finalOrderedList = normalizeNsspLocations(metadata.locations).sort(
            (a, b) => {
              const isADefault = a.abbreviation === "US_All";
              const isBDefault = b.abbreviation === "US_All";
              if (isADefault) return -1;
              if (isBDefault) return 1;
              return (a.location_name || "").localeCompare(
                b.location_name || "",
              );
            },
          );
        } else {
          finalOrderedList = metadata.locations.sort((a, b) => {
            const isA_Default = a.abbreviation === "US";
            const isB_Default = b.abbreviation === "US";
            if (isA_Default) return -1;
            if (isB_Default) return 1;
            return (a.location_name || "").localeCompare(b.location_name || "");
          });
        }

        setStates(finalOrderedList);
      } catch (err) {
        if (err.name === "AbortError") return;
        setError(err.message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };

    fetchStates();

    return () => controller.abort();
  }, [viewType, currentDataset]);

  useEffect(() => {
    if (states.length > 0) {
      const index = states.findIndex(
        (state) => state.abbreviation === selectedLocation,
      );
      setHighlightedIndex(index >= 0 ? index : 0);
    }
  }, [states, selectedLocation]);

  const filteredStates = states.filter(
    (state) =>
      state.location_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      state.abbreviation.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  const handleSearchChange = (e) => {
    const newSearchTerm = e.currentTarget.value;
    setSearchTerm(newSearchTerm);

    if (newSearchTerm.length > 0 && filteredStates.length > 0) {
      setHighlightedIndex(0);
    } else if (newSearchTerm.length === 0) {
      const index = states.findIndex(
        (state) => state.abbreviation === selectedLocation,
      );
      setHighlightedIndex(index >= 0 ? index : 0);
    }
  };

  const handleKeyDown = (event) => {
    if (filteredStates.length === 0) return;

    let newIndex = highlightedIndex;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      newIndex = (highlightedIndex + 1) % filteredStates.length;
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      newIndex =
        (highlightedIndex - 1 + filteredStates.length) % filteredStates.length;
    } else if (event.key === "Enter") {
      event.preventDefault();
      const selectedState = filteredStates[highlightedIndex];

      if (selectedState) {
        handleLocationSelect(selectedState.abbreviation);
        setSearchTerm("");
        setHighlightedIndex(
          states.findIndex(
            (s) => s.abbreviation === selectedState.abbreviation,
          ),
        );
        event.currentTarget.blur();
      }
      return; // Exit early if Enter is pressed
    }

    setHighlightedIndex(newIndex);
  };

  if (loading) {
    return (
      <Center>
        <Loader />
      </Center>
    );
  }

  if (error) {
    return (
      <Alert color="red" title="Error" icon={<IconAlertTriangle />}>
        {error}
      </Alert>
    );
  }

  return (
    <Stack
      gap="md"
      style={{ height: "100%", display: "flex", flexDirection: "column" }}
    >
      <Stack gap="xs" style={{ flexShrink: 0 }}>
        <ViewSelector />
      </Stack>

      <Divider />

      <Stack>
        <TargetSelector />
      </Stack>

      {viewType !== "frontpage" && (
        <Accordion
          variant="separated"
          radius="md"
          styles={{
            control: { padding: "6px 8px" },
            label: { fontSize: "0.875rem", fontWeight: 500 },
            panel: { padding: "6px 8px 8px" },
          }}
        >
          <Accordion.Item value="advanced-controls">
            <Accordion.Control icon={<IconAdjustmentsHorizontal size={14} />}>
              Advanced controls
            </Accordion.Control>
            <Accordion.Panel>
              <ForecastChartControls
                chartScale={chartScale}
                setChartScale={setChartScale}
                intervalVisibility={intervalVisibility}
                setIntervalVisibility={setIntervalVisibility}
                showLegend={showLegend}
                setShowLegend={setShowLegend}
                showIntervals={viewType !== "nhsnall" && viewType !== "nsspall"}
              />
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      )}

      <Stack
        gap="xs"
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <TextInput
          label="Search locations"
          placeholder="Search locations..."
          value={searchTerm}
          onChange={handleSearchChange}
          onKeyDown={handleKeyDown}
          leftSection={<IconSearch size={16} />}
          autoFocus
          aria-label="Search locations"
        />
        <ScrollArea style={{ flex: 1, minHeight: 0 }} type="auto">
          <Stack gap="xs">
            {filteredStates.map((state, index) => {
              const isSelected = selectedLocation === state.abbreviation;
              const isKeyboardHighlighted =
                (searchTerm.length > 0 || index === highlightedIndex) &&
                index === highlightedIndex &&
                !isSelected;

              // Only apply nested styling in Metrocast view
              const isCity =
                viewType === "metrocast_forecasts" &&
                state.location_name.includes(",");

              let variant = "subtle";
              let color = "blue";

              if (isSelected) {
                variant = "filled";
                color = "blue";
              } else if (isKeyboardHighlighted) {
                variant = "light";
                color = "blue";
              }

              return (
                <Button
                  key={state.abbreviation}
                  variant={variant}
                  color={color}
                  onClick={() => {
                    handleLocationSelect(state.abbreviation);
                    setSearchTerm("");
                    setHighlightedIndex(
                      states.findIndex(
                        (s) => s.abbreviation === state.abbreviation,
                      ),
                    );
                  }}
                  justify="start"
                  size="sm"
                  fullWidth
                  onMouseEnter={() => {
                    if (searchTerm.length > 0) {
                      setHighlightedIndex(index);
                    }
                  }}
                  pl={isCity ? 28 : 10}
                  styles={{
                    label: {
                      fontWeight: isCity ? 400 : 700,
                      fontSize: isCity ? "13px" : "14px",
                    },
                  }}
                >
                  {state.location_name}
                </Button>
              );
            })}
            {filteredStates.length === 0 && (
              <Center p="md">
                <Text c="dimmed">No locations found.</Text>
              </Center>
            )}
          </Stack>
        </ScrollArea>
      </Stack>
    </Stack>
  );
};

export default StateSelector;
