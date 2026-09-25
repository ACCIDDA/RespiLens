import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { URLParameterManager } from "../utils/urlManager";
import { useForecastData } from "../hooks/useForecastData";
import { ViewContext } from "./ViewContextObject";
import { APP_CONFIG, DATASETS } from "../config";
import { getDataPath } from "../utils/paths";
import {
  buildForecastUrl,
  getForecastRouteError,
  isForecastPathname,
  parseForecastUrlState,
} from "../utils/forecastRoutes";

const DATE_URL_DELAY_MS = 250;

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

const METRO_STATE_ABBREVIATION_TO_LOCATION = Object.fromEntries(
  Object.entries(METRO_STATE_MAP).map(([stateName, abbreviation]) => [
    abbreviation,
    stateName.toLowerCase().replace(/\s+/g, "-"),
  ]),
);

const METRO_SUBAREA_TO_STATE_ABBREVIATION = {
  nyc: "NY",
};

const STATE_NAME_TO_ABBREVIATION = {
  "United States": "US",
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
  Missouri: "MO",
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
  "Puerto Rico": "PR",
  "Rhode Island": "RI",
  "South Carolina": "SC",
  "South Dakota": "SD",
  Tennessee: "TN",
  Texas: "TX",
  Utah: "UT",
  Vermont: "VT",
  Virginia: "VA",
  Washington: "WA",
  "West Virginia": "WV",
  Wisconsin: "WI",
  Wyoming: "WY",
};

const STATE_ABBREVIATION_TO_NAME = Object.fromEntries(
  Object.entries(STATE_NAME_TO_ABBREVIATION).map(([name, abbreviation]) => [
    abbreviation,
    name,
  ]),
);

const parseStateAbbreviationFromLocationName = (locationName = "") => {
  const stateSuffixMatch = String(locationName).match(/,\s*([A-Z]{2})$/);
  if (stateSuffixMatch) {
    return stateSuffixMatch[1];
  }

  return STATE_NAME_TO_ABBREVIATION[locationName] || null;
};

const normalizeLocationEntry = (entry) => {
  if (!entry) {
    return null;
  }

  if (typeof entry === "string") {
    return {
      abbreviation: entry,
      location_name: entry,
    };
  }

  if (Array.isArray(entry)) {
    const [locationName, abbreviation] = entry;
    return {
      abbreviation: abbreviation || locationName,
      location_name: locationName || abbreviation,
    };
  }

  if (typeof entry === "object") {
    return {
      ...entry,
      abbreviation: entry.abbreviation || entry.location || entry.location_name,
      location_name:
        entry.location_name || entry.abbreviation || entry.location,
    };
  }

  return null;
};

const buildStandardLocationCatalog = (metadata) => {
  const exactLocations = new Set();
  const parentStateByLocation = {};
  const stateLocationByAbbreviation = {};

  (metadata?.locations || [])
    .map(normalizeLocationEntry)
    .filter(Boolean)
    .forEach((entry) => {
      const locationId = entry.abbreviation;
      const locationName = entry.location_name;
      const parentStateAbbreviation =
        locationId === "US"
          ? "US"
          : parseStateAbbreviationFromLocationName(locationName);

      exactLocations.add(locationId);

      if (parentStateAbbreviation) {
        parentStateByLocation[locationId] = parentStateAbbreviation;
      }

      const isStateLevelLocation =
        locationId === "US" ||
        (!String(locationName).includes(",") &&
          Boolean(parentStateAbbreviation));

      if (isStateLevelLocation && parentStateAbbreviation) {
        stateLocationByAbbreviation[parentStateAbbreviation] = locationId;
      }
    });

  return {
    exactLocations,
    parentStateByLocation,
    stateLocationByAbbreviation,
  };
};

const buildMetroLocationCatalog = (metadata) => {
  const exactLocations = new Set();
  const parentStateByLocation = {};
  const stateLocationByAbbreviation = {};

  (metadata?.locations || [])
    .map(normalizeLocationEntry)
    .filter(Boolean)
    .forEach((entry) => {
      const locationId = entry.abbreviation;
      const locationName = entry.location_name;
      const parentStateAbbreviation =
        METRO_SUBAREA_TO_STATE_ABBREVIATION[locationId] ||
        METRO_STATE_MAP[locationName] ||
        parseStateAbbreviationFromLocationName(locationName);

      exactLocations.add(locationId);

      if (parentStateAbbreviation) {
        parentStateByLocation[locationId] = parentStateAbbreviation;
      }

      if (METRO_STATE_MAP[locationName] && parentStateAbbreviation) {
        stateLocationByAbbreviation[parentStateAbbreviation] = locationId;
      }
    });

  return {
    exactLocations,
    parentStateByLocation,
    stateLocationByAbbreviation,
  };
};

const buildNsspLocationCatalog = (metadata) => {
  const exactLocations = new Set();
  const parentStateByLocation = {};
  const stateLocationByAbbreviation = {};

  (metadata?.locations || []).forEach(([stateName, subLocation]) => {
    const stateAbbreviation = STATE_NAME_TO_ABBREVIATION[stateName];
    if (!stateAbbreviation) {
      return;
    }

    const locationId = `${stateAbbreviation}_${subLocation}`;
    exactLocations.add(locationId);
    parentStateByLocation[locationId] = stateAbbreviation;

    if (subLocation === "All") {
      stateLocationByAbbreviation[stateAbbreviation] = locationId;
    }
  });

  return {
    exactLocations,
    parentStateByLocation,
    stateLocationByAbbreviation,
  };
};

const getStateAbbreviationForLocation = (location, data, sourceCatalog) => {
  if (!location) {
    return null;
  }

  if (location === "US" || location === "US_All") {
    return "US";
  }

  if (location.includes("_")) {
    return location.split("_")[0] || null;
  }

  if (/^[A-Z]{2}$/.test(location)) {
    return location;
  }

  if (sourceCatalog?.parentStateByLocation?.[location]) {
    return sourceCatalog.parentStateByLocation[location];
  }

  if (METRO_STATE_ABBREVIATION_TO_LOCATION[location]) {
    return location;
  }

  return parseStateAbbreviationFromLocationName(data?.metadata?.location_name);
};

const getHeuristicStateLocationForView = (viewType, stateAbbreviation) => {
  if (!stateAbbreviation) {
    return null;
  }

  if (viewType === "metrocast_forecasts") {
    return METRO_STATE_ABBREVIATION_TO_LOCATION[stateAbbreviation] || null;
  }

  if (viewType === "nsspall") {
    return `${stateAbbreviation}_All`;
  }

  if (stateAbbreviation === "US") {
    return "US";
  }

  return stateAbbreviation;
};

const resolveLocationForView = ({
  nextView,
  currentView,
  currentLocation,
  currentData,
  destinationDefaultLocation,
  locationCatalogs,
}) => {
  const destinationCatalog = locationCatalogs[nextView];
  const sourceCatalog = locationCatalogs[currentView];

  const parentStateAbbreviation = getStateAbbreviationForLocation(
    currentLocation,
    currentData,
    sourceCatalog,
  );

  if (destinationCatalog?.exactLocations?.has(currentLocation)) {
    return { nextLocation: currentLocation, locationMessage: null };
  }

  const stateLevelLocation =
    (parentStateAbbreviation &&
      destinationCatalog?.stateLocationByAbbreviation?.[
        parentStateAbbreviation
      ]) ||
    getHeuristicStateLocationForView(nextView, parentStateAbbreviation);

  if (
    stateLevelLocation &&
    stateLevelLocation !== currentLocation &&
    (destinationCatalog
      ? destinationCatalog.exactLocations.has(stateLevelLocation)
      : true)
  ) {
    return { nextLocation: stateLevelLocation, locationMessage: null };
  }

  const locationMessage =
    nextView === "nsspall" &&
    parentStateAbbreviation &&
    parentStateAbbreviation !== "US"
      ? `There is no NSSP data for ${STATE_ABBREVIATION_TO_NAME[parentStateAbbreviation] || parentStateAbbreviation}.`
      : null;

  return {
    nextLocation: destinationDefaultLocation,
    locationMessage,
  };
};

export const ViewProvider = ({ children }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  // An address that is not a page (a mistyped hub, view or state) is left
  // exactly as typed: reconciling it would redirect to a working view and
  // swallow the error the reader needs to see.
  const isForecastPage =
    isForecastPathname(location.pathname) &&
    !getForecastRouteError(location.pathname);

  const urlManager = useMemo(
    () =>
      new URLParameterManager(
        searchParams,
        setSearchParams,
        location.pathname,
        navigate,
      ),
    [searchParams, setSearchParams, location.pathname, navigate],
  );

  const [viewType, setViewTypeState] = useState(() => urlManager.getView());
  const [selectedLocation, setSelectedLocation] = useState(() => {
    const { location: urlLoc, viewType: currentView } = parseForecastUrlState(
      location.pathname,
    );
    const dataset = urlManager.getDatasetFromView(currentView);
    if (dataset?.defaultLocation && urlLoc === APP_CONFIG.defaultLocation) {
      return dataset.defaultLocation;
    }

    return urlLoc;
  });
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedDates, setSelectedDates] = useState([]);
  const [activeDate, setActiveDate] = useState(null);
  // Holding an arrow key steps dates far faster than browsers allow history
  // writes (Safari throws past 100 per 10 s), so the URL trails the state
  const dateUrlTimerRef = useRef(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [locationMessage, setLocationMessage] = useState(null);
  // NSSP data comes per group of counties (HSA): remember which county the
  // user picked so the county picker and map can name it
  const [nsspCounty, setNsspCounty] = useState(null);
  const [locationCatalogs, setLocationCatalogs] = useState({});
  const [chartScale, setChartScale] = useState(
    () => urlManager.getAdvancedParams().chartScale,
  );
  const [intervalVisibility, setIntervalVisibility] = useState(
    () => urlManager.getAdvancedParams().intervalVisibility,
  );
  const [showLegend, setShowLegend] = useState(
    () => urlManager.getAdvancedParams().showLegend,
  );
  const [showOtherGroundTruthSeasons, setShowOtherGroundTruthSeasons] =
    useState(() => urlManager.getAdvancedParams().showOtherGroundTruthSeasons);
  const [showHistogram, setShowHistogram] = useState(
    () => urlManager.getAdvancedParams().showHistogram,
  );

  const {
    data,
    metadata,
    loading,
    error,
    availableDates,
    models,
    availableTargets,
    modelsByTarget,
    peaks,
    availablePeakDates,
    availablePeakModels,
  } = useForecastData(selectedLocation, viewType);

  useEffect(() => {
    let isActive = true;

    const loadLocationCatalogs = async () => {
      try {
        const datasetEntries = await Promise.all(
          Object.values(DATASETS).map(async (dataset) => {
            if (!dataset?.dataPath) {
              return null;
            }

            // One missing dataset shouldn't wipe every catalog (which would
            // snap all views back to their default location).
            try {
              const response = await fetch(
                getDataPath(`${dataset.dataPath}/metadata.json`),
              );

              if (!response.ok) {
                throw new Error(
                  `Failed to fetch location metadata for ${dataset.shortName}: ${response.status}`,
                );
              }

              const metadata = await response.json();
              const catalog =
                dataset.shortName === "nssp"
                  ? buildNsspLocationCatalog(metadata)
                  : dataset.shortName === "metrocast"
                    ? buildMetroLocationCatalog(metadata)
                    : buildStandardLocationCatalog(metadata);

              return [dataset.shortName, catalog];
            } catch (datasetError) {
              console.error(
                `Error loading location catalog for ${dataset.shortName}:`,
                datasetError,
              );
              return null;
            }
          }),
        );

        if (!isActive) {
          return;
        }

        const catalogsByView = {};
        datasetEntries.filter(Boolean).forEach(([shortName, catalog]) => {
          const dataset = DATASETS[shortName];
          dataset?.views?.forEach((view) => {
            catalogsByView[view.value] = catalog;
          });
        });

        setLocationCatalogs(catalogsByView);
      } catch (catalogError) {
        console.error("Error loading location catalogs:", catalogError);
      }
    };

    loadLocationCatalogs();

    return () => {
      isActive = false;
    };
  }, []);

  // filter flu_peak dates based on current season
  const availableDatesToExpose = useMemo(() => {
    if (viewType === "flu_peak") {
      return availablePeakDates || [];
    }
    return availableDates || [];
  }, [viewType, availablePeakDates, availableDates]);

  // A pending date write belongs to the view it was made in
  useEffect(() => () => clearTimeout(dateUrlTimerRef.current), [viewType]);

  const updateDatasetParams = useCallback(
    (params) => {
      const currentDataset = urlManager.getDatasetFromView(viewType);
      if (currentDataset)
        urlManager.updateDatasetParams(currentDataset, params);
    },
    [viewType, urlManager],
  );

  const modelsForView = useMemo(() => {
    if (viewType === "fludetailed") {
      const target1Models = new Set(modelsByTarget["wk inc flu hosp"] || []);
      const target2Models = new Set(
        modelsByTarget["wk flu hosp rate change"] || [],
      );
      return Array.from(new Set([...target1Models, ...target2Models])).sort();
    }

    if (viewType === "flu_peak") {
      return availablePeakModels || [];
    }

    if (selectedTarget && modelsByTarget[selectedTarget]) {
      return modelsByTarget[selectedTarget];
    }

    return [];
  }, [selectedTarget, modelsByTarget, viewType, availablePeakModels]);

  const availableTargetsToExpose = useMemo(() => {
    if (viewType === "flu_peak") {
      return [];
    }

    const peakTargets = ["peak inc flu hosp", "peak week inc flu hosp"];

    return availableTargets.filter((target) => !peakTargets.includes(target));
  }, [availableTargets, viewType]);

  useEffect(() => {
    if (!isForecastPage) {
      return;
    }
    const currentDataset = urlManager.getDatasetFromView(viewType);
    if (
      loading ||
      !currentDataset ||
      modelsForView.length === 0 ||
      availableDatesToExpose.length === 0 ||
      availableTargets.length === 0
    ) {
      return;
    }

    const params = urlManager.getDatasetParams(currentDataset);
    let needsModelUrlUpdate = false;
    let dateUrlUpdate = null;

    let modelsToSet = [];
    const validUrlModels =
      params.models?.filter((m) => modelsForView.includes(m)) || [];
    if (validUrlModels.length > 0) {
      modelsToSet = validUrlModels;
    } else if (
      currentDataset.defaultModel &&
      modelsForView.includes(currentDataset.defaultModel)
    ) {
      modelsToSet = [currentDataset.defaultModel];
      needsModelUrlUpdate = true;
    } else if (modelsForView.length > 0) {
      modelsToSet = [modelsForView[0]];
      needsModelUrlUpdate = true;
    }
    // Drop a URL that only names the default model. Not one where other
    // requested models just have no forecast here: they come back at the
    // next location that has them.
    if (
      params.models?.length === 1 &&
      params.models[0] === currentDataset.defaultModel
    ) {
      needsModelUrlUpdate = true;
    }

    let datesToSet = [];
    const validUrlDates =
      params.dates?.filter((date) => availableDatesToExpose.includes(date)) ||
      [];
    if (validUrlDates.length > 0) {
      datesToSet = validUrlDates;
    } else {
      const latestDate =
        availableDatesToExpose[availableDatesToExpose.length - 1];
      if (latestDate) {
        datesToSet = [latestDate];
      }
    }

    const latestDate =
      availableDatesToExpose[availableDatesToExpose.length - 1];
    const isDefaultDate =
      datesToSet.length === 1 && datesToSet[0] === latestDate;
    const requestedDates = params.dates || [];
    if (
      requestedDates.length > 0 &&
      (isDefaultDate || requestedDates.length !== validUrlDates.length)
    ) {
      dateUrlUpdate = isDefaultDate ? [] : datesToSet;
    }

    const urlTarget = params.target;
    let targetToSet = null;
    let needsTargetUrlUpdate = false;
    if (urlTarget && availableTargets.includes(urlTarget)) {
      targetToSet = urlTarget;
    }
    if (urlTarget && urlTarget === availableTargetsToExpose[0]) {
      needsTargetUrlUpdate = true;
    }

    setSelectedModels((current) =>
      JSON.stringify(current) !== JSON.stringify(modelsToSet)
        ? modelsToSet
        : current,
    );
    setSelectedDates((current) =>
      JSON.stringify(current) !== JSON.stringify(datesToSet)
        ? datesToSet
        : current,
    );
    setActiveDate((currentActive) => {
      if (currentActive && datesToSet.includes(currentActive)) {
        return currentActive;
      }
      return datesToSet.length > 0 ? datesToSet[datesToSet.length - 1] : null;
    });

    if (targetToSet && targetToSet !== selectedTarget) {
      setSelectedTarget(targetToSet);
    }

    if (needsModelUrlUpdate || dateUrlUpdate || needsTargetUrlUpdate) {
      updateDatasetParams({
        ...(needsModelUrlUpdate ? { models: [] } : {}),
        ...(dateUrlUpdate ? { dates: dateUrlUpdate } : {}),
        ...(needsTargetUrlUpdate ? { target: null } : {}),
      });
    }
  }, [
    isForecastPage,
    loading,
    viewType,
    models,
    availableTargets,
    urlManager,
    updateDatasetParams,
    selectedTarget,
    modelsForView,
    availableDatesToExpose,
    availableTargetsToExpose,
  ]);

  useEffect(() => {
    const availableModelsSet = new Set(modelsForView);
    const cleanedSelectedModels = selectedModels.filter((model) =>
      availableModelsSet.has(model),
    );

    if (cleanedSelectedModels.length !== selectedModels.length) {
      setSelectedModels(cleanedSelectedModels);
    }
  }, [modelsForView, selectedModels]);

  useEffect(() => {
    if (loading || !availableTargets || availableTargets.length === 0) {
      return;
    }
    const isCurrentTargetValid =
      selectedTarget && availableTargets.includes(selectedTarget);
    if (!isCurrentTargetValid) {
      setSelectedTarget(availableTargets[0]);
    }
  }, [loading, availableTargets, selectedTarget]);

  const handleLocationSelect = (newLocation, county = null) => {
    const currentDataset = urlManager.getDatasetFromView(viewType);
    setLocationMessage(null);
    setNsspCounty(county);
    const nextUrl = buildForecastUrl({
      viewType,
      location:
        newLocation ||
        currentDataset?.defaultLocation ||
        APP_CONFIG.defaultLocation,
      searchParams,
    });
    navigate(nextUrl, { replace: true });
    setSelectedLocation(newLocation);
  };

  const handleTargetSelect = (target) => {
    if (!target) return;
    setSelectedTarget(target);
    updateDatasetParams({
      target: target === availableTargetsToExpose[0] ? null : target,
    });
  };

  const handleViewLocationChange = useCallback(
    (newView, explicitLocation) => {
      const oldView = viewType;
      if (!newView) {
        return;
      }

      const oldDataset = urlManager.getDatasetFromView(oldView);
      const newDataset = urlManager.getDatasetFromView(newView);
      const newSearchParams = new URLSearchParams(searchParams);
      const effectiveDefault =
        newDataset?.defaultLocation || APP_CONFIG.defaultLocation;
      const { nextLocation, locationMessage: nextLocationMessage } =
        explicitLocation
          ? { nextLocation: explicitLocation, locationMessage: null }
          : resolveLocationForView({
              nextView: newView,
              currentView: oldView,
              currentLocation: selectedLocation,
              currentData: data,
              destinationDefaultLocation: effectiveDefault,
              locationCatalogs,
            });

      setLocationMessage(nextLocationMessage);
      setSelectedLocation(nextLocation);

      const isDatasetChange = oldDataset?.shortName !== newDataset?.shortName;
      const isPeakTransition = oldView === "flu_peak" || newView === "flu_peak";

      if (isDatasetChange || isPeakTransition) {
        setSelectedDates([]);
        setSelectedModels([]);
        setActiveDate(null);
        setSelectedTarget(null);

        if (oldDataset) {
          newSearchParams.delete(`${oldDataset.prefix}_models`);
          newSearchParams.delete("dates");
          newSearchParams.delete(`${oldDataset.prefix}_target`);
        }

        if (oldDataset?.shortName === "nhsn") {
          newSearchParams.delete("nhsn_target");
          newSearchParams.delete("nhsn_cols");
        }
        if (oldDataset?.shortName === "nssp") {
          newSearchParams.delete("nssp_cols");
        }
      } else {
        if (newDataset) {
          newSearchParams.delete(`${newDataset.prefix}_target`);
        }
        setSelectedTarget(null);
      }

      setViewTypeState(newView);
      const nextUrl = buildForecastUrl({
        viewType: newView,
        location: nextLocation || effectiveDefault,
        searchParams: newSearchParams,
      });
      navigate(nextUrl, { replace: false });
    },
    [
      viewType,
      urlManager,
      searchParams,
      selectedLocation,
      data,
      locationCatalogs,
      navigate,
    ],
  );

  const handleViewChange = useCallback(
    (newView) => {
      // Off the chart pages (Forecastle, My Plots...) the remembered view is
      // not on screen, so picking it again still has to navigate there
      if (isForecastPage && viewType === newView) return;
      handleViewLocationChange(newView);
    },
    [handleViewLocationChange, viewType, isForecastPage],
  );

  useEffect(() => {
    if (!isForecastPage) {
      return;
    }

    const currentDataset = urlManager.getDatasetFromView(viewType);
    if (!currentDataset) {
      return;
    }

    // Wait for this view's catalog: validating against a missing one sends
    // every deep link (e.g. /forecasts/flusight/MA) back to the default.
    if (!locationCatalogs[viewType]) {
      return;
    }

    const effectiveDefault =
      currentDataset.defaultLocation || APP_CONFIG.defaultLocation;
    const resolution = resolveLocationForView({
      nextView: viewType,
      currentView: viewType,
      currentLocation: selectedLocation,
      currentData: data,
      destinationDefaultLocation: effectiveDefault,
      locationCatalogs,
    });

    const shouldPreserveExistingNsspMessage =
      viewType === "nsspall" &&
      Boolean(locationMessage) &&
      resolution.nextLocation === selectedLocation &&
      resolution.locationMessage === null;

    if (
      !shouldPreserveExistingNsspMessage &&
      resolution.locationMessage !== locationMessage
    ) {
      setLocationMessage(resolution.locationMessage);
    }

    if (resolution.nextLocation === selectedLocation) {
      return;
    }

    setSelectedLocation(resolution.nextLocation);
    const nextUrl = buildForecastUrl({
      viewType,
      location: resolution.nextLocation,
      searchParams,
    });
    navigate(nextUrl, { replace: true });
  }, [
    isForecastPage,
    viewType,
    selectedLocation,
    data,
    locationCatalogs,
    locationMessage,
    navigate,
    searchParams,
    urlManager,
  ]);

  useEffect(() => {
    const viewFromUrl = urlManager.getView();
    if (viewFromUrl !== viewType) {
      setViewTypeState(viewFromUrl);
    }
  }, [searchParams, location.pathname, urlManager, viewType]);

  useEffect(() => {
    const locationFromUrl = urlManager.getLocation();
    if (locationFromUrl !== selectedLocation) {
      setSelectedLocation(locationFromUrl);
    }
  }, [searchParams, location.pathname, selectedLocation, urlManager]);

  useEffect(() => {
    if (!isForecastPage) {
      return;
    }

    const isSurveillanceView = viewType === "nhsnall" || viewType === "nsspall";
    const {
      chartScale: urlScale,
      intervalVisibility: urlIntervals,
      showLegend: urlLegend,
      showOtherGroundTruthSeasons: urlShowOtherGroundTruthSeasons,
      showHistogram: urlShowHistogram,
    } = urlManager.getAdvancedParams();

    if (isSurveillanceView && urlShowOtherGroundTruthSeasons) {
      if (showOtherGroundTruthSeasons) {
        setShowOtherGroundTruthSeasons(false);
      }
      urlManager.updateAdvancedParams({
        showOtherGroundTruthSeasons: false,
      });
      return;
    }

    if (urlScale !== chartScale) {
      setChartScale(urlScale);
    }
    if (JSON.stringify(urlIntervals) !== JSON.stringify(intervalVisibility)) {
      setIntervalVisibility(urlIntervals);
    }
    if (urlLegend !== showLegend) {
      setShowLegend(urlLegend);
    }
    if (urlShowOtherGroundTruthSeasons !== showOtherGroundTruthSeasons) {
      setShowOtherGroundTruthSeasons(urlShowOtherGroundTruthSeasons);
    }
    if (urlShowHistogram !== showHistogram) {
      setShowHistogram(urlShowHistogram);
    }
  }, [
    searchParams,
    location.pathname,
    urlManager,
    isForecastPage,
    viewType,
    chartScale,
    intervalVisibility,
    showLegend,
    showOtherGroundTruthSeasons,
    showHistogram,
  ]);

  const setChartScaleWithUrl = useCallback(
    (nextScale) => {
      setChartScale(nextScale);
      if (isForecastPage) {
        urlManager.updateAdvancedParams({ chartScale: nextScale });
      }
    },
    [urlManager, isForecastPage],
  );

  const setIntervalVisibilityWithUrl = useCallback(
    (updater) => {
      setIntervalVisibility((prev) => {
        const next = typeof updater === "function" ? updater(prev) : updater;
        if (isForecastPage) {
          urlManager.updateAdvancedParams({ intervalVisibility: next });
        }
        return next;
      });
    },
    [urlManager, isForecastPage],
  );

  const setShowLegendWithUrl = useCallback(
    (nextShowLegend) => {
      setShowLegend(nextShowLegend);
      if (isForecastPage) {
        urlManager.updateAdvancedParams({ showLegend: nextShowLegend });
      }
    },
    [urlManager, isForecastPage],
  );

  const setShowOtherGroundTruthSeasonsWithUrl = useCallback(
    (nextValue) => {
      setShowOtherGroundTruthSeasons(nextValue);
      if (isForecastPage) {
        urlManager.updateAdvancedParams({
          showOtherGroundTruthSeasons: nextValue,
        });
      }
    },
    [urlManager, isForecastPage],
  );

  const setShowHistogramWithUrl = useCallback(
    (nextValue) => {
      setShowHistogram(nextValue);
      if (isForecastPage) {
        urlManager.updateAdvancedParams({ showHistogram: nextValue });
      }
    },
    [urlManager, isForecastPage],
  );

  const contextValue = {
    selectedLocation,
    locationMessage,
    handleLocationSelect,
    nsspCounty,
    data,
    metadata,
    loading,
    error,
    availableDates: availableDatesToExpose,
    models: modelsForView,
    selectedModels,
    setSelectedModels: (updater) => {
      const resolveModels = (prevModels) =>
        typeof updater === "function" ? updater(prevModels) : updater;
      const currentDataset = urlManager.getDatasetFromView(viewType);
      setSelectedModels((prevModels) => {
        const nextModels = resolveModels(prevModels);
        const defaultModel = currentDataset?.defaultModel
          ? [currentDataset.defaultModel]
          : [];
        const isDefault =
          JSON.stringify(nextModels.slice().sort()) ===
          JSON.stringify(defaultModel.slice().sort());
        updateDatasetParams({ models: isDefault ? [] : nextModels });
        return nextModels;
      });
    },
    selectedDates,
    setSelectedDates: (updater) => {
      const nextDates =
        typeof updater === "function" ? updater(selectedDates) : updater;
      setSelectedDates(nextDates);
      clearTimeout(dateUrlTimerRef.current);
      dateUrlTimerRef.current = setTimeout(() => {
        const latestDate =
          availableDatesToExpose[availableDatesToExpose.length - 1];
        const isDefault = nextDates.length === 1 && nextDates[0] === latestDate;
        updateDatasetParams({ dates: isDefault ? [] : nextDates });
      }, DATE_URL_DELAY_MS);
    },
    activeDate,
    setActiveDate,
    viewType,
    setViewType: handleViewChange,
    setViewAndLocation: handleViewLocationChange,
    currentDataset: urlManager.getDatasetFromView(viewType),
    availableTargets: availableTargetsToExpose,

    selectedTarget,
    handleTargetSelect,
    peaks,
    availablePeakDates: availablePeakDates || [],
    availablePeakModels,
    chartScale,
    setChartScale: setChartScaleWithUrl,
    intervalVisibility,
    setIntervalVisibility: setIntervalVisibilityWithUrl,
    showLegend,
    setShowLegend: setShowLegendWithUrl,
    showOtherGroundTruthSeasons,
    setShowOtherGroundTruthSeasons: setShowOtherGroundTruthSeasonsWithUrl,
    showHistogram,
    setShowHistogram: setShowHistogramWithUrl,
  };

  return (
    <ViewContext.Provider value={contextValue}>{children}</ViewContext.Provider>
  );
};
