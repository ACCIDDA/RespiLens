import { useState, useCallback, useEffect, useMemo } from "react";
import { useLocation, useSearchParams } from "react-router-dom";
import { URLParameterManager } from "../utils/urlManager";
import { useForecastData } from "../hooks/useForecastData";
import { ViewContext } from "./ViewContextObject";
import { APP_CONFIG } from "../config";

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

const STATE_NAME_TO_ABBREVIATION = {
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

const NSSP_UNAVAILABLE_STATE_ABBREVIATIONS = new Set(["MO", "PR"]);

const parseStateAbbreviationFromLocationName = (locationName = "") => {
  const stateSuffixMatch = String(locationName).match(/,\s*([A-Z]{2})$/);
  if (stateSuffixMatch) {
    return stateSuffixMatch[1];
  }

  return STATE_NAME_TO_ABBREVIATION[locationName] || null;
};

const getStateAbbreviationForLocation = (location, data) => {
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

  if (METRO_STATE_ABBREVIATION_TO_LOCATION[location]) {
    return location;
  }

  return parseStateAbbreviationFromLocationName(data?.metadata?.location_name);
};

const getStandardLocationForView = (location, data) => {
  const stateAbbreviation = getStateAbbreviationForLocation(location, data);
  return stateAbbreviation && stateAbbreviation !== "US"
    ? stateAbbreviation
    : APP_CONFIG.defaultLocation;
};

const getMetroLocationForView = (location, data, defaultLocation) => {
  const stateAbbreviation = getStateAbbreviationForLocation(location, data);
  if (!stateAbbreviation || stateAbbreviation === "US") {
    return defaultLocation;
  }

  return (
    METRO_STATE_ABBREVIATION_TO_LOCATION[stateAbbreviation] || defaultLocation
  );
};

const getNsspLocationForView = (location, data, defaultLocation) => {
  const stateAbbreviation = getStateAbbreviationForLocation(location, data);

  if (!stateAbbreviation || stateAbbreviation === "US") {
    return { nextLocation: defaultLocation, locationMessage: null };
  }

  if (NSSP_UNAVAILABLE_STATE_ABBREVIATIONS.has(stateAbbreviation)) {
    return {
      nextLocation: defaultLocation,
      locationMessage: `There is no NSSP data for ${STATE_ABBREVIATION_TO_NAME[stateAbbreviation] || stateAbbreviation}.`,
    };
  }

  return {
    nextLocation: `${stateAbbreviation}_All`,
    locationMessage: null,
  };
};

export const ViewProvider = ({ children }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const isForecastPage = location.pathname === "/";

  const urlManager = useMemo(
    () => new URLParameterManager(searchParams, setSearchParams),
    [searchParams, setSearchParams],
  );

  const [viewType, setViewTypeState] = useState(() => urlManager.getView());
  const [selectedLocation, setSelectedLocation] = useState(() => {
    const urlLoc = urlManager.getLocation();
    const currentView = urlManager.getView();
    const dataset = urlManager.getDatasetFromView(currentView);
    if (dataset?.defaultLocation && urlLoc === APP_CONFIG.defaultLocation) {
      return dataset.defaultLocation;
    }

    return urlLoc;
  });
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedDates, setSelectedDates] = useState([]);
  const [activeDate, setActiveDate] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [locationMessage, setLocationMessage] = useState(null);
  const [chartScale, setChartScale] = useState(
    () => urlManager.getAdvancedParams().chartScale,
  );
  const [intervalVisibility, setIntervalVisibility] = useState(
    () => urlManager.getAdvancedParams().intervalVisibility,
  );
  const [showLegend, setShowLegend] = useState(
    () => urlManager.getAdvancedParams().showLegend,
  );
  const CURRENT_FLU_SEASON_START = "2025-11-01"; // !! CRITICAL !!: need to change this manually based on the season (for flu peak view)

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

  // filter flu_peak dates based on current season
  const availableDatesToExpose = useMemo(() => {
    if (viewType === "flu_peak") {
      return (availablePeakDates || []).filter(
        (date) => date >= CURRENT_FLU_SEASON_START,
      );
    }
    return availableDates || [];
  }, [viewType, availablePeakDates, availableDates]);

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

    const urlTarget = params.target;
    let targetToSet = null;
    if (urlTarget && availableTargets.includes(urlTarget)) {
      targetToSet = urlTarget;
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

    if (needsModelUrlUpdate) {
      updateDatasetParams({ models: [] });
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

  const handleLocationSelect = (newLocation) => {
    const currentDataset = urlManager.getDatasetFromView(viewType);
    const effectiveDefault =
      currentDataset?.defaultLocation || APP_CONFIG.defaultLocation;
    setLocationMessage(null);
    urlManager.updateLocation(newLocation, effectiveDefault);
    setSelectedLocation(newLocation);
  };

  const handleTargetSelect = (target) => {
    if (!target) return;
    setSelectedTarget(target);
    updateDatasetParams({ target: target });
  };

  const handleViewChange = useCallback(
    (newView) => {
      const oldView = viewType;
      if (oldView === newView) return;

      const oldDataset = urlManager.getDatasetFromView(oldView);
      const newDataset = urlManager.getDatasetFromView(newView);
      const newSearchParams = new URLSearchParams(searchParams);
      const effectiveDefault =
        newDataset?.defaultLocation || APP_CONFIG.defaultLocation;

      let nextLocation = selectedLocation;
      let nextLocationMessage = null;

      if (newView === "metrocast_forecasts") {
        nextLocation = getMetroLocationForView(
          selectedLocation,
          data,
          effectiveDefault,
        );
      } else if (newView === "nsspall") {
        const nsspResolution = getNsspLocationForView(
          selectedLocation,
          data,
          effectiveDefault,
        );
        nextLocation = nsspResolution.nextLocation;
        nextLocationMessage = nsspResolution.locationMessage;
      } else {
        nextLocation = getStandardLocationForView(selectedLocation, data);
      }

      setLocationMessage(nextLocationMessage);
      setSelectedLocation(nextLocation);

      if (nextLocation && nextLocation !== effectiveDefault) {
        newSearchParams.set("location", nextLocation);
      } else {
        newSearchParams.delete("location");
      }

      if (
        newView !== APP_CONFIG.defaultView ||
        newSearchParams.toString().length > 0
      ) {
        newSearchParams.set("view", newView);
      } else {
        newSearchParams.delete("view");
      }

      const isDatasetChange = oldDataset?.shortName !== newDataset?.shortName;
      const isPeakTransition = oldView === "flu_peak" || newView === "flu_peak"; // reset if coming or going from flu_peak

      if (isDatasetChange || isPeakTransition) {
        setSelectedDates([]);
        setSelectedModels([]);
        setActiveDate(null);
        setSelectedTarget(null);

        if (oldDataset) {
          newSearchParams.delete(`${oldDataset.prefix}_models`);
          newSearchParams.delete(`${oldDataset.prefix}_dates`);
          newSearchParams.delete(`${oldDataset.prefix}_target`);
        }

        if (oldDataset?.shortName === "nhsn") {
          newSearchParams.delete("nhsn_target");
          newSearchParams.delete("nhsn_cols");
        }
      } else {
        if (newDataset) {
          newSearchParams.delete(`${newDataset.prefix}_target`);
        }
        setSelectedTarget(null);
      }

      setViewTypeState(newView);
      // Push history for view changes so browser back works between forecast views.
      setSearchParams(newSearchParams, { replace: false });
    },
    [
      viewType,
      searchParams,
      setSearchParams,
      urlManager,
      selectedLocation,
      data,
    ],
  );

  useEffect(() => {
    const viewFromUrl = urlManager.getView();
    if (viewFromUrl !== viewType) {
      setViewTypeState(viewFromUrl);
    }
  }, [searchParams, urlManager, viewType]);

  useEffect(() => {
    if (!isForecastPage) {
      return;
    }
    const {
      chartScale: urlScale,
      intervalVisibility: urlIntervals,
      showLegend: urlLegend,
    } = urlManager.getAdvancedParams();
    if (urlScale !== chartScale) {
      setChartScale(urlScale);
    }
    if (JSON.stringify(urlIntervals) !== JSON.stringify(intervalVisibility)) {
      setIntervalVisibility(urlIntervals);
    }
    if (urlLegend !== showLegend) {
      setShowLegend(urlLegend);
    }
  }, [
    searchParams,
    urlManager,
    isForecastPage,
    chartScale,
    intervalVisibility,
    showLegend,
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

  const contextValue = {
    selectedLocation,
    locationMessage,
    handleLocationSelect,
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
      setSelectedDates((prevDates) => {
        const nextDates =
          typeof updater === "function" ? updater(prevDates) : updater;
        updateDatasetParams({ dates: nextDates });
        return nextDates;
      });
    },
    activeDate,
    setActiveDate,
    viewType,
    setViewType: handleViewChange,
    currentDataset: urlManager.getDatasetFromView(viewType),
    availableTargets: availableTargetsToExpose,

    selectedTarget,
    handleTargetSelect,
    peaks,
    availablePeakDates: (availablePeakDates || []).filter(
      (date) => date >= CURRENT_FLU_SEASON_START,
    ),
    availablePeakModels,
    chartScale,
    setChartScale: setChartScaleWithUrl,
    intervalVisibility,
    setIntervalVisibility: setIntervalVisibilityWithUrl,
    showLegend,
    setShowLegend: setShowLegendWithUrl,
  };

  return (
    <ViewContext.Provider value={contextValue}>{children}</ViewContext.Provider>
  );
};
