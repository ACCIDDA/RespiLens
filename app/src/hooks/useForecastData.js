import { useState, useEffect } from "react";
import { getDataPath } from "../utils/paths";

export const useForecastData = (location, viewType) => {
  const [data, setData] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [models, setModels] = useState([]);
  const [availableTargets, setAvailableTargets] = useState([]);
  const [modelsByTarget, setModelsByTarget] = useState({});

  const [availablePeakDates, setAvailablePeakDates] = useState([]);
  const [availablePeakModels, setAvailablePeakModels] = useState([]);
  const peaks = data?.peaks || null;

  useEffect(() => {
    const isMetrocastView = viewType === "metrocast_forecasts";
    const isDefaultUS = location === "US";
    if (
      (isMetrocastView && isDefaultUS) ||
      (!isMetrocastView && location === "colorado")
    ) {
      setLoading(false);
      return;
    }
    if (!location || !viewType || viewType === "frontpage") {
      setLoading(false);
      setError(null);
      setData(null);
      setMetadata(null);
      setAvailableDates([]);
      setModels([]);
      setAvailableTargets([]);
      setModelsByTarget({});
      return;
    }

    const fetchData = async () => {
      console.log(
        `[useForecastData] Triggered for View: ${viewType}, Location: ${location}`,
      ); // REMOVE after debug
      setLoading(true);
      setError(null);
      setData(null);
      setMetadata(null);
      setAvailableTargets([]);

      try {
        const datasetMap = {
          fludetailed: { directory: "flusight", suffix: "flu" },
          flu_forecasts: { directory: "flusight", suffix: "flu" },
          flu_peak: { directory: "flusight", suffix: "flu" },
          covid_forecasts: {
            directory: "covid19forecasthub",
            suffix: "covid19",
          },
          rsv_forecasts: { directory: "rsvforecasthub", suffix: "rsv" },
          nhsnall: { directory: "nhsn", suffix: "nhsn" },
          nsspall: { directory: "nssp", suffix: "nssp" },
          metrocast_forecasts: {
            directory: "flumetrocast",
            suffix: "flu_metrocast",
          },
        };

        const datasetConfig = datasetMap[viewType];
        if (!datasetConfig) throw new Error(`Unknown view type: ${viewType}`);

        const dataPath = getDataPath(
          `${datasetConfig.directory}/${location}_${datasetConfig.suffix}.json`,
        );
        console.log(`[useForecastData] Attempting fetch from: ${dataPath}`); // REMOVE after debug
        const metadataPath = getDataPath(
          `${datasetConfig.directory}/metadata.json`,
        );

        const [dataResponse, metadataResponse] = await Promise.all([
          fetch(dataPath),
          fetch(metadataPath),
        ]);
        console.log(
          `[useForecastData] Data Response OK: ${dataResponse.ok}, Status: ${dataResponse.status}`,
        );

        if (!dataResponse.ok)
          throw new Error(`Failed to fetch data: ${dataResponse.status}`);
        if (!metadataResponse.ok)
          throw new Error(
            `Failed to fetch metadata: ${metadataResponse.status}`,
          );

        const jsonData = await dataResponse.json();
        console.log(
          `[useForecastData] Successfully parsed JSON for ${viewType}`,
        );
        const jsonMetadata = await metadataResponse.json();

        setData(jsonData);
        setMetadata(jsonMetadata);

        if (jsonData.forecasts) {
          const dates = Object.keys(jsonData.forecasts).sort();
          setAvailableDates(dates);

          const modelSet = new Set();
          Object.values(jsonData.forecasts).forEach((dateData) => {
            Object.values(dateData).forEach((targetData) => {
              Object.keys(targetData).forEach((model) => modelSet.add(model));
            });
          });
          setModels(Array.from(modelSet).sort());
        }

        // if flu, rsv, or covid data: // loop through JSON data to get model lists by target (for dynamic model selection)
        if (jsonData.forecasts) {
          const modelsByTargetMap = new Map();

          Object.values(jsonData.forecasts).forEach((dateData) => {
            // Loop over [target, targetData] entries
            Object.entries(dateData).forEach(([target, targetData]) => {
              if (!modelsByTargetMap.has(target)) {
                modelsByTargetMap.set(target, new Set());
              }
              const modelSetForTarget = modelsByTargetMap.get(target);

              Object.keys(targetData).forEach((model) => {
                modelSetForTarget.add(model);
              });
            });
          });

          const modelsByTargetState = {};
          for (const [target, modelSet] of modelsByTargetMap.entries()) {
            modelsByTargetState[target] = Array.from(modelSet).sort();
          }
          setModelsByTarget(modelsByTargetState);
        }

        let targets = [];
        if (jsonData?.ground_truth) {
          targets = Object.keys(jsonData.ground_truth).filter(
            (key) => key !== "dates",
          );
        }
        setAvailableTargets(targets);
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [location, viewType]);

  useEffect(() => {
    if (!peaks || typeof peaks !== "object") {
      setAvailablePeakDates([]);
      setAvailablePeakModels([]);
      return;
    }

    const dates = Object.keys(peaks).sort();
    setAvailablePeakDates(dates);

    const models = new Set();
    const targets = new Set();
    Object.values(peaks).forEach((dateData) => {
      Object.entries(dateData).forEach(([targetName, targetData]) => {
        targets.add(targetName);
        Object.keys(targetData).forEach((modelId) => {
          models.add(modelId);
        });
      });
    });
    setAvailablePeakModels(Array.from(models).sort());
    // setAvailablePeakTargets(Array.from(targets).sort());, leave this for now
  }, [peaks]);

  return {
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
  };
};
