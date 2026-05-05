export const DATASETS = {
  flu: {
    shortName: "flu",
    fullName: "Flu Forecasts",
    titleName: "Flusight Forecasts",
    views: [
      {
        key: "detailed",
        label: "Detailed Forecasts View",
        value: "fludetailed",
      },
      { key: "forecasts", label: "Forecasts", value: "flu_forecasts" },
      { key: "peak", label: "Peak Forecasts", value: "flu_peak" },
    ],
    defaultView: "flu_forecasts",
    defaultModel: "FluSight-ensemble",
    hasDateSelector: true,
    hasModelSelector: true,
    prefix: "flu",
    dataPath: "flusight",
    targetLineDayOfWeek: 3, // Wednesday (0=Sunday, 3=Wednesday)
  },
  rsv: {
    shortName: "rsv",
    fullName: "RSV Forecasts",
    titleName: "RSV Forecast Hub",
    views: [{ key: "forecasts", label: "Forecasts", value: "rsv_forecasts" }],
    defaultView: "rsv_forecasts",
    defaultModel: "RSVHub-ensemble",
    hasDateSelector: true,
    hasModelSelector: true,
    prefix: "rsv",
    dataPath: "rsv",
    targetLineDayOfWeek: 3, // Wednesday (0=Sunday, 3=Wednesday)
  },
  covid: {
    shortName: "covid",
    fullName: "COVID-19 Forecasts",
    titleName: "COVID-19 Forecast Hub",
    views: [{ key: "forecasts", label: "Forecasts", value: "covid_forecasts" }],
    defaultView: "covid_forecasts",
    defaultModel: "CovidHub-ensemble",
    hasDateSelector: true,
    hasModelSelector: true,
    prefix: "covid",
    dataPath: "covid19",
    targetLineDayOfWeek: 3, // Wednesday (0=Sunday, 3=Wednesday)
  },
  nhsn: {
    shortName: "nhsn",
    fullName: "NHSN Surveillance Data",
    titleName: "NHSN Surveillance Data",
    views: [{ key: "all", label: "All Data", value: "nhsnall" }],
    defaultView: "nhsnall",
    defaultColumn: "Number of Adult COVID-19 Admissions, 18-49 years",
    hasDateSelector: false,
    hasModelSelector: false,
    prefix: "nhsn",
    dataPath: "nhsn",
  },
  metrocast: {
    shortName: "metrocast",
    fullName: "Flu MetroCast Forecasts",
    titleName: "Flu MetroCast Forecasts",
    views: [
      { key: "forecasts", label: "Forecasts", value: "metrocast_forecasts" },
    ],
    defaultView: "metrocast_forecasts",
    defaultModel: "epiENGAGE-ensemble_mean",
    defaultLocation: "colorado",
    hasDateSelector: true,
    hasModelSelector: true,
    prefix: "metrocast",
    dataPath: "flumetrocast",
    targetLineDayOfWeek: 3,
  },
  nssp: {
    shortName: "nssp",
    fullName: "NSSP Surveillance Data",
    titleName: "NSSP Surveillance Data",
    views: [{ key: "all", label: "All Data", value: "nsspall" }],
    defaultView: "nsspall",
    defaultLocation: "US_All",
    defaultColumn: "TO DO",
    hasDateSelector: false,
    hasModelSelector: false,
    prefix: "nssp",
    dataPath: "nssp",
  },
};

// Helper function to get all valid view values
export const getAllViewValues = () => {
  return Object.values(DATASETS).flatMap((dataset) =>
    dataset.views.map((view) => view.value),
  );
};

// Import centralized colors from theme
import { MODEL_COLORS, getModelColor } from "../theme/mantine.js";

// Re-export for backward compatibility
export { MODEL_COLORS, getModelColor };
