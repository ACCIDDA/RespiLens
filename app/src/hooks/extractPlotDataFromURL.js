import { savePlot } from "../utils/plotStorage";
import { resolvePlotLocationDisplayName } from "../utils/plotLocationDisplay";

const NSSP_COLUMN_LABELS = {
  percent_visits_covid: "COVID-19",
  percent_visits_influenza: "Influenza",
  percent_visits_rsv: "RSV",
};

const NSSP_DEFAULT_COLUMNS = Object.keys(NSSP_COLUMN_LABELS);

/**
 * Parses the current URL and viewType to extract a serialized state
 * for the "My Plots" persistence feature.
 * * @param {string} viewType - The current active view
 * @param {string} href - The full window.location.href string
 * @returns {Object} The processed plotData object
 */
export const extractPlotData = (viewType, href, data) => {
  const url = new URL(href);
  const params = url.searchParams;
  const id = crypto.randomUUID();
  const currentDate = new Date().toISOString().split("T")[0];
  let dataSuffix = "";
  let fileName = "";
  let fullDataPath = "";

  // plot settings
  let location = "";
  let target = "";
  let columns = [];
  let models = [];
  let dates = [];
  let scale = "";
  let intervals = [];
  let viewDisplayName = "";
  let locationDisplayName = "";

  // forecast views set date dynamically if it is the default date (not present in URL)
  switch (viewType) {
    case "covid_forecasts": {
      dataSuffix = "covid19";
      location = params.has("location") ? params.get("location") : "US";
      fileName = `${location}_${dataSuffix}.json`;
      fullDataPath = `covid19forecasthub/${fileName}`;
      target = params.has("covid_target")
        ? params.get("covid_target")
        : "wk inc covid hosp";
      const covidModelsString = params.get("covid_models");
      models = covidModelsString
        ? covidModelsString.split(",")
        : ["CovidHub-ensemble"];
      const covidDatesString = params.get("covid_dates");
      if (covidDatesString) {
        dates = covidDatesString.split(",");
      } else {
        const availableDates = Object.keys(data?.forecasts || {});
        if (availableDates.length > 0) {
          const mostRecent = availableDates.sort().pop();
          dates = [mostRecent];
        } else {
          throw new Error(
            `Unable to extract plot data: No dates found in URL and no forecast data available for ${viewType}.`,
          );
        }
      }
      scale = params.has("scale") ? params.get("scale") : "linear";
      const covidIntervalsString = params.get("intervals");
      intervals = covidIntervalsString
        ? covidIntervalsString.split(",")
        : ["median", "ci50", "ci95"];
      viewDisplayName = "COVID-19 Forecasts";
      break;
    }

    case "flu_forecasts":
    case "fludetailed": {
      dataSuffix = "flu";
      location = params.has("location") ? params.get("location") : "US";
      fileName = `${location}_${dataSuffix}.json`;
      fullDataPath = `flusight/${fileName}`;
      target = params.has("flu_target")
        ? params.get("flu_target")
        : "wk inc flu hosp";
      const fluModelsString = params.get("flu_models");
      models = fluModelsString
        ? fluModelsString.split(",")
        : ["FluSight-ensemble"];
      const fluDatesString = params.get("flu_dates");
      if (fluDatesString) {
        dates = fluDatesString.split(",");
      } else {
        const availableDates = Object.keys(data?.forecasts || {});
        if (availableDates.length > 0) {
          const mostRecent = availableDates.sort().pop();
          dates = [mostRecent];
        } else {
          throw new Error(
            `Unable to extract plot data: No dates found in URL and no forecast data available for ${viewType}.`,
          );
        }
      }
      scale = params.has("scale") ? params.get("scale") : "linear";
      const fluIntervalsString = params.get("intervals");
      intervals = fluIntervalsString
        ? fluIntervalsString.split(",")
        : ["median", "ci50", "ci95"];
      viewDisplayName = "Flu Forecasts";
      break;
    }

    case "rsv_forecasts": {
      dataSuffix = "rsv";
      location = params.has("location") ? params.get("location") : "US";
      fileName = `${location}_${dataSuffix}.json`;
      fullDataPath = `rsvforecasthub/${fileName}`;
      target = params.has("rsv_target")
        ? params.get("rsv_target")
        : "wk inc rsv hosp";
      const rsvModelsString = params.get("rsv_models");
      models = rsvModelsString
        ? rsvModelsString.split(",")
        : ["RSVHub-ensemble"];
      const rsvDatesString = params.get("rsv_dates");
      if (rsvDatesString) {
        dates = rsvDatesString.split(",");
      } else {
        const availableDates = Object.keys(data?.forecasts || {});
        if (availableDates.length > 0) {
          const mostRecent = availableDates.sort().pop();
          dates = [mostRecent];
        } else {
          throw new Error(
            `Unable to extract plot data: No dates found in URL and no forecast data available for ${viewType}.`,
          );
        }
      }
      scale = params.has("scale") ? params.get("scale") : "linear";
      const rsvIntervalsString = params.get("intervals");
      intervals = rsvIntervalsString
        ? rsvIntervalsString.split(",")
        : ["median", "ci50", "ci95"];
      viewDisplayName = "RSV Forecasts";
      break;
    }

    case "metrocast_forecasts": {
      dataSuffix = "flu_metrocast";
      location = params.has("location") ? params.get("location") : "colorado";
      fileName = `${location}_${dataSuffix}.json`;
      fullDataPath = `flumetrocast/${fileName}`;
      target = "Flu ED visits pct";
      const metrocastModelsString = params.get("metrocast_models");
      models = metrocastModelsString
        ? metrocastModelsString.split(",")
        : ["epiENGAGE-ensemble_mean"];
      const metrocastDatesString = params.get("metrocast_dates");
      if (metrocastDatesString) {
        dates = metrocastDatesString.split(",");
      } else {
        const availableDates = Object.keys(data?.forecasts || {});
        if (availableDates.length > 0) {
          const mostRecent = availableDates.sort().pop();
          dates = [mostRecent];
        } else {
          throw new Error(
            `Unable to extract plot data: No dates found in URL and no forecast data available for ${viewType}.`,
          );
        }
      }
      scale = params.has("scale") ? params.get("scale") : "linear";
      const metrocastIntervalsString = params.get("intervals");
      intervals = metrocastIntervalsString
        ? metrocastIntervalsString.split(",")
        : ["median", "ci50", "ci95"];
      viewDisplayName = "Flu Forecasts";
      break;
    }

    case "nhsnall": {
      dataSuffix = "nhsn";
      location = params.has("location") ? params.get("location") : "US";
      fileName = `${location}_${dataSuffix}.json`;
      fullDataPath = `nhsn/${fileName}`;
      target = params.has("nhsn_target")
        ? params.get("nhsn_target")
        : "Hospital Admissions (count)";
      const nhsnColsFromUrl = params.getAll("nhsn_cols");
      const defaultColumnSlugsByTarget = {
        "Hospital Admissions (count)": [
          "totalconfc19newadm",
          "totalconfflunewadm",
          "totalconfrsvnewadm",
        ],
        "Hospital Admissions (rate)": [
          "totalconfc19newadmper100k",
          "totalconfflunewadmper100k",
          "totalconfrsvnewadmper100k",
        ],
        "Hospital Admissions (%)": [
          "pctconfc19newadmadult",
          "pctconfflunewadmadult",
          "pctconfrsvnewadmadult",
        ],
        "Bed Capacity (count)": ["numinptbeds", "numinptbedsocc"],
        "Bed Capacity (%)": ["pctinptbedsocc"],
      };
      if (nhsnColsFromUrl.length > 0) {
        columns = nhsnColsFromUrl;
        // default columns are dependent on target (aka column unit)
      } else {
        columns = defaultColumnSlugsByTarget[target] || [];
      }
      dates = [currentDate];
      scale = params.has("scale") ? params.get("scale") : "linear";
      viewDisplayName = "NHSN Data";
      break;
    }

    case "nsspall": {
      dataSuffix = "nssp";
      location = params.has("location") ? params.get("location") : "US_All";
      fileName = `${location}_${dataSuffix}.json`;
      fullDataPath = `nssp/${fileName}`;
      target = "Percent of visits";
      const nsspColumnsFromUrl = params.getAll("nssp_cols");
      const availableColumns = Object.keys(data?.series || {}).filter(
        (key) => key !== "dates" && key,
      );
      const validUrlColumns = nsspColumnsFromUrl.filter((column) =>
        availableColumns.includes(column),
      );
      const isExplicitlyEmpty = nsspColumnsFromUrl.includes("none");

      if (validUrlColumns.length > 0) {
        columns = validUrlColumns;
      } else if (isExplicitlyEmpty) {
        columns = [];
      } else {
        const defaultColumns = NSSP_DEFAULT_COLUMNS.filter((column) =>
          availableColumns.includes(column),
        );
        columns = defaultColumns.length > 0 ? defaultColumns : availableColumns;
      }

      dates = [currentDate];
      scale = params.has("scale") ? params.get("scale") : "linear";
      viewDisplayName = "NSSP Data";
      break;
    }

    default:
      throw new Error(`Unknown view type: ${viewType}`);
  }

  const plotData = {
    viewType: viewType,
    fullUrl: href,
    viewDisplayName,
    id,
    currentDate,
    dataSuffix,
    fileName,
    fullDataPath,
    locationDisplayName:
      locationDisplayName || resolvePlotLocationDisplayName(location, data),
    settings: {
      location,
      target,
      columns,
      models,
      dates,
      scale,
      intervals,
    },
  };

  savePlot(plotData);

  return plotData;
};
