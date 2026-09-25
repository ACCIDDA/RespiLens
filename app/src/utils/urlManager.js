import { DATASETS } from "../config";
import { parseForecastUrlState } from "./forecastRoutes";
import { DEFAULT_CHART_SCALE, normalizeChartScale } from "./scaleUtils";
const DEFAULT_INTERVAL_VISIBILITY = {
  median: true,
  ci50: true,
  ci95: true,
};
const DEFAULT_SHOW_LEGEND = true;
const DEFAULT_SHOW_OTHER_GT = false;

export class URLParameterManager {
  constructor(
    searchParams,
    setSearchParams,
    pathname = window.location?.pathname,
    navigate = null,
  ) {
    this.searchParams = searchParams;
    this.setSearchParams = setSearchParams;
    this.pathname = pathname;
    this.navigate = navigate;
  }

  // Get dataset from view type
  getDatasetFromView(viewType) {
    for (const dataset of Object.values(DATASETS)) {
      const hasMatchingView = dataset.views.some(
        (view) => view.value === viewType,
      );
      if (hasMatchingView) {
        return dataset;
      }
    }
    return null;
  }

  getDatasetParams(dataset) {
    if (!dataset) return {};

    const prefix = dataset.prefix;
    const params = {};
    const currentView = this.getView();

    if (dataset.hasDateSelector) {
      const dates = this.searchParams.get("dates");
      params.dates = dates ? dates.split(",") : [];
    }

    if (dataset.hasModelSelector) {
      const models = this.searchParams.get(`${prefix}_models`);
      params.models = models ? models.split(",") : [];
    }

    if (currentView !== "nhsnall") {
      const target = this.searchParams.get(`${prefix}_target`);
      if (target) {
        params.target = target;
      }
    }

    // Special case for NHSN
    if (dataset.shortName === "nhsn") {
      params.columns = this.searchParams.get("nhsn_columns")?.split(",") || [];
    }

    return params;
  }

  getAdvancedParams() {
    const scaleParam = this.searchParams.get("scale");
    const intervalsParam = this.searchParams.get("intervals");
    const legendParam = this.searchParams.get("legend");
    const otherGtParam = this.searchParams.get("other_gt");
    const histogramParam = this.searchParams.get("histogram");

    const chartScale = normalizeChartScale(scaleParam || DEFAULT_CHART_SCALE);
    let intervalVisibility = { ...DEFAULT_INTERVAL_VISIBILITY };

    if (intervalsParam === "none") {
      intervalVisibility = {
        median: false,
        ci50: false,
        ci95: false,
      };
    } else if (intervalsParam) {
      const enabled = new Set(intervalsParam.split(",").filter(Boolean));
      intervalVisibility = {
        median: enabled.has("median"),
        ci50: enabled.has("ci50"),
        ci95: enabled.has("ci95"),
      };
    }

    let showLegend = DEFAULT_SHOW_LEGEND;
    if (legendParam === "0" || legendParam === "false") {
      showLegend = false;
    } else if (legendParam === "1" || legendParam === "true") {
      showLegend = true;
    }

    let showOtherGroundTruthSeasons = DEFAULT_SHOW_OTHER_GT;
    if (otherGtParam === "1" || otherGtParam === "true") {
      showOtherGroundTruthSeasons = true;
    } else if (otherGtParam === "0" || otherGtParam === "false") {
      showOtherGroundTruthSeasons = false;
    }

    return {
      chartScale,
      intervalVisibility,
      showLegend,
      showOtherGroundTruthSeasons,
      showHistogram: histogramParam === "1" || histogramParam === "true",
    };
  }

  // Clear parameters for a specific dataset
  clearDatasetParams(dataset) {
    if (!dataset) {
      return;
    }
    const newParams = new URLSearchParams(this.searchParams);
    const prefix = dataset.prefix;

    if (dataset.hasDateSelector) {
      newParams.delete("dates");
    }
    if (dataset.hasModelSelector) {
      newParams.delete(`${prefix}_models`);
    }

    newParams.delete(`${prefix}_target`);

    if (dataset.shortName === "nhsn") {
      newParams.delete("nhsn_columns");
    }

    this.setSearchParams(newParams, { replace: true });
  }

  updateAdvancedParams({
    chartScale,
    intervalVisibility,
    showLegend,
    showOtherGroundTruthSeasons,
    showHistogram,
  }) {
    const updatedParams = new URLSearchParams(this.searchParams);

    if (chartScale) {
      if (chartScale !== DEFAULT_CHART_SCALE) {
        updatedParams.set("scale", chartScale);
      } else {
        updatedParams.delete("scale");
      }
    }

    if (intervalVisibility) {
      const enabled = ["median", "ci50", "ci95"].filter(
        (key) => intervalVisibility[key],
      );
      if (enabled.length === 0) {
        updatedParams.set("intervals", "none");
      } else if (enabled.length === 3) {
        updatedParams.delete("intervals");
      } else {
        updatedParams.set("intervals", enabled.join(","));
      }
    }

    if (typeof showLegend === "boolean") {
      if (showLegend !== DEFAULT_SHOW_LEGEND) {
        updatedParams.set("legend", showLegend ? "1" : "0");
      } else {
        updatedParams.delete("legend");
      }
    }

    if (typeof showOtherGroundTruthSeasons === "boolean") {
      if (showOtherGroundTruthSeasons !== DEFAULT_SHOW_OTHER_GT) {
        updatedParams.set("other_gt", showOtherGroundTruthSeasons ? "1" : "0");
      } else {
        updatedParams.delete("other_gt");
      }
    }

    if (typeof showHistogram === "boolean") {
      if (showHistogram) {
        updatedParams.set("histogram", "1");
      } else {
        updatedParams.delete("histogram");
      }
    }

    if (updatedParams.toString() !== this.searchParams.toString()) {
      this.setSearchParams(updatedParams, { replace: true });
    }
  }

  updateDatasetParams(dataset, newParams) {
    if (!dataset) {
      return;
    }
    const updatedParams = new URLSearchParams(this.searchParams);
    const prefix = dataset.prefix;
    const currentView = this.getView();

    if (
      dataset.hasDateSelector &&
      Object.prototype.hasOwnProperty.call(newParams, "dates")
    ) {
      if (newParams.dates && newParams.dates.length > 0) {
        updatedParams.set("dates", newParams.dates.join(","));
      } else {
        updatedParams.delete("dates");
      }
    }

    if (
      dataset.hasModelSelector &&
      Object.prototype.hasOwnProperty.call(newParams, "models")
    ) {
      if (newParams.models && newParams.models.length > 0) {
        updatedParams.set(`${prefix}_models`, newParams.models.join(","));
      } else {
        updatedParams.delete(`${prefix}_models`);
      }
    }

    if (
      currentView !== "nhsnall" &&
      Object.prototype.hasOwnProperty.call(newParams, "target")
    ) {
      if (newParams.target) {
        updatedParams.set(`${prefix}_target`, newParams.target);
      } else {
        updatedParams.delete(`${prefix}_target`);
      }
    }

    if (
      dataset.shortName === "nhsn" &&
      Object.prototype.hasOwnProperty.call(newParams, "columns")
    ) {
      if (newParams.columns && newParams.columns.length > 0) {
        updatedParams.set("nhsn_columns", newParams.columns.join(","));
      } else {
        updatedParams.delete("nhsn_columns");
      }
    }

    // Only call setSearchParams if the parameters actually changed
    if (updatedParams.toString() !== this.searchParams.toString()) {
      this.setSearchParams(updatedParams, { replace: true });
    }
  }

  // Get current location from URL
  getLocation() {
    return parseForecastUrlState(this.pathname).location;
  }

  // Get current view from URL
  getView() {
    return parseForecastUrlState(this.pathname).viewType;
  }
  initializeDefaults() {}
}
