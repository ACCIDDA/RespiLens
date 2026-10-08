import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActionIcon,
  Alert,
  Anchor,
  Badge,
  Box,
  Button,
  Checkbox,
  Container,
  Group,
  Loader,
  List,
  Paper,
  Popover,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  ThemeIcon,
  Tooltip,
  Title,
  useMantineColorScheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useNavigate, useParams } from "react-router-dom";
import {
  IconAdjustmentsHorizontal,
  IconAlertCircle,
  IconArrowLeft,
  IconFileText,
  IconBrandGithub,
  IconInfoCircle,
  IconUpload,
  IconPlus,
  IconZoomReset,
  IconDownload,
} from "@tabler/icons-react";
import Plot from "react-plotly.js";
import Plotly from "plotly.js/dist/plotly";
import { parquetReadObjects } from "hyparquet";
import DateSelector from "../DateSelector";
import InlinePicker from "../InlinePicker";
import { ChartInfoButton } from "../KeyboardShortcutsModal";
import ShortcutHint from "../ShortcutHint";
import { useKeyboardShortcut } from "../../hooks/useKeyboardShortcut";
import ModelSelector from "../ModelSelector";
import ForecastChartControls from "../controls/ForecastChartControls";
import Seo from "../Seo";
import useQuantileForecastTraces from "../../hooks/useQuantileForecastTraces";
import { APP_CONFIG } from "../../config/app";
import { DATASETS, MODEL_COLORS } from "../../config/datasets";
import { nextIntervalVisibility } from "../../utils/intervalCycle";
import { downloadChartPng } from "../../utils/downloadChartPng";
import {
  CHART_CONSTANTS,
  RANGESLIDER_STYLE,
  GROUND_TRUTH_LINE_WIDTH,
  GROUND_TRUTH_MARKER_SIZE,
  PLOT_CONFIG,
  getBaseChartLayout,
  getChartInk,
  getForecastDateLineStyle,
} from "../../constants/chart";
import { extendStableModelOrder } from "../../utils/modelColorUtils";
import { targetDisplayNameMap } from "../../utils/mapUtils";
import {
  getScaleYAxis,
  isPlotlyLogScale,
  normalizeChartScale,
  transformValueForScale,
} from "../../utils/scaleUtils";

const HUB_OPTIONS = [
  {
    slug: "flusight",
    label: "FluSight Forecast Hub",
    githubUrl: "https://github.com/cdcepi/fluSight-forecast-hub",
    pathogenKey: "flu",
    processedDataPath: "processed_data/forecastchecker/flusight",
    fileSuffix: "flu",
    datasetLabel: "flusight forecasts",
    groundTruthMinDate: "2022-10-01",
  },
  {
    slug: "covid19forecasthub",
    label: "COVID-19 Forecast Hub",
    githubUrl: "https://github.com/CDCgov/covid19-forecast-hub",
    pathogenKey: "covid19forecasthub",
    processedDataPath: "processed_data/forecastchecker/covid19forecasthub",
    fileSuffix: "covid19",
    datasetLabel: "covid19 forecast hub",
    groundTruthMinDate: "2023-10-01",
  },
  {
    slug: "rsvforecasthub",
    label: "RSV Forecast Hub",
    githubUrl: "https://github.com/CDCgov/rsv-forecast-hub",
    pathogenKey: "rsvforecasthub",
    processedDataPath: "processed_data/forecastchecker/rsvforecasthub",
    fileSuffix: "rsv",
    datasetLabel: "rsv forecast hub",
    groundTruthMinDate: "2023-10-01",
  },
  {
    slug: "flumetrocast",
    label: "Flu Metrocast Hub",
    githubUrl: "https://github.com/reichlab/flu-metrocast",
    pathogenKey: "flumetrocast",
    processedDataPath: "processed_data/forecastchecker/flumetrocast",
    fileSuffix: "flu_metrocast",
    datasetLabel: "flu metrocast forecasts",
    groundTruthMinDate: "2024-08-01",
  },
];

const EXTRA_HUB_OPTIONS = [
  {
    slug: "other-hub",
    label: "Other hub",
    icon: IconPlus,
  },
];

const OTHER_HUB_CONFIG = {
  slug: "other-hub",
  label: "Other hub",
  githubUrl: null,
  pathogenKey: "other-hub",
  processedDataPath: null,
  fileSuffix: "other_hub",
  datasetLabel: "user supplied forecasts",
  groundTruthMinDate: null,
};

const FORECAST_REQUIRED_COLUMNS = [
  "location",
  "reference_date",
  "target",
  "horizon",
  "output_type",
  "output_type_id",
  "value",
  "target_end_date",
];

const GROUND_TRUTH_REQUIRED_COLUMNS = [
  "target_end_date",
  "location",
  "observation",
  "target",
];

const PEAK_TARGETS = new Set(["peak inc flu hosp", "peak week inc flu hosp"]);
const CATEGORICAL_OUTPUT_TYPE_IDS = new Set([
  "decrease",
  "increase",
  "large_decrease",
  "large_increase",
  "stable",
]);
const DEFAULT_MODEL_ID = "user-uploaded-model";
const FLU_METROCAST_MIN_TARGET_END_DATE = "2025-11-22";
const HUB_PATHOGEN_KEYWORDS = {
  flusight: "flu",
  flumetrocast: "flu",
  covid19forecasthub: "covid",
  rsvforecasthub: "rsv",
};

const csvDateLikeRegex = /^\d{4}-\d{2}-\d{2}$/;

const countCsvDataRows = (text) => {
  const rows = parseCsv(text);
  return Math.max(0, rows.length - 1);
};

const readTabularUpload = async (file) => {
  const lowerName = file.name.toLowerCase();

  if (lowerName.endsWith(".csv")) {
    const text = await file.text();
    const rows = parseCsv(text);
    const { headers, records } = toObjects(rows);
    return { headers, records };
  }

  if (lowerName.endsWith(".parquet") || lowerName.endsWith(".pq")) {
    const buffer = await file.arrayBuffer();
    const records = await parquetReadObjects({ file: buffer });
    const headers = records[0] ? Object.keys(records[0]) : [];
    return { headers, records };
  }

  throw new Error(
    `Unsupported file type "${file.name}". Please upload a .csv or .parquet file.`,
  );
};

const normalizeDateString = (value) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date.toISOString().slice(0, 10);
};

const shiftDateStringByDays = (dateString, days) => {
  const [year, month, day] = dateString.split("-").map(Number);
  const shiftedDate = new Date(Date.UTC(year, month - 1, day + days));
  return shiftedDate.toISOString().slice(0, 10);
};

const buildRequiredColumnError = (fileLabel, missingColumns) =>
  `${fileLabel} is missing required columns: ${missingColumns.join(", ")}.`;

const uniqueValuesInOrder = (values) => Array.from(new Set(values));

const getFileRelativePath = (file) => file.webkitRelativePath || file.name;

const readDirectoryEntry = (entry) =>
  new Promise((resolve, reject) => {
    const reader = entry.createReader();
    const allEntries = [];

    const readBatch = () => {
      reader.readEntries(
        (entries) => {
          if (entries.length === 0) {
            resolve(allEntries);
            return;
          }
          allEntries.push(...entries);
          readBatch();
        },
        (error) => reject(error),
      );
    };

    readBatch();
  });

const fileFromEntry = (entry) =>
  new Promise((resolve, reject) => {
    entry.file(resolve, reject);
  });

const collectFilesFromEntry = async (entry) => {
  if (entry.isFile) {
    const file = await fileFromEntry(entry);
    return [file];
  }

  if (entry.isDirectory) {
    const entries = await readDirectoryEntry(entry);
    const nestedFiles = await Promise.all(
      entries.map((nestedEntry) => collectFilesFromEntry(nestedEntry)),
    );
    return nestedFiles.flat();
  }

  return [];
};

const collectDroppedFiles = async (dataTransfer) => {
  const items = Array.from(dataTransfer?.items ?? []);
  const directoryEntries = items
    .map((item) =>
      typeof item.webkitGetAsEntry === "function"
        ? item.webkitGetAsEntry()
        : null,
    )
    .filter(Boolean);

  if (directoryEntries.length > 0) {
    const nestedFiles = await Promise.all(
      directoryEntries.map((entry) => collectFilesFromEntry(entry)),
    );
    return nestedFiles.flat();
  }

  return Array.from(dataTransfer?.files ?? []);
};

const fetchReferenceFile = async (path, fileKind) => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${fileKind} from ${path}.`);
  }

  const blob = await response.blob();
  const fileName = path.split("/").pop() ?? path;

  if (fileName.endsWith(".csv")) {
    const text = await blob.text();
    const rows = parseCsv(text);
    const { headers, records } = toObjects(rows);
    return {
      path,
      fileName,
      size: blob.size,
      rowCount: countCsvDataRows(text),
      headers,
      records,
    };
  }

  return {
    path,
    fileName,
    size: blob.size,
    rowCount: null,
    headers: [],
    records: null,
  };
};

const loadHubReferenceData = async (hubConfig) => {
  const basePath = `/${hubConfig.processedDataPath}`;
  const locationsPath = `${basePath}/locations.csv`;
  const locations = await fetchReferenceFile(locationsPath, "locations.csv");

  try {
    const timeSeriesCsvPath = `${basePath}/time-series.csv`;
    const timeSeries = await fetchReferenceFile(
      timeSeriesCsvPath,
      "time-series.csv",
    );
    return { locations, timeSeries };
  } catch {
    const timeSeriesParquetPath = `${basePath}/time-series.parquet`;
    const timeSeries = await fetchReferenceFile(
      timeSeriesParquetPath,
      "time-series.parquet",
    );
    return { locations, timeSeries };
  }
};

const fetchProjectionJson = async (path) => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load projections JSON from ${path}.`);
  }

  return response.json();
};

const parseCsv = (text) => {
  const rows = [];
  let currentCell = "";
  let currentRow = [];
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const nextChar = text[index + 1];

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        currentCell += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      currentRow.push(currentCell);
      currentCell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && nextChar === "\n") {
        index += 1;
      }
      currentRow.push(currentCell);
      rows.push(currentRow);
      currentRow = [];
      currentCell = "";
      continue;
    }

    currentCell += char;
  }

  if (currentCell.length > 0 || currentRow.length > 0) {
    currentRow.push(currentCell);
    rows.push(currentRow);
  }

  if (inQuotes) {
    throw new Error("The CSV appears to have an unmatched quote.");
  }

  return rows
    .map((row) => row.map((cell) => cell.trim()))
    .filter((row) => row.some((cell) => cell !== ""));
};

const toObjects = (rows) => {
  if (rows.length === 0) {
    return { headers: [], records: [] };
  }

  const headers = rows[0];
  const records = rows.slice(1).map((row) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = row[index] ?? "";
    });
    return record;
  });

  return { headers, records };
};

const isDateLikeValue = (value) => {
  if (!value || !csvDateLikeRegex.test(value)) {
    return false;
  }
  const parsed = new Date(value);
  return !Number.isNaN(parsed.getTime());
};

const normalizeOutputTypeId = (value) => {
  const numericValue = Number(value);
  if (!Number.isNaN(numericValue)) {
    return numericValue;
  }
  return value;
};

const isSupportedQuantileOutputTypeId = (outputType, outputTypeId) =>
  outputType === "quantile" &&
  typeof outputTypeId === "number" &&
  outputTypeId >= 0 &&
  outputTypeId <= 1;

const buildQuantileIntervalKey = (lowerQuantile, upperQuantile) =>
  `${lowerQuantile}_${upperQuantile}`;

const formatIntervalPercentage = (value) => {
  const rounded = Math.round(value * 100) / 100;
  return Number.isInteger(rounded) ? String(rounded) : String(rounded);
};

const getPredictionIntervalDefinitions = (locationData, target) => {
  if (!locationData?.forecasts || !target) {
    return { hasMedian: false, intervalDefinitions: [] };
  }

  const intervalMap = new Map();
  let hasMedian = false;

  Object.values(locationData.forecasts).forEach((dateData) => {
    const targetData = dateData?.[target];
    if (!targetData) {
      return;
    }

    Object.values(targetData).forEach((modelData) => {
      Object.values(modelData?.predictions || {}).forEach((prediction) => {
        const quantiles = (prediction.quantiles || []).map((value) =>
          Number(value),
        );
        const quantileSet = new Set(
          quantiles.filter((value) => !Number.isNaN(value)),
        );

        if (quantileSet.has(0.5)) {
          hasMedian = true;
        }

        quantileSet.forEach((quantile) => {
          if (quantile >= 0.5) {
            return;
          }

          const pairedQuantile = Number((1 - quantile).toFixed(10));
          if (!quantileSet.has(pairedQuantile)) {
            return;
          }

          const key = buildQuantileIntervalKey(quantile, pairedQuantile);
          if (!intervalMap.has(key)) {
            intervalMap.set(key, {
              key: `interval_${String(quantile).replace(".", "p")}_${String(
                pairedQuantile,
              ).replace(".", "p")}`,
              label: `${formatIntervalPercentage(
                (pairedQuantile - quantile) * 100,
              )}% interval`,
              lowerQuantile: quantile,
              upperQuantile: pairedQuantile,
            });
          }
        });
      });
    });
  });

  return {
    hasMedian,
    intervalDefinitions: [...intervalMap.values()].sort(
      (left, right) => left.lowerQuantile - right.lowerQuantile,
    ),
  };
};

const validateHubverseCsv = (records, hubConfig) => {
  const summary = {
    totalRows: records.length,
    rowsMissingHorizon: 0,
    rowsFilteredAsNowcasts: 0,
    rowsFilteredAsSamples: 0,
    rowsFilteredByOutputTypeId: 0,
    rowsFilteredAsPeakTargets: 0,
    rowsFilteredAsDuplicates: 0,
    usableRows: 0,
  };

  const errors = [];
  const sampleProblems = [];
  let hasFluMetrocastEarlyDate = false;

  if (records.length === 0) {
    errors.push("The CSV has headers but no forecast rows.");
    return { ok: false, errors, summary };
  }

  const usableRows = records.flatMap((record, index) => {
    const target = String(record.target ?? "");
    const outputType = String(record.output_type ?? "");
    const outputTypeId = String(record.output_type_id ?? "");
    const rawHorizon = String(record.horizon ?? "");
    const modelId = String(record.model_id ?? DEFAULT_MODEL_ID);
    const normalizedTargetEndDate = normalizeDateString(record.target_end_date);
    const numericValue = Number(record.value);

    if (hubConfig?.slug === "flusight" && PEAK_TARGETS.has(target)) {
      summary.rowsFilteredAsPeakTargets += 1;
      return [];
    }

    if (!normalizedTargetEndDate) {
      sampleProblems.push(
        `Row ${index + 2} has an invalid target_end_date of "${record.target_end_date}".`,
      );
      return [];
    }

    if (
      hubConfig?.slug === "flumetrocast" &&
      normalizedTargetEndDate < FLU_METROCAST_MIN_TARGET_END_DATE
    ) {
      hasFluMetrocastEarlyDate = true;
      return [];
    }

    if (Number.isNaN(numericValue)) {
      sampleProblems.push(
        `Row ${index + 2} has a non-numeric value of "${record.value}".`,
      );
      return [];
    }

    let horizonValue = rawHorizon;
    if (PEAK_TARGETS.has(target)) {
      horizonValue = "50";
    }

    if (horizonValue === "") {
      summary.rowsMissingHorizon += 1;
      return [];
    }

    const numericHorizon = Number(horizonValue);
    if (!Number.isInteger(numericHorizon)) {
      sampleProblems.push(
        `Row ${index + 2} has a horizon of "${rawHorizon}", which is not an integer.`,
      );
      return [];
    }

    if (numericHorizon < 0) {
      summary.rowsFilteredAsNowcasts += 1;
      return [];
    }

    if (outputType === "sample") {
      summary.rowsFilteredAsSamples += 1;
      return [];
    }

    const normalizedOutputTypeId = normalizeOutputTypeId(outputTypeId);
    const isCategorical = CATEGORICAL_OUTPUT_TYPE_IDS.has(outputTypeId);
    const isPeakWeekTarget = target.includes("peak week inc flu hosp");
    const isValidPeakWeekDate =
      isPeakWeekTarget && isDateLikeValue(outputTypeId);
    const isSupportedQuantile = isSupportedQuantileOutputTypeId(
      outputType,
      normalizedOutputTypeId,
    );

    if (isCategorical || !(isSupportedQuantile || isValidPeakWeekDate)) {
      summary.rowsFilteredByOutputTypeId += 1;
      return [];
    }

    return [
      {
        ...record,
        target,
        output_type: outputType,
        output_type_id:
          outputType === "quantile"
            ? Number(normalizedOutputTypeId)
            : outputTypeId,
        horizon: numericHorizon,
        model_id: modelId,
        value: numericValue,
        target_end_date: normalizedTargetEndDate,
        reference_date:
          normalizeDateString(record.reference_date) ?? record.reference_date,
      },
    ];
  });

  if (sampleProblems.length > 0) {
    errors.push(...sampleProblems.slice(0, 5));
  }

  if (hasFluMetrocastEarlyDate) {
    errors.push(
      "Your upload contains dates before 2025-11-22 (Forecast Checker flu-metrocast can only process target_end_dates after this day)",
    );
  }

  const seenCompositeKeys = new Set();
  const deduplicatedRows = usableRows.filter((row) => {
    const compositeKey = [
      String(row.reference_date),
      String(row.target_end_date),
      String(row.location ?? ""),
      String(row.horizon),
      String(row.target),
      String(row.output_type),
      String(row.output_type_id),
      String(row.model_id ?? DEFAULT_MODEL_ID),
    ].join("__");

    if (seenCompositeKeys.has(compositeKey)) {
      summary.rowsFilteredAsDuplicates += 1;
      return false;
    }

    seenCompositeKeys.add(compositeKey);
    return true;
  });

  summary.usableRows = deduplicatedRows.length;

  if (deduplicatedRows.length === 0) {
    errors.push("No usable forecast rows remain after preprocessing.");
  }

  return {
    ok: errors.length === 0,
    errors,
    summary,
    usableRows: deduplicatedRows,
  };
};

const buildHubPathogenWarning = (forecastRows, hubConfig) => {
  const expectedKeyword = HUB_PATHOGEN_KEYWORDS[hubConfig?.slug];

  if (!expectedKeyword || !forecastRows?.length) {
    return null;
  }

  const hasExpectedPathogen = forecastRows.some((row) =>
    String(row.target ?? "")
      .toLowerCase()
      .includes(expectedKeyword),
  );

  if (hasExpectedPathogen) {
    return null;
  }

  return `Your uploaded target names do not appear to mention "${expectedKeyword}". Please double-check your hub selection.`;
};

const buildGroundTruthOutput = (
  targetRows,
  hubConfig,
  allowedTargets = null,
) => {
  const requiredColumns = GROUND_TRUTH_REQUIRED_COLUMNS;
  const missingColumns = requiredColumns.filter(
    (column) => !(targetRows[0] ? column in targetRows[0] : true),
  );
  if (missingColumns.length > 0) {
    throw new Error(
      buildRequiredColumnError("time-series data", missingColumns),
    );
  }

  const minDate = hubConfig.groundTruthMinDate
    ? new Date(hubConfig.groundTruthMinDate)
    : null;
  const hasAsOfColumn = Boolean(targetRows[0] && "as_of" in targetRows[0]);
  const latestByKey = new Map();
  const allowedTargetSet =
    allowedTargets && allowedTargets.length > 0
      ? new Set(allowedTargets)
      : null;

  targetRows.forEach((row) => {
    const normalizedTargetEndDate = normalizeDateString(row.target_end_date);
    const normalizedAsOf = hasAsOfColumn
      ? normalizeDateString(row.as_of)
      : null;
    const observation =
      row.observation === null ? null : Number(row.observation);
    const target = String(row.target);

    if (
      (allowedTargetSet && !allowedTargetSet.has(target)) ||
      !normalizedTargetEndDate ||
      (hasAsOfColumn && !normalizedAsOf) ||
      (observation !== null && Number.isNaN(observation)) ||
      (minDate && new Date(normalizedTargetEndDate) < minDate)
    ) {
      return;
    }

    const dedupeKey = `${target}__${normalizedTargetEndDate}`;
    const existing = latestByKey.get(dedupeKey);
    if (!existing || !hasAsOfColumn || normalizedAsOf >= existing.as_of) {
      latestByKey.set(dedupeKey, {
        target,
        target_end_date: normalizedTargetEndDate,
        as_of: normalizedAsOf,
        observation,
      });
    }
  });

  const dedupedRows = Array.from(latestByKey.values()).sort((left, right) => {
    const leftTime = new Date(left.target_end_date).getTime();
    const rightTime = new Date(right.target_end_date).getTime();
    if (leftTime !== rightTime) {
      return leftTime - rightTime;
    }
    return left.target.localeCompare(right.target);
  });

  const dates = uniqueValuesInOrder(
    dedupedRows.map((row) => row.target_end_date),
  );
  const targets = uniqueValuesInOrder(dedupedRows.map((row) => row.target));
  const groundTruth = { dates };

  targets.forEach((target) => {
    const valuesByDate = new Map(
      dedupedRows
        .filter((row) => row.target === target)
        .map((row) => [row.target_end_date, row.observation]),
    );
    groundTruth[target] = dates.map((date) =>
      valuesByDate.has(date) ? valuesByDate.get(date) : null,
    );
  });

  return groundTruth;
};

const buildForecastOutput = (forecastRows) => {
  const forecasts = {};

  forecastRows.forEach((row) => {
    const outputType = row.output_type;
    if (outputType !== "quantile" && outputType !== "pmf") {
      throw new Error(
        `Unsupported output_type "${outputType}" found while building projections JSON.`,
      );
    }

    const referenceDate = String(row.reference_date);
    const target = String(row.target);
    const modelId = String(row.model_id);
    const horizon = String(row.horizon);

    forecasts[referenceDate] ??= {};
    forecasts[referenceDate][target] ??= {};
    forecasts[referenceDate][target][modelId] ??= {
      type: outputType,
      predictions: {},
    };

    forecasts[referenceDate][target][modelId].predictions[horizon] ??= {
      date: String(row.target_end_date),
      ...(outputType === "quantile"
        ? { quantiles: [], values: [] }
        : { categories: [], probabilities: [] }),
    };

    if (outputType === "quantile") {
      forecasts[referenceDate][target][modelId].predictions[
        horizon
      ].quantiles.push(row.output_type_id);
      forecasts[referenceDate][target][modelId].predictions[
        horizon
      ].values.push(row.value);
    } else {
      forecasts[referenceDate][target][modelId].predictions[
        horizon
      ].categories.push(row.output_type_id);
      forecasts[referenceDate][target][modelId].predictions[
        horizon
      ].probabilities.push(row.value);
    }
  });

  return forecasts;
};

const buildProjectionOutputs = ({
  hubConfig,
  forecastRows,
  locationsRows,
  targetRows,
}) => {
  if (!targetRows) {
    throw new Error("Reference data did not include parsed CSV records.");
  }

  if (hubConfig.slug === "other-hub") {
    const outputs = {};
    const locationIds = uniqueValuesInOrder(
      forecastRows.map((row) => String(row.location)),
    );

    locationIds.forEach((locationId) => {
      const locationForecastRows = forecastRows.filter(
        (row) => String(row.location) === locationId,
      );
      const locationTargetRows = targetRows.filter(
        (row) => String(row.location) === locationId,
      );
      const forecastTargets = uniqueValuesInOrder(
        locationForecastRows.map((row) => String(row.target)),
      );

      outputs[locationId] = {
        metadata: {
          location: locationId,
          abbreviation: locationId,
          location_name: locationId,
          population: null,
          dataset: hubConfig.datasetLabel,
          series_type: "projection",
          hubverse_keys: {
            models: uniqueValuesInOrder(
              locationForecastRows.map((row) => String(row.model_id)),
            ),
            targets: forecastTargets,
            horizons: uniqueValuesInOrder(
              locationForecastRows.map((row) => String(row.horizon)),
            ),
            output_types: uniqueValuesInOrder(
              locationForecastRows.map((row) => String(row.output_type)),
            ).filter((value) => value !== "sample"),
          },
        },
        ground_truth: buildGroundTruthOutput(
          locationTargetRows,
          hubConfig,
          forecastTargets,
        ),
        forecasts: buildForecastOutput(locationForecastRows),
      };
    });

    outputs["metadata.json"] = {
      last_updated: new Date().toISOString(),
      models: [
        ...uniqueValuesInOrder(forecastRows.map((row) => String(row.model_id))),
      ].sort(),
      locations: locationIds.map((locationId) => ({
        location: locationId,
        abbreviation: locationId,
        location_name: locationId,
        population: null,
      })),
    };

    return outputs;
  }

  if (!locationsRows) {
    throw new Error("Reference data did not include parsed CSV records.");
  }

  const requiredLocationColumns =
    hubConfig.slug === "flumetrocast"
      ? [
          "location",
          "original_location_code",
          "state",
          "state_abb",
          "location_name",
          "population",
        ]
      : ["location", "abbreviation", "location_name", "population"];
  const missingLocationColumns = requiredLocationColumns.filter(
    (column) => !(locationsRows[0] ? column in locationsRows[0] : true),
  );
  if (missingLocationColumns.length > 0) {
    throw new Error(
      buildRequiredColumnError("locations.csv", missingLocationColumns),
    );
  }

  const locationMap = new Map(
    locationsRows.map((row) => [String(row.location), row]),
  );

  const missingLocations = forecastRows
    .map((row) => String(row.location))
    .filter((location, index, all) => all.indexOf(location) === index)
    .filter((location) => !locationMap.has(location))
    .sort();
  if (missingLocations.length > 0) {
    throw new Error(
      `The following locations are missing from locations.csv: ${missingLocations.join(", ")}.`,
    );
  }

  const outputs = {};
  const locationIds = uniqueValuesInOrder(
    forecastRows.map((row) => String(row.location)),
  );

  locationIds.forEach((locationId) => {
    const locationInfo = locationMap.get(locationId);
    const locationForecastRows = forecastRows.filter(
      (row) => String(row.location) === locationId,
    );
    const locationTargetRows = targetRows.filter(
      (row) => String(row.location) === locationId,
    );

    const metadata = {
      location:
        hubConfig.slug === "flumetrocast"
          ? String(locationInfo.original_location_code)
          : locationId,
      abbreviation:
        hubConfig.slug === "flumetrocast"
          ? String(locationInfo.location)
          : String(locationInfo.abbreviation),
      location_name: String(locationInfo.location_name),
      population: Number(locationInfo.population),
      dataset: hubConfig.datasetLabel,
      series_type: "projection",
      hubverse_keys: {
        models: uniqueValuesInOrder(
          locationForecastRows.map((row) => String(row.model_id)),
        ),
        targets: uniqueValuesInOrder(
          locationForecastRows.map((row) => String(row.target)),
        ),
        horizons: uniqueValuesInOrder(
          locationForecastRows.map((row) => String(row.horizon)),
        ),
        output_types: uniqueValuesInOrder(
          locationForecastRows.map((row) => String(row.output_type)),
        ).filter((value) => value !== "sample"),
      },
    };

    if (hubConfig.slug === "flumetrocast") {
      metadata.state = String(locationInfo.state);
      metadata.state_abb = String(locationInfo.state_abb);
      metadata.location_type = String(locationInfo.location_type ?? "");
    }

    const forecastTargets = uniqueValuesInOrder(
      locationForecastRows.map((row) => String(row.target)),
    );
    const groundTruth = buildGroundTruthOutput(
      locationTargetRows,
      hubConfig,
      forecastTargets,
    );
    const forecasts = buildForecastOutput(locationForecastRows);
    const fileName =
      hubConfig.slug === "flumetrocast"
        ? `${String(locationInfo.location)}_${hubConfig.fileSuffix}.json`
        : `${metadata.abbreviation}_${hubConfig.fileSuffix}.json`;

    outputs[fileName] = {
      metadata,
      ground_truth: groundTruth,
      forecasts,
    };
  });

  const metadataFile = {
    last_updated: new Date().toISOString(),
    models: [
      ...uniqueValuesInOrder(forecastRows.map((row) => String(row.model_id))),
    ].sort(),
    locations: locationIds.map((locationId) => {
      const row = locationMap.get(locationId);
      if (hubConfig.slug === "flumetrocast") {
        return {
          location: String(row.original_location_code),
          abbreviation: String(row.location),
          location_name: String(row.location_name),
          population: Number(row.population),
          state: String(row.state),
          state_abb: String(row.state_abb),
          location_type: String(row.location_type ?? ""),
        };
      }

      return {
        location: String(row.location),
        abbreviation: String(row.abbreviation),
        location_name: String(row.location_name),
        population: Number(row.population),
      };
    }),
  };

  outputs["metadata.json"] = metadataFile;

  return outputs;
};

const getHubConfig = (slug) =>
  HUB_OPTIONS.find((hub) => hub.slug === slug) ?? null;

const ValidationSummary = ({ summary }) => (
  <Group gap="xs">
    <Badge color="blue" variant="light">
      {summary.totalRows} rows read
    </Badge>
    <Badge color="green" variant="light">
      {summary.usableRows} rows usable
    </Badge>
    {summary.rowsMissingHorizon > 0 && (
      <Badge color="yellow" variant="light">
        {summary.rowsMissingHorizon} missing horizon
      </Badge>
    )}
    {summary.rowsFilteredAsNowcasts > 0 && (
      <Badge color="yellow" variant="light">
        {summary.rowsFilteredAsNowcasts} nowcasts filtered
      </Badge>
    )}
    {summary.rowsFilteredAsSamples > 0 && (
      <Badge color="yellow" variant="light">
        {summary.rowsFilteredAsSamples} sample rows filtered
      </Badge>
    )}
    {summary.rowsFilteredByOutputTypeId > 0 && (
      <Badge color="yellow" variant="light">
        {summary.rowsFilteredByOutputTypeId} unsupported output_type_id
      </Badge>
    )}
    {summary.rowsFilteredAsPeakTargets > 0 && (
      <Badge color="yellow" variant="light">
        {summary.rowsFilteredAsPeakTargets} flu peak rows filtered
      </Badge>
    )}
    {summary.rowsFilteredAsDuplicates > 0 && (
      <Badge color="yellow" variant="light">
        {summary.rowsFilteredAsDuplicates} duplicate rows filtered
      </Badge>
    )}
  </Group>
);

// Same order as the hub's location picker: the default location (US) first,
// then the rest by name. The first option is therefore the default.
const buildLocationOptions = (projectionOutputs) =>
  Object.entries(projectionOutputs)
    .filter(([fileName]) => fileName !== "metadata.json")
    .map(([fileName, payload]) => {
      const abbreviation = payload?.metadata?.abbreviation;
      const isDefault = abbreviation === APP_CONFIG.defaultLocation;
      const locationName = isDefault
        ? "United States"
        : payload?.metadata?.location_name;
      return {
        value: fileName,
        isDefault,
        name: locationName || abbreviation || fileName,
        label:
          locationName && abbreviation && locationName !== abbreviation
            ? `${locationName} (${abbreviation})`
            : locationName || abbreviation || fileName,
      };
    })
    .sort((left, right) => {
      if (left.isDefault !== right.isDefault) return left.isDefault ? -1 : 1;
      return left.name.localeCompare(right.name);
    })
    .map(({ value, label }) => ({ value, label }));

const validateGroundTruthCsv = (
  records,
  { allowNaObservation = false } = {},
) => {
  const summary = {
    totalRows: records.length,
    rowsDroppedInvalidDates: 0,
    rowsDroppedInvalidObservation: 0,
    rowsKeptAsGaps: 0,
    rowsDroppedMissingLocation: 0,
    rowsDroppedMissingTarget: 0,
    rowsCollapsedToLatestAsOf: 0,
    rowsFlaggedAsDuplicatesWithoutAsOf: 0,
    usableRows: 0,
  };

  const errors = [];
  const sampleProblems = [];

  if (records.length === 0) {
    errors.push("The ground truth CSV has headers but no data rows.");
    return { ok: false, errors, summary };
  }

  const hasAsOfColumn = Boolean(records[0] && "as_of" in records[0]);
  const usableRows = records.flatMap((record, index) => {
    const normalizedTargetEndDate = normalizeDateString(record.target_end_date);
    const normalizedAsOf = hasAsOfColumn
      ? normalizeDateString(record.as_of)
      : null;
    const isNaObservation =
      allowNaObservation &&
      String(record.observation ?? "")
        .trim()
        .toUpperCase() === "NA";
    const observation = isNaObservation ? null : Number(record.observation);
    const location = String(record.location ?? "").trim();
    const target = String(record.target ?? "").trim();

    if (!location) {
      summary.rowsDroppedMissingLocation += 1;
      sampleProblems.push(`Row ${index + 2} is missing a location value.`);
      return [];
    }

    if (!target) {
      summary.rowsDroppedMissingTarget += 1;
      sampleProblems.push(`Row ${index + 2} is missing a target value.`);
      return [];
    }

    if (!normalizedTargetEndDate || (hasAsOfColumn && !normalizedAsOf)) {
      summary.rowsDroppedInvalidDates += 1;
      sampleProblems.push(
        hasAsOfColumn
          ? `Row ${index + 2} has an invalid as_of or target_end_date value.`
          : `Row ${index + 2} has an invalid target_end_date value.`,
      );
      return [];
    }

    if (Number.isNaN(observation)) {
      summary.rowsDroppedInvalidObservation += 1;
      sampleProblems.push(
        `Row ${index + 2} has a non-numeric observation of "${record.observation}".`,
      );
      return [];
    }

    if (isNaObservation) {
      summary.rowsKeptAsGaps += 1;
    }

    return [
      {
        ...record,
        as_of: normalizedAsOf,
        target_end_date: normalizedTargetEndDate,
        observation,
        location,
        target,
      },
    ];
  });

  if (sampleProblems.length > 0) {
    errors.push(...sampleProblems.slice(0, 5));
  }

  const latestByKey = new Map();
  if (hasAsOfColumn) {
    usableRows.forEach((row) => {
      const dedupeKey = `${row.location}__${row.target}__${row.target_end_date}`;
      const existing = latestByKey.get(dedupeKey);
      if (!existing || row.as_of >= existing.as_of) {
        latestByKey.set(dedupeKey, row);
      }
    });
    summary.rowsCollapsedToLatestAsOf = Math.max(
      0,
      usableRows.length - latestByKey.size,
    );
  } else {
    const seenKeys = new Set();
    const duplicateKeys = new Set();

    usableRows.forEach((row) => {
      const dedupeKey = `${row.location}__${row.target}__${row.target_end_date}`;
      if (seenKeys.has(dedupeKey)) {
        duplicateKeys.add(dedupeKey);
        return;
      }
      seenKeys.add(dedupeKey);
      latestByKey.set(dedupeKey, row);
    });

    if (duplicateKeys.size > 0) {
      summary.rowsFlaggedAsDuplicatesWithoutAsOf = duplicateKeys.size;
      errors.push(
        "Your ground truth file does not include an as_of column, so duplicate rows with the same location, target, and target_end_date are not allowed.",
      );
    }
  }
  summary.usableRows = latestByKey.size;

  if (latestByKey.size === 0) {
    errors.push("No usable ground truth rows remain after validation.");
  }

  return {
    ok: errors.length === 0,
    errors,
    summary,
    usableRows: Array.from(latestByKey.values()).sort((left, right) => {
      const leftLocation = left.location.localeCompare(right.location);
      if (leftLocation !== 0) return leftLocation;
      const leftDate = new Date(left.target_end_date).getTime();
      const rightDate = new Date(right.target_end_date).getTime();
      if (leftDate !== rightDate) return leftDate - rightDate;
      return left.target.localeCompare(right.target);
    }),
  };
};

const buildGroundTruthTargetWarning = (forecastRows, groundTruthRows) => {
  if (!forecastRows?.length || !groundTruthRows?.length) {
    return null;
  }

  const forecastTargets = uniqueValuesInOrder(
    forecastRows.map((row) => String(row.target ?? "").trim()).filter(Boolean),
  );
  const groundTruthTargets = new Set(
    groundTruthRows
      .map((row) => String(row.target ?? "").trim())
      .filter(Boolean),
  );
  const unmatchedTargets = forecastTargets.filter(
    (target) => !groundTruthTargets.has(target),
  );

  if (unmatchedTargets.length === 0) {
    return null;
  }

  return `Some forecast targets do not appear in your ground truth data: ${unmatchedTargets.join(", ")}. Please ensure your ground truth and forecast data match.`;
};

const buildMetroHierarchy = (projectionOutputs) => {
  const metadataLocations =
    projectionOutputs?.["metadata.json"]?.locations ?? [];
  const statesByAbbreviation = new Map();

  metadataLocations.forEach((location) => {
    const stateAbbreviation = String(location.state_abb ?? "");
    const stateName = String(location.state ?? "");
    if (!stateAbbreviation || !stateName) {
      return;
    }
    if (!statesByAbbreviation.has(stateAbbreviation)) {
      statesByAbbreviation.set(stateAbbreviation, {
        value: stateAbbreviation,
        label: stateName,
      });
    }
  });

  const stateOptions = [...statesByAbbreviation.values()].sort((left, right) =>
    left.label.localeCompare(right.label),
  );

  const locationsByState = {};

  stateOptions.forEach((state) => {
    const stateLocations = metadataLocations
      .filter((location) => location.state_abb === state.value)
      .sort((left, right) =>
        left.location_name.localeCompare(right.location_name),
      );

    const topLevelLocation = stateLocations.find(
      (location) => String(location.location_name ?? "") === state.label,
    );
    const childLocations = stateLocations.filter(
      (location) => location.abbreviation !== topLevelLocation?.abbreviation,
    );

    locationsByState[state.value] = [
      ...(topLevelLocation
        ? [
            {
              value: `${topLevelLocation.abbreviation}_flu_metrocast.json`,
              label: `${topLevelLocation.location_name} (statewide)`,
            },
          ]
        : []),
      ...childLocations.map((location) => ({
        value: `${location.abbreviation}_flu_metrocast.json`,
        label: location.location_name,
      })),
    ];
  });

  // The state of the hub's default MetroCast location, else the first state
  const defaultLocation = metadataLocations.find(
    (location) => location.abbreviation === DATASETS.metrocast.defaultLocation,
  );
  const defaultState =
    stateOptions.find((state) => state.value === defaultLocation?.state_abb)
      ?.value ??
    stateOptions[0]?.value ??
    null;

  return {
    stateOptions,
    defaultState,
    locationsByState,
  };
};

const getAllForecastDates = (projectionOutputs) => {
  const dateSet = new Set();
  Object.entries(projectionOutputs).forEach(([fileName, payload]) => {
    if (fileName === "metadata.json") return;
    Object.keys(payload?.forecasts || {}).forEach((date) => dateSet.add(date));
  });
  return [...dateSet].sort((left, right) => new Date(left) - new Date(right));
};

const getTargetOptions = (locationData) => {
  const targetSet = new Set();

  Object.values(locationData?.forecasts || {}).forEach((dateData) => {
    Object.keys(dateData || {}).forEach((target) => targetSet.add(target));
  });

  return [...targetSet]
    .map((target) => ({
      value: target,
      label: targetDisplayNameMap[target] || target,
    }))
    .sort((left, right) => left.label.localeCompare(right.label));
};

const getModelsForTarget = (locationData, target) => {
  if (!locationData?.forecasts || !target) return [];
  const modelSet = new Set();
  Object.values(locationData.forecasts).forEach((dateData) => {
    const targetData = dateData?.[target];
    if (!targetData) return;
    Object.keys(targetData).forEach((model) => modelSet.add(model));
  });
  return [...modelSet].sort();
};

const buildComparisonHoverText = ({
  model,
  pointDate,
  formattedMedian,
  formattedIntervals,
  issuedDate,
}) => {
  const rows = [`<b>Submitted model: ${model}</b><br>Date: ${pointDate}<br>`];

  if (formattedMedian !== null) {
    rows.push(`Median: <b>${formattedMedian}</b><br>`);
  }

  formattedIntervals.forEach((interval) => {
    rows.push(`${interval.label}: [${interval.formattedRange}]<br>`);
  });

  rows.push(
    `<span style="color: rgba(255,255,255,0.8); font-size: 0.8em">submitted as of ${issuedDate}</span><extra></extra>`,
  );

  return rows.join("");
};

const filterComparisonLocationData = (
  comparisonLocationData,
  userLocationData,
) => {
  if (!comparisonLocationData?.forecasts || !userLocationData?.forecasts) {
    return comparisonLocationData;
  }

  const filteredForecasts = {};

  Object.entries(userLocationData.forecasts).forEach(([date, targetData]) => {
    const comparisonDateData = comparisonLocationData.forecasts?.[date];
    if (!comparisonDateData) {
      return;
    }

    Object.keys(targetData || {}).forEach((target) => {
      const comparisonTargetData = comparisonDateData[target];
      if (!comparisonTargetData) {
        return;
      }

      filteredForecasts[date] ??= {};
      filteredForecasts[date][target] = comparisonTargetData;
    });
  });

  return {
    ...comparisonLocationData,
    forecasts: filteredForecasts,
  };
};

const getDefaultViewerRange = (
  groundTruthDates,
  selectedDates,
  forRangeslider = false,
) => {
  if (!groundTruthDates?.length || !selectedDates?.length) {
    return undefined;
  }

  const firstGroundTruthDate = new Date(groundTruthDates[0]);
  const lastGroundTruthDate = new Date(
    groundTruthDates[groundTruthDates.length - 1],
  );

  if (forRangeslider) {
    const sliderEnd = new Date(lastGroundTruthDate);
    sliderEnd.setDate(
      sliderEnd.getDate() + CHART_CONSTANTS.RANGESLIDER_WEEKS_AFTER * 7,
    );
    return [
      firstGroundTruthDate.toISOString().split("T")[0],
      sliderEnd.toISOString().split("T")[0],
    ];
  }

  const firstSelectedDate = new Date(selectedDates[0]);
  const lastSelectedDate = new Date(selectedDates[selectedDates.length - 1]);
  const rangeStart = new Date(firstSelectedDate);
  const rangeEnd = new Date(lastSelectedDate);
  rangeStart.setDate(
    rangeStart.getDate() - CHART_CONSTANTS.DEFAULT_WEEKS_BEFORE * 7,
  );
  rangeEnd.setDate(
    rangeEnd.getDate() + CHART_CONSTANTS.DEFAULT_WEEKS_AFTER * 7,
  );

  return [
    rangeStart.toISOString().split("T")[0],
    rangeEnd.toISOString().split("T")[0],
  ];
};

const SUBMITTED_INTERVAL_OPTIONS = [
  { value: "median", label: "Median" },
  { value: "ci50", label: "50% interval" },
  { value: "ci95", label: "95% interval" },
];

const MyRespiVisualizationPanel = ({
  projectionOutputs,
  hubConfig,
  comparisonEligibility,
}) => {
  const { colorScheme } = useMantineColorScheme();
  const isMetrocast = hubConfig?.slug === "flumetrocast";
  const comparisonEnabled = hubConfig?.slug !== "other-hub";
  const [selectedMetroState, setSelectedMetroState] = useState(null);
  const [selectedLocationFile, setSelectedLocationFile] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedDates, setSelectedDates] = useState([]);
  const [activeDate, setActiveDate] = useState(null);
  const [chartScale, setChartScale] = useState("linear");
  const normalizedChartScale = normalizeChartScale(chartScale);
  const [intervalVisibility, setIntervalVisibility] = useState({});
  const [submittedIntervalVisibility, setSubmittedIntervalVisibility] =
    useState({
      median: true,
      ci50: true,
      ci95: true,
    });
  const [showLegend, setShowLegend] = useState(true);
  const [xAxisRange, setXAxisRange] = useState(null);
  const [yAxisRange, setYAxisRange] = useState(null);
  // On by default for supported hubs, regardless of uploaded forecast dates.
  const [compareWithSubmittingModels, setCompareWithSubmittingModels] =
    useState(Boolean(comparisonEligibility?.isEligible));
  const [comparisonDataState, setComparisonDataState] = useState({
    status: "idle",
    data: null,
    error: null,
    locationFile: null,
  });
  const [preferredSubmittedModels, setPreferredSubmittedModels] = useState([]);
  const plotRef = useRef(null);
  const isResettingRef = useRef(false);
  const comparisonCacheRef = useRef(new Map());
  const userModelOrderRef = useRef([]);
  const submittedModelOrderRef = useRef([]);

  const locationOptions = useMemo(
    () => buildLocationOptions(projectionOutputs),
    [projectionOutputs],
  );
  const metroHierarchy = useMemo(
    () => (isMetrocast ? buildMetroHierarchy(projectionOutputs) : null),
    [isMetrocast, projectionOutputs],
  );

  useEffect(() => {
    if (!isMetrocast) {
      setSelectedMetroState(null);
      return;
    }

    const stateOptions = metroHierarchy?.stateOptions ?? [];
    if (!stateOptions.length) {
      setSelectedMetroState(null);
      return;
    }

    setSelectedMetroState((current) =>
      current && stateOptions.some((option) => option.value === current)
        ? current
        : (metroHierarchy?.defaultState ?? stateOptions[0].value),
    );
  }, [isMetrocast, metroHierarchy]);

  useEffect(() => {
    if (isMetrocast) {
      const scopedOptions =
        metroHierarchy?.locationsByState?.[selectedMetroState] ?? [];
      if (!scopedOptions.length) {
        setSelectedLocationFile(null);
        return;
      }
      setSelectedLocationFile((current) =>
        current && scopedOptions.some((option) => option.value === current)
          ? current
          : scopedOptions[0].value,
      );
      return;
    }

    if (!locationOptions.length) {
      setSelectedLocationFile(null);
      return;
    }
    setSelectedLocationFile((current) =>
      current && locationOptions.some((option) => option.value === current)
        ? current
        : locationOptions[0].value,
    );
  }, [isMetrocast, locationOptions, metroHierarchy, selectedMetroState]);

  const locationData = selectedLocationFile
    ? projectionOutputs[selectedLocationFile]
    : null;
  const scopedMetroLocationOptions = useMemo(
    () => metroHierarchy?.locationsByState?.[selectedMetroState] ?? [],
    [metroHierarchy, selectedMetroState],
  );

  useEffect(() => {
    setCompareWithSubmittingModels(Boolean(comparisonEligibility?.isEligible));
  }, [comparisonEligibility]);

  useEffect(() => {
    if (!compareWithSubmittingModels || !selectedLocationFile) {
      setComparisonDataState({
        status: "idle",
        data: null,
        error: null,
        locationFile: selectedLocationFile,
      });
      return;
    }

    const cachedData = comparisonCacheRef.current.get(selectedLocationFile);
    if (cachedData) {
      setComparisonDataState({
        status: "success",
        data: cachedData,
        error: null,
        locationFile: selectedLocationFile,
      });
      return;
    }

    let cancelled = false;
    const comparisonPath = `/processed_data/${hubConfig.slug}/${selectedLocationFile}`;

    setComparisonDataState({
      status: "loading",
      data: null,
      error: null,
      locationFile: selectedLocationFile,
    });

    fetchProjectionJson(comparisonPath)
      .then((data) => {
        if (cancelled) {
          return;
        }

        comparisonCacheRef.current.set(selectedLocationFile, data);
        setComparisonDataState({
          status: "success",
          data,
          error: null,
          locationFile: selectedLocationFile,
        });
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }

        setComparisonDataState({
          status: "error",
          data: null,
          error:
            error instanceof Error
              ? error.message
              : "Could not load submitted model comparison data.",
          locationFile: selectedLocationFile,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [compareWithSubmittingModels, hubConfig.slug, selectedLocationFile]);

  const comparisonLocationData =
    compareWithSubmittingModels &&
    comparisonDataState.status === "success" &&
    comparisonDataState.locationFile === selectedLocationFile
      ? comparisonDataState.data
      : null;
  const filteredComparisonLocationData = useMemo(
    () => filterComparisonLocationData(comparisonLocationData, locationData),
    [comparisonLocationData, locationData],
  );

  const targetOptions = useMemo(
    () => getTargetOptions(locationData),
    [locationData],
  );
  const { hasMedian, intervalDefinitions } = useMemo(
    () => getPredictionIntervalDefinitions(locationData, selectedTarget),
    [locationData, selectedTarget],
  );
  const intervalOptions = useMemo(
    () => [
      ...(hasMedian ? [{ value: "median", label: "Median" }] : []),
      ...intervalDefinitions.map((definition) => ({
        value: definition.key,
        label: definition.label,
      })),
    ],
    [hasMedian, intervalDefinitions],
  );

  useEffect(() => {
    if (!targetOptions.length) {
      setSelectedTarget(null);
      return;
    }
    setSelectedTarget((current) =>
      current && targetOptions.some((option) => option.value === current)
        ? current
        : targetOptions[0].value,
    );
  }, [targetOptions]);

  useEffect(() => {
    if (!intervalOptions.length) {
      setIntervalVisibility({});
      return;
    }

    setIntervalVisibility((current) => {
      const nextVisibility = {};
      intervalOptions.forEach((option) => {
        nextVisibility[option.value] =
          current?.[option.value] === undefined ? true : current[option.value];
      });
      return nextVisibility;
    });
  }, [intervalOptions]);

  const models = useMemo(
    () => getModelsForTarget(locationData, selectedTarget),
    [locationData, selectedTarget],
  );
  const submittedModels = useMemo(
    // Keep unavailable models visible; activeSubmittedModels disables them.
    () => getModelsForTarget(comparisonLocationData, selectedTarget),
    [comparisonLocationData, selectedTarget],
  );
  const selectedSubmittedIntervals = useMemo(
    () =>
      SUBMITTED_INTERVAL_OPTIONS.filter(
        (option) => submittedIntervalVisibility?.[option.value],
      ).map((option) => option.value),
    [submittedIntervalVisibility],
  );

  useEffect(() => {
    if (!models.length) {
      setSelectedModels([]);
      return;
    }
    setSelectedModels((current) => {
      const stillValid = current.filter((model) => models.includes(model));
      return stillValid.length ? stillValid : [models[0]];
    });
  }, [models]);

  const selectedSubmittedModels = useMemo(() => {
    if (!submittedModels.length) {
      return [];
    }

    return preferredSubmittedModels.filter((model) =>
      submittedModels.includes(model),
    );
  }, [preferredSubmittedModels, submittedModels]);

  const handleSubmittedModelSelectionChange = useCallback((nextModels) => {
    setPreferredSubmittedModels(nextModels);
  }, []);

  const stableSubmittedModelOrder = useMemo(() => {
    const nextOrder = extendStableModelOrder(
      submittedModelOrderRef.current,
      selectedSubmittedModels,
    );
    submittedModelOrderRef.current = nextOrder;
    return nextOrder;
  }, [selectedSubmittedModels]);

  const stableUserModelOrder = useMemo(() => {
    const nextOrder = extendStableModelOrder(
      userModelOrderRef.current,
      selectedModels,
    );
    userModelOrderRef.current = nextOrder;
    return nextOrder;
  }, [selectedModels]);

  const submittedModelPalette = useMemo(() => {
    const reservedColors = new Set(
      selectedModels
        .map((model) => {
          const index = stableUserModelOrder.indexOf(model);
          if (index < 0) {
            return null;
          }

          return MODEL_COLORS[index % MODEL_COLORS.length];
        })
        .filter(Boolean),
    );

    const availableColors = MODEL_COLORS.filter(
      (color) => !reservedColors.has(color),
    );

    return availableColors.length ? availableColors : MODEL_COLORS;
  }, [selectedModels, stableUserModelOrder]);

  const submittedModelColorFn = useCallback(
    (model) => {
      const index = stableSubmittedModelOrder.indexOf(model);
      if (index < 0) {
        return null;
      }

      return submittedModelPalette[index % submittedModelPalette.length];
    },
    [stableSubmittedModelOrder, submittedModelPalette],
  );

  const availableDates = useMemo(
    () => getAllForecastDates(projectionOutputs),
    [projectionOutputs],
  );

  useEffect(() => {
    if (!availableDates.length) {
      setSelectedDates([]);
      setActiveDate(null);
      return;
    }
    const latestDate = availableDates[availableDates.length - 1];
    setSelectedDates((current) =>
      current.length
        ? current.filter((date) => availableDates.includes(date))
        : [latestDate],
    );
    setActiveDate((current) =>
      current && availableDates.includes(current) ? current : latestDate,
    );
  }, [availableDates]);

  const activeModels = useMemo(() => {
    const activeModelSet = new Set();
    if (!locationData?.forecasts || !selectedTarget || !selectedDates.length) {
      return activeModelSet;
    }

    selectedDates.forEach((date) => {
      const targetData = locationData.forecasts?.[date]?.[selectedTarget];
      if (!targetData) return;
      Object.keys(targetData).forEach((model) => activeModelSet.add(model));
    });

    return activeModelSet;
  }, [locationData, selectedTarget, selectedDates]);
  const activeSubmittedModels = useMemo(() => {
    const activeModelSet = new Set();
    if (
      !filteredComparisonLocationData?.forecasts ||
      !selectedTarget ||
      !selectedDates.length
    ) {
      return activeModelSet;
    }

    selectedDates.forEach((date) => {
      const targetData =
        filteredComparisonLocationData.forecasts?.[date]?.[selectedTarget];
      if (!targetData) return;
      Object.keys(targetData).forEach((model) => activeModelSet.add(model));
    });

    return activeModelSet;
  }, [filteredComparisonLocationData, selectedTarget, selectedDates]);

  const manualScaleTransform = useMemo(() => {
    if (normalizedChartScale !== "sqrt" && normalizedChartScale !== "log2") {
      return null;
    }
    return (value) => transformValueForScale(value, normalizedChartScale);
  }, [normalizedChartScale]);

  const { traces, rawYRange } = useQuantileForecastTraces({
    groundTruth: locationData?.ground_truth,
    forecasts: locationData?.forecasts,
    selectedDates,
    selectedModels,
    target: selectedTarget,
    showLegendForFirstDate: showLegend,
    groundTruthLineWidth: GROUND_TRUTH_LINE_WIDTH,
    groundTruthMarkerSize: GROUND_TRUTH_MARKER_SIZE,
    groundTruthColor: getChartInk(colorScheme).text,
    showMedian: intervalVisibility.median ?? hasMedian,
    fillMissingQuantiles: false,
    intervalDefinitions,
    intervalVisibility,
    transformY: manualScaleTransform,
  });
  const {
    traces: submittedModelTraceBundle,
    rawYRange: submittedModelRawYRange,
  } = useQuantileForecastTraces({
    groundTruth: locationData?.ground_truth,
    forecasts: filteredComparisonLocationData?.forecasts,
    selectedDates,
    selectedModels: selectedSubmittedModels,
    target: selectedTarget,
    showLegendForFirstDate: showLegend,
    showMedian: submittedIntervalVisibility.median,
    show50: submittedIntervalVisibility.ci50,
    show95: submittedIntervalVisibility.ci95,
    intervalVisibility: submittedIntervalVisibility,
    modelColorFn: submittedModelColorFn,
    modelOrder: stableSubmittedModelOrder,
    modelHoverBuilder: buildComparisonHoverText,
    transformY: manualScaleTransform,
  });
  const submittedModelTraces = useMemo(
    () => submittedModelTraceBundle.slice(1),
    [submittedModelTraceBundle],
  );
  const allTraces = useMemo(
    () => [...traces, ...submittedModelTraces],
    [submittedModelTraces, traces],
  );
  const combinedRawYRange = useMemo(() => {
    const ranges = [rawYRange, submittedModelRawYRange].filter(Boolean);
    if (!ranges.length) {
      return null;
    }

    return [
      Math.min(...ranges.map((range) => range[0])),
      Math.max(...ranges.map((range) => range[1])),
    ];
  }, [rawYRange, submittedModelRawYRange]);

  const calculateYRange = useCallback((chartData, currentXRange) => {
    if (
      !chartData ||
      !currentXRange ||
      !Array.isArray(chartData) ||
      chartData.length === 0
    ) {
      return null;
    }

    const [startX, endX] = currentXRange;
    const startDate = new Date(startX);
    const endDate = new Date(endX);
    let minY = Infinity;
    let maxY = -Infinity;

    chartData.forEach((trace) => {
      if (!trace?.x || !trace?.y) return;
      trace.x.forEach((xValue, index) => {
        const pointDate = new Date(xValue);
        const yValue = Number(trace.y[index]);
        if (
          pointDate >= startDate &&
          pointDate <= endDate &&
          !Number.isNaN(yValue)
        ) {
          minY = Math.min(minY, yValue);
          maxY = Math.max(maxY, yValue);
        }
      });
    });

    if (minY === Infinity || maxY === -Infinity) return null;
    const padding = maxY * (CHART_CONSTANTS.Y_AXIS_PADDING_PERCENT / 100);
    return [Math.max(0, minY - padding), maxY + padding];
  }, []);

  const defaultRange = useMemo(
    () =>
      getDefaultViewerRange(
        locationData?.ground_truth?.dates,
        selectedDates,
        false,
      ),
    [locationData, selectedDates],
  );

  useEffect(() => {
    setXAxisRange(null);
  }, [selectedLocationFile, selectedTarget]);

  useEffect(() => {
    if (isPlotlyLogScale(normalizedChartScale)) {
      setYAxisRange(null);
      return;
    }
    const currentRange = xAxisRange || defaultRange;
    if (allTraces.length > 0 && currentRange) {
      setYAxisRange(calculateYRange(allTraces, currentRange));
    } else {
      setYAxisRange(null);
    }
  }, [
    allTraces,
    xAxisRange,
    defaultRange,
    calculateYRange,
    normalizedChartScale,
  ]);

  const handlePlotUpdate = useCallback((figure) => {
    if (isResettingRef.current) {
      isResettingRef.current = false;
      return;
    }
    if (figure?.["xaxis.range"]) {
      setXAxisRange(figure["xaxis.range"]);
    }
  }, []);

  const layout = useMemo(
    () => ({
      ...getBaseChartLayout(colorScheme),
      showlegend: showLegend,
      hovermode: "closest",
      dragmode: false,
      margin: { l: 60, r: 30, t: 30, b: 30 },
      xaxis: {
        ...getBaseChartLayout(colorScheme).xaxis,
        rangeslider: {
          ...RANGESLIDER_STYLE,
          range: getDefaultViewerRange(
            locationData?.ground_truth?.dates,
            selectedDates,
            true,
          ),
        },
        range: xAxisRange || defaultRange,
      },
      yaxis: {
        ...getBaseChartLayout(colorScheme).yaxis,
        ...getScaleYAxis({
          scale: normalizedChartScale,
          rawRange: combinedRawYRange,
          range: yAxisRange,
          title:
            targetDisplayNameMap[selectedTarget] || selectedTarget || "Value",
        }),
      },
      shapes: selectedDates.map((date) => {
        const shiftedDate = shiftDateStringByDays(date, -3);
        return {
          type: "line",
          x0: shiftedDate,
          x1: shiftedDate,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: getForecastDateLineStyle(colorScheme),
        };
      }),
    }),
    [
      colorScheme,
      showLegend,
      locationData,
      selectedDates,
      xAxisRange,
      defaultRange,
      selectedTarget,
      normalizedChartScale,
      yAxisRange,
      combinedRawYRange,
    ],
  );

  const handleResetView = () => {
    const gd = plotRef.current?.el;
    if (!gd) return;
    const range = getDefaultViewerRange(
      locationData?.ground_truth?.dates,
      selectedDates,
      false,
    );
    const nextYRange =
      isPlotlyLogScale(normalizedChartScale) || !range
        ? null
        : calculateYRange(allTraces, range);
    isResettingRef.current = true;
    setXAxisRange(null);
    setYAxisRange(nextYRange);
    Plotly.relayout(gd, {
      "xaxis.range": range,
      "yaxis.range": nextYRange,
      "yaxis.autorange":
        isPlotlyLogScale(normalizedChartScale) || nextYRange === null,
    });
  };

  // Same PNG as the hub views' chart header (the Plotly toolbar is hidden)
  const handleDownload = () =>
    downloadChartPng(plotRef.current?.el, "forecast-checker");

  useKeyboardShortcut("r", handleResetView);
  useKeyboardShortcut("d", handleDownload);
  // I: every interval, then 95% + 50% + median, 50% + median, median alone
  useKeyboardShortcut("i", () =>
    setIntervalVisibility((current) => {
      const keys = intervalOptions.map((option) => option.value);
      const keyForWidth = (lower) =>
        intervalDefinitions.find(
          (definition) => Math.abs(definition.lowerQuantile - lower) < 1e-9,
        )?.key;
      const ci95 = keyForWidth(0.025);
      const ci50 = keyForWidth(0.25);
      return nextIntervalVisibility(current, keys, [
        keys,
        ["median", ci50, ci95],
        ["median", ci50],
        ["median"],
      ]);
    }),
  );

  if ((!isMetrocast && !locationOptions.length) || !locationData) {
    return null;
  }

  const isComparing =
    compareWithSubmittingModels && comparisonDataState.status === "success";

  // Same layout as the forecast pages: the title sentence holds the
  // pickers, display options and actions are quiet icons on the right
  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="md">
        <Stack gap={6} style={{ minWidth: 0 }}>
          <Title
            order={2}
            fz={{ base: 20, sm: 24 }}
            fw={600}
            lh={1.35}
            style={{ textWrap: "balance" }}
          >
            <InlinePicker
              value={selectedTarget}
              data={targetOptions}
              onChange={setSelectedTarget}
              dropdownWidth={360}
              shortcut="t"
              shortcutLabel="Change target"
              aria-label="Select target"
            />{" "}
            <Text span inherit c="dimmed" fw={400}>
              in
            </Text>{" "}
            {isMetrocast && (
              <>
                <InlinePicker
                  value={selectedMetroState}
                  data={metroHierarchy?.stateOptions ?? []}
                  onChange={setSelectedMetroState}
                  searchable
                  shortcut="l"
                  shortcutLabel="Change state"
                  stepKeys
                  aria-label="Select state"
                />
                <Text span inherit c="dimmed" fw={400}>
                  ,
                </Text>{" "}
              </>
            )}
            <InlinePicker
              value={selectedLocationFile}
              data={isMetrocast ? scopedMetroLocationOptions : locationOptions}
              onChange={setSelectedLocationFile}
              searchable
              shortcut={isMetrocast ? null : "l"}
              shortcutLabel="Change location"
              stepKeys={isMetrocast ? "first" : true}
              aria-label="Select location"
            />
          </Title>
          <Group gap={6} wrap="nowrap" align="flex-start" fz="sm" c="dimmed">
            <Text span size="sm" c="dimmed" style={{ whiteSpace: "nowrap" }}>
              Forecast date
            </Text>
            <DateSelector
              compact
              availableDates={availableDates}
              selectedDates={selectedDates}
              setSelectedDates={setSelectedDates}
              activeDate={activeDate}
              setActiveDate={setActiveDate}
            />
          </Group>
        </Stack>

        <Group gap={2} wrap="nowrap">
          {comparisonEnabled && (
            <Tooltip
              label={comparisonEligibility?.reason}
              disabled={
                comparisonEligibility?.isEligible || !comparisonEligibility
              }
              multiline
              w={260}
            >
              <div style={{ marginRight: 8 }}>
                <Switch
                  label="Compare with hub models"
                  checked={compareWithSubmittingModels}
                  onChange={(event) =>
                    setCompareWithSubmittingModels(event.currentTarget.checked)
                  }
                  disabled={!comparisonEligibility?.isEligible}
                  size="sm"
                />
              </div>
            </Tooltip>
          )}
          <Popover position="bottom-end" shadow="md" width={380}>
            <Popover.Target>
              <Tooltip label="Display options" openDelay={300}>
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="lg"
                  aria-label="Display options"
                >
                  <IconAdjustmentsHorizontal size={18} />
                </ActionIcon>
              </Tooltip>
            </Popover.Target>
            <Popover.Dropdown>
              <Stack gap="md">
                <ForecastChartControls
                  chartScale={chartScale}
                  setChartScale={setChartScale}
                  intervalVisibility={intervalVisibility}
                  setIntervalVisibility={setIntervalVisibility}
                  showLegend={showLegend}
                  setShowLegend={setShowLegend}
                  intervalOptions={intervalOptions}
                />
                {comparisonEnabled && isComparing && (
                  <Group align="center" gap="md" wrap="wrap">
                    <Text size="xs" c="dimmed">
                      Hub model intervals
                    </Text>
                    <Checkbox.Group
                      value={selectedSubmittedIntervals}
                      onChange={(values) => {
                        const nextVisibility = {};
                        SUBMITTED_INTERVAL_OPTIONS.forEach((option) => {
                          nextVisibility[option.value] = values.includes(
                            option.value,
                          );
                        });
                        setSubmittedIntervalVisibility(nextVisibility);
                      }}
                    >
                      <Group gap="sm" wrap="wrap">
                        {SUBMITTED_INTERVAL_OPTIONS.map((option) => (
                          <Checkbox
                            key={option.value}
                            value={option.value}
                            label={option.label}
                            size="xs"
                          />
                        ))}
                      </Group>
                    </Checkbox.Group>
                  </Group>
                )}
              </Stack>
            </Popover.Dropdown>
          </Popover>
          <Tooltip
            label={<ShortcutHint label="Download chart as PNG" shortcut="d" />}
            openDelay={300}
          >
            <ActionIcon
              variant="subtle"
              size="lg"
              color="gray"
              onClick={handleDownload}
              aria-label="Download chart as PNG"
            >
              <IconDownload size={18} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label={<ShortcutHint label="Reset axes" shortcut="r" />}>
            <ActionIcon
              variant="subtle"
              size="lg"
              color="gray"
              onClick={handleResetView}
              aria-label="Reset axes"
            >
              <IconZoomReset size={18} />
            </ActionIcon>
          </Tooltip>
          <ChartInfoButton
            dragDates={false}
            hasHubModels={compareWithSubmittingModels}
          />
        </Group>
      </Group>

      {compareWithSubmittingModels &&
        comparisonDataState.status === "loading" && (
          <Group gap="xs">
            <Loader size="sm" color="blue" />
            <Text size="sm" c="dimmed">
              Loading submitted model data for this location...
            </Text>
          </Group>
        )}
      {compareWithSubmittingModels &&
        comparisonDataState.status === "error" && (
          <Alert
            color="red"
            variant="light"
            radius="md"
            icon={<IconAlertCircle size={16} />}
          >
            {comparisonDataState.error}
          </Alert>
        )}

      <div
        style={{
          width: "100%",
          height: "min(780px, 66vh)",
          minHeight: 320,
        }}
      >
        <Plot
          ref={plotRef}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          data={allTraces}
          layout={layout}
          config={PLOT_CONFIG}
          onRelayout={handlePlotUpdate}
        />
      </div>

      <ModelSelector
        title={comparisonEnabled ? "User submitted models" : "Models"}
        models={models}
        selectedModels={selectedModels}
        setSelectedModels={setSelectedModels}
        activeModels={activeModels}
        keyboardShortcut
      />

      {compareWithSubmittingModels && (
        <ModelSelector
          title="Hub models"
          keyboardShortcut="M"
          models={submittedModels}
          selectedModels={selectedSubmittedModels}
          setSelectedModels={handleSubmittedModelSelectionChange}
          activeModels={activeSubmittedModels}
          modelColorFn={submittedModelColorFn}
        />
      )}
    </Stack>
  );
};

const HubSelectionScreen = () => {
  const navigate = useNavigate();
  const [opened, { toggle }] = useDisclosure(false);
  const [hoveredHub, setHoveredHub] = useState(null);

  return (
    <>
      <Seo
        title="RespiLens | Forecast Checker"
        description="Validate and prepare Hubverse forecast CSV files for use in Forecast Checker."
        canonicalPath="/toolbox/forecast-checker"
      />
      <Container size="xl" pt="md" pb="xl" style={{ maxWidth: "1500px" }}>
        <Stack gap="xl" maw={1320}>
          <Stack gap={4}>
            <Group gap="xs" align="center" wrap="nowrap">
              <Tooltip label="Back to toolbox" withArrow>
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  onClick={() => navigate("/toolbox")}
                  aria-label="Back to toolbox"
                >
                  <IconArrowLeft size={18} />
                </ActionIcon>
              </Tooltip>
              <Title order={1}>Forecast Checker</Title>
            </Group>
            <Text size="sm" c="dimmed">
              Select a hub and drop your data for instant visualization.
            </Text>
          </Stack>

          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3, xl: 5 }} spacing="lg">
            {[...HUB_OPTIONS, ...EXTRA_HUB_OPTIONS].map((hub) => {
              const HubIcon = hub.icon ?? IconBrandGithub;

              return (
                <Paper
                  key={hub.slug}
                  withBorder
                  radius="xl"
                  p="xl"
                  onMouseEnter={() => setHoveredHub(hub.slug)}
                  onMouseLeave={() => setHoveredHub(null)}
                  onClick={() =>
                    navigate(`/toolbox/forecast-checker/${hub.slug}`)
                  }
                  style={{
                    aspectRatio: "1 / 1",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    textAlign: "center",
                    transform:
                      hoveredHub === hub.slug
                        ? "translateY(-6px)"
                        : "translateY(0)",
                    boxShadow:
                      hoveredHub === hub.slug
                        ? "0 14px 30px rgba(37, 99, 235, 0.18)"
                        : "0 4px 12px rgba(15, 23, 42, 0.06)",
                    borderColor:
                      hoveredHub === hub.slug
                        ? "var(--mantine-color-blue-4)"
                        : undefined,
                    transition: "transform 160ms ease, box-shadow 160ms ease",
                  }}
                >
                  <Stack align="center" gap="sm">
                    <ThemeIcon
                      size={56}
                      radius="xl"
                      variant="light"
                      color="blue"
                    >
                      <HubIcon size={28} />
                    </ThemeIcon>
                    <Text fw={700} size="lg" tt="none">
                      {hub.label}
                    </Text>
                  </Stack>
                </Paper>
              );
            })}
          </SimpleGrid>

          <Group justify="center">
            <Button
              variant="light"
              color="blue"
              leftSection={<IconInfoCircle size={16} />}
              onClick={toggle}
            >
              What is Forecast Checker?
            </Button>
            <Button
              component="a"
              href="/toolbox/forecast-checker/documentation"
              target="_blank"
              variant="light"
              color="blue"
              leftSection={<IconFileText size={16} />}
            >
              View documentation
            </Button>
          </Group>

          {opened && (
            <Alert
              icon={<IconInfoCircle size={16} />}
              title="About Forecast Checker"
              color="blue"
              radius="lg"
            >
              Forecast Checker is a tool that allows you to visualize your own
              respiratory disease forecast data <b>instantly</b> and{" "}
              <b>privately.</b> Simply <b>select the hub</b> your data
              corresponds to, <b>drag 'n drop your forecast data</b>, and then{" "}
              <b>interact with your personalized visualization dashboard</b>.
              Your data will not leave your machine, which means that your
              visualizations are not shareable via URL. User-provided model data
              must be in the Hubverse <code>.csv</code> format in order to be
              visualized with Forecast Checker.
            </Alert>
          )}
        </Stack>
      </Container>
    </>
  );
};

const OtherHubScreen = () => {
  const navigate = useNavigate();
  const [
    groundTruthRequirementsOpened,
    { toggle: toggleGroundTruthRequirements },
  ] = useDisclosure(false);
  const [groundTruthDragActive, setGroundTruthDragActive] = useState(false);
  const [forecastDragActive, setForecastDragActive] = useState(false);
  const [isGroundTruthProcessing, setIsGroundTruthProcessing] = useState(false);
  const [isForecastProcessing, setIsForecastProcessing] = useState(false);
  const [groundTruthState, setGroundTruthState] = useState({
    status: "idle",
    fileName: null,
    rows: null,
    summary: null,
    error: null,
  });
  const [validationState, setValidationState] = useState(null);
  const [projectionBuildState, setProjectionBuildState] = useState({
    status: "idle",
    outputs: null,
    error: null,
    comparisonEligibility: null,
  });
  const [groundTruthTargetWarning, setGroundTruthTargetWarning] =
    useState(null);

  const isShowingVisualization = projectionBuildState.status === "success";
  const isOnForecastStep =
    groundTruthState.status === "success" && !isShowingVisualization;
  const backArrowLabel = isShowingVisualization
    ? "Back to forecast upload"
    : isOnForecastStep
      ? "Back to ground truth upload"
      : "Back to hub selection";

  const handleResetForecastStep = useCallback(() => {
    setForecastDragActive(false);
    setIsForecastProcessing(false);
    setValidationState(null);
    setProjectionBuildState({
      status: "idle",
      outputs: null,
      error: null,
      comparisonEligibility: null,
    });
  }, []);

  const handleResetGroundTruthStep = useCallback(() => {
    setGroundTruthDragActive(false);
    setIsGroundTruthProcessing(false);
    setGroundTruthState({
      status: "idle",
      fileName: null,
      rows: null,
      summary: null,
      error: null,
    });
    setGroundTruthTargetWarning(null);
    handleResetForecastStep();
  }, [handleResetForecastStep]);

  const handleBackArrowClick = useCallback(() => {
    if (isShowingVisualization) {
      handleResetForecastStep();
      return;
    }

    if (isOnForecastStep) {
      handleResetGroundTruthStep();
      return;
    }

    navigate("/toolbox/forecast-checker");
  }, [
    handleResetForecastStep,
    handleResetGroundTruthStep,
    isOnForecastStep,
    isShowingVisualization,
    navigate,
  ]);

  const processGroundTruthFiles = useCallback(async (incomingFiles) => {
    setIsGroundTruthProcessing(true);
    const files = Array.from(incomingFiles ?? []);

    if (files.length !== 1) {
      setGroundTruthState({
        status: "error",
        fileName: null,
        rows: null,
        summary: null,
        error: "Please upload exactly one ground truth file at a time.",
      });
      setIsGroundTruthProcessing(false);
      return;
    }

    const [file] = files;

    try {
      const { headers, records } = await readTabularUpload(file);
      const missingColumns = GROUND_TRUTH_REQUIRED_COLUMNS.filter(
        (column) => !headers.includes(column),
      );

      if (missingColumns.length > 0) {
        setGroundTruthState({
          status: "error",
          fileName: file.name,
          rows: null,
          summary: null,
          error: buildRequiredColumnError("ground truth data", missingColumns),
        });
        setIsGroundTruthProcessing(false);
        return;
      }

      const validation = validateGroundTruthCsv(records, {
        allowNaObservation: true,
      });
      if (!validation.ok) {
        setGroundTruthState({
          status: "error",
          fileName: file.name,
          rows: null,
          summary: validation.summary,
          error: validation.errors.join(" "),
        });
        setIsGroundTruthProcessing(false);
        return;
      }

      setGroundTruthState({
        status: "success",
        fileName: file.name,
        rows: validation.usableRows,
        summary: validation.summary,
        error: null,
      });
      setGroundTruthTargetWarning(null);
      setValidationState(null);
      setProjectionBuildState({
        status: "idle",
        outputs: null,
        error: null,
        comparisonEligibility: null,
      });
    } catch (error) {
      setGroundTruthState({
        status: "error",
        fileName: file.name,
        rows: null,
        summary: null,
        error:
          error instanceof Error
            ? error.message
            : "The ground truth file could not be processed.",
      });
    } finally {
      setIsGroundTruthProcessing(false);
    }
  }, []);

  const processForecastFiles = useCallback(
    async (incomingFiles) => {
      if (!groundTruthState.rows) {
        return;
      }

      setIsForecastProcessing(true);
      const files = Array.from(incomingFiles ?? []);
      const csvFiles = files.filter((file) =>
        file.name.toLowerCase().endsWith(".csv"),
      );

      if (csvFiles.length === 0) {
        setValidationState({
          status: "error",
          errors: [
            "Please upload at least one Hubverse forecast file in `.csv` format.",
          ],
        });
        setProjectionBuildState({
          status: "idle",
          outputs: null,
          error: null,
          comparisonEligibility: null,
        });
        setIsForecastProcessing(false);
        return;
      }

      setValidationState(null);
      setProjectionBuildState({
        status: "idle",
        outputs: null,
        error: null,
        comparisonEligibility: null,
      });

      try {
        const parsedFiles = await Promise.all(
          csvFiles.map(async (file) => {
            const text = await file.text();
            const rows = parseCsv(text);
            const { headers, records } = toObjects(rows);
            return {
              fileName: getFileRelativePath(file),
              headers,
              records,
            };
          }),
        );

        const missingColumnsByFile = parsedFiles
          .map((parsedFile) => ({
            fileName: parsedFile.fileName,
            missingColumns: FORECAST_REQUIRED_COLUMNS.filter(
              (column) => !parsedFile.headers.includes(column),
            ),
          }))
          .filter((entry) => entry.missingColumns.length > 0);

        if (missingColumnsByFile.length > 0) {
          setValidationState({
            status: "error",
            errors: missingColumnsByFile
              .slice(0, 5)
              .map(
                (entry) =>
                  `${entry.fileName} is missing required forecast columns: ${entry.missingColumns.join(", ")}.`,
              ),
          });
          setIsForecastProcessing(false);
          return;
        }

        const concatenatedRecords = parsedFiles.flatMap(
          (parsedFile) => parsedFile.records,
        );
        const validation = validateHubverseCsv(
          concatenatedRecords,
          OTHER_HUB_CONFIG,
        );

        if (!validation.ok) {
          setValidationState({
            status: "error",
            errors: validation.errors,
            summary: validation.summary,
          });
          setIsForecastProcessing(false);
          return;
        }

        setValidationState({
          status: "success",
          summary: validation.summary,
        });

        const outputs = buildProjectionOutputs({
          hubConfig: OTHER_HUB_CONFIG,
          forecastRows: validation.usableRows,
          locationsRows: null,
          targetRows: groundTruthState.rows,
        });
        setGroundTruthTargetWarning(
          buildGroundTruthTargetWarning(
            validation.usableRows,
            groundTruthState.rows,
          ),
        );

        setProjectionBuildState({
          status: "success",
          outputs,
          error: null,
          comparisonEligibility: {
            isEligible: false,
            reason:
              "Comparison with submitting models is not available for custom hubs.",
          },
        });
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "The forecast file could not be processed.";

        setProjectionBuildState({
          status: "error",
          outputs: null,
          error: message,
          comparisonEligibility: null,
        });
        setValidationState({
          status: "error",
          errors: [message],
        });
      } finally {
        setIsForecastProcessing(false);
      }
    },
    [groundTruthState.rows],
  );

  const handleGroundTruthDrop = useCallback(
    async (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (isGroundTruthProcessing) {
        return;
      }
      setGroundTruthDragActive(false);
      const droppedFiles = await collectDroppedFiles(event.dataTransfer);
      if (droppedFiles.length > 0) {
        processGroundTruthFiles(droppedFiles);
      }
    },
    [isGroundTruthProcessing, processGroundTruthFiles],
  );

  const handleGroundTruthFileSelect = useCallback(
    (event) => {
      if (isGroundTruthProcessing) {
        event.target.value = "";
        return;
      }
      if (event.target.files?.length) {
        processGroundTruthFiles(event.target.files);
      }
      event.target.value = "";
    },
    [isGroundTruthProcessing, processGroundTruthFiles],
  );

  const handleForecastDrop = useCallback(
    async (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (isForecastProcessing) {
        return;
      }
      setForecastDragActive(false);
      const droppedFiles = await collectDroppedFiles(event.dataTransfer);
      if (droppedFiles.length > 0) {
        processForecastFiles(droppedFiles);
      }
    },
    [isForecastProcessing, processForecastFiles],
  );

  const handleForecastFileSelect = useCallback(
    (event) => {
      if (isForecastProcessing) {
        event.target.value = "";
        return;
      }
      if (event.target.files?.length) {
        processForecastFiles(event.target.files);
      }
      event.target.value = "";
    },
    [isForecastProcessing, processForecastFiles],
  );

  return (
    <>
      <Seo
        title="RespiLens | Forecast Checker | Other Hub"
        description="Provide ground truth and forecast data for a hub that is not currently listed in Forecast Checker."
        canonicalPath="/toolbox/forecast-checker/other-hub"
      />
      <Container size="xl" pt="md" pb="xl" fluid>
        <Stack gap="lg">
          {isShowingVisualization ? (
            <Group justify="space-between" align="center">
              <Group gap="xs" align="center">
                <Tooltip label={backArrowLabel} withArrow>
                  <ActionIcon
                    variant="subtle"
                    color="blue"
                    size="xl"
                    radius="xl"
                    onClick={handleBackArrowClick}
                    aria-label={backArrowLabel}
                  >
                    <IconArrowLeft size={24} stroke={2.25} />
                  </ActionIcon>
                </Tooltip>
                <Title order={1}>Other hub</Title>
              </Group>
            </Group>
          ) : (
            <Group justify="center">
              <Box w="100%" maw={860}>
                <Group justify="space-between" align="center">
                  <Group gap="xs" align="center">
                    <Tooltip label={backArrowLabel} withArrow>
                      <ActionIcon
                        variant="subtle"
                        color="blue"
                        size="xl"
                        radius="xl"
                        onClick={handleBackArrowClick}
                        aria-label={backArrowLabel}
                      >
                        <IconArrowLeft size={24} stroke={2.25} />
                      </ActionIcon>
                    </Tooltip>
                    <Title order={1}>Other hub</Title>
                  </Group>
                </Group>
              </Box>
            </Group>
          )}

          {isShowingVisualization && groundTruthTargetWarning && (
            <Alert
              color="red"
              variant="light"
              radius="lg"
              icon={<IconInfoCircle size={16} />}
              title="Possible target mismatch"
            >
              {groundTruthTargetWarning}
            </Alert>
          )}

          {isShowingVisualization && (
            <MyRespiVisualizationPanel
              projectionOutputs={projectionBuildState.outputs}
              hubConfig={OTHER_HUB_CONFIG}
              comparisonEligibility={projectionBuildState.comparisonEligibility}
            />
          )}

          {!isShowingVisualization && (
            <Group justify="center">
              <Box w="100%" maw={860}>
                <Stack gap="md">
                  <Text size="lg">
                    If you wish to check forecasts for a hub that is not listed
                    on RespiLens, you may do so by:
                  </Text>
                  <List type="ordered" spacing="xs">
                    <List.Item>
                      <b>Providing the corresponding ground truth data</b>
                    </List.Item>
                    <List.Item>
                      <b>Providing your forecast data</b>
                    </List.Item>
                  </List>
                  <Text size="lg">
                    For ground truth data requirements, see below. For forecast
                    data requirements, see the Forecast Checker{" "}
                    <Anchor
                      href="/toolbox/forecast-checker/documentation"
                      target="_blank"
                      rel="noreferrer"
                    >
                      documentation
                    </Anchor>{" "}
                    page.
                  </Text>

                  {!isOnForecastStep && (
                    <Stack gap="md">
                      <Paper
                        withBorder
                        radius="xl"
                        p="xl"
                        onDragEnter={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          if (isGroundTruthProcessing) {
                            return;
                          }
                          setGroundTruthDragActive(true);
                        }}
                        onDragLeave={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          setGroundTruthDragActive(false);
                        }}
                        onDragOver={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                        }}
                        onDrop={handleGroundTruthDrop}
                        onClick={() => {
                          if (isGroundTruthProcessing) {
                            return;
                          }
                          document
                            .getElementById(
                              "forecast-checker-other-ground-truth-input",
                            )
                            ?.click();
                        }}
                        style={{
                          cursor: isGroundTruthProcessing
                            ? "progress"
                            : "pointer",
                          border: groundTruthDragActive
                            ? "2px dashed var(--mantine-color-blue-6)"
                            : "2px dashed var(--mantine-color-gray-4)",
                          backgroundColor: groundTruthDragActive
                            ? "var(--mantine-color-blue-light)"
                            : "transparent",
                          transition:
                            "border-color 160ms ease, background-color 160ms ease",
                        }}
                      >
                        <Stack align="center" gap="lg" py="xl">
                          {isGroundTruthProcessing ? (
                            <Loader color="blue" size="xl" />
                          ) : (
                            <ThemeIcon
                              size={84}
                              radius="xl"
                              variant="light"
                              color={groundTruthDragActive ? "blue" : "gray"}
                            >
                              <IconUpload size={40} />
                            </ThemeIcon>
                          )}
                          <Stack gap="xs" ta="center">
                            <Title order={2}>
                              {isGroundTruthProcessing
                                ? "Validating ground truth data"
                                : "Step 1: Add your ground truth data"}
                            </Title>
                            <Text c="dimmed">
                              {isGroundTruthProcessing
                                ? "This could take a moment..."
                                : "Drop a single ground truth CSV or parquet file here, or click to select it from your device."}
                            </Text>
                          </Stack>
                          <input
                            id="forecast-checker-other-ground-truth-input"
                            type="file"
                            accept=".csv,.parquet,.pq,text/csv,application/octet-stream"
                            style={{ display: "none" }}
                            onChange={handleGroundTruthFileSelect}
                          />
                        </Stack>
                      </Paper>

                      <Group justify="center">
                        <Button
                          variant="light"
                          color="blue"
                          leftSection={<IconInfoCircle size={16} />}
                          onClick={toggleGroundTruthRequirements}
                        >
                          Ground truth data requirements
                        </Button>
                      </Group>

                      {groundTruthRequirementsOpened && (
                        <Alert
                          icon={<IconInfoCircle size={16} />}
                          title="Ground truth data requirements"
                          color="blue"
                          radius="lg"
                        >
                          <Stack gap="sm">
                            <Text>
                              Upload exactly one ground truth file at a time.
                              Forecast Checker currently accepts{" "}
                              <code>.csv</code>, <code>.parquet</code>, or{" "}
                              <code>.pq</code>.
                            </Text>
                            <Text fw={600}>Required columns</Text>
                            <List spacing="xs">
                              <List.Item>
                                <code>target_end_date</code> (must be valid
                                date)
                              </List.Item>
                              <List.Item>
                                <code>location</code>
                              </List.Item>
                              <List.Item>
                                <code>observation</code> (must be numeric;{" "}
                                <code>NA</code> is accepted)
                              </List.Item>
                              <List.Item>
                                <code>target</code>
                              </List.Item>
                            </List>
                            <Text>
                              <code>as_of</code> is an optional column. If it is
                              present, it will be used to filter duplicate rows
                              (i.e., for each duplicate, the latest{" "}
                              <code>as_of</code> value will be kept. In its
                              absence, duplicate rows will cause validation
                              failure. Must be a valid date.
                            </Text>
                            <Text fw={600}>Other validation behavior</Text>
                            <List spacing="xs">
                              <List.Item>
                                Rows with missing or invalid required values are
                                removed during validation.
                              </List.Item>
                              <List.Item>
                                <code>NA</code> observations are retained as
                                gaps in the ground truth plot.
                              </List.Item>
                              <List.Item>
                                The upload fails if no usable ground truth rows
                                remain after validation.
                              </List.Item>
                              <List.Item>
                                Ground truth data may contain multiple targets,
                                but only those found in your forecast data will
                                be shown in the target selector.
                              </List.Item>
                            </List>
                          </Stack>
                        </Alert>
                      )}
                    </Stack>
                  )}

                  {isOnForecastStep && (
                    <Paper
                      withBorder
                      radius="xl"
                      p="xl"
                      onDragEnter={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (isForecastProcessing) {
                          return;
                        }
                        setForecastDragActive(true);
                      }}
                      onDragLeave={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        setForecastDragActive(false);
                      }}
                      onDragOver={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                      }}
                      onDrop={handleForecastDrop}
                      onClick={() => {
                        if (isForecastProcessing) {
                          return;
                        }
                        document
                          .getElementById(
                            "forecast-checker-other-forecast-input",
                          )
                          ?.click();
                      }}
                      style={{
                        cursor: isForecastProcessing ? "progress" : "pointer",
                        border: forecastDragActive
                          ? "2px dashed var(--mantine-color-blue-6)"
                          : "2px dashed var(--mantine-color-gray-4)",
                        backgroundColor: forecastDragActive
                          ? "var(--mantine-color-blue-light)"
                          : "transparent",
                        transition:
                          "border-color 160ms ease, background-color 160ms ease",
                      }}
                    >
                      <Stack align="center" gap="lg" py="xl">
                        {isForecastProcessing ? (
                          <Loader color="blue" size="xl" />
                        ) : (
                          <ThemeIcon
                            size={84}
                            radius="xl"
                            variant="light"
                            color={forecastDragActive ? "blue" : "gray"}
                          >
                            <IconUpload size={40} />
                          </ThemeIcon>
                        )}
                        <Stack gap="xs" ta="center">
                          <Title order={2}>
                            {isForecastProcessing
                              ? "Processing your uploaded forecasts"
                              : "Step 2: Add your forecast data"}
                          </Title>
                          <Text c="dimmed">
                            {isForecastProcessing
                              ? "This could take a moment..."
                              : `Ground truth file loaded: ${groundTruthState.fileName}. Now drop your Hubverse-style forecast CSV file(s) here.`}
                          </Text>
                        </Stack>
                        <input
                          id="forecast-checker-other-forecast-input"
                          type="file"
                          accept=".csv,text/csv"
                          multiple
                          style={{ display: "none" }}
                          onChange={handleForecastFileSelect}
                        />
                      </Stack>
                    </Paper>
                  )}
                </Stack>
              </Box>
            </Group>
          )}

          {groundTruthState.status === "error" && (
            <Alert
              color="red"
              radius="lg"
              title="Ground truth validation failed"
              icon={<IconAlertCircle size={16} />}
            >
              <Stack gap="sm">
                <Text>{groundTruthState.error}</Text>
                {groundTruthState.summary && (
                  <Group gap="xs">
                    <Badge color="blue" variant="light">
                      {groundTruthState.summary.totalRows} rows read
                    </Badge>
                    <Badge color="green" variant="light">
                      {groundTruthState.summary.usableRows} rows usable
                    </Badge>
                  </Group>
                )}
              </Stack>
            </Alert>
          )}

          {projectionBuildState.status === "error" && (
            <Alert
              color="red"
              radius="lg"
              title="Could not create projections JSON"
              icon={<IconAlertCircle size={16} />}
            >
              {projectionBuildState.error}
            </Alert>
          )}

          {validationState?.status === "error" && (
            <Alert
              color="red"
              radius="lg"
              title="Validation failed"
              icon={<IconAlertCircle size={16} />}
            >
              <Stack gap="sm">
                <Text>CSV validation failed:</Text>
                {validationState.summary && (
                  <ValidationSummary summary={validationState.summary} />
                )}
                <List spacing="xs">
                  {validationState.errors.map((error) => (
                    <List.Item key={error}>{error}</List.Item>
                  ))}
                </List>
              </Stack>
            </Alert>
          )}
        </Stack>
      </Container>
    </>
  );
};

const HubUploadScreen = () => {
  const navigate = useNavigate();
  const { hub } = useParams();
  const hubConfig = useMemo(() => getHubConfig(hub), [hub]);
  const referenceDataPromiseRef = useRef(null);

  const [dragActive, setDragActive] = useState(false);
  const [isUploadProcessing, setIsUploadProcessing] = useState(false);
  const [validationState, setValidationState] = useState(null);
  const [pathogenWarning, setPathogenWarning] = useState(null);
  const [projectionBuildState, setProjectionBuildState] = useState({
    status: "idle",
    outputs: null,
    error: null,
    comparisonEligibility: null,
  });
  const [referenceDataState, setReferenceDataState] = useState({
    status: "idle",
    data: null,
    error: null,
  });

  useEffect(() => {
    setDragActive(false);
    setIsUploadProcessing(false);
    setValidationState(null);
    setPathogenWarning(null);
    setProjectionBuildState({
      status: "idle",
      outputs: null,
      error: null,
      comparisonEligibility: null,
    });
    referenceDataPromiseRef.current = null;
    setReferenceDataState({
      status: "idle",
      data: null,
      error: null,
    });
  }, [hub]);

  useEffect(() => {
    if (referenceDataState.status === "error") {
      setIsUploadProcessing(false);
    }
  }, [referenceDataState.status]);

  const ensureReferenceDataLoaded = useCallback(async () => {
    if (!hubConfig) {
      throw new Error("This hub is not configured.");
    }

    if (referenceDataState.status === "success" && referenceDataState.data) {
      return referenceDataState.data;
    }

    if (referenceDataPromiseRef.current) {
      return referenceDataPromiseRef.current;
    }

    setReferenceDataState({
      status: "loading",
      data: null,
      error: null,
    });

    const loadPromise = loadHubReferenceData(hubConfig)
      .then((data) => {
        setReferenceDataState({
          status: "success",
          data,
          error: null,
        });
        return data;
      })
      .catch((error) => {
        const message =
          error instanceof Error
            ? error.message
            : "Could not load the hub reference files.";

        setReferenceDataState({
          status: "error",
          data: null,
          error: message,
        });
        throw new Error(message);
      })
      .finally(() => {
        referenceDataPromiseRef.current = null;
      });

    referenceDataPromiseRef.current = loadPromise;
    return loadPromise;
  }, [hubConfig, referenceDataState.data, referenceDataState.status]);

  const processFiles = useCallback(
    async (incomingFiles) => {
      if (!hubConfig) {
        return;
      }

      setIsUploadProcessing(true);

      const files = Array.from(incomingFiles ?? []);
      const csvFiles = files.filter((file) =>
        file.name.toLowerCase().endsWith(".csv"),
      );

      if (csvFiles.length === 0) {
        setValidationState({
          status: "error",
          errors: [
            "Please upload at least one Hubverse forecast file in `.csv` format.",
          ],
        });
        setProjectionBuildState({
          status: "idle",
          outputs: null,
          error: null,
          comparisonEligibility: null,
        });
        setPathogenWarning(null);
        setIsUploadProcessing(false);
        return;
      }

      setValidationState(null);
      setPathogenWarning(null);
      setProjectionBuildState({
        status: "idle",
        outputs: null,
        error: null,
        comparisonEligibility: null,
      });

      try {
        const parsedFiles = await Promise.all(
          csvFiles.map(async (file) => {
            const text = await file.text();
            const rows = parseCsv(text);
            const { headers, records } = toObjects(rows);
            return {
              fileName: getFileRelativePath(file),
              headers,
              records,
            };
          }),
        );

        const missingColumnsByFile = parsedFiles
          .map((parsedFile) => ({
            fileName: parsedFile.fileName,
            missingColumns: FORECAST_REQUIRED_COLUMNS.filter(
              (column) => !parsedFile.headers.includes(column),
            ),
          }))
          .filter((entry) => entry.missingColumns.length > 0);

        if (missingColumnsByFile.length > 0) {
          setValidationState({
            status: "error",
            errors: missingColumnsByFile
              .slice(0, 5)
              .map(
                (entry) =>
                  `${entry.fileName} is missing required forecast columns: ${entry.missingColumns.join(", ")}.`,
              ),
          });
          setIsUploadProcessing(false);
          return;
        }

        const concatenatedRecords = parsedFiles.flatMap(
          (parsedFile) => parsedFile.records,
        );
        const validation = validateHubverseCsv(concatenatedRecords, hubConfig);

        if (!validation.ok) {
          setValidationState({
            status: "error",
            errors: validation.errors,
            summary: validation.summary,
          });
          setIsUploadProcessing(false);
          return;
        }

        setValidationState({
          status: "success",
          summary: validation.summary,
        });
        setPathogenWarning(
          buildHubPathogenWarning(validation.usableRows, hubConfig),
        );

        const referenceData = await ensureReferenceDataLoaded();

        if (!referenceData.timeSeries.records) {
          setIsUploadProcessing(false);
          setProjectionBuildState({
            status: "error",
            outputs: null,
            error:
              "This hub's time-series file was found, but it is not currently in a CSV format the frontend can process.",
            comparisonEligibility: null,
          });
          return;
        }

        const outputs = buildProjectionOutputs({
          hubConfig,
          forecastRows: validation.usableRows,
          locationsRows: referenceData.locations.records,
          targetRows: referenceData.timeSeries.records,
        });

        setProjectionBuildState({
          status: "success",
          outputs,
          error: null,
          comparisonEligibility: {
            isEligible: true,
            reason: null,
          },
        });
        setIsUploadProcessing(false);
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "The file could not be processed.";

        setIsUploadProcessing(false);
        setProjectionBuildState({
          status: "error",
          outputs: null,
          error: message,
          comparisonEligibility: null,
        });

        setValidationState((previousState) =>
          previousState?.status === "success"
            ? previousState
            : {
                status: "error",
                errors: [message],
              },
        );
      }
    },
    [ensureReferenceDataLoaded, hubConfig],
  );

  const handleDrop = useCallback(
    async (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (isUploadProcessing) {
        return;
      }
      setDragActive(false);
      const droppedFiles = await collectDroppedFiles(event.dataTransfer);
      if (droppedFiles.length) {
        processFiles(droppedFiles);
      }
    },
    [isUploadProcessing, processFiles],
  );

  const handleFileSelect = useCallback(
    (event) => {
      if (isUploadProcessing) {
        event.target.value = "";
        return;
      }
      if (event.target.files?.length) {
        processFiles(event.target.files);
      }
      event.target.value = "";
    },
    [isUploadProcessing, processFiles],
  );

  const handleResetUpload = useCallback(() => {
    setDragActive(false);
    setIsUploadProcessing(false);
    setValidationState(null);
    setPathogenWarning(null);
    setProjectionBuildState({
      status: "idle",
      outputs: null,
      error: null,
      comparisonEligibility: null,
    });
  }, []);
  const isShowingVisualization = projectionBuildState.status === "success";
  const backArrowLabel = isShowingVisualization
    ? "Back to hub upload"
    : "Back to hub selection";
  const handleBackArrowClick = isShowingVisualization
    ? handleResetUpload
    : () => navigate("/toolbox/forecast-checker");

  if (!hubConfig) {
    return (
      <Container size="md" py="xl">
        <Alert
          color="red"
          title="Unknown hub"
          icon={<IconAlertCircle size={16} />}
        >
          The Forecast Checker hub route <strong>{hub}</strong> is not
          supported.
        </Alert>
      </Container>
    );
  }

  return (
    <>
      <Seo
        title={`RespiLens | Forecast Checker | ${hubConfig.label}`}
        description={`Validate Hubverse CSV data for ${hubConfig.label} before Forecast Checker conversion.`}
        canonicalPath={`/toolbox/forecast-checker/${hubConfig.slug}`}
      />
      <Container size="xl" pt="md" pb="xl" fluid>
        <Stack gap="lg">
          {isShowingVisualization ? (
            <Group justify="space-between" align="center">
              <Stack gap={4}>
                <Group gap="xs" align="center">
                  <Tooltip label={backArrowLabel} withArrow>
                    <ActionIcon
                      variant="subtle"
                      color="blue"
                      size="xl"
                      radius="xl"
                      onClick={handleBackArrowClick}
                      aria-label={backArrowLabel}
                    >
                      <IconArrowLeft size={24} stroke={2.25} />
                    </ActionIcon>
                  </Tooltip>
                  <Title order={1}>{hubConfig.label}</Title>
                  <Tooltip label="Open hub in GitHub" withArrow>
                    <ActionIcon
                      component="a"
                      href={hubConfig.githubUrl}
                      target="_blank"
                      rel="noreferrer"
                      variant="subtle"
                      color="blue"
                      radius="xl"
                      aria-label={`Open ${hubConfig.label} GitHub`}
                    >
                      <IconBrandGithub size={20} />
                    </ActionIcon>
                  </Tooltip>
                </Group>
              </Stack>
            </Group>
          ) : (
            <Group justify="center">
              <Box w="100%" maw={860}>
                <Group justify="space-between" align="center">
                  <Stack gap={4}>
                    <Group gap="xs" align="center">
                      <Tooltip label={backArrowLabel} withArrow>
                        <ActionIcon
                          variant="subtle"
                          color="blue"
                          size="xl"
                          radius="xl"
                          onClick={handleBackArrowClick}
                          aria-label={backArrowLabel}
                        >
                          <IconArrowLeft size={24} stroke={2.25} />
                        </ActionIcon>
                      </Tooltip>
                      <Title order={1}>{hubConfig.label}</Title>
                      <Tooltip label="Open hub in GitHub" withArrow>
                        <ActionIcon
                          component="a"
                          href={hubConfig.githubUrl}
                          target="_blank"
                          rel="noreferrer"
                          variant="subtle"
                          color="blue"
                          radius="xl"
                          aria-label={`Open ${hubConfig.label} GitHub`}
                        >
                          <IconBrandGithub size={20} />
                        </ActionIcon>
                      </Tooltip>
                    </Group>
                  </Stack>
                </Group>
              </Box>
            </Group>
          )}

          {referenceDataState.status === "error" && (
            <Alert
              color="red"
              radius="lg"
              icon={<IconAlertCircle size={16} />}
              title="Could not load hub reference files"
            >
              {referenceDataState.error}
            </Alert>
          )}

          {isShowingVisualization && (
            <>
              {pathogenWarning && (
                <Alert
                  color="red"
                  variant="light"
                  radius="lg"
                  icon={<IconInfoCircle size={16} />}
                  title="Possible hub mismatch"
                >
                  {pathogenWarning}
                </Alert>
              )}
              <MyRespiVisualizationPanel
                projectionOutputs={projectionBuildState.outputs}
                hubConfig={hubConfig}
                comparisonEligibility={
                  projectionBuildState.comparisonEligibility
                }
              />
            </>
          )}

          {projectionBuildState.status !== "success" && (
            <Group justify="center">
              <Box w="100%" maw={860}>
                <Paper
                  withBorder
                  radius="xl"
                  p="xl"
                  onDragEnter={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (isUploadProcessing) {
                      return;
                    }
                    setDragActive(true);
                  }}
                  onDragLeave={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    setDragActive(false);
                  }}
                  onDragOver={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                  }}
                  onDrop={handleDrop}
                  onClick={() => {
                    if (isUploadProcessing) {
                      return;
                    }
                    document
                      .getElementById("myrespi-hubverse-files-input")
                      ?.click();
                  }}
                  style={{
                    cursor: isUploadProcessing ? "progress" : "pointer",
                    border: dragActive
                      ? "2px dashed var(--mantine-color-blue-6)"
                      : "2px dashed var(--mantine-color-gray-4)",
                    backgroundColor: dragActive
                      ? "var(--mantine-color-blue-light)"
                      : "transparent",
                    transition:
                      "border-color 160ms ease, background-color 160ms ease",
                  }}
                >
                  <Stack align="center" gap="lg" py="xl">
                    {isUploadProcessing ? (
                      <Loader color="blue" size="xl" />
                    ) : (
                      <ThemeIcon
                        size={84}
                        radius="xl"
                        variant="light"
                        color={dragActive ? "blue" : "gray"}
                      >
                        <IconUpload size={40} />
                      </ThemeIcon>
                    )}
                    <Stack gap="xs" ta="center">
                      <Title order={2}>
                        {isUploadProcessing
                          ? "Processing your uploaded data"
                          : "Drop your Hubverse-style CSV file(s) here"}
                      </Title>
                      <Text c="dimmed">
                        {isUploadProcessing ? (
                          "This could take a moment..."
                        ) : (
                          <>
                            Visit the Forecast Checker{" "}
                            <Anchor
                              href="/toolbox/forecast-checker/documentation"
                              target="_blank"
                              rel="noreferrer"
                              onClick={(event) => event.stopPropagation()}
                            >
                              documentation
                            </Anchor>{" "}
                            page to learn more about what makes your data valid.
                          </>
                        )}
                      </Text>
                    </Stack>
                    <input
                      id="myrespi-hubverse-files-input"
                      type="file"
                      accept=".csv,text/csv"
                      multiple
                      style={{ display: "none" }}
                      onChange={handleFileSelect}
                    />
                  </Stack>
                </Paper>
              </Box>
            </Group>
          )}

          {projectionBuildState.status === "error" && (
            <Alert
              color="red"
              radius="lg"
              title="Could not create projections JSON"
              icon={<IconAlertCircle size={16} />}
            >
              {projectionBuildState.error}
            </Alert>
          )}

          {validationState?.status === "error" && (
            <Alert
              color="red"
              radius="lg"
              title="Validation failed"
              icon={<IconAlertCircle size={16} />}
            >
              <Stack gap="sm">
                <Text>CSV validation failed:</Text>
                {validationState.summary && (
                  <ValidationSummary summary={validationState.summary} />
                )}

                <List spacing="xs">
                  {validationState.errors.map((error) => (
                    <List.Item key={error}>{error}</List.Item>
                  ))}
                </List>
                <Text size="sm" c="dimmed">
                  Check the Forecast Checker{" "}
                  <Anchor
                    href="/toolbox/forecast-checker/documentation"
                    target="_blank"
                    rel="noreferrer"
                  >
                    documentation
                  </Anchor>{" "}
                  for the expected upload format and validation rules.
                </Text>
              </Stack>
            </Alert>
          )}
        </Stack>
      </Container>
    </>
  );
};

const ForecastCheckerDashboard = () => {
  const { hub } = useParams();

  if (hub === "other-hub") {
    return <OtherHubScreen />;
  }

  if (hub) {
    return <HubUploadScreen />;
  }

  return <HubSelectionScreen />;
};

export {
  FORECAST_REQUIRED_COLUMNS,
  GROUND_TRUTH_REQUIRED_COLUMNS,
  MyRespiVisualizationPanel,
  OTHER_HUB_CONFIG,
  // Shared with My Private Hub to keep both upload paths behaviorally aligned.
  // eslint-disable-next-line react-refresh/only-export-components
  buildGroundTruthOutput,
  // eslint-disable-next-line react-refresh/only-export-components
  readTabularUpload,
  // eslint-disable-next-line react-refresh/only-export-components
  validateGroundTruthCsv,
  // eslint-disable-next-line react-refresh/only-export-components
  validateHubverseCsv,
};

export default ForecastCheckerDashboard;
