import { useState, useEffect, useRef } from "react";
import {
  Stack,
  Container,
  Group,
  Title,
  Anchor,
  List,
  Text,
} from "@mantine/core";
import { useView } from "../hooks/useView";
import { extractPlotData } from "../hooks/extractPlotDataFromURL";
import ChartHeader from "./ChartHeader";
import { ChartAboutContext } from "../contexts/ChartAboutContext";
import { ChartResetContext } from "../contexts/ChartResetContext";
import { ChartRangeContext } from "../contexts/ChartRangeContext";
import LocationPicker from "./LocationPicker";
import KeyboardShortcutsModal from "./KeyboardShortcutsModal";
import ViewSwitchboard from "./ViewSwitchboard";
import ErrorBoundary from "./ErrorBoundary";
import FrontPage, {
  FrontPageAnnouncements,
  FluSightStartAnnouncement,
  MetroCastPausedAnnouncement,
} from "./FrontPage";
import { IconBrandGithub } from "@tabler/icons-react";
import { useClipboard } from "@mantine/hooks";
import Seo from "./Seo";
import { resolvePlotLocationDisplayName } from "../utils/plotLocationDisplay";
import { downloadChartPng } from "../utils/downloadChartPng";

const DataVisualizationContainer = ({ disableSeo = false }) => {
  const {
    selectedLocation,
    data,
    metadata,
    loading,
    error,
    availableDates,
    models,
    selectedModels,
    setSelectedModels,
    selectedDates,
    setSelectedDates,
    activeDate,
    setActiveDate,
    viewType,
    currentDataset,
    selectedColumns,
    setSelectedColumns,
    selectedTarget,
    peaks,
    availablePeakDates,
    availablePeakModels,
  } = useView();

  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  const clipboard = useClipboard({ timeout: 2000 });
  const chartAreaRef = useRef(null);

  // The chart header's download: the view's main chart, as a PNG
  const handleDownload = () => downloadChartPng(chartAreaRef.current, viewType);

  // Each chart's zoomed time range, kept while views remount
  const rangeStore = useRef({});

  const [resetCount, setResetCount] = useState(0);
  const handleResetView = () => setResetCount((count) => count + 1);

  const [isAdded, setIsAdded] = useState(false);
  const handleSaveToMyPlots = () => {
    const plotData = extractPlotData(viewType, window.location.href, data);
    // visual cue to signal it has been added
    setIsAdded(true);
    // Reset text after 2 seconds
    setTimeout(() => setIsAdded(false), 2000);
    // Temporary console feedback for development
    console.log("Saved the plot", plotData);
  };

  // Configuration for AboutHubOverlay based on viewType
  const aboutHubConfig = {
    covid_forecasts: {
      title: (
        <Group gap="sm">
          <Title order={4}>COVID-19 Forecast Hub</Title>
          <Anchor
            href="https://github.com/CDCgov/covid19-forecast-hub"
            target="_blank"
            rel="noopener noreferrer"
            c="dimmed"
          >
            <IconBrandGithub size={20} />
          </Anchor>
        </Group>
      ),
      buttonLabel: "About COVID-19 Forecast Hub",
      content: (
        <>
          <div>
            Data for the RespiLens COVID-19 Forecasts view is retrieved from the
            COVID-19 Forecast Hub, which is an open challenge organized by the{" "}
            <a
              href="https://www.cdc.gov/cfa-modeling-and-forecasting/covid19-data-vis/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              US CDC
            </a>{" "}
            designed to collect forecasts for the following two targets:
            <List spacing="xs" size="sm">
              <List.Item>Weekly new hospitalizations due to COVID-19</List.Item>
              <List.Item>
                Weekly incident percentage of emergency department visits due to
                COVID-19
              </List.Item>
            </List>
            RespiLens displays forecasts for all models, dates and targets. For
            attribution and more information, please visit the COVID-19 Forecast
            Hub{" "}
            <a
              href="https://github.com/CDCgov/covid19-forecast-hub"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub repository
            </a>
            .
          </div>
          <div>
            <Title order={4} mb="xs">
              Forecasts
            </Title>
            <p>
              Forecasting teams submit a probabilistic forecasts of these
              targets every Wednesday. RespiLens displays the 50% and 95%
              confidence intervals for each model's forecast for a chosen date,
              shown on the plot with a shadow.
            </p>
          </div>
        </>
      ),
    },
    rsv_forecasts: {
      title: (
        <Group gap="sm">
          <Title order={4}>RSV Forecast Hub</Title>
          <Anchor
            href="https://github.com/CDCgov/rsv-forecast-hub"
            target="_blank"
            rel="noopener noreferrer"
            c="dimmed"
          >
            <IconBrandGithub size={20} />
          </Anchor>
        </Group>
      ),
      buttonLabel: "About RSV Forecast Hub",
      content: (
        <>
          <div>
            Data for the RespiLens RSV Forecasts view is retrieved from the RSV
            Forecast Hub, which is an open challenge organized by the US CDC
            designed to collect forecasts for the following two targets:
            <List spacing="xs" size="sm">
              <List.Item>Weekly new hospitalizations due to RSV</List.Item>
              <List.Item>
                Weekly incident percentage of emergency department visits due to
                RSV
              </List.Item>
            </List>
            RespiLens displays forecasts for all models, dates and targets. For
            attribution and more information, please visit the RSV Forecast Hub{" "}
            <a
              href="https://github.com/CDCgov/rsv-forecast-hub"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub repository
            </a>
            .
          </div>
          <div>
            <Title order={4} mb="xs">
              Forecasts
            </Title>
            <p>
              Forecasting teams submit a probabilistic forecasts of these
              targets every Wednesday of the RSV season. RespiLens displays the
              50% and 95% confidence intervals for each model's forecast for a
              chosen date, shown on the plot with a shadow.
            </p>
          </div>
        </>
      ),
    },
    flu_peak: {
      title: (
        <Group gap="sm">
          <Title order={4}>FluSight Forecast Hub</Title>
          <Anchor
            href="https://github.com/cdcepi/FluSight-forecast-hub"
            target="_blank"
            rel="noopener noreferrer"
            c="dimmed"
          >
            <IconBrandGithub size={20} />
          </Anchor>
        </Group>
      ),
      buttonLabel: "About FluSight",
      content: (
        <>
          <p>
            Data for the RespiLens Flu Peaks view is retrieved from FluSight,
            which is an open challenge organized by the US CDC designed to
            collect{" "}
            <a
              href="https://www.cdc.gov/flu-forecasting/about/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              influenza forecasts
            </a>{" "}
            during the flu season. RespiLens displays forecasts for all models,
            dates and targets. For attribution and more information, please
            visit the FluSight{" "}
            <a
              href="https://github.com/cdcepi/FluSight-forecast-hub"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub repository
            </a>
            .
          </p>
          <div>
            <Title order={4} mb="xs">
              Forecasts
            </Title>
            <p>
              Forecasting teams submit a probabilistic forecasts of these
              targets every Wednesday of the flu season. RespiLens displays the
              50% and 95% confidence intervals for each model's forecast for a
              chosen date, shown on the plot with a shadow.
            </p>
            <Title order={4} mb="xs">
              Peaks
            </Title>
            <p>
              Some teams elect to submit predictions for peak influenza burden
              (for which week the peak is expected, and what the hospitalization
              burden is projected to be). This data is displayed in our Flu
              Peaks view, where you can view participating models' median
              forecast for peak flu burden during the current season.
            </p>
          </div>
        </>
      ),
    },
    flu_forecasts: {
      title: (
        <Group gap="sm">
          <Title order={4}>FluSight Forecast Hub</Title>
          <Anchor
            href="https://github.com/cdcepi/FluSight-forecast-hub"
            target="_blank"
            rel="noopener noreferrer"
            c="dimmed"
          >
            <IconBrandGithub size={20} />
          </Anchor>
        </Group>
      ),
      buttonLabel: "About FluSight",
      content: (
        <>
          <p>
            Data for the RespiLens Flu Projections view is retrieved from
            FluSight, which is an open challenge organized by the US CDC
            designed to collect{" "}
            <a
              href="https://www.cdc.gov/flu-forecasting/about/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              influenza forecasts
            </a>{" "}
            during the flu season. RespiLens displays forecasts for all models,
            dates and targets. For attribution and more information, please
            visit the FluSight{" "}
            <a
              href="https://github.com/cdcepi/FluSight-forecast-hub"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub repository
            </a>
            .
          </p>
          <div>
            <Title order={4} mb="xs">
              Forecasts
            </Title>
            <p>
              Forecasting teams submit a probabilistic forecasts of these
              targets every Wednesday of the flu season. RespiLens displays the
              50% and 95% confidence intervals for each model's forecast for a
              chosen date, shown on the plot with a shadow.
            </p>
            <Title order={4} mb="xs">
              Peaks
            </Title>
            <p>
              Some teams elect to submit predictions for peak influenza burden
              (for which week the peak is expected, and what the hospitalization
              burden is projected to be). This data is displayed in our Flu
              Peaks view, where you can view participating models' median
              forecast for peak flu burden during the current season.
            </p>
          </div>
        </>
      ),
    },
    fludetailed: {
      title: (
        <Group gap="sm">
          <Title order={4}>FluSight Forecast Hub</Title>
          <Anchor
            href="https://github.com/cdcepi/FluSight-forecast-hub"
            target="_blank"
            rel="noopener noreferrer"
            c="dimmed"
          >
            <IconBrandGithub size={20} />
          </Anchor>
        </Group>
      ),
      buttonLabel: "About FluSight",
      content: (
        <>
          <p>
            Data for the RespiLens Flu Detailed View is retrieved from FluSight,
            which is an open challenge organized by the US CDC designed to
            collect{" "}
            <a
              href="https://www.cdc.gov/flu-forecasting/about/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              influenza forecasts
            </a>{" "}
            during the flu season. RespiLens displays forecasts for all models,
            dates and targets. For attribution and more information, please
            visit the FluSight{" "}
            <a
              href="https://github.com/cdcepi/FluSight-forecast-hub"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub repository
            </a>
            .
          </p>
          <div>
            <Title order={4} mb="xs">
              Forecasts
            </Title>
            <p>
              Forecasting teams submit a probabilistic forecasts of these
              targets every Wednesday of the flu season. RespiLens displays the
              50% and 95% confidence intervals for each model's forecast for a
              chosen date, shown on the plot with a shadow.
            </p>
            <Title order={4} mb="xs">
              Peaks
            </Title>
            <p>
              Some teams elect to submit predictions for peak influenza burden
              (for which week the peak is expected, and what the hospitalization
              burden is projected to be). This data is displayed in our Flu
              Peaks view, where you can view participating models' median
              forecast for peak flu burden during the current season.
            </p>
          </div>
        </>
      ),
    },
    metrocast_forecasts: {
      title: (
        <Group gap="sm">
          <Title order={4}>Flu MetroCast</Title>
        </Group>
      ),
      buttonLabel: "About MetroCast",
      content: (
        <>
          <p>
            Data for the RespiLens Flu Metrocast view is retrieved from the Flu
            MetroCast Hub, which is a collaborative modeling project that
            collects and shares weekly probabilistic forecasts of influenza
            activity at the metropolitan level in the United States. The hub is
            run by{" "}
            <a
              href="https://epiengage.org/"
              target="_blank"
              rel="noopener noreferrer"
            >
              epiENGAGE
            </a>{" "}
            – an{" "}
            <a
              href="https://www.cdc.gov/insight-net/php/about/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Insight Net
            </a>{" "}
            Center for Implementation within the U.S. Centers for Disease
            Control and Prevention (CDC)’s{" "}
            <a
              href="https://www.cdc.gov/forecast-outbreak-analytics/about/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Center for Forecasting and Outbreak Analytics
            </a>{" "}
            (CFA).
          </p>
          <p>
            For more info and attribution on the Flu MetroCast Hub, please visit
            their{" "}
            <a
              href="https://reichlab.io/metrocast-dashboard/"
              target="_blank"
              rel="noopener noreferrer"
            >
              site
            </a>
            , or visit their{" "}
            <a
              href="https://reichlab.io/metrocast-dashboard/forecast.html?as_of=2026-01-24&interval=95%25&target_var=ILI+ED+visits+pct&xaxis_range=2025-08-01&xaxis_range=2026-07-01&yaxis_range=0.5955774343586175&yaxis_range=11.579180135033756&model=epiENGAGE-ensemble_mean&location=nyc"
              target="_blank"
              rel="noopener noreferrer"
            >
              visualization dashboard
            </a>{" "}
            to engage with their original visualization scheme.
          </p>
          <div>
            <Title order={4} mb="xs">
              Forecasts
            </Title>
            <p>
              Forecasting teams submit a probabilistic forecasts of targets
              every week of the flu season. RespiLens displays the 50% and 95%
              confidence intervals for each model's forecast for a chosen date,
              shown on the plot with a shadow.
            </p>
          </div>
        </>
      ),
    },
    nhsnall: {
      title: (
        <Group gap="sm">
          <Title order={4}>National Healthcare Safety Network (NHSN)</Title>
        </Group>
      ),
      buttonLabel: "About NHSN Data",
      content: (
        <>
          <p>
            Data for the RespiLens NHSN view comes from the CDC's{" "}
            <a
              href="https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data"
              target="_blank"
              rel="noopener noreferrer"
            >
              National Healthcare Safety Network
            </a>{" "}
            weekly "Hospital Respiratory Data" (HRD) dataset. This dataset
            represents metrics aggregated to national and state/territory levels
            beginning in August 2020.
          </p>
          <div>
            <Title order={4} mb="xs">
              Columns
            </Title>
            <p>
              The NHSN dataset contains ~300 columns for plotting data with a
              variety of scales, including hospitalization admission counts,
              percent of admissions by pathogen, hospitalization rates per 100k,
              raw bed capacity numbers, bed capacity percents, and absolute
              percentage of change. On RespiLens, you can use the timeseries
              unit selector to switch between data scales and view similar
              columns on the same plot.
            </p>
          </div>
        </>
      ),
    },
    nsspall: {
      title: (
        <Group gap="sm">
          <Title order={4}>
            National Syndromic Surveillance Program (NSSP)
          </Title>
        </Group>
      ),
      buttonLabel: "About NSSP Data",
      content: (
        <>
          <p>
            Data for the RespiLens NSSP view comes from the CDC's{" "}
            <a
              href="https://www.cdc.gov/nssp/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              National Syndromic Surveillance Program,
            </a>{" "}
            which integrates data from many sources to help consolidate
            complete, accurate, and timely health data. Syndromic surveillance
            is a collaborative effort among local and state health departments,
            CDC, and partners.
          </p>
          <div>
            <Title order={4} mb="xs">
              Data
            </Title>
            <p>
              RespiLens' display of NSSP data shows percent of hospital visits
              attributable to RSV, COVID-19, and influenza across 50 states +
              Washington D.C.; where 25 states have county-level visualization.
            </p>
          </div>
        </>
      ),
    },
  };

  const currentAboutConfig = aboutHubConfig[viewType];
  const chartAbout = currentAboutConfig && {
    title: currentAboutConfig.title,
    content: currentAboutConfig.content,
    sourceLabel: currentAboutConfig.buttonLabel.replace(/^About\s+/, ""),
  };

  useEffect(() => {
    const handleResize = () =>
      setWindowSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleShare = () => {
    const url = window.location.href;
    clipboard.copy(url);
  };

  // for when switching from one flu view to flu_peak and there are multiple dates selected
  useEffect(() => {
    if (viewType === "flu_peak" && selectedDates.length > 1) {
      // keep only the active date, or the first date if activeDate isn't set
      const singleDate = activeDate || selectedDates[0];
      setSelectedDates([singleDate]);
    }
  }, [viewType, selectedDates, activeDate, setSelectedDates]);

  // "RespiLens | Flu Forecasts | Peak Forecasts | California": names, not
  // codes, and one title per address so analytics can tell the pages apart
  const locationName = resolvePlotLocationDisplayName(selectedLocation);
  const variantLabel =
    currentDataset && viewType !== currentDataset.defaultView
      ? currentDataset.views.find((view) => view.value === viewType)?.label
      : null;
  const forecastTitle = [
    "RespiLens",
    currentDataset?.fullName || "Forecasts",
    variantLabel,
    locationName,
  ]
    .filter(Boolean)
    .join(" | ");

  if (viewType === "frontpage") {
    return (
      <ErrorBoundary onReset={() => window.location.reload()}>
        {!disableSeo && (
          <Seo
            title={`RespiLens | Forecasts | ${locationName}`}
            description="Explore respiratory disease forecasts and surveillance data for influenza, COVID-19, RSV, and hospital respiratory metrics across the United States."
          />
        )}
        <FrontPageAnnouncements />
        <Container size="xl" pt="md" pb="xl" style={{ maxWidth: "1400px" }}>
          <Stack gap="lg">
            <Title order={2} fz={{ base: 20, sm: 24 }} fw={600}>
              <Text span inherit c="dimmed" fw={400}>
                Overview for
              </Text>{" "}
              <LocationPicker />
            </Title>
            <FrontPage />
          </Stack>
          <KeyboardShortcutsModal
            hasTargets={false}
            hasDates={false}
            hasModels={false}
            hasIntervals={false}
            hasDownload={false}
          />
        </Container>
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary onReset={() => window.location.reload()}>
      {!disableSeo && (
        <Seo
          title={forecastTitle}
          description={`View ${currentDataset?.fullName || "respiratory disease forecasts"} in RespiLens with model projections, observed trends, and state-level respiratory disease activity.`}
        />
      )}
      {currentDataset?.shortName === "flu" && <FluSightStartAnnouncement />}
      {currentDataset?.shortName === "metrocast" && (
        <MetroCastPausedAnnouncement />
      )}
      <ChartAboutContext.Provider value={chartAbout}>
        <ChartResetContext.Provider value={resetCount}>
          <ChartRangeContext.Provider value={rangeStore}>
            <Container size="xl" pt="md" pb="xl" style={{ maxWidth: "1400px" }}>
              <Stack gap="lg">
                <Stack gap="md" style={{ minHeight: "78vh" }}>
                  <ChartHeader
                    onSave={handleSaveToMyPlots}
                    isAdded={isAdded}
                    onShare={handleShare}
                    shareCopied={clipboard.copied}
                    onDownload={handleDownload}
                    onResetView={handleResetView}
                  />
                  <div ref={chartAreaRef} style={{ flex: 1, minHeight: 0 }}>
                    <ViewSwitchboard
                      viewType={viewType}
                      location={selectedLocation}
                      data={data}
                      metadata={metadata}
                      loading={loading}
                      error={error}
                      availableDates={availableDates}
                      models={models}
                      selectedDates={selectedDates}
                      selectedModels={selectedModels}
                      setSelectedDates={setSelectedDates}
                      setActiveDate={setActiveDate}
                      setSelectedModels={setSelectedModels}
                      selectedColumns={selectedColumns}
                      setSelectedColumns={setSelectedColumns}
                      windowSize={windowSize}
                      selectedTarget={selectedTarget}
                      peaks={peaks}
                      availablePeakDates={availablePeakDates}
                      availablePeakModels={availablePeakModels}
                    />
                  </div>
                </Stack>
              </Stack>
            </Container>
          </ChartRangeContext.Provider>
        </ChartResetContext.Provider>
      </ChartAboutContext.Provider>
    </ErrorBoundary>
  );
};

export default DataVisualizationContainer;
