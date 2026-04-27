import { useState, useEffect } from "react";
import {
  Stack,
  Container,
  Paper,
  Group,
  Button,
  Tooltip,
  Title,
  Anchor,
  List,
} from "@mantine/core";
import { useView } from "../hooks/useView";
import { extractPlotData } from "../hooks/extractPlotDataFromURL";
import DateSelector from "./DateSelector";
import ViewSwitchboard from "./ViewSwitchboard";
import ErrorBoundary from "./ErrorBoundary";
import AboutHubOverlay from "./AboutHubOverlay";
import FrontPage from "./FrontPage";
import {
  IconShare,
  IconBrandGithub,
  IconChartScatter,
} from "@tabler/icons-react";
import { useClipboard } from "@mantine/hooks";
import Seo from "./Seo";

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
          <p>
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
            <p></p>
            <List spacing="xs" size="sm">
              <List.Item>Weekly new hospitalizations due to COVID-19</List.Item>
              <List.Item>
                Weekly incident percentage of emergency department visits due to
                COVID-19
              </List.Item>
            </List>
            <p></p>
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
          </p>
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
          <p>
            Data for the RespiLens RSV Forecasts view is retrieved from the RSV
            Forecast Hub, which is an open challenge organized by the US CDC
            designed to collect forecasts for the following two targets:
            <p></p>
            <List spacing="xs" size="sm">
              <List.Item>Weekly new hospitalizations due to RSV</List.Item>
              <List.Item>
                Weekly incident percentage of emergency department visits due to
                RSV
              </List.Item>
            </List>
            <p></p>
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
          </p>
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
  };

  const currentAboutConfig = aboutHubConfig[viewType];

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

  if (viewType === "frontpage") {
    return (
      <ErrorBoundary onReset={() => window.location.reload()}>
        {!disableSeo && (
          <Seo
            title="RespiLens | Forecasts"
            description="Explore respiratory disease forecasts and surveillance data for influenza, COVID-19, RSV, and hospital respiratory metrics across the United States."
          />
        )}
        <Container size="xl" py="xl" style={{ maxWidth: "1400px" }}>
          <Stack gap="lg">
            <FrontPage />
          </Stack>
        </Container>
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary onReset={() => window.location.reload()}>
      {!disableSeo && (
        <Seo
          title={`RespiLens | ${currentDataset?.fullName || "Forecasts"}`}
          description={`View ${currentDataset?.fullName || "respiratory disease forecasts"} in RespiLens with model projections, observed trends, and state-level respiratory disease activity.`}
        />
      )}
      <Container size="xl" py="xl" style={{ maxWidth: "1400px" }}>
        <Stack gap="lg">
          <Paper shadow="sm" p="lg" radius="md" withBorder>
            <Stack gap="md" style={{ minHeight: "70vh" }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    windowSize.width > 800 ? "auto 1fr auto" : "1fr",
                  gap: "0.5rem",
                  alignItems: "center",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gridColumn: windowSize.width > 800 ? "auto" : "1",
                  }}
                >
                  {currentAboutConfig && (
                    <AboutHubOverlay
                      title={currentAboutConfig.title}
                      buttonLabel={currentAboutConfig.buttonLabel}
                    >
                      {currentAboutConfig.content}
                    </AboutHubOverlay>
                  )}
                  {windowSize.width <= 800 && (
                    <Group gap="xs">
                      {viewType !== "flu_peak" && (
                        <Button
                          variant="light"
                          size="xs"
                          mr="xs"
                          color={isAdded ? "green" : "blue"}
                          className={isAdded ? "added-text-pulse" : ""}
                          leftSection={<IconChartScatter size={16} />}
                          onClick={handleSaveToMyPlots}
                        >
                          {isAdded ? "Added!" : "Add to My Plots"}
                        </Button>
                      )}
                      <Tooltip
                        label={
                          clipboard.copied
                            ? "Link copied"
                            : "Copy link to this view"
                        }
                      >
                        <Button
                          variant="light"
                          size="xs"
                          color={clipboard.copied ? "green" : "blue"}
                          leftSection={<IconShare size={16} />}
                          onClick={handleShare}
                        >
                          {clipboard.copied ? "URL Copied!" : "Share View"}
                        </Button>
                      </Tooltip>
                    </Group>
                  )}
                </div>
                {currentDataset?.hasDateSelector && windowSize.width > 800 && (
                  <div style={{ display: "flex", justifyContent: "center" }}>
                    <DateSelector
                      selectedDates={selectedDates}
                      setSelectedDates={setSelectedDates}
                      availableDates={availableDates}
                      activeDate={activeDate}
                      setActiveDate={setActiveDate}
                      loading={loading}
                      multi={viewType !== "flu_peak"}
                    />
                  </div>
                )}
                {windowSize.width > 800 && (
                  <div style={{ display: "flex", justifyContent: "flex-end" }}>
                    {viewType !== "flu_peak" && (
                      <Button
                        variant="light"
                        size="xs"
                        mr="xs"
                        color={isAdded ? "green" : "blue"}
                        className={isAdded ? "added-text-pulse" : ""}
                        leftSection={<IconChartScatter size={16} />}
                        onClick={handleSaveToMyPlots}
                      >
                        {isAdded ? "Added!" : "Add to My Plots"}
                      </Button>
                    )}
                    <Tooltip
                      label={
                        clipboard.copied
                          ? "Link copied"
                          : "Copy link to this view"
                      }
                    >
                      <Button
                        variant="light"
                        size="xs"
                        color={clipboard.copied ? "green" : "blue"}
                        leftSection={<IconShare size={16} />}
                        onClick={handleShare}
                      >
                        {clipboard.copied ? "URL Copied!" : "Share View"}
                      </Button>
                    </Tooltip>
                  </div>
                )}
                {currentDataset?.hasDateSelector && windowSize.width <= 800 && (
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      marginTop: "0.5rem",
                    }}
                  >
                    <DateSelector
                      selectedDates={selectedDates}
                      setSelectedDates={setSelectedDates}
                      availableDates={availableDates}
                      activeDate={activeDate}
                      setActiveDate={setActiveDate}
                      loading={loading}
                      multi={viewType !== "flu_peak"} //this disables multi date select if flu peak
                    />
                  </div>
                )}
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
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
          </Paper>
        </Stack>
      </Container>
    </ErrorBoundary>
  );
};

export default DataVisualizationContainer;
