import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Container,
  Paper,
  Title,
  Text,
  Group,
  Stack,
  Badge,
  ThemeIcon,
  Loader,
  Center,
  Button,
  ActionIcon,
  Box,
  Divider,
} from "@mantine/core";
import {
  IconBook,
  IconCalendar,
  IconUser,
  IconChevronLeft,
  IconChevronRight,
  IconCode,
} from "@tabler/icons-react";
import DataVisualizationContainer from "../DataVisualizationContainer";
import Seo from "../Seo";

// Plotly Gaussian Chart Component
const PlotlyGaussianChart = () => {
  const [chartData, setChartData] = useState(null);
  const [plotlyLoaded, setPlotlyLoaded] = useState(false);
  const chartRef = useRef(null);

  // Load Plotly if not already loaded
  useEffect(() => {
    if (window.Plotly) {
      setPlotlyLoaded(true);
      return;
    }

    if (!document.querySelector('script[src*="plotly"]')) {
      const script = document.createElement("script");
      script.src = "https://cdn.plot.ly/plotly-2.27.0.min.js"; // Use specific version
      script.onload = () => {
        setPlotlyLoaded(true);
      };
      script.onerror = () => {
        console.error("Failed to load Plotly");
      };
      document.head.appendChild(script);
    }
  }, []);

  // Generate chart data once
  useEffect(() => {
    if (!plotlyLoaded) return;

    // Generate Gaussian data
    const generateGaussian = (mean = 0, std = 1, points = 200) => {
      const x = [];
      const y = [];

      for (let i = 0; i < points; i++) {
        const xi = ((i - points / 2) * 6) / points; // Range from -3 to 3
        x.push(xi);
        y.push(
          (1 / (std * Math.sqrt(2 * Math.PI))) *
            Math.exp(-0.5 * Math.pow((xi - mean) / std, 2)),
        );
      }

      return { x, y };
    };

    const gaussianData = generateGaussian(0, 1, 200);

    const plotData = [
      {
        x: gaussianData.x,
        y: gaussianData.y,
        type: "scatter",
        mode: "lines",
        name: "Gaussian Distribution",
        line: {
          color: "#228be6",
          width: 3,
        },
        fill: "tonexty",
        fillcolor: "rgba(34, 139, 230, 0.2)",
      },
    ];

    const layout = {
      title: {
        text: "Standard Normal Distribution (μ=0, σ=1)",
        font: { size: 16 },
      },
      xaxis: {
        title: "Value",
        showgrid: true,
        gridcolor: "#f0f0f0",
      },
      yaxis: {
        title: "Probability Density",
        showgrid: true,
        gridcolor: "#f0f0f0",
      },
      margin: { t: 50, r: 20, b: 50, l: 60 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, sans-serif" },
    };

    setChartData({ data: plotData, layout });
  }, [plotlyLoaded]);

  // Render chart when data is ready
  useEffect(() => {
    if (chartData && plotlyLoaded && window.Plotly && chartRef.current) {
      window.Plotly.newPlot(
        chartRef.current,
        chartData.data,
        chartData.layout,
        {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
          displaylogo: false,
        },
      ).catch((error) => {
        console.error("Error creating Plotly chart:", error);
      });
    }
  }, [chartData, plotlyLoaded]);

  if (!plotlyLoaded || !chartData) {
    return (
      <Center h="100%">
        <Stack align="center" gap="md">
          <Loader size="lg" />
          <Text>Loading Plotly chart...</Text>
        </Stack>
      </Center>
    );
  }

  return (
    <div style={{ height: "100%", padding: "20px" }}>
      <div ref={chartRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
};

const SlideNarrativeViewer = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [slides, setSlides] = useState([]);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [loading, setLoading] = useState(true);
  const [metadata, setMetadata] = useState({});
  const [currentVisualization, setCurrentVisualization] = useState(null);

  const parseVisualizationUrl = useCallback((url) => {
    if (!url) return null;
    if (url.startsWith("javascript:")) {
      return { type: "custom", code: url.replace("javascript:", "") };
    }

    const urlObj = new URL(url, window.location.origin);
    const params = new URLSearchParams(urlObj.search);

    return {
      type: "respilens",
      location: params.get("location") || "US",
      view: params.get("view") || "fludetailed",
      dates: params.get("dates")?.split(",") || [],
      models: params.get("models")?.split(",") || [],
    };
  }, []);

  const parseNarrative = useCallback(
    (content) => {
      try {
        const parts = content.split("---");

        if (parts.length >= 3) {
          const frontmatterLines = parts[1].trim().split("\n");
          const parsedMetadata = {};
          frontmatterLines.forEach((line) => {
            const [key, ...valueParts] = line.split(":");
            if (key && valueParts.length > 0) {
              parsedMetadata[key.trim()] = valueParts
                .join(":")
                .trim()
                .replace(/"/g, "");
            }
          });
          setMetadata(parsedMetadata);

          const slideContent = parts.slice(2).join("---");

          const slideMatches = slideContent
            .split(/\n# /)
            .filter((s) => s.trim());

          const parsedSlides = slideMatches.map((slide, index) => {
            let normalizedSlide = slide;
            if (index === 0) {
              normalizedSlide = normalizedSlide.replace(/^# /, "");
            }

            const lines = normalizedSlide.split("\n");
            const titleLine = lines[0];
            const titleMatch = titleLine.match(/^(.*?)\s*\[(.*?)\]$/);

            const title = titleMatch ? titleMatch[1].trim() : titleLine.trim();
            const url = titleMatch ? titleMatch[2].trim() : null;
            const body = lines.slice(1).join("\n").trim();

            return { title, url, content: body };
          });

          setSlides(parsedSlides);

          const initialVisualizationUrl =
            parsedSlides[0]?.url || parsedMetadata.dataset;
          setCurrentVisualization(
            initialVisualizationUrl
              ? parseVisualizationUrl(initialVisualizationUrl)
              : null,
          );
        } else {
          console.error(
            "Invalid narrative format - not enough parts after splitting by ---",
          );
          setMetadata({});
          setSlides([]);
          setCurrentVisualization(null);
        }
      } catch (error) {
        console.error("Error parsing narrative:", error);
        setMetadata({});
        setSlides([]);
        setCurrentVisualization(null);
      }

      setLoading(false);
    },
    [parseVisualizationUrl],
  );

  useEffect(() => {
    const narrativeId = id || "flu-winter-2024-25-slides";

    const loadNarrative = async () => {
      try {
        const narrativeModule = await import(
          `../../data/narratives/${narrativeId}.js`
        );
        parseNarrative(narrativeModule.narrativeContent);
      } catch (error) {
        console.error("Error loading narrative module:", error);

        const fallbackContent = `---
title: "Flu Season Winter 2024-25: A Data Story"
authors: "RespiLens Analytics Team"
date: "December 24, 2024"
abstract: "An interactive narrative exploring the 2024-25 flu season trends, forecasting insights, and public health implications using RespiLens visualization tools."
dataset: "/?location=US&view=fludetailed&dates=2024-12-14,2024-12-21&models=FluSight-ensemble,CU-ensemble"
---

# Introduction: The 2024-25 Flu Season [/?location=US&view=fludetailed&dates=2024-12-14,2024-12-21&models=FluSight-ensemble]

The 2024-25 influenza season has shown unique patterns compared to previous years. This narrative will walk you through the key insights from our forecasting models and surveillance data.

**Key Highlights:**
- Early season onset in several regions
- Unusual strain circulation patterns  
- Model performance variations across states
- Public health response adaptations

Let's explore the data together and understand what it tells us about this flu season. The visualization on the right shows the current national flu detailed view with our ensemble forecasting models.

# National Trends and Patterns [/?location=US&view=flutimeseries&dates=2024-11-30,2024-12-07,2024-12-14,2024-12-21&models=FluSight-ensemble,CU-ensemble,CMU-TimeSeries]

At the national level, we're seeing several interesting patterns emerge.

**Rising Activity:**
The flu activity has been steadily increasing since early December, with hospitalizations showing a steep upward trajectory. This aligns with typical seasonal patterns but with some notable accelerations.

**Model Convergence:**
Most forecasting models are showing good agreement on the near-term trajectory, which increases our confidence in the predictions. The time series view shows how different models compare over the past month.

**Regional Variations:**
While the national picture shows clear trends, there's significant variation at the state level that we'll explore next.

# Regional Spotlight: Northeast [/?location=NY&view=fludetailed&dates=2024-12-14,2024-12-21&models=FluSight-ensemble,CU-ensemble]

Let's examine New York as an example of the Northeast pattern.

**Early Season Onset:**
New York experienced one of the earliest flu season onsets this year, with activity ramping up in late October - nearly a month earlier than typical.

**High Model Confidence:**
The forecasting models show strong agreement for New York, suggesting the trajectory is well-established and predictable in the near term.

**Rapid Trajectory Changes:**
Despite the early onset, the rate of increase has been steeper than historical averages, putting additional strain on healthcare systems.

The visualization shows the detailed forecast view for New York, highlighting the ensemble model predictions and confidence intervals.

# Regional Spotlight: Southeast [/?location=FL&view=fludetailed&dates=2024-12-14,2024-12-21&models=FluSight-ensemble,CU-ensemble,CMU-TimeSeries]

Now let's compare this with Florida, representing the Southeast pattern.

**Delayed but Accelerating Growth:**
Florida showed a delayed onset compared to the Northeast, but is now experiencing rapid acceleration in flu activity.

**Higher Uncertainty:**
The forecasting models show more uncertainty for Florida, with wider confidence intervals reflecting the less predictable trajectory.

**Weather Pattern Influences:**
The delayed onset may be related to warmer weather patterns that persisted longer than usual in the Southeast region.

This comparison highlights how local factors can significantly impact disease dynamics and forecasting accuracy.

# Forecasting Model Performance [javascript:custom-accuracy-chart]

Our ensemble of forecasting models has performed with varying degrees of accuracy this season.

**Performance Metrics:**
- **Short-term forecasts (1-2 weeks)**: 85% accuracy
- **Medium-term forecasts (3-4 weeks)**: 72% accuracy  
- **Long-term forecasts (5+ weeks)**: 58% accuracy

**Model Insights:**
The top-performing models consistently incorporated:
1. Real-time syndromic surveillance data
2. Search trend analytics
3. Social mobility patterns
4. Weather and environmental factors

The custom visualization shows a detailed breakdown of model accuracy across different time horizons and regions.

# Public Health Implications [/?location=US&view=fludetailed&dates=2024-12-21&models=FluSight-ensemble]

Based on our analysis, several key implications emerge for public health decision-making.

**Timing of Interventions:**
The early onset in some regions suggests that intervention timing needs to be region-specific rather than following a national schedule.

**Resource Allocation:**
States showing rapid acceleration may need additional resource allocation for the coming weeks.

**Forecasting Insights:**
The good model agreement provides confidence for short-term planning, though longer-term uncertainty remains elevated.

**Recommendations:**
1. Enhanced surveillance in high-trajectory states
2. Early communication about vaccine importance
3. Flexible resource distribution strategies

The final view returns to the national perspective with our latest forecasts, showing the overall trajectory as we move through the peak season.`;

        parseNarrative(fallbackContent);
      }
    };

    loadNarrative();
  }, [id, parseNarrative]);

  const renderMarkdown = (content) => {
    return content.split("\n").map((line, index) => {
      if (line.startsWith("**") && line.endsWith("**")) {
        return (
          <Title key={index} order={4} mb="sm" mt="md">
            {line.slice(2, -2)}
          </Title>
        );
      }
      if (line.startsWith("- ")) {
        const text = line.substring(2);
        const parts = text.split(/(\*\*.*?\*\*)/);
        return (
          <Text key={index} component="li" mb="xs" ml="md">
            {parts.map((part, i) =>
              part.startsWith("**") && part.endsWith("**") ? (
                <strong key={i}>{part.slice(2, -2)}</strong>
              ) : (
                part
              ),
            )}
          </Text>
        );
      }
      if (line.match(/^\d+\./)) {
        return (
          <Text key={index} component="li" mb="xs" ml="md">
            {line.substring(line.indexOf(".") + 2)}
          </Text>
        );
      }
      if (line.trim()) {
        const parts = line.split(/(\*\*.*?\*\*)/);
        return (
          <Text key={index} mb="md">
            {parts.map((part, i) =>
              part.startsWith("**") && part.endsWith("**") ? (
                <strong key={i}>{part.slice(2, -2)}</strong>
              ) : (
                part
              ),
            )}
          </Text>
        );
      }
      return <div key={index} style={{ height: "0.5rem" }} />;
    });
  };

  const goToSlide = (index) => {
    if (index >= 0 && index < slides.length) {
      setCurrentSlide(index);
      const slide = slides[index];
      if (slide.url) {
        setCurrentVisualization(parseVisualizationUrl(slide.url));
      }
    }
  };

  const visualizationDetails = useMemo(() => {
    if (!currentVisualization || currentVisualization.type !== "respilens") {
      return null;
    }

    const params = new URLSearchParams();
    if (currentVisualization.location) {
      params.set("location", currentVisualization.location);
    }
    if (currentVisualization.view) {
      params.set("view", currentVisualization.view);
    }
    if (currentVisualization.dates?.length) {
      params.set("dates", currentVisualization.dates.join(","));
    }
    if (currentVisualization.models?.length) {
      params.set("models", currentVisualization.models.join(","));
    }

    return {
      url: `/?${params.toString()}`,
      location: currentVisualization.location,
      view: currentVisualization.view,
      dates: currentVisualization.dates,
      models: currentVisualization.models,
    };
  }, [currentVisualization]);

  const renderVisualization = () => {
    if (!currentVisualization) {
      return (
        <Center h="100%">
          <Text c="dimmed">No visualization specified for this slide</Text>
        </Center>
      );
    }

    if (currentVisualization.type === "custom") {
      // Handle specific custom visualizations
      if (currentVisualization.code === "plotly-gaussian") {
        return <PlotlyGaussianChart />;
      }

      // Default custom visualization placeholder
      return (
        <Center h="100%">
          <Stack align="center" gap="md">
            <ThemeIcon size="xl" variant="light">
              <IconCode size={32} />
            </ThemeIcon>
            <div style={{ textAlign: "center" }}>
              <Text fw={500} mb="xs">
                Custom Visualization
              </Text>
              <Text size="sm" c="dimmed">
                {currentVisualization.code}
              </Text>
              <Text size="xs" c="dimmed" mt="md">
                Custom JavaScript visualizations would be rendered here
              </Text>
            </div>
          </Stack>
        </Center>
      );
    }

    // Render RespiLens visualization
    return (
      <Stack gap="sm" style={{ height: "100%" }}>
        <Group justify="space-between" align="center">
          <Badge variant="light" size="sm">
            RespiLens View
          </Badge>
          {visualizationDetails?.url && (
            <Text size="xs" c="dimmed" style={{ fontFamily: "monospace" }}>
              {visualizationDetails.url}
            </Text>
          )}
        </Group>
        <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
          <DataVisualizationContainer
            location={currentVisualization.location}
            disableSeo
            // Additional props would be passed here to control the view
          />
        </div>
      </Stack>
    );
  };

  if (loading) {
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: "50vh" }}>
          <Stack align="center" gap="md">
            <Loader size="lg" />
            <Text>Loading narrative...</Text>
          </Stack>
        </Center>
      </Container>
    );
  }

  const currentSlideData = slides[currentSlide];

  return (
    <>
      <Seo
        title={
          metadata.title
            ? `RespiLens | ${metadata.title}`
            : "RespiLens | Narrative Viewer"
        }
        description={
          metadata.abstract ||
          metadata.description ||
          "Read an interactive RespiLens narrative with embedded respiratory disease data visualizations."
        }
        canonicalPath={`/narratives/${id || "flu-winter-2024-25-slides"}`}
        type="article"
      />
      <div
        style={{ height: "100vh", display: "flex", flexDirection: "column" }}
      >
        {/* Header */}
        <Paper shadow="sm" p="md" style={{ flexShrink: 0 }}>
          <Group justify="space-between" align="center">
            <Button
              variant="light"
              leftSection={<IconChevronLeft size={16} />}
              onClick={() => navigate("/narratives")}
            >
              Back
            </Button>
            <div>
              <Group gap="xs" mb="xs">
                <ThemeIcon size="sm" variant="light">
                  <IconBook size={16} />
                </ThemeIcon>
                <Text size="sm" c="dimmed">
                  Interactive Narrative
                </Text>
              </Group>
              <Title order={2}>{metadata.title}</Title>
            </div>
            <Group gap="xs">
              <Badge variant="light" size="sm">
                Slide {currentSlide + 1} of {slides.length}
              </Badge>
            </Group>
          </Group>
        </Paper>

        {/* Main Content */}
        <div
          style={{
            flex: 1,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            minHeight: 0,
          }}
        >
          {/* Left Panel - Slide Content */}
          <Paper
            p="xl"
            style={{
              overflow: "auto",
              borderRight: "1px solid var(--mantine-color-gray-3)",
            }}
          >
            <Stack gap="md" style={{ maxWidth: "600px" }}>
              <div>
                <Title order={3} mb="lg">
                  {currentSlideData?.title}
                </Title>
                <div style={{ lineHeight: 1.6 }}>
                  {renderMarkdown(currentSlideData?.content || "")}
                </div>
              </div>

              <Divider my="xl" />

              {/* Navigation */}
              <Group justify="space-between" align="center">
                <Button
                  variant="subtle"
                  leftSection={<IconChevronLeft size={16} />}
                  onClick={() => goToSlide(currentSlide - 1)}
                  disabled={currentSlide === 0}
                >
                  Previous
                </Button>

                <Group gap="xs">
                  {slides.map((_, index) => (
                    <ActionIcon
                      key={index}
                      variant={index === currentSlide ? "filled" : "subtle"}
                      size="sm"
                      onClick={() => goToSlide(index)}
                      aria-label={`Go to slide ${index + 1}`}
                      aria-current={index === currentSlide ? "true" : "false"}
                    >
                      {index + 1}
                    </ActionIcon>
                  ))}
                </Group>

                <Button
                  rightSection={<IconChevronRight size={16} />}
                  onClick={() => goToSlide(currentSlide + 1)}
                  disabled={currentSlide === slides.length - 1}
                >
                  Next
                </Button>
              </Group>

              {/* Slide metadata */}
              <Group gap="md" mt="xl">
                <Group gap="xs">
                  <IconUser size={16} />
                  <Text size="sm" c="dimmed">
                    {metadata.authors}
                  </Text>
                </Group>
                <Group gap="xs">
                  <IconCalendar size={16} />
                  <Text size="sm" c="dimmed">
                    {metadata.date}
                  </Text>
                </Group>
              </Group>
            </Stack>
          </Paper>

          {/* Right Panel - Visualization */}
          <Box style={{ position: "relative", overflow: "hidden" }}>
            {renderVisualization()}
          </Box>
        </div>
      </div>
    </>
  );
};

export default SlideNarrativeViewer;
