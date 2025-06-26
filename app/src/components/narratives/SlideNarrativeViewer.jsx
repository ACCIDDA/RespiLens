import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
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
  Divider
} from '@mantine/core';
import { 
  IconBook,
  IconCalendar,
  IconUser,
  IconChevronLeft,
  IconChevronRight,
  IconCode
} from '@tabler/icons-react';
import ForecastViz from '../ForecastViz';

const SlideNarrativeViewer = () => {
  const { id } = useParams();
  const [slides, setSlides] = useState([]);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [loading, setLoading] = useState(true);
  const [metadata, setMetadata] = useState({});
  const [currentVisualization, setCurrentVisualization] = useState(null);

  useEffect(() => {
    // Load narrative using dynamic import approach (works better with Vite)
    const narrativeId = id || 'flu-winter-2024-25-slides';
    
    console.log('Loading narrative:', narrativeId);
    
    const loadNarrative = async () => {
      try {
        // Try to import the narrative as a module
        const narrativeModule = await import(`../../data/narratives/${narrativeId}.js`);
        console.log('Loaded narrative module');
        parseNarrative(narrativeModule.narrativeContent);
      } catch (error) {
        console.error('Error loading narrative module:', error);
        
        // Fallback to embedded content
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
        
        console.log('Using fallback content');
        parseNarrative(fallbackContent);
      }
    };

    loadNarrative();
  }, [id]);

  const parseNarrative = (content) => {
    console.log('parseNarrative called with content length:', content?.length);
    
    try {
      // Split into frontmatter and content
      const parts = content.split('---');
      console.log('Split into parts:', parts.length);
      
      if (parts.length >= 3) {
        // Parse YAML frontmatter
        const frontmatterLines = parts[1].trim().split('\n');
        const parsedMetadata = {};
        frontmatterLines.forEach(line => {
          const [key, ...valueParts] = line.split(':');
          if (key && valueParts.length > 0) {
            parsedMetadata[key.trim()] = valueParts.join(':').trim().replace(/"/g, '');
          }
        });
        console.log('Parsed metadata:', parsedMetadata);
        setMetadata(parsedMetadata);

        // Parse slides
        const slideContent = parts.slice(2).join('---');
        console.log('Slide content length:', slideContent.length);
        
        const slideMatches = slideContent.split(/\n# /).filter(s => s.trim());
        console.log('Found slide matches:', slideMatches.length);
        
        const parsedSlides = slideMatches.map((slide, index) => {
          if (index === 0) {
            slide = slide.replace(/^# /, '');
          }
          
          // Extract title and URL from heading
          const lines = slide.split('\n');
          const titleLine = lines[0];
          const titleMatch = titleLine.match(/^(.*?)\s*\[(.*?)\]$/);
          
          let title, url;
          if (titleMatch) {
            title = titleMatch[1].trim();
            url = titleMatch[2].trim();
          } else {
            title = titleLine.trim();
            url = null;
          }

          const content = lines.slice(1).join('\n').trim();
          
          return { title, url, content };
        });

        console.log('Parsed slides:', parsedSlides.length, parsedSlides.map(s => s.title));
        setSlides(parsedSlides);
        
        // Set initial visualization
        if (parsedSlides[0]?.url) {
          setCurrentVisualization(parseVisualizationUrl(parsedSlides[0].url));
        } else if (parsedMetadata.dataset) {
          setCurrentVisualization(parseVisualizationUrl(parsedMetadata.dataset));
        }
      } else {
        console.error('Invalid narrative format - not enough parts after splitting by ---');
      }
    } catch (error) {
      console.error('Error parsing narrative:', error);
    }

    setLoading(false);
  };

  const parseVisualizationUrl = (url) => {
    if (!url) return null;
    
    if (url.startsWith('javascript:')) {
      return { type: 'custom', code: url.replace('javascript:', '') };
    }
    
    // Parse RespiLens URL parameters
    const urlObj = new URL(url, window.location.origin);
    const params = new URLSearchParams(urlObj.search);
    
    return {
      type: 'respilens',
      location: params.get('location') || 'US',
      view: params.get('view') || 'fludetailed',
      dates: params.get('dates')?.split(',') || [],
      models: params.get('models')?.split(',') || []
    };
  };

  const renderMarkdown = (content) => {
    return content
      .split('\n')
      .map((line, index) => {
        if (line.startsWith('**') && line.endsWith('**')) {
          return <Title key={index} order={4} mb="sm" mt="md">{line.slice(2, -2)}</Title>;
        }
        if (line.startsWith('- ')) {
          const text = line.substring(2);
          const parts = text.split(/(\*\*.*?\*\*)/);
          return (
            <Text key={index} component="li" mb="xs" ml="md">
              {parts.map((part, i) => 
                part.startsWith('**') && part.endsWith('**') 
                  ? <strong key={i}>{part.slice(2, -2)}</strong>
                  : part
              )}
            </Text>
          );
        }
        if (line.match(/^\d+\./)) {
          return <Text key={index} component="li" mb="xs" ml="md">{line.substring(line.indexOf('.') + 2)}</Text>;
        }
        if (line.trim()) {
          const parts = line.split(/(\*\*.*?\*\*)/);
          return (
            <Text key={index} mb="md">
              {parts.map((part, i) => 
                part.startsWith('**') && part.endsWith('**') 
                  ? <strong key={i}>{part.slice(2, -2)}</strong>
                  : part
              )}
            </Text>
          );
        }
        return <div key={index} style={{ height: '0.5rem' }} />;
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

  const renderVisualization = () => {
    if (!currentVisualization) {
      return (
        <Center h="100%">
          <Text c="dimmed">No visualization specified for this slide</Text>
        </Center>
      );
    }

    if (currentVisualization.type === 'custom') {
      return (
        <Center h="100%">
          <Stack align="center" gap="md">
            <ThemeIcon size="xl" variant="light">
              <IconCode size={32} />
            </ThemeIcon>
            <div style={{ textAlign: 'center' }}>
              <Text fw={500} mb="xs">Custom Visualization</Text>
              <Text size="sm" c="dimmed">{currentVisualization.code}</Text>
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
      <div style={{ height: '100%', overflow: 'hidden' }}>
        <ForecastViz 
          location={currentVisualization.location}
          // Additional props would be passed here to control the view
        />
      </div>
    );
  };

  if (loading) {
    return (
      <Container size="xl" py="xl">
        <Center style={{ minHeight: '50vh' }}>
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
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper shadow="sm" p="md" style={{ flexShrink: 0 }}>
        <Group justify="space-between" align="center">
          <div>
            <Group gap="xs" mb="xs">
              <ThemeIcon size="sm" variant="light">
                <IconBook size={16} />
              </ThemeIcon>
              <Text size="sm" c="dimmed">Interactive Narrative</Text>
            </Group>
            <Title order={2}>{metadata.title}</Title>
          </div>
          <Group gap="xs">
            <Badge variant="light" size="sm">Slide {currentSlide + 1} of {slides.length}</Badge>
          </Group>
        </Group>
      </Paper>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: 0 }}>
        {/* Left Panel - Slide Content */}
        <Paper p="xl" style={{ overflow: 'auto', borderRight: '1px solid var(--mantine-color-gray-3)' }}>
          <Stack gap="md" style={{ maxWidth: '600px' }}>
            <div>
              <Title order={3} mb="lg">{currentSlideData?.title}</Title>
              <div style={{ lineHeight: 1.6 }}>
                {renderMarkdown(currentSlideData?.content || '')}
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
                    variant={index === currentSlide ? 'filled' : 'subtle'}
                    size="sm"
                    onClick={() => goToSlide(index)}
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
                <Text size="sm" c="dimmed">{metadata.authors}</Text>
              </Group>
              <Group gap="xs">
                <IconCalendar size={16} />
                <Text size="sm" c="dimmed">{metadata.date}</Text>
              </Group>
            </Group>
          </Stack>
        </Paper>

        {/* Right Panel - Visualization */}
        <Box style={{ position: 'relative', overflow: 'hidden' }}>
          {renderVisualization()}
        </Box>
      </div>
    </div>
  );
};

export default SlideNarrativeViewer;