import React, { useState, useEffect, useCallback } from 'react';
import StateSelector from './StateSelector';
import RSVDefaultView from './RSVDefaultView';
import FluView from './FluView';
import InfoOverlay from './InfoOverlay';
import { useView } from '../contexts/ViewContext';
import { useSearchParams } from 'react-router-dom';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import ViewSelector from './ViewSelector';
import Plot from 'react-plotly.js';
import NHSNRawView from './NHSNRawView';
import DateSelector from './DateSelector';
import { getDataPath } from '../utils/paths';
import { DATASETS } from '../config/datasets';

// Color palette for model visualization
const MODEL_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
];

const getModelColor = (model, selectedModels) => {
  const index = selectedModels.indexOf(model);
  return index >= 0 ? MODEL_COLORS[index % MODEL_COLORS.length] : null;
};

const ForecastViz = ({ location, handleStateSelect }) => {
  // 1. First declare all hooks
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const {
    selectedModels, setSelectedModels,
    selectedDates, setSelectedDates,
    activeDate, setActiveDate,
    viewType, setViewType,  // Add this line
    currentDataset
  } = useView();
  const [availableDates, setAvailableDates] = useState([]);
  const [models, setModels] = useState([]);
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });
  const [searchParams, setSearchParams] = useSearchParams();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // 2. Define all useEffects
  useEffect(() => {
    const urlView = searchParams.get('view');
    if (urlView && ['fludetailed', 'flutimeseries', 'rsvdetailed', 'nhsnall'].includes(urlView)) {
      setViewType(urlView);
    }
  }, []);

  useEffect(() => {
    if (selectedDates.length > 0 && selectedModels.length > 0) {
      const newParams = new URLSearchParams(searchParams);
      const prefix = viewType === 'rsvdetailed' ? 'rsv' : viewType === 'nhsnall' ? 'nhsn' : 'flu';
      newParams.set(`${prefix}_dates`, selectedDates.join(','));
      newParams.set(`${prefix}_models`, selectedModels.join(','));
      newParams.set('location', location);
      newParams.set('view', viewType);
      setSearchParams(newParams, { replace: true });
    }
  }, [selectedDates, selectedModels, viewType, location, searchParams, setSearchParams]);

  useEffect(() => {
    if (!loading && data && availableDates.length > 0 && models.length > 0 &&
        (selectedDates.length === 0 || selectedModels.length === 0)) {

      const prefix = viewType === 'rsvdetailed' ? 'rsv' : viewType === 'nhsnall' ? 'nhsn' : 'flu';
      const urlDates = searchParams.get(`${prefix}_dates`)?.split(',') || [];
      const urlModels = searchParams.get(`${prefix}_models`)?.split(',') || [];

      // Only set dates if none are selected for current view type
      if (selectedDates.length === 0) {
        const validDates = urlDates
          .filter(date => availableDates.includes(date))
          .sort();  // Sort the dates chronologically

        if (validDates.length > 0) {
          // Set all valid dates from URL instead of just the first one
          setSelectedDates(validDates);
          // Set active date to the most recent one
          setActiveDate(validDates[validDates.length - 1]);
        } else {
          const latestDate = availableDates[availableDates.length - 1];
          setSelectedDates([latestDate]);
          setActiveDate(latestDate);
        }
      }

      // Only set models if none are selected for current view type
      if (selectedModels.length === 0) {
        const requestedModels = urlModels.filter(Boolean); // Filter out any empty strings
        console.log('Initializing models:', {
          requestedModels,
          availableModels: models,
          currentSelection: selectedModels
        });

        if (requestedModels.length > 0) {
          // Try to match each requested model exactly
          const validModels = requestedModels.filter(model => models.includes(model));
          console.log('Found valid models:', validModels);

          if (validModels.length > 0) {
            setSelectedModels(validModels);
            console.log('Setting models from URL:', validModels);
            return;  // Exit early if we found valid models
          }
        }

        // If no valid models found, set default
        const defaultModel = viewType === 'rsvdetailed' ?
          (models.includes('hub-ensemble') ? 'hub-ensemble' : models[0]) :
          (models.includes('FluSight-ensemble') ? 'FluSight-ensemble' : models[0]);
        console.log('Setting default model:', defaultModel);
        setSelectedModels([defaultModel]);
      }
    }
  }, [loading, data, availableDates, models, viewType]);

  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    console.log('ForecastViz useEffect triggered:', { viewType, location });

    // Skip loading forecast data if we're in NHSN view
    if (viewType === 'nhsnall') {
      return;
    }

    const fetchData = async () => {
      try {
        // Don't clear state when fetching new data
        setData(null);
        setError(null);
        setLoading(true);

        // Don't clear these anymore:
        // setSelectedDates([]);
        // setSelectedModels([]);
        // setAvailableDates([]);
        // setModels([]);

        // Determine which file to load based on view type
        const dataPathForFetching = currentDataset?.dataPath;
        if (!dataPathForFetching) {
            console.error("dataPath not found in currentDataset", currentDataset);
            setError(`Data path not configured for dataset: ${currentDataset?.shortName || 'current view'}`);
            setLoading(false);
            return;
        }

        const url = getDataPath(`${dataPathForFetching}/${location}.json`);
        console.log('Attempting to fetch:', url);

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to load data from ${url} for ${location} (status ${response.status})`);
        }

        const text = await response.text();
        console.log('Raw response text:', text.slice(0, 500) + '...');

        const parsedData = JSON.parse(text);
        console.log('Parsed JSON structure:', {
          hasMetadata: !!parsedData.metadata,
          hasGroundTruth: !!parsedData.ground_truth,
          topLevelKeys: Object.keys(parsedData)
        });

        if (!parsedData || typeof parsedData !== 'object') {
          throw new Error('Invalid JSON response: not an object');
        }
        if (!parsedData.metadata) {
          throw new Error('Invalid JSON response: missing metadata');
        }
        if (!parsedData.ground_truth) {
          throw new Error('Invalid JSON response: missing ground_truth');
        }

        setData(parsedData);

        // Initialize dates and models
        if (viewType === 'rsvdetailed') {
          const dates = Object.keys(parsedData.forecasts || {}).sort();
          setAvailableDates(dates);
          if (dates.length > 0) {
            setSelectedDates([dates[dates.length - 1]]);
            setActiveDate(dates[dates.length - 1]);
          }
        } else {
          // For flu view
          const dates = Object.keys(parsedData.forecasts || {}).sort();
          const extractedModels = new Set();

          dates.forEach(date => {
            ['wk inc flu hosp', 'wk flu hosp rate change'].forEach(type => {
              const typeForecast = parsedData.forecasts[date]?.[type] || {};
              Object.keys(typeForecast).forEach(model => extractedModels.add(model));
            });
          });

          const modelList = Array.from(extractedModels).sort((a, b) => a.localeCompare(b));
          console.log('Extracted models from data:', {
            modelList,
            dates: dates.map(date => ({
              date,
              models: Object.keys(parsedData.forecasts[date]?.['wk inc flu hosp'] || {})
            }))
          });
          setModels(modelList);
          setAvailableDates(dates);
        }
      } catch (err) {
        console.error('Data loading error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (location) {
      fetchData();
    }
  }, [location, viewType]);

  // 3. Define callbacks
  const getDefaultRange = useCallback((forRangeslider = false) => {
    if (!data?.ground_truth?.dates?.length || selectedDates.length === 0) return undefined;

    const firstGroundTruthDate = new Date(data.ground_truth.dates[0]);
    const lastGroundTruthDate = new Date(data.ground_truth.dates[data.ground_truth.dates.length - 1]);

    if (forRangeslider) {
      const rangesliderEnd = new Date(lastGroundTruthDate);
      rangesliderEnd.setDate(rangesliderEnd.getDate() + (5 * 7));
      return [firstGroundTruthDate, rangesliderEnd];
    }

    const firstDate = new Date(selectedDates[0]);
    const lastDate = new Date(selectedDates[selectedDates.length - 1]);

    const startDate = new Date(firstDate);
    const endDate = new Date(lastDate);

    startDate.setDate(startDate.getDate() - (8 * 7));
    endDate.setDate(endDate.getDate() + (5 * 7));

    return [startDate, endDate];
  }, [data, selectedDates]);

  const getTimeSeriesData = useCallback(() => {
    if (!data || selectedDates.length === 0) return null;

    // Ground truth trace
    const groundTruthTrace = {
      x: data.ground_truth.dates,
      y: data.ground_truth.values,
      name: 'Observed',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#8884d8', width: 2 },
      marker: { size: 6 }
    };

    // Generate traces for each selected model and date combination
    const modelTraces = selectedModels.flatMap(model =>
      selectedDates.flatMap((date) => {
        const forecasts = data.forecasts[date] || {};
        const forecast =
          forecasts['wk inc flu hosp']?.[model] ||
          forecasts['wk flu hosp rate change']?.[model];

        if (!forecast) return [];

        const forecastDates = [];
        const medianValues = [];
        const ci95Upper = [];
        const ci95Lower = [];
        const ci50Upper = [];
        const ci50Lower = [];

        // Process all horizons and sort by target date
        const sortedPredictions = Object.entries(forecast.predictions || {})
          .sort((a, b) => new Date(a[1].date) - new Date(b[1].date));

        sortedPredictions.forEach(([horizon, pred]) => {
          forecastDates.push(pred.date);

          if (forecast.type !== 'quantile') {
            return;
          }
          const quantiles = pred.quantiles || [];
          const values = pred.values || [];

          // Default to 0 if quantile not found
          const q95Lower = values[quantiles.indexOf(0.025)] || 0;
          const q50Lower = values[quantiles.indexOf(0.25)] || 0;
          const median = values[quantiles.indexOf(0.5)] || 0;
          const q50Upper = values[quantiles.indexOf(0.75)] || 0;
          const q95Upper = values[quantiles.indexOf(0.975)] || 0;

          ci95Lower.push(q95Lower);
          ci50Lower.push(q50Lower);
          medianValues.push(median);
          ci50Upper.push(q50Upper);
          ci95Upper.push(q95Upper);
        });

        const modelColor = MODEL_COLORS[selectedModels.indexOf(model) % MODEL_COLORS.length];

        return [
          {
            x: [...forecastDates, ...forecastDates.slice().reverse()],
            y: [...ci95Upper, ...ci95Lower.slice().reverse()],
            fill: 'toself',
            fillcolor: `${modelColor}10`,
            line: { color: 'transparent' },
            showlegend: false,
            type: 'scatter',
            name: `${model} (${date}) 95% CI`
          },
          {
            x: [...forecastDates, ...forecastDates.slice().reverse()],
            y: [...ci50Upper, ...ci50Lower.slice().reverse()],
            fill: 'toself',
            fillcolor: `${modelColor}30`,
            line: { color: 'transparent' },
            showlegend: false,
            type: 'scatter',
            name: `${model} (${date}) 50% CI`
          },
          {
            x: forecastDates,
            y: medianValues,
            name: `${model} (${date})`,
            type: 'scatter',
            mode: 'lines+markers',
            line: {
              color: modelColor,
              width: 2,
              dash: 'solid'
            },
            marker: { size: 6, color: modelColor },
            showlegend: true
          }
        ];
      })
    );

    return [groundTruthTrace, ...modelTraces];
  }, [data, selectedDates, selectedModels]);

  const getRateChangeData = useCallback(() => {
    if (!data || selectedDates.length === 0) return null;

    const categoryOrder = [
      'large_decrease',
      'decrease',
      'stable',
      'increase',
      'large_increase'
    ];

    // Only show rate change data for most recent selected date
    const lastSelectedDate = selectedDates.slice().sort().pop();

    return selectedModels.map(model => {
      const forecast = data.forecasts[lastSelectedDate]?.['wk flu hosp rate change']?.[model];
      if (!forecast) return null;

      const horizon0 = forecast.predictions['0'];
      if (!horizon0) return null;

      const modelColor = MODEL_COLORS[selectedModels.indexOf(model) % MODEL_COLORS.length];

      const orderedData = categoryOrder.map(cat => ({
        category: cat.replace('_', '<br>'),
        value: horizon0.probabilities[horizon0.categories.indexOf(cat)] * 100
      }));

      return {
        name: `${model} (${lastSelectedDate})`,
        y: orderedData.map(d => d.category),
        x: orderedData.map(d => d.value),
        type: 'bar',
        orientation: 'h',
        marker: { color: modelColor },
        showlegend: true,
        legendgroup: 'histogram',
        xaxis: 'x2',
        yaxis: 'y2'
      };
    }).filter(Boolean);
  }, [data, selectedDates, selectedModels]);

  // Get dataset-specific view component
  const getDatasetView = useCallback(() => {
    switch(currentDataset?.shortName) {
      case 'rsv':
        return (
          <RSVDefaultView
            location={location}
            selectedDates={selectedDates}
            availableDates={availableDates}
            setSelectedDates={setSelectedDates}
            setActiveDate={setActiveDate}
            selectedModels={selectedModels}
            setSelectedModels={setSelectedModels}
          />
        );
      case 'nhsn':
        return <NHSNRawView location={location} />;
      case 'flu':
      default:
        return (
          <FluView
            data={data}
            selectedDates={selectedDates}
            selectedModels={selectedModels}
            models={models}
            setSelectedModels={setSelectedModels}
            viewType={viewType}
            windowSize={windowSize}
            getDefaultRange={getDefaultRange}
          />
        );
    }
  }, [currentDataset, location, data, selectedDates, availableDates, selectedModels, viewType, windowSize, getDefaultRange]);

  // 5. Rest of the component logic
  if (loading) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex justify-center items-center h-96">
          <div className="animate-pulse text-lg">Loading forecast data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex flex-col items-center justify-center h-96">
          <div className="text-red-500 text-center mb-4">
            Error loading forecast data: {error}
          </div>
          <div className="text-sm text-gray-600 max-w-lg text-center">
            Please ensure that:
            <ul className="list-disc text-left mt-2 space-y-1">
              <li>The data processing scripts have been run</li>
              <li>Data files are present in processed_data/</li>
              <li>The selected location has forecast data</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex justify-center items-center h-96">
          <div className="text-gray-500">No forecast data available</div>
        </div>
      </div>
    );
  }

  const timeSeriesData = getTimeSeriesData();
  const rateChangeData = getRateChangeData();

  console.log('ForecastViz render state:', {
    viewType,
    data,
    selectedDates,
    selectedModels,
    timeSeriesData,
    rateChangeData
  });

  return (
    <div className="flex h-screen">
      {!sidebarCollapsed && (
        <StateSelector
          onStateSelect={handleStateSelect}
          currentLocation={location}
          sidebarMode={true}
        />
      )}
      <div className="flex-1 overflow-auto relative">
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="absolute top-4 left-4 z-10 p-2 rounded-full bg-white shadow hover:bg-gray-50"
        >
          {sidebarCollapsed ? <ChevronRight /> : <ChevronLeft />}
        </button>
        <div className="container mx-auto p-4">
          <div className="border rounded-lg shadow-sm bg-white">
            {currentDataset?.hasDateSelector && (
              <div className="p-4 border-b">
                <DateSelector
                  availableDates={availableDates}
                  selectedDates={selectedDates}
                  setSelectedDates={setSelectedDates}
                  activeDate={activeDate}
                  setActiveDate={setActiveDate}
                />
              </div>
            )}

            <div className="p-4">
              {getDatasetView()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastViz;
