import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import ModelSelector from './ModelSelector';

const MODEL_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
];

const RSVDefaultView = ({ 
  location, 
  selectedDates,
  availableDates,
  setSelectedDates,
  setActiveDate,
  ageGroups = ["0-130", "0-0.99", "1-4", "5-64", "65-130"],
  getModelColor = (model, selectedModels) => {
    const index = selectedModels.indexOf(model);
    return MODEL_COLORS[index % MODEL_COLORS.length];
  }
}) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`./processed_data/rsv/${location}_rsv.json`);
        console.log('RSV fetch response:', response);
        if (!response.ok) {
          throw new Error(`No RSV data available for ${location} (status ${response.status})`);
        }
        const jsonData = await response.json();
        
        console.log('RSV data structure:', {
          metadata: jsonData.metadata ? 'present' : 'missing',
          ground_truth: Object.keys(jsonData.ground_truth || {}),
          forecasts: Object.keys(jsonData.forecasts || {}),
          ageGroups: ageGroups
        });

        // Validate forecast data structure
        if (jsonData.forecasts) {
          for (const [date, dateData] of Object.entries(jsonData.forecasts)) {
            for (const [age, ageData] of Object.entries(dateData)) {
              for (const [target, targetData] of Object.entries(ageData)) {
                for (const [model, modelData] of Object.entries(targetData)) {
                  console.log(`Model data for ${model} on ${date}:`, {
                    type: modelData.type,
                    predictions: Object.keys(modelData.predictions || {}).length
                  });
                }
              }
            }
          }
        }

        setData(jsonData);
        
        // Extract available models with more fludetailed logging
        const availableModels = new Set();
        Object.values(jsonData.forecasts || {}).forEach(dateData => {
          Object.values(dateData).forEach(ageData => {
            Object.values(ageData).forEach(targetData => {
              Object.keys(targetData).forEach(model => {
                console.log(`Found model ${model} with data:`, {
                  type: targetData[model].type,
                  predictions: Object.keys(targetData[model].predictions || {}).length
                });
                availableModels.add(model);
              });
            });
          });
        });
        
        const sortedModels = Array.from(availableModels).sort();
        console.log('Available models:', sortedModels);
        setModels(sortedModels);
        
        // Set default model selection if none selected
        if (selectedModels.length === 0 && sortedModels.length > 0) {
          const defaultModel = sortedModels.includes('hub-ensemble') ? 
            'hub-ensemble' : 
            sortedModels[0];
          console.log(`Setting default model: ${defaultModel}`);
          setSelectedModels([defaultModel]);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [location]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error || !data || !data.ground_truth || Object.keys(data.ground_truth).length === 0) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-gray-100 bg-opacity-50 rounded-lg">
        <div className="text-gray-500 text-center p-4">
          No RSV forecast data available for this location
        </div>
      </div>
    );
  }

  const getDefaultRange = (forRangeslider = false) => {
    if (!data?.ground_truth || !selectedDates.length) return undefined;
    
    const firstGroundTruthDate = new Date(data.ground_truth[ageGroups[0]].dates[0]);
    const lastGroundTruthDate = new Date(data.ground_truth[ageGroups[0]].dates.slice(-1)[0]);
    
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
  };

  console.log('Creating RSV traces:', {
    ageGroups,
    groundTruth: data.ground_truth,
    firstAge: data.ground_truth[ageGroups[0]]
  });

  // Create subplot traces for each age group
  const traces = ageGroups.map((age, index) => {
    // Get the age-specific ground truth data
    const ageData = data.ground_truth[age] || {};
    
    // Create base ground truth trace for this age group
    const groundTruthTrace = {
      x: ageData.dates || [],
      y: ageData.values || [],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Observed',  // Remove age group from name since it's in subplot title
      xaxis: `x${index + 1}`,  // Use consistent indexing
      yaxis: `y${index + 1}`,  // Use consistent indexing
      showlegend: index === 0,  // Only show in legend for first subplot
      line: { color: '#8884d8', width: 2 }
    };

    // Get model traces for this specific age group
    const modelTraces = selectedModels.flatMap(model => {
      const modelColor = getModelColor(model, selectedModels); // Use the passed in color function
      
      // Get all available forecast dates for this model and age group
      const availableForecastDates = Object.keys(data.forecasts || {})
        .filter(date => data.forecasts[date]?.[age]?.['inc hosp']?.[model])
        .sort();
      
      if (availableForecastDates.length === 0) {
        console.log(`No forecasts found for model ${model} and age group ${age}`);
        return [];
      }

      // Map over all selected dates to get forecasts for each
      return selectedDates.flatMap(forecastDate => {
        const forecastData = data.forecasts[forecastDate]?.[age]?.['inc hosp']?.[model];
        if (!forecastData?.type || !forecastData?.predictions) {
          console.log(`No valid forecast data for model ${model}, age group ${age}, date ${forecastDate}`);
          return [];
        }
        
        console.log(`Model: ${model}, Age Group: ${age}, Date: ${forecastDate}`);
      console.log('Forecast data structure:', {
        type: forecastData.type,
        predictions: Object.keys(forecastData.predictions || {}).length
      });

      if (!forecastData || forecastData.type !== 'quantile') {
        console.log(`Skipping model ${model} - no valid quantile forecast data`);
        return [];
      }

      // Process predictions
      const predictions = Object.entries(forecastData.predictions || {})
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0])); // Sort by horizon

      const forecastDates = [];
      const medianValues = [];
      const ci95Upper = [];
      const ci95Lower = [];
      const ci50Upper = [];
      const ci50Lower = [];

      predictions.forEach(([horizon, pred]) => {
        // Calculate target date based on forecast date + horizon weeks
        const targetDate = new Date(forecastDate);  // Use forecastDate from the outer scope
        targetDate.setDate(targetDate.getDate() + parseInt(horizon) * 7);
        forecastDates.push(targetDate.toISOString().split('T')[0]);
        
        // Extract quantiles
        const { quantiles, values } = pred;
        if (!quantiles || !values) {
          console.warn(`Missing quantiles/values for model ${model}, horizon ${horizon}`);
          return;
        }

        // Get specific quantile values
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

      return [
        {
          x: [...forecastDates, ...forecastDates.slice().reverse()],
          y: [...ci95Upper, ...ci95Lower.slice().reverse()],
          fill: 'toself',
          fillcolor: `${modelColor}10`,
          line: { color: 'transparent' },
          showlegend: false,
          type: 'scatter',
          name: `${model} 95% CI`,
          xaxis: `x${index + 1}`,
          yaxis: `y${index + 1}`
        },
        {
          x: [...forecastDates, ...forecastDates.slice().reverse()],
          y: [...ci50Upper, ...ci50Lower.slice().reverse()],
          fill: 'toself',
          fillcolor: `${modelColor}30`,
          line: { color: 'transparent' },
          showlegend: false,
          type: 'scatter',
          name: `${model} 50% CI`,
          xaxis: `x${index + 1}`,
          yaxis: `y${index + 1}`
        },
        {
          x: forecastDates,
          y: medianValues,
          name: `${model}`,
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: modelColor, width: 2 },
          marker: { size: 6 },
          showlegend: index === 0, // Only show in legend for first age group to avoid duplicates
          xaxis: `x${index + 1}`,
          yaxis: `y${index + 1}`
        }];
      }); // Close the selectedDates.flatMap
    });

    return [groundTruthTrace, ...modelTraces];
  }).flat();

  const layout = {
    grid: {
      rows: 3,
      columns: 2,
      pattern: 'independent',
      roworder: 'top to bottom',
      subplots: [
        ['xy'],          // First row spans full width
        ['x2y2', 'x3y3'], // Second row for first two age groups
        ['x4y4', 'x5y5']  // Third row for last two age groups
      ],
      rowheights: [0.6, 0.2, 0.2], // Adjusted to make first row significantly taller
      columnwidths: [0.5, 0.5]
    },
    height: 1000,
    margin: { l: 60, r: 30, t: 50, b: 30 },
    showlegend: false, // Remove legend
    // Update domain ranges for subplots
    xaxis: { 
      domain: [0, 1],
      rangeslider: {
        range: getDefaultRange(true)
      },
      range: getDefaultRange(),
      rangeselector: {
        buttons: [
          {count: 1, label: '1m', step: 'month', stepmode: 'backward'},
          {count: 6, label: '6m', step: 'month', stepmode: 'backward'},
          {step: 'all', label: 'all'}
        ]
      }
    },
    xaxis2: { 
      domain: [0, 0.48],
      range: getDefaultRange()
    },
    xaxis3: { 
      domain: [0.52, 1],
      range: getDefaultRange()
    },
    xaxis4: { 
      domain: [0, 0.48],
      range: getDefaultRange()
    },
    xaxis5: { 
      domain: [0.52, 1],
      range: getDefaultRange()
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: 1.05,
      x: 0.5,
      xanchor: 'center'
    },
    annotations: ageGroups.map((age, index) => {
      if (index === 0) {
        return {
          text: `Overall (Age ${age})`,
          xref: 'paper',
          yref: 'paper',
          x: 0.5,
          y: 0.95,
          showarrow: false,
          font: { size: 16, weight: 'bold' }
        };
      } else {
        const row = Math.floor((index - 1) / 2) + 1;  // 1 for second row, 2 for third row
        const col = ((index - 1) % 2);  // 0 for left, 1 for right
        return {
          text: `Age ${age}`,
          xref: 'paper',
          yref: 'paper',
          x: col === 0 ? 0.24 : 0.76,   // Adjusted x positions
          y: row === 1 ? 0.6 : 0.25,    // Adjusted y positions
          showarrow: false,
          font: { size: 14, weight: 'bold' }
        };
      }
    })
  };

  return (
    <div>
      <Plot
        data={traces}
        layout={layout}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToAdd: [{
            name: 'Reset view',
            click: function(gd) {
              const range = getDefaultRange();
              if (range) {
                Plotly.relayout(gd, {
                  'xaxis.range': range,
                  'xaxis2.range': range,
                  'xaxis3.range': range,
                  'xaxis4.range': range,
                  'xaxis5.range': range,
                  'xaxis.rangeslider.range': getDefaultRange(true)
                });
              }
            }
          }]
        }}
        className="w-full"
      />
      <ModelSelector 
        models={models}
        selectedModels={selectedModels}
        setSelectedModels={setSelectedModels}
        getModelColor={(model, selectedModels) => {
          const index = selectedModels.indexOf(model);
          return MODEL_COLORS[index % MODEL_COLORS.length];
        }}
      />
    </div>
  );
};

export default RSVDefaultView;
