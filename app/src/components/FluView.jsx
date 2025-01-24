import React from 'react';
import Plot from 'react-plotly.js';
import ModelSelector from './ModelSelector';

export const MODEL_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
];

const FluView = ({ data, selectedDates, selectedModels, models, setSelectedModels, viewType, windowSize, getDefaultRange }) => {
  const getTimeSeriesData = () => {
    if (!data || selectedDates.length === 0) {
      console.log('Early return from getTimeSeriesData:', { data, selectedDates });
      return null;
    }
    
    console.log('Ground truth data:', data.ground_truth);

    const groundTruthTrace = {
      x: data.ground_truth.dates,
      y: data.ground_truth.values,
      name: 'Observed',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#8884d8', width: 2 },
      marker: { size: 6 }
    };

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

        const sortedPredictions = Object.entries(forecast.predictions || {})
          .sort((a, b) => new Date(a[1].date) - new Date(b[1].date));
        
        sortedPredictions.forEach(([horizon, pred]) => {
          forecastDates.push(pred.date);
          
          if (forecast.type !== 'quantile') {
            return;
          }
          const quantiles = pred.quantiles || [];
          const values = pred.values || [];
          
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
  };

  const getRateChangeData = () => {
    if (!data || selectedDates.length === 0) return null;

    const categoryOrder = [
      'large_decrease',
      'decrease',
      'stable',
      'increase',
      'large_increase'
    ];

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
  };

  const timeSeriesData = getTimeSeriesData() || [];
  const rateChangeData = getRateChangeData() || [];

  console.log('FluView plotting data:', {
    traces: [...timeSeriesData, ...(viewType === 'fludetailed' ? rateChangeData.map(trace => ({
      ...trace,
      orientation: 'h',
      xaxis: 'x2',
      yaxis: 'y2'
    })) : [])],
    selectedDates,
    data: data?.ground_truth
  });

  const layout = {
    width: Math.min(1200, windowSize.width * 0.8),
    height: Math.min(800, windowSize.height * 0.6),
    autosize: true,
    grid: viewType === 'fludetailed' ? {
      columns: 1,
      rows: 1,
      pattern: 'independent',
      subplots: [['xy'], ['x2y2']],
      xgap: 0.15
    } : undefined,
    showlegend: false,
    hovermode: 'x unified',
    margin: { l: 60, r: 30, t: 30, b: 30 },
    xaxis: {
      domain: viewType === 'fludetailed' ? [0, 0.8] : [0, 1],
      rangeslider: {
        range: getDefaultRange(true)
      },
      rangeselector: {
        buttons: [
          {count: 1, label: '1m', step: 'month', stepmode: 'backward'},
          {count: 6, label: '6m', step: 'month', stepmode: 'backward'},
          {step: 'all', label: 'all'}
        ]
      },
      range: getDefaultRange()
    },
    shapes: selectedDates.map(date => ({
      type: 'line',
      x0: date,
      x1: date,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: {
        color: 'red',
        width: 1,
        dash: 'dash'
      }
    })),
    yaxis: {
      title: 'Hospitalizations'
    },
    ...(viewType === 'fludetailed' ? {
      xaxis2: {
        domain: [0.85, 1],
        showgrid: false
      },
      yaxis2: {
        title: '',
        showticklabels: true,
        type: 'category',
        side: 'right',
        automargin: true,
        tickfont: { align: 'right' }
      }
    } : {}),
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarPosition: 'left',
    showSendToCloud: false,
    plotlyServerURL: "",
    toImageButtonOptions: {
      format: 'png',
      filename: 'forecast_plot'
    },
    modeBarButtonsToAdd: [{
      name: 'Reset view',
      click: function(gd) {
        const range = getDefaultRange();
        if (range) {
          Plotly.relayout(gd, {
            'xaxis.range': range,
            'xaxis.rangeslider.range': range
          });
        }
      }
    }]
  };

  return (
    <div>
      <div className="w-full" style={{ height: Math.min(800, windowSize.height * 0.6) }}>
        <Plot
          style={{ width: '100%', height: '100%' }}
        data={[
          ...timeSeriesData,
          ...(viewType === 'fludetailed' 
            ? rateChangeData.map(trace => ({
                ...trace,
                orientation: 'h',
                xaxis: 'x2',
                yaxis: 'y2'
              }))
            : [])
        ]}
        layout={{
          width: Math.min(1200, windowSize.width * 0.8),
          height: Math.min(800, windowSize.height * 0.6),
          autosize: true,
          grid: viewType === 'fludetailed' ? {
            columns: 1,
            rows: 1,
            pattern: 'independent',
            subplots: [['xy'], ['x2y2']],
            xgap: 0.15
          } : undefined,
          showlegend: false,
          hovermode: 'x unified',
          margin: { l: 60, r: 30, t: 30, b: 30 },
          xaxis: {
            domain: viewType === 'fludetailed' ? [0, 0.8] : [0, 1],
            rangeslider: {
              range: getDefaultRange(true)
            },
            rangeselector: {
              buttons: [
                {count: 1, label: '1m', step: 'month', stepmode: 'backward'},
                {count: 6, label: '6m', step: 'month', stepmode: 'backward'},
                {step: 'all', label: 'all'}
              ]
            },
            range: getDefaultRange()
          },
          shapes: selectedDates.map(date => ({
            type: 'line',
            x0: date,
            x1: date,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
              color: 'red',
              width: 1,
              dash: 'dash'
            }
          })),
          yaxis: {
            title: 'Hospitalizations'
          },
          ...(viewType === 'fludetailed' ? {
            xaxis2: {
              domain: [0.85, 1],
              showgrid: false
            },
            yaxis2: {
              title: '',
              showticklabels: true,
              type: 'category',
              side: 'right',
              automargin: true,
              tickfont: { align: 'right' }
            }
          } : {}),
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarPosition: 'left',
          showSendToCloud: false,
          plotlyServerURL: "",
          toImageButtonOptions: {
            format: 'png',
            filename: 'forecast_plot'
          },
          modeBarButtonsToAdd: [{
            name: 'Reset view',
            click: function(gd) {
              const range = getDefaultRange();
              if (range) {
                Plotly.relayout(gd, {
                  'xaxis.range': range,
                  'xaxis.rangeslider.range': range
                });
              }
            }
          }]
        }}
      />
      </div>
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

export default FluView;
