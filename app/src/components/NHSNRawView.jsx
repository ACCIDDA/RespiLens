import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import ModelSelector from './ModelSelector';
import { getDataPath } from '../utils/paths';
import { useSearchParams } from 'react-router-dom';
import { ChevronLeft } from 'lucide-react';
import ViewSelector from './ViewSelector';
import InfoOverlay from './InfoOverlay';
import { useView } from '../contexts/ViewContext';
import NHSNColumnSelector from './NHSNColumnSelector';
import { VISUALIZATION_COLORS } from '../config/datasets';

const NHSNRawView = ({ location }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { currentDataset } = useView();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedColumns, setSelectedColumns] = useState(() => {
    return searchParams.get('nhsn_columns')?.split(',') || ['totalconfflunewadm'];
  });
  const [availableColumns, setAvailableColumns] = useState({
    official: [],
    preliminary: []
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Use currentDataset.dataPath for the URL
        const url = getDataPath(`${currentDataset.dataPath}/${location}.json`);
        console.log('Fetching NHSN data from:', url);

        const response = await fetch(url);
        console.log('NHSN response status:', response.status);

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('No NHSN data available for this location');
          }
          throw new Error('Failed to load NHSN data');
        }

        const text = await response.text();
        console.log('Raw NHSN response:', text.slice(0, 500) + '...');

        const jsonData = JSON.parse(text);
        console.log('Parsed NHSN data structure (V2):', {
          hasOfficial: !!jsonData.official,
          hasPreliminary: !!jsonData.preliminary,
          topLevelKeys: Object.keys(jsonData)
        });

        // Validate the data structure (V2)
        if (!jsonData.official && !jsonData.preliminary) {
          throw new Error('Invalid data format: Missing both official and preliminary data.');
        }

        setData(jsonData);

        // Get available columns (only those with data) (V2)
        const officialCols = Object.keys(jsonData.official?.series?.columns || {}).sort();
        const prelimCols = Object.keys(jsonData.preliminary?.series?.columns || {}).sort();

        setAvailableColumns({
          official: officialCols,
          preliminary: prelimCols
        });

        // Get columns from URL if any, otherwise select only totalconfflunewadm
        const urlColumns = searchParams.get('nhsn_columns')?.split(',').filter(Boolean);
        if (urlColumns?.length > 0) {
          const validColumns = urlColumns.filter(col =>
            officialCols.includes(col) || prelimCols.includes(col)
          );
          setSelectedColumns(validColumns);
        } else {
          // By default, select only totalconfflunewadm
          const defaultColumn = officialCols.find(col => col === 'totalconfflunewadm') || officialCols[0];
          setSelectedColumns([defaultColumn]);
        }

      } catch (err) {
        console.error('Error loading NHSN data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (location) {
      fetchData();
    }
  }, [location]);

  // Update URL when columns change
  useEffect(() => {
    const newParams = new URLSearchParams(searchParams);
    if (selectedColumns.length > 0) {
      newParams.set('nhsn_columns', selectedColumns.join(','));
    } else {
      newParams.delete('nhsn_columns');
    }
    setSearchParams(newParams, { replace: true });
  }, [selectedColumns]);

  if (loading) return <div className="p-4">Loading NHSN data...</div>;
  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!data) return <div className="p-4">No NHSN data available for this location</div>;

  const traces = selectedColumns.map((columnName, index) => {
    let seriesData;
    let seriesDates;
    let seriesTypeLabel = '';
    let titleLocationName = 'N/A';

    if (data.official?.series?.columns[columnName] && data.official?.series?.dates) {
      seriesData = data.official.series.columns[columnName];
      seriesDates = data.official.series.dates;
      seriesTypeLabel = ' (Official)';
      titleLocationName = data.official.metadata.location_name || titleLocationName;
    } else if (data.preliminary?.series?.columns[columnName] && data.preliminary?.series?.dates) {
      seriesData = data.preliminary.series.columns[columnName];
      seriesDates = data.preliminary.series.dates;
      seriesTypeLabel = ' (Preliminary)';
      titleLocationName = data.preliminary.metadata.location_name || titleLocationName;
    }

    if (!seriesData || !seriesDates || seriesDates.length === 0) return null;

    const colorIndex = [...availableColumns.official, ...availableColumns.preliminary].indexOf(columnName);

    return {
      x: seriesDates,
      y: seriesData,
      name: `${columnName}${seriesTypeLabel}`,
      type: 'scatter',
      mode: 'lines+markers',
      line: {
        color: VISUALIZATION_COLORS[colorIndex % VISUALIZATION_COLORS.length],
        width: 2
      },
      marker: { size: 6 }
    };
  }).filter(Boolean); // Remove null traces

  // Determine title and x-axis range from the first available series or data
  let plotTitle = 'NHSN Raw Data';
  let xaxisRange = [null, null]; // Default to auto-range

  if (traces.length > 0 && traces[0].x && traces[0].x.length > 0) {
    // Attempt to get location name from the first trace's source metadata
    const firstTraceName = traces[0].name;
    let firstTraceSourceMetadata;
    if (firstTraceName.includes('(Official)') && data.official?.metadata) {
        firstTraceSourceMetadata = data.official.metadata;
    } else if (firstTraceName.includes('(Preliminary)') && data.preliminary?.metadata) {
        firstTraceSourceMetadata = data.preliminary.metadata;
    } else if (data.official?.metadata) { // Fallback if label is missing for some reason
        firstTraceSourceMetadata = data.official.metadata;
    } else if (data.preliminary?.metadata) {
        firstTraceSourceMetadata = data.preliminary.metadata;
    }

    if (firstTraceSourceMetadata?.location_name) {
        plotTitle = `NHSN Raw Data for ${firstTraceSourceMetadata.location_name}`;
    }
    xaxisRange = [traces[0].x[0], traces[0].x[traces[0].x.length - 1]];
  } else if (data.official?.metadata?.location_name) {
    plotTitle = `NHSN Raw Data for ${data.official.metadata.location_name}`;
  } else if (data.preliminary?.metadata?.location_name) {
    plotTitle = `NHSN Raw Data for ${data.preliminary.metadata.location_name}`;
  }


  const layout = {
    title: plotTitle,
    xaxis: {
      title: 'Date',
      rangeslider: {
        visible: true
      },
      range: xaxisRange
    },
    yaxis: {
      title: 'Value'
    },
    height: 600,
    showlegend: false,  // Hide legend
    margin: { t: 40, r: 10, l: 60, b: 120 }  // Adjust margins to fit everything
  };

  return (
    <div className="w-full">
      <Plot
        data={traces}
        layout={layout}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToAdd: ['resetScale2d']
        }}
        className="w-full"
      />

      <NHSNColumnSelector
        availableColumns={availableColumns}
        selectedColumns={selectedColumns}
        setSelectedColumns={setSelectedColumns}
      />
    </div>
  );
};

export default NHSNRawView;
