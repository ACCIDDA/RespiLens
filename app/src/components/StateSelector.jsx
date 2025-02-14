import React, { useState, useEffect } from 'react';
import ViewSelector from './ViewSelector';
import InfoOverlay from './InfoOverlay';
import { getDataPath } from '../utils/paths';

const StateSelector = ({ onStateSelect, currentLocation = null, sidebarMode = false }) => {
  const [states, setStates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const filteredStates = states.filter(state =>
    state.location_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    state.abbreviation.toLowerCase().includes(searchTerm.toLowerCase())
  );

  useEffect(() => {
    const fetchStates = async () => {
      try {
        // Fetch manifest using getDataPath
        const manifestResponse = await fetch(getDataPath('flusight/metadata.json'));
        if (!manifestResponse.ok) {
          throw new Error(`Failed to fetch metadata: ${manifestResponse.statusText}`);
        }

        const metadata = await manifestResponse.json();
        console.log('Loaded metadata:', metadata);

        if (!metadata.locations || !Array.isArray(metadata.locations)) {
          throw new Error('Invalid metadata format');
        }

        // Use the locations data directly from metadata
        const sortedLocations = metadata.locations
          .sort((a, b) => {
            if (a.abbreviation === 'US') return -1;
            if (b.abbreviation === 'US') return 1;
            return (a.location_name || '').localeCompare(b.location_name || '');
          });

        setStates(sortedLocations);
        setLoading(false);
      } catch (err) {
        console.error('Error in data loading:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchStates();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-center">
          <div className="mb-2">Loading locations...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen p-4">
        <div className="text-red-500 text-center mb-4">
          Error: {error}
        </div>
        <div className="text-sm text-gray-600 max-w-lg text-center">
          Please ensure that:
          <ul className="list-disc text-left mt-2 space-y-1">
            <li>The process_flusight_data.py script has been run</li>
            <li>Data files are present in app/public/processed_data/</li>
            <li>manifest.json contains valid location data</li>
          </ul>
        </div>
      </div>
    );
  }

  if (sidebarMode) {
    return (
      <div className="w-64 bg-white border-r shadow-lg flex flex-col">
        <div className="p-4 border-b">
          <h3 className="font-bold mb-4 text-gray-700">Select View</h3>
          <ViewSelector />
        </div>
        <div className="p-4">
          <h3 className="font-bold mb-4 text-gray-700">Select Location</h3>
          <input
            type="text"
            placeholder="Search states..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full p-2 border rounded mb-4"
          />
          <div className="max-h-[calc(100vh-300px)] overflow-y-auto">
            {filteredStates.map((state) => (
              <div
                key={state.location}
                onClick={() => onStateSelect(state.abbreviation)}
                className={`p-2 cursor-pointer rounded transition-colors ${
                  currentLocation === state.abbreviation
                    ? 'bg-blue-100 text-blue-800'
                    : 'hover:bg-gray-100'
                }`}
              >
                <div className="font-medium">{state.location_name}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <div className="flex items-center justify-center gap-4 mb-6">
        <img src="respilens-logo.svg" alt="RespiLens Logo" className="h-14 w-14" />
        <h1 className="text-3xl font-bold text-blue-600">
          RespiLens<sup className="text-red-500 text-xs">α</sup>
        </h1>
        <InfoOverlay />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
        {states.map((state) => (
          <div
            key={state.location}
            onClick={() => onStateSelect(state.abbreviation)}
            className="border rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition-colors shadow-sm"
          >
            <h2 className="text-xl font-semibold">{state.location_name || state.location}</h2>
            {state.population && (
              <p className="text-gray-600 text-sm">
                Population: {state.population.toLocaleString()}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default StateSelector;
