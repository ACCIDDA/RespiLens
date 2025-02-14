import React from 'react';
import { useView } from '../contexts/ViewContext';
import { DATASETS } from '../config/datasets';

const ViewSelector = () => {
  const { viewType, setViewType, currentDataset } = useView();

  // Generate all possible view options
  const viewOptions = Object.values(DATASETS).flatMap(dataset =>
    dataset.views.map(view => ({
      value: `${dataset.shortName}${view}`,
      label: `${dataset.fullName} - ${view}`,
      dataset: dataset.shortName
    }))
  );

  return (
    <div className="flex flex-col gap-2">
      {viewOptions.map(option => (
        <div
          key={option.value}
          onClick={() => setViewType(option.value)}
          className={`p-2 cursor-pointer rounded transition-colors ${
            viewType === option.value
              ? 'bg-blue-100 text-blue-800'
              : 'hover:bg-gray-100'
          }`}
        >
          <div className="font-medium">{option.label}</div>
        </div>
      ))}
    </div>
  );
};

export default ViewSelector;
