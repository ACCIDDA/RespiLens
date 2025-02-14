import React from 'react';

const COLUMN_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
];

const NHSNColumnSelector = ({ 
  availableColumns,
  selectedColumns,
  setSelectedColumns,
}) => {
  const toggleColumn = (column, isPreview) => {
    if (selectedColumns.includes(column)) {
      setSelectedColumns(selectedColumns.filter(c => c !== column));
    } else {
      setSelectedColumns([...selectedColumns, column]);
    }
  };

  return (
    <div className="mt-4 grid grid-cols-2 gap-4">
      <div>
        <h3 className="font-bold mb-2">Official Data Columns</h3>
        <div className="flex flex-wrap gap-2">
          {availableColumns.official.map((column, index) => (
            <div
              key={column}
              onClick={() => toggleColumn(column, false)}
              className={`px-3 py-1 rounded cursor-pointer text-sm transition-colors ${
                selectedColumns.includes(column)
                  ? 'text-white'
                  : 'border hover:bg-gray-100'
              }`}
              style={
                selectedColumns.includes(column) 
                  ? { backgroundColor: COLUMN_COLORS[index % COLUMN_COLORS.length] }
                  : undefined
              }
            >
              {column}
            </div>
          ))}
        </div>
      </div>
      
      <div>
        <h3 className="font-bold mb-2">Preliminary Data Columns</h3>
        <div className="flex flex-wrap gap-2">
          {availableColumns.preliminary.map((column, index) => (
            <div
              key={column}
              onClick={() => toggleColumn(column, true)}
              className={`px-3 py-1 rounded cursor-pointer text-sm transition-colors ${
                selectedColumns.includes(column)
                  ? 'text-white'
                  : 'border hover:bg-gray-100'
              }`}
              style={
                selectedColumns.includes(column) 
                  ? { backgroundColor: COLUMN_COLORS[(index + availableColumns.official.length) % COLUMN_COLORS.length] }
                  : undefined
              }
            >
              {column}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default NHSNColumnSelector;
