import { DATASETS } from '../config/datasets';

export class URLParameterManager {
  constructor(searchParams, setSearchParams) {
    this.searchParams = searchParams;
    this.setSearchParams = setSearchParams;
  }

  // Get dataset from view type
  getDatasetFromView(viewType) {
    for (const [key, dataset] of Object.entries(DATASETS)) {
      if (viewType.includes(dataset.shortName)) {
        return dataset;
      }
    }
    return null;
  }

  // Get all parameters for a specific dataset
  getDatasetParams(dataset) {
    const prefix = dataset.prefix;
    const params = {};
    
    if (dataset.hasDateSelector) {
      const dates = this.searchParams.get(`${prefix}_dates`);
      params.dates = dates ? dates.split(',') : [];
    }
    
    if (dataset.hasModelSelector) {
      const models = this.searchParams.get(`${prefix}_models`);
      params.models = models ? models.split(',') : [];
    }

    // Special case for NHSN
    if (dataset.shortName === 'nhsn') {
      params.columns = this.searchParams.get('nhsn_columns')?.split(',') || [];
    }

    return params;
  }

  // Clear parameters for a specific dataset
  clearDatasetParams(dataset) {
    const newParams = new URLSearchParams(this.searchParams);
    const prefix = dataset.prefix;
    
    if (dataset.hasDateSelector) {
      newParams.delete(`${prefix}_dates`);
    }
    if (dataset.hasModelSelector) {
      newParams.delete(`${prefix}_models`);
    }
    if (dataset.shortName === 'nhsn') {
      newParams.delete('nhsn_columns');
    }
    
    this.setSearchParams(newParams, { replace: true });
  }

  // Update parameters for a dataset
  updateDatasetParams(dataset, newParams) {
    const updatedParams = new URLSearchParams(this.searchParams);
    const prefix = dataset.prefix;

    // Update dates if present and dataset supports it
    if (dataset.hasDateSelector && newParams.dates) {
      updatedParams.set(`${prefix}_dates`, newParams.dates.join(','));
    }

    // Update models if present and dataset supports it
    if (dataset.hasModelSelector && newParams.models) {
      updatedParams.set(`${prefix}_models`, newParams.models.join(','));
    }

    // Special case for NHSN columns
    if (dataset.shortName === 'nhsn' && newParams.columns) {
      updatedParams.set('nhsn_columns', newParams.columns.join(','));
    }

    this.setSearchParams(updatedParams, { replace: true });
  }

  // Handle view type changes
  handleViewChange(oldView, newView) {
    const oldDataset = this.getDatasetFromView(oldView);
    const newDataset = this.getDatasetFromView(newView);
    
    // If switching between datasets, clear old dataset's parameters
    if (oldDataset?.shortName !== newDataset?.shortName) {
      this.clearDatasetParams(oldDataset);
    }

    // Update view parameter
    const newParams = new URLSearchParams(this.searchParams);
    newParams.set('view', newView);
    this.setSearchParams(newParams, { replace: true });
  }
}
