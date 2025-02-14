export const DATASETS = {
  flu: {
    shortName: 'flu',
    fullName: 'FluSight',
    views: ['detailed', 'timeseries'],
    defaultView: 'detailed',
    defaultModel: 'FluSight-ensemble',
    hasDateSelector: true,
    hasModelSelector: true,
    prefix: 'flu',
    dataPath: 'flusight'
  },
  rsv: {
    shortName: 'rsv', 
    fullName: 'RSV Forecast Hub',
    views: ['detailed'],
    defaultView: 'detailed',
    defaultModel: 'hub-ensemble',
    hasDateSelector: true,
    hasModelSelector: true,
    prefix: 'rsv',
    dataPath: 'rsv'
  },
  nhsn: {
    shortName: 'nhsn',
    fullName: 'NHSN Raw Data',
    views: ['raw'],
    defaultView: 'raw',
    hasDateSelector: false,
    hasModelSelector: false,
    prefix: 'nhsn',
    dataPath: 'nhsn'
  }
};
