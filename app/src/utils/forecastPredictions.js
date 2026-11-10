// Keep observations separate: this filter applies only to model predictions.
export const getVisiblePredictions = (
  predictions = {},
  referenceDate,
  showNowcasts = false,
) =>
  Object.entries(predictions).flatMap(([horizon, prediction]) => {
    const isNowcast = Number(horizon) < 0 || prediction.date < referenceDate;
    return showNowcasts || !isNowcast ? [prediction] : [];
  });
