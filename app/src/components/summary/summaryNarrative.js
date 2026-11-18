// Plain-language comparisons for the state summary: where a series stands,
// how it has moved, and how that compares with last season. Built from the
// data at the chosen forecast date, so the text follows the date picker.
import { shiftDateStringByDays } from "../../constants/chart";

// Seasons run August through July (as in the peak view)
const seasonStartYear = (date) => {
  const [year, month] = date.split("-").map(Number);
  return month >= 8 ? year : year - 1;
};

const formatWeek = (date) =>
  new Date(`${date}T00:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

const pct = (ratio) => `${Math.round(Math.abs(ratio) * 100)}%`;

const observations = (groundTruth, target) =>
  (groundTruth?.dates || [])
    .map((date, i) => ({ date, value: groundTruth[target]?.[i] }))
    .filter((p) => p.value !== null && p.value !== undefined);

const valueOn = (points, date) => points.find((p) => p.date === date)?.value;

/**
 * Facts about one observed series at a forecast date: the latest value, its
 * change over three weeks, the same week a year earlier and last season's
 * peak. `floor` is the level below which changes are noise (small counts).
 */
export const describeSeries = (groundTruth, target, date, floor) => {
  const points = observations(groundTruth, target).filter((p) => p.date < date);
  const latest = points[points.length - 1];
  if (!latest) return null;
  const threeWeeksAgo = valueOn(
    points,
    shiftDateStringByDays(latest.date, -21),
  );
  const lastYear = valueOn(points, shiftDateStringByDays(latest.date, -364));

  const start = seasonStartYear(latest.date);
  const lastSeason = points.filter(
    (p) => p.date >= `${start - 1}-08-01` && p.date <= `${start}-07-31`,
  );
  const peak = lastSeason.length
    ? lastSeason.reduce((a, b) => (b.value > a.value ? b : a))
    : null;

  return { latest, threeWeeksAgo, lastYear, peak, floor };
};

const trendPhrase = ({ latest, threeWeeksAgo, floor }, fmt) => {
  if (threeWeeksAgo === undefined) return null;
  if (Math.max(latest.value, threeWeeksAgo) < floor)
    return `low and roughly flat over the past 3 weeks (${fmt(threeWeeksAgo)} → ${fmt(latest.value)})`;
  if (threeWeeksAgo <= 0) return `up from ${fmt(threeWeeksAgo)} 3 weeks ago`;
  const change = latest.value / threeWeeksAgo - 1;
  if (Math.abs(change) < 0.1)
    return `roughly stable over the past 3 weeks (${fmt(threeWeeksAgo)} → ${fmt(latest.value)})`;
  return `${change > 0 ? "up" : "down"} ${pct(change)} over the past 3 weeks (${fmt(threeWeeksAgo)} → ${fmt(latest.value)})`;
};

const lastYearPhrase = ({ latest, lastYear, floor }, fmt) => {
  if (lastYear === undefined) return null;
  const last = `the same week last year (${fmt(lastYear)})`;
  // Ratios off a near-zero base mean little: say it in words
  if (Math.max(latest.value, lastYear) < floor) return `similar to ${last}`;
  if (lastYear < floor) return `far above ${last}`;
  if (latest.value < floor) return `far below ${last}`;
  const ratio = latest.value / lastYear;
  if (ratio >= 1.15) return `${ratio.toFixed(1)}× ${last}`;
  if (ratio <= 0.87) return `${pct(1 - ratio)} below ${last}`;
  return `similar to ${last}`;
};

const peakPhrase = ({ latest, peak }, fmt) => {
  if (!peak || peak.value <= 0) return null;
  const share = latest.value / peak.value;
  const peakText = `${fmt(peak.value)}, week ending ${formatWeek(peak.date)}`;
  return share >= 1
    ? `already above last season's peak (${peakText})`
    : `${pct(share)} of last season's peak (${peakText})`;
};

/**
 * One sentence: "<label> <verb> <value> (week ending ...), <trend>. That is
 * <vs last year> and <share of last season's peak>."
 */
export const seriesSentence = (label, facts, fmt) => {
  if (!facts) return null;
  const { latest } = facts;
  const trend = trendPhrase(facts, fmt);
  const comparisons = [
    lastYearPhrase(facts, fmt),
    peakPhrase(facts, fmt),
  ].filter(Boolean);
  let text = `${label} ${fmt(latest.value)} in the week ending ${formatWeek(latest.date)}`;
  if (trend) text += `, ${trend}`;
  text += ".";
  if (comparisons.length) text += ` That is ${comparisons.join(", and ")}.`;
  return text;
};

// Weeks between two dates once both are placed in the same season
const seasonWeekOffset = (forecastWeek, lastPeakWeek) => {
  const aligned = shiftDateStringByDays(lastPeakWeek, 364);
  return Math.round(
    (Date.parse(forecastWeek) - Date.parse(aligned)) / (7 * 86400000),
  );
};

/**
 * The FluSight ensemble's peak forecast against last season's observed peak.
 * `peakSize` is the quantile forecast of the peak, `peakWeek` the pmf over
 * weeks.
 */
export const peakSentence = (peakSize, peakWeek, lastPeak, fmt) => {
  const q = (level) => {
    const i = peakSize?.quantiles?.findIndex((x) => Number(x) === level);
    return i >= 0 ? peakSize.values[i] : null;
  };
  const median = q(0.5);
  if (median === null) return null;
  let text = `The FluSight ensemble expects this season's hospital admissions to peak at about ${fmt(median)} a week`;
  if (q(0.25) !== null && q(0.75) !== null)
    text += ` (50% interval ${fmt(q(0.25))}–${fmt(q(0.75))})`;
  if (lastPeak?.value > 0) {
    const ratio = median / lastPeak.value;
    const vs =
      Math.abs(ratio - 1) < 0.1
        ? "close to"
        : ratio > 1
          ? `${pct(ratio - 1)} above`
          : `${pct(1 - ratio)} below`;
    text += `, ${vs} last season's peak of ${fmt(lastPeak.value)}`;
  }
  const weeks = peakWeek?.["peak week"] || [];
  const probs = peakWeek?.probabilities || [];
  if (weeks.length && probs.length) {
    const best = probs.indexOf(Math.max(...probs));
    text += `. The most likely peak week is the week ending ${formatWeek(weeks[best])} (${pct(probs[best])} chance)`;
    if (lastPeak) {
      const offset = seasonWeekOffset(weeks[best], lastPeak.date);
      text +=
        offset === 0
          ? ", the same time of year as last season's peak"
          : `, about ${Math.abs(offset)} week${Math.abs(offset) === 1 ? "" : "s"} ${offset < 0 ? "earlier" : "later"} than last season's (week ending ${formatWeek(lastPeak.date)})`;
    }
  }
  return `${text}.`;
};
