import { useEffect, useMemo, useRef, useState } from "react";
import { useView } from "./useView";

const DAY_MS = 24 * 60 * 60 * 1000;
// How close (px) the pointer must be to a date line to grab it
const GRAB_TOLERANCE_PX = 6;
// A pointer that moves less than this between down and up is a click
const CLICK_SLOP_PX = 4;
// A click only adds a date if an available forecast date is this close
const MAX_CLICK_SNAP_DAYS = 7;
// Same cap as the date selector's "Add date" button
const DEFAULT_MAX_DATES = 20;
// Two clicks on the same line within this window are a double-click
const DOUBLE_CLICK_MS = 400;

const toMs = (dateString) => Date.parse(`${dateString}T00:00:00Z`);

// Ignore pointer events on Plotly's own controls drawn over the plot area
const PLOTLY_CONTROLS =
  ".legend, .rangeselector, .rangeslider-container, .modebar, .updatemenu-container";

// The main x/y axes of a Plotly graph div and the pointer's position in
// their plot area (null until Plotly has laid the chart out)
const locatePointer = (gd, clientX, clientY) => {
  const xa = gd?._fullLayout?.xaxis;
  const ya = gd?._fullLayout?.yaxis;
  if (!xa?.p2l || !ya) return null;
  const rect = gd.getBoundingClientRect();
  const px = clientX - rect.left - xa._offset;
  const py = clientY - rect.top - ya._offset;
  return {
    xa,
    px,
    inside: px >= 0 && px <= xa._length && py >= 0 && py <= ya._length,
  };
};

const nearestDate = (dates, ms, maxDistanceMs = Infinity) => {
  let best = null;
  let bestDistance = maxDistanceMs;
  dates.forEach((date) => {
    const distance = Math.abs(toMs(date) - ms);
    if (distance <= bestDistance) {
      best = date;
      bestDistance = distance;
    }
  });
  return best;
};

/**
 * Lets people place forecast dates directly on a forecast chart:
 * clicking the plot area adds the nearest available forecast date, a date
 * line can be dragged to another date (it snaps week by week), and
 * double-clicking a line removes that date (while more than one is shown).
 *
 * `lineOffsetDays` is how far the chart draws each date line from its date.
 * With `maxDates` of 1 (a single-date chart) a click moves the date there.
 * `onBeforeCommit(gd)` runs before the selection changes (e.g. to pin the
 * current x range so the chart does not jump to re-centre on the dates).
 *
 * Returns a ref for the element wrapping the chart, the dates to draw (with
 * a line being dragged at its preview position) and the date currently
 * being dragged, if any.
 */
const useForecastDateDrag = ({
  selectedDates,
  lineOffsetDays = 0,
  maxDates = DEFAULT_MAX_DATES,
  enabled = true,
  onBeforeCommit,
}) => {
  const { availableDates, setSelectedDates, setActiveDate } = useView();
  const [drag, setDrag] = useState(null); // { from, to }
  // A callback ref: the chart's wrapper can mount after the first render
  const [container, containerRef] = useState(null);

  // Handlers are attached once; read the latest values through a ref
  const latest = useRef({});
  latest.current = {
    availableDates: availableDates || [],
    selectedDates,
    setSelectedDates,
    setActiveDate,
    onBeforeCommit,
    maxDates,
    offsetMs: lineOffsetDays * DAY_MS,
  };

  useEffect(() => {
    if (!enabled || !container) return undefined;

    const getGraph = () => container.querySelector(".js-plotly-plot");
    const getDragLayer = () =>
      container.querySelector(".draglayer .xy .nsewdrag");

    // The selected date whose line is under the pointer, if any
    const lineAt = (xa, px) => {
      const { selectedDates: dates, offsetMs } = latest.current;
      let hit = null;
      let hitDistance = GRAB_TOLERANCE_PX;
      dates.forEach((date) => {
        const distance = Math.abs(xa.l2p(toMs(date) + offsetMs) - px);
        if (distance <= hitDistance) {
          hit = date;
          hitDistance = distance;
        }
      });
      return hit;
    };

    // The available date a pointer x position points at
    const dateAt = (xa, px, maxDistanceMs) => {
      const { availableDates: dates, offsetMs } = latest.current;
      const clampedPx = Math.min(Math.max(px, 0), xa._length);
      return nearestDate(dates, xa.p2l(clampedPx) - offsetMs, maxDistanceMs);
    };

    let gesture = null; // { startX, grabbed, target }
    let lastLineClick = null; // { date, time }

    const setCursor = (cursor) => {
      const dragLayer = getDragLayer();
      if (dragLayer && dragLayer.style.cursor !== cursor) {
        dragLayer.style.cursor = cursor;
      }
    };

    // Plotly counts clicks on mousedown (double-clicks trigger its own handling);
    // keep presses on a date line away from it
    const onMouseDown = (event) => {
      const at = locatePointer(getGraph(), event.clientX, event.clientY);
      if (at?.inside && lineAt(at.xa, at.px)) event.stopPropagation();
    };

    const onPointerDown = (event) => {
      if (event.button !== 0 || event.target.closest?.(PLOTLY_CONTROLS)) {
        return;
      }
      const gd = getGraph();
      const at = locatePointer(gd, event.clientX, event.clientY);
      if (!at?.inside) return;

      const grabbed = lineAt(at.xa, at.px);
      gesture = { startX: event.clientX, grabbed, target: grabbed };
      if (grabbed) {
        // Keep Plotly from treating this as the start of its own drag
        event.stopPropagation();
        event.preventDefault();
        setCursor("grabbing");
        setDrag({ from: grabbed, to: grabbed });
      }
    };

    const onPointerMove = (event) => {
      const gd = getGraph();
      const at = locatePointer(gd, event.clientX, event.clientY);

      if (!gesture?.grabbed) {
        // Hover feedback: grab a line, or place a new date
        if (!gesture && at?.inside) {
          setCursor(lineAt(at.xa, at.px) ? "ew-resize" : "crosshair");
        }
        return;
      }
      if (!at) return;

      const { selectedDates: dates } = latest.current;
      const next = dateAt(at.xa, at.px);
      // Lines cannot be dropped onto another selected date
      if (next && (next === gesture.grabbed || !dates.includes(next))) {
        if (next !== gesture.target) {
          gesture.target = next;
          setDrag({ from: gesture.grabbed, to: next });
        }
      }
    };

    const onPointerUp = (event) => {
      if (!gesture) return;
      const { grabbed, target, startX } = gesture;
      gesture = null;
      const isClick = Math.abs(event.clientX - startX) < CLICK_SLOP_PX;
      const {
        selectedDates: dates,
        setSelectedDates: setDates,
        setActiveDate: setActive,
        onBeforeCommit: beforeCommit,
        maxDates: max,
      } = latest.current;
      const gd = getGraph();

      if (grabbed) {
        setDrag(null);
        setCursor("ew-resize");
        if (target && target !== grabbed) {
          beforeCommit?.(gd);
          setDates(dates.map((d) => (d === grabbed ? target : d)).sort());
          setActive(target);
        } else if (isClick) {
          const now = Date.now();
          const isDoubleClick =
            lastLineClick?.date === grabbed &&
            now - lastLineClick.time < DOUBLE_CLICK_MS;
          lastLineClick = isDoubleClick ? null : { date: grabbed, time: now };
          if (isDoubleClick && dates.length > 1) {
            const remaining = dates.filter((d) => d !== grabbed);
            beforeCommit?.(gd);
            setDates(remaining);
            setActive(remaining[remaining.length - 1]);
          } else {
            setActive(grabbed);
          }
        }
        return;
      }

      if (!isClick) return;
      const at = locatePointer(gd, event.clientX, event.clientY);
      if (!at?.inside) return;
      const clicked = dateAt(at.xa, at.px, MAX_CLICK_SNAP_DAYS * DAY_MS);
      if (!clicked) return;
      if (dates.includes(clicked)) {
        setActive(clicked);
      } else if (max === 1) {
        beforeCommit?.(gd);
        setDates([clicked]);
        setActive(clicked);
      } else if (dates.length < max) {
        beforeCommit?.(gd);
        setDates([...dates, clicked].sort());
        setActive(clicked);
      }
    };

    const onPointerCancel = () => {
      gesture = null;
      setDrag(null);
    };

    // Capture phase so a grabbed line never reaches Plotly's drag handling
    container.addEventListener("pointerdown", onPointerDown, true);
    container.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerCancel);
    return () => {
      container.removeEventListener("pointerdown", onPointerDown, true);
      container.removeEventListener("mousedown", onMouseDown, true);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerCancel);
    };
  }, [container, enabled]);

  const displayDates = useMemo(() => {
    if (!drag) return selectedDates;
    return selectedDates.map((d) => (d === drag.from ? drag.to : d));
  }, [drag, selectedDates]);

  return { containerRef, displayDates, draggingDate: drag?.to ?? null };
};

export default useForecastDateDrag;
