import { APP_CONFIG, DATASETS } from "../config";
import { NSSP_STATE_INFO } from "./nsspGeo";

const FORECAST_ROOT = "/forecasts";
const SURVEILLANCE_ROOT = "/surveillance";

const PATH_VIEW_CONFIG = {
  covid_forecasts: { pathogen: "covid" },
  flu_forecasts: { pathogen: "flusight" },
  fludetailed: { pathogen: "flusight", variant: "detailed" },
  flu_peak: { pathogen: "flusight", variant: "peak" },
  rsv_forecasts: { pathogen: "rsv" },
  metrocast_forecasts: { pathogen: "flu-metrocast" },
  nhsnall: { section: "surveillance", source: "nhsn" },
  nsspall: { section: "surveillance", source: "nssp" },
};

const PATHOGEN_VARIANT_TO_VIEW = {
  covid: {
    default: "covid_forecasts",
  },
  flusight: {
    default: "flu_forecasts",
    detailed: "fludetailed",
    peak: "flu_peak",
  },
  "flu-metrocast": {
    default: "metrocast_forecasts",
  },
  rsv: {
    default: "rsv_forecasts",
  },
};

const SURVEILLANCE_SOURCE_TO_VIEW = {
  nhsn: "nhsnall",
  nssp: "nsspall",
};

// The front page takes its location as a bare state code: /AR, /US. Only
// codes the hubs actually carry are routes; anything else is a 404 rather
// than a front page under a URL that means nothing.
export const FRONT_PAGE_LOCATIONS = new Set([
  ...NSSP_STATE_INFO.map((state) => state.abbreviation),
  // Not in the NSSP map, but present in the forecast hubs
  "PR",
]);

const FRONTPAGE_LOCATION_PATTERN = /^\/([A-Za-z]{2})\/?$/;

export const isFrontPageLocation = (location) =>
  FRONT_PAGE_LOCATIONS.has(String(location || "").toUpperCase());

const parseFrontPageLocation = (pathname = "") => {
  const code = pathname.match(FRONTPAGE_LOCATION_PATTERN)?.[1]?.toUpperCase();
  return code && FRONT_PAGE_LOCATIONS.has(code) ? code : null;
};

const RESERVED_VARIANTS = new Set(["detailed", "peak", "metrocast"]);

const METROCAST_STATE_URL_TO_LOCATION = {
  CA: "california",
  CO: "colorado",
  GA: "georgia",
  IN: "indiana",
  ME: "maine",
  MD: "maryland",
  MA: "massachusetts",
  MN: "minnesota",
  NC: "north-carolina",
  OR: "oregon",
  SC: "south-carolina",
  TX: "texas",
  UT: "utah",
  VA: "virginia",
};

const METROCAST_LOCATION_TO_STATE_URL = Object.fromEntries(
  Object.entries(METROCAST_STATE_URL_TO_LOCATION).map(
    ([stateCode, location]) => [location, stateCode],
  ),
);

const METROCAST_SUBLOCATION_TO_STATE_URL = {
  "bay-area": "CA",
  "central-california": "CA",
  "greater-sierra-sacramento": "CA",
  "los-angeles": "CA",
  "rural-north": "CA",
  "southern-california": "CA",
  denver: "CO",
  mesa: "CO",
  "colorado-springs": "CO",
  weld: "CO",
  boulder: "CO",
  larimer: "CO",
  savannah: "GA",
  floyd: "GA",
  hall: "GA",
  marietta: "GA",
  macon: "GA",
  henry: "GA",
  cherokee: "GA",
  athens: "GA",
  "south-augusta": "GA",
  columbus: "GA",
  "la-grange": "GA",
  indianapolis: "IN",
  bangor: "ME",
  portland: "ME",
  baltimore: "MD",
  frederick: "MD",
  montgomery: "MD",
  harford: "MD",
  worcester: "MA",
  pittsfield: "MA",
  boston: "MA",
  springfield: "MA",
  "new-bedford": "MA",
  lynn: "MA",
  "st-paul": "MN",
  duluth: "MN",
  minneapolis: "MN",
  "st-cloud": "MN",
  rochester: "MN",
  nenc: "NC",
  senc: "NC",
  "fay-area": "NC",
  "rtp-area": "NC",
  "triad-area": "NC",
  wnc: "NC",
  "clt-area": "NC",
  "portland-or": "OR",
  salem: "OR",
  deschutes: "OR",
  eugene: "OR",
  medford: "OR",
  columbia: "SC",
  greenville: "SC",
  florence: "SC",
  charleston: "SC",
  "rock-hill": "SC",
  horry: "SC",
  houston: "TX",
  "san-antonio": "TX",
  beaumont: "TX",
  "el-paso": "TX",
  austin: "TX",
  dallas: "TX",
  provo: "UT",
  "salt-lake-city": "UT",
  ogden: "UT",
  roanoke: "VA",
  nyc: "NY",
};

const sanitizeLocationSegment = (location) => {
  if (!location) {
    return null;
  }

  return String(location).trim();
};

const serializeMetrocastLocationForPath = (location) => {
  if (!location) {
    return null;
  }

  const stateCode = METROCAST_LOCATION_TO_STATE_URL[location];
  if (stateCode) {
    return stateCode;
  }

  const parentStateCode = METROCAST_SUBLOCATION_TO_STATE_URL[location];
  if (parentStateCode) {
    return `${parentStateCode}_${location}`;
  }

  return location;
};

const deserializeMetrocastLocationFromPath = (locationSegment) => {
  const sanitizedLocation = sanitizeLocationSegment(locationSegment);
  if (!sanitizedLocation) {
    return null;
  }

  if (METROCAST_STATE_URL_TO_LOCATION[sanitizedLocation]) {
    return METROCAST_STATE_URL_TO_LOCATION[sanitizedLocation];
  }

  const [stateCode, subLocation] = sanitizedLocation.split(/_(.+)/);
  if (
    stateCode &&
    subLocation &&
    METROCAST_SUBLOCATION_TO_STATE_URL[subLocation] === stateCode
  ) {
    return subLocation;
  }

  return sanitizedLocation;
};

const isPathBasedForecastView = (viewType) =>
  Object.prototype.hasOwnProperty.call(PATH_VIEW_CONFIG, viewType);

export const isForecastPathname = (pathname = "") =>
  pathname === "/" ||
  Boolean(parseFrontPageLocation(pathname)) ||
  pathname === FORECAST_ROOT ||
  pathname.startsWith(`${FORECAST_ROOT}/`) ||
  pathname === SURVEILLANCE_ROOT ||
  pathname.startsWith(`${SURVEILLANCE_ROOT}/`);

export const getDefaultLocationForView = (viewType) => {
  const dataset =
    Object.values(DATASETS).find((entry) =>
      entry.views.some((view) => view.value === viewType),
    ) || null;

  return dataset?.defaultLocation || APP_CONFIG.defaultLocation;
};

export const buildForecastPath = (viewType, location) => {
  if (!isPathBasedForecastView(viewType)) {
    return "/";
  }

  const config = PATH_VIEW_CONFIG[viewType];
  const defaultLocation = getDefaultLocationForView(viewType);

  if (config.section === "surveillance") {
    const segments = [SURVEILLANCE_ROOT, config.source];
    if (location && location !== defaultLocation) {
      segments.push(encodeURIComponent(location));
    }
    return segments.join("/");
  }

  const segments = [FORECAST_ROOT, config.pathogen];

  if (config.variant) {
    segments.push(config.variant);
  }

  if (location && location !== defaultLocation) {
    const serializedLocation =
      viewType === "metrocast_forecasts"
        ? serializeMetrocastLocationForPath(location)
        : location;
    segments.push(encodeURIComponent(serializedLocation));
  }

  return segments.join("/");
};

const parsePathBasedForecastState = (pathname) => {
  if (pathname.startsWith(SURVEILLANCE_ROOT)) {
    const trimmedPath = pathname.replace(/\/+$/g, "");
    const relativePath = trimmedPath
      .slice(SURVEILLANCE_ROOT.length)
      .replace(/^\/+/g, "");
    const segments = relativePath ? relativePath.split("/") : [];

    if (segments.length === 0) {
      return null;
    }

    const [source, locationSegment] = segments.map((segment) =>
      decodeURIComponent(segment),
    );
    const viewType = SURVEILLANCE_SOURCE_TO_VIEW[source];

    if (!viewType) {
      return null;
    }

    return {
      viewType,
      location:
        sanitizeLocationSegment(locationSegment) ||
        getDefaultLocationForView(viewType),
    };
  }

  const trimmedPath = pathname.replace(/\/+$/g, "");
  const relativePath = trimmedPath
    .slice(FORECAST_ROOT.length)
    .replace(/^\/+/g, "");
  const segments = relativePath ? relativePath.split("/") : [];

  if (segments.length === 0) {
    return null;
  }

  const [pathogen, secondSegment, thirdSegment] = segments.map((segment) =>
    decodeURIComponent(segment),
  );
  const pathogenConfig = PATHOGEN_VARIANT_TO_VIEW[pathogen];

  if (!pathogenConfig) {
    return null;
  }

  let viewType = pathogenConfig.default;
  let location = null;

  if (
    pathogen === "flusight" &&
    secondSegment &&
    RESERVED_VARIANTS.has(secondSegment)
  ) {
    viewType = pathogenConfig[secondSegment] || pathogenConfig.default;
    location =
      viewType === "metrocast_forecasts"
        ? deserializeMetrocastLocationFromPath(thirdSegment)
        : sanitizeLocationSegment(thirdSegment);
  } else if (pathogen === "flu-metrocast") {
    viewType = "metrocast_forecasts";
    location = deserializeMetrocastLocationFromPath(secondSegment);
  } else if (secondSegment) {
    location = sanitizeLocationSegment(secondSegment);
  }

  return {
    viewType,
    location: location || getDefaultLocationForView(viewType),
  };
};

export const parseForecastUrlState = (pathname) => {
  const frontPageLocation = parseFrontPageLocation(pathname);
  if (frontPageLocation) {
    return { viewType: "frontpage", location: frontPageLocation };
  }

  if (
    pathname &&
    (pathname.startsWith(FORECAST_ROOT) ||
      pathname.startsWith(SURVEILLANCE_ROOT))
  ) {
    const pathState = parsePathBasedForecastState(pathname);
    if (pathState) {
      return pathState;
    }
  }

  return {
    viewType: APP_CONFIG.defaultView,
    location: getDefaultLocationForView(APP_CONFIG.defaultView),
  };
};

export const buildForecastUrl = ({ viewType, location, searchParams }) => {
  const search = new URLSearchParams(searchParams).toString();
  const pathname =
    viewType === "frontpage"
      ? location && location !== APP_CONFIG.defaultLocation
        ? `/${encodeURIComponent(location)}`
        : "/"
      : buildForecastPath(viewType, location);

  return { pathname, search: search ? `?${search}` : "" };
};

// Views whose location segment is a plain state code, so a wrong one can be
// caught from the URL alone. MetroCast (city slugs) and NSSP (a state code
// plus a county) are checked on their state part only.
const STATE_CODED_VIEWS = new Set([
  "flu_forecasts",
  "fludetailed",
  "flu_peak",
  "covid_forecasts",
  "rsv_forecasts",
  "nhsnall",
]);

const KNOWN_PATHOGENS = Object.keys(PATHOGEN_VARIANT_TO_VIEW).join(", ");
const KNOWN_SOURCES = Object.keys(SURVEILLANCE_SOURCE_TO_VIEW).join(", ");

/**
 * Why a /forecasts/... or /surveillance/... address is not a page, as
 * something to show the reader: `{ title, detail }`, or null when the path
 * is fine. Called before rendering a view so a mistyped hub or state says
 * so, instead of quietly turning into the US front page.
 */
export const getForecastRouteError = (pathname = "") => {
  const isSurveillance = pathname.startsWith(SURVEILLANCE_ROOT);
  if (!isSurveillance && !pathname.startsWith(FORECAST_ROOT)) {
    return null;
  }

  const segments = pathname
    .replace(/\/+$/g, "")
    .slice((isSurveillance ? SURVEILLANCE_ROOT : FORECAST_ROOT).length)
    .replace(/^\/+/g, "")
    .split("/")
    .filter(Boolean)
    .map((segment) => decodeURIComponent(segment));

  // "/forecasts" and "/surveillance" alone are handled by their redirects
  if (segments.length === 0) {
    return null;
  }

  const [first, second, third] = segments;

  if (isSurveillance) {
    if (!SURVEILLANCE_SOURCE_TO_VIEW[first]) {
      return {
        title: "No such data source",
        detail: `RespiLens has no surveillance source called "${first}". The sources are: ${KNOWN_SOURCES}.`,
      };
    }
  } else if (!PATHOGEN_VARIANT_TO_VIEW[first]) {
    return {
      title: "No such forecast hub",
      detail: `RespiLens has no forecast hub called "${first}". The hubs are: ${KNOWN_PATHOGENS}.`,
    };
  } else if (
    second &&
    RESERVED_VARIANTS.has(second) &&
    !PATHOGEN_VARIANT_TO_VIEW[first][second]
  ) {
    return {
      title: "No such view",
      detail: `${first} has no "${second}" view.`,
    };
  }

  const pathState = parsePathBasedForecastState(pathname);
  if (!pathState) {
    return {
      title: "Page not found",
      detail: "This address is not a RespiLens view.",
    };
  }

  // The location segment, where there is one to check
  const locationSegment = isSurveillance
    ? second
    : RESERVED_VARIANTS.has(second)
      ? third
      : second;

  if (!locationSegment) {
    return null;
  }

  const { viewType } = pathState;
  const stateCode = STATE_CODED_VIEWS.has(viewType)
    ? locationSegment
    : viewType === "nsspall"
      ? locationSegment.split("_")[0]
      : null;

  if (stateCode && !FRONT_PAGE_LOCATIONS.has(stateCode.toUpperCase())) {
    return {
      title: "No such location",
      detail: `"${stateCode}" is not a US state or territory RespiLens covers. Pick a location from the chart's location menu.`,
    };
  }

  return null;
};
