import { getDataPath } from "./paths";
import { feature as topojsonFeature } from "topojson-client";

export const NSSP_STATE_INFO = [
  { name: "United States", abbreviation: "US", fips: "US" },
  { name: "Alabama", abbreviation: "AL", fips: "01" },
  { name: "Alaska", abbreviation: "AK", fips: "02" },
  { name: "Arizona", abbreviation: "AZ", fips: "04" },
  { name: "Arkansas", abbreviation: "AR", fips: "05" },
  { name: "California", abbreviation: "CA", fips: "06" },
  { name: "Colorado", abbreviation: "CO", fips: "08" },
  { name: "Connecticut", abbreviation: "CT", fips: "09" },
  { name: "Delaware", abbreviation: "DE", fips: "10" },
  { name: "District of Columbia", abbreviation: "DC", fips: "11" },
  { name: "Florida", abbreviation: "FL", fips: "12" },
  { name: "Georgia", abbreviation: "GA", fips: "13" },
  { name: "Hawaii", abbreviation: "HI", fips: "15" },
  { name: "Idaho", abbreviation: "ID", fips: "16" },
  { name: "Illinois", abbreviation: "IL", fips: "17" },
  { name: "Indiana", abbreviation: "IN", fips: "18" },
  { name: "Iowa", abbreviation: "IA", fips: "19" },
  { name: "Kansas", abbreviation: "KS", fips: "20" },
  { name: "Kentucky", abbreviation: "KY", fips: "21" },
  { name: "Louisiana", abbreviation: "LA", fips: "22" },
  { name: "Maine", abbreviation: "ME", fips: "23" },
  { name: "Maryland", abbreviation: "MD", fips: "24" },
  { name: "Massachusetts", abbreviation: "MA", fips: "25" },
  { name: "Michigan", abbreviation: "MI", fips: "26" },
  { name: "Minnesota", abbreviation: "MN", fips: "27" },
  { name: "Mississippi", abbreviation: "MS", fips: "28" },
  { name: "Missouri", abbreviation: "MO", fips: "29" },
  { name: "Montana", abbreviation: "MT", fips: "30" },
  { name: "Nebraska", abbreviation: "NE", fips: "31" },
  { name: "Nevada", abbreviation: "NV", fips: "32" },
  { name: "New Hampshire", abbreviation: "NH", fips: "33" },
  { name: "New Jersey", abbreviation: "NJ", fips: "34" },
  { name: "New Mexico", abbreviation: "NM", fips: "35" },
  { name: "New York", abbreviation: "NY", fips: "36" },
  { name: "North Carolina", abbreviation: "NC", fips: "37" },
  { name: "North Dakota", abbreviation: "ND", fips: "38" },
  { name: "Ohio", abbreviation: "OH", fips: "39" },
  { name: "Oklahoma", abbreviation: "OK", fips: "40" },
  { name: "Oregon", abbreviation: "OR", fips: "41" },
  { name: "Pennsylvania", abbreviation: "PA", fips: "42" },
  { name: "Rhode Island", abbreviation: "RI", fips: "44" },
  { name: "South Carolina", abbreviation: "SC", fips: "45" },
  { name: "South Dakota", abbreviation: "SD", fips: "46" },
  { name: "Tennessee", abbreviation: "TN", fips: "47" },
  { name: "Texas", abbreviation: "TX", fips: "48" },
  { name: "Utah", abbreviation: "UT", fips: "49" },
  { name: "Vermont", abbreviation: "VT", fips: "50" },
  { name: "Virginia", abbreviation: "VA", fips: "51" },
  { name: "Washington", abbreviation: "WA", fips: "53" },
  { name: "West Virginia", abbreviation: "WV", fips: "54" },
  { name: "Wisconsin", abbreviation: "WI", fips: "55" },
  { name: "Wyoming", abbreviation: "WY", fips: "56" },
];

export const NSSP_STATE_NAME_TO_INFO = Object.fromEntries(
  NSSP_STATE_INFO.map((entry) => [entry.name, entry]),
);

export const NSSP_STATE_ABBREVIATION_TO_INFO = Object.fromEntries(
  NSSP_STATE_INFO.map((entry) => [entry.abbreviation, entry]),
);

const NSSP_STATE_FIPS_TO_ABBREVIATION = Object.fromEntries(
  NSSP_STATE_INFO.filter((entry) => entry.fips !== "US").map((entry) => [
    entry.fips,
    entry.abbreviation,
  ]),
);

const NATIVE_HAWAIIAN = "native hawaiian";

const fetchJson = async (pathOrUrl) => {
  const response = await fetch(pathOrUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${pathOrUrl}: ${response.status}`);
  }
  return response.json();
};

const getPublicAssetPath = (path) => {
  const baseUrl = import.meta.env.BASE_URL || "";
  const cleanBaseUrl = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
  return `${cleanBaseUrl}/${path}`;
};

export const getNsspStateAbbreviationFromLocation = (location) =>
  location?.split("_")?.[0] || null;

export const getNsspTopLevelLocation = (location) => {
  const stateAbbreviation = getNsspStateAbbreviationFromLocation(location);
  return stateAbbreviation ? `${stateAbbreviation}_All` : location;
};

export const isNsspUnitedStatesLocation = (location) => location === "US_All";

export const isNsspStatewideLocation = (location) =>
  Boolean(location) &&
  location.endsWith("_All") &&
  !isNsspUnitedStatesLocation(location);

export const normalizeCountyBasename = (value = "") =>
  value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/city and borough/gi, "")
    .replace(/city and county/gi, "")
    .replace(/county|parish|borough|census area|municipality/gi, "")
    .replace(/city$/gi, "")
    .replace(/saint/gi, "st")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

let nsspMetadataPromise;
let statesTopoPromise;
let countiesTopoPromise;
let countyMetadataPromise;
const stateCountyAssignmentPromises = new Map();

const getNsspMetadata = () => {
  if (!nsspMetadataPromise) {
    nsspMetadataPromise = fetchJson(getDataPath("nssp/metadata.json"));
  }
  return nsspMetadataPromise;
};

const getStatesTopology = () => {
  if (!statesTopoPromise) {
    statesTopoPromise = fetchJson(
      getPublicAssetPath("maps/nssp/us-states-10m.topo.json"),
    );
  }
  return statesTopoPromise;
};

const getCountiesTopology = () => {
  if (!countiesTopoPromise) {
    countiesTopoPromise = fetchJson(
      getPublicAssetPath("maps/nssp/us-counties-10m.topo.json"),
    );
  }
  return countiesTopoPromise;
};

const getCountyMetadata = () => {
  if (!countyMetadataPromise) {
    countyMetadataPromise = fetchJson(
      getPublicAssetPath("maps/nssp/us-counties-metadata.json"),
    );
  }
  return countyMetadataPromise;
};

export const fetchNsspTopLevelLocations = async () => {
  const metadata = await getNsspMetadata();
  return metadata.locations
    .filter(([, subLocation]) => subLocation === "All")
    .map(([stateName, subLocation]) => {
      const stateInfo = NSSP_STATE_NAME_TO_INFO[stateName];
      if (!stateInfo) {
        return null;
      }

      return {
        abbreviation: `${stateInfo.abbreviation}_${subLocation}`,
        location_name: stateInfo.name,
      };
    })
    .filter(Boolean)
    .sort((a, b) => {
      const isADefault = a.abbreviation === "US_All";
      const isBDefault = b.abbreviation === "US_All";
      if (isADefault) return -1;
      if (isBDefault) return 1;
      return a.location_name.localeCompare(b.location_name);
    });
};

export const fetchNsspStatesGeoJson = async (allowedAbbreviations = []) => {
  const topology = await getStatesTopology();
  const geoJson = topojsonFeature(topology, topology.objects.states);
  const allowedSet = new Set(allowedAbbreviations);
  return {
    ...geoJson,
    features: geoJson.features
      .map((stateFeature) => {
        const fips = String(stateFeature.id).padStart(2, "0");
        const abbreviation = NSSP_STATE_FIPS_TO_ABBREVIATION[fips];
        return {
          ...stateFeature,
          properties: {
            ...stateFeature.properties,
            GEOID: fips,
            STUSAB: abbreviation,
            NAME: stateFeature.properties?.name,
          },
        };
      })
      .filter((stateFeature) =>
        allowedSet.size === 0
          ? true
          : allowedSet.has(stateFeature.properties?.STUSAB),
      ),
  };
};

export const fetchNsspCountiesGeoJson = async (stateAbbreviation) => {
  const stateInfo = NSSP_STATE_ABBREVIATION_TO_INFO[stateAbbreviation];
  if (!stateInfo?.fips || stateInfo.fips === "US") {
    throw new Error(`No FIPS code available for state ${stateAbbreviation}`);
  }

  const [countiesTopology, countyMetadata] = await Promise.all([
    getCountiesTopology(),
    getCountyMetadata(),
  ]);

  const countyNameByGeoid = Object.fromEntries(
    (countyMetadata.features || []).map((feature) => [
      feature.attributes?.GEOID,
      feature.attributes,
    ]),
  );

  const countyFeatureCollection = topojsonFeature(
    countiesTopology,
    countiesTopology.objects.counties,
  );

  return {
    ...countyFeatureCollection,
    features: countyFeatureCollection.features
      .filter((countyFeature) =>
        String(countyFeature.id).startsWith(stateInfo.fips),
      )
      .map((countyFeature) => {
        const geoid = String(countyFeature.id).padStart(5, "0");
        const metadataEntry = countyNameByGeoid[geoid];
        return {
          ...countyFeature,
          id: geoid,
          properties: {
            ...countyFeature.properties,
            GEOID: geoid,
            STATE: geoid.slice(0, 2),
            BASENAME:
              metadataEntry?.BASENAME ||
              countyFeature.properties?.name ||
              geoid,
            NAME:
              metadataEntry?.NAME || countyFeature.properties?.name || geoid,
          },
        };
      }),
  };
};

export const fetchNsspCountyAssignments = async (stateAbbreviation) => {
  if (!stateCountyAssignmentPromises.has(stateAbbreviation)) {
    stateCountyAssignmentPromises.set(
      stateAbbreviation,
      (async () => {
        const metadata = await getNsspMetadata();
        const stateInfo = NSSP_STATE_ABBREVIATION_TO_INFO[stateAbbreviation];
        if (!stateInfo) {
          throw new Error(
            `Unknown NSSP state abbreviation: ${stateAbbreviation}`,
          );
        }

        const relevantLocations = metadata.locations.filter(([stateName]) => {
          const info = NSSP_STATE_NAME_TO_INFO[stateName];
          return info?.abbreviation === stateAbbreviation;
        });

        const countyAssignments = {};
        let statewideLocationId = `${stateAbbreviation}_All`;

        await Promise.all(
          relevantLocations.map(async ([, subLocation]) => {
            const locationId = `${stateAbbreviation}_${subLocation}`;
            if (subLocation === "All") {
              statewideLocationId = locationId;
              return;
            }

            const data = await fetchJson(
              getDataPath(`nssp/${locationId}_nssp.json`),
            );

            const countyNames = String(data?.metadata?.location_name || "")
              .split(",")
              .map((countyName) => countyName.trim())
              .filter(Boolean);

            countyNames.forEach((countyName) => {
              countyAssignments[normalizeCountyBasename(countyName)] = {
                countyName,
                locationId,
                hsaId: String(subLocation),
                groupLabel: data?.metadata?.location_name || countyName,
              };
            });
          }),
        );

        return {
          stateName: stateInfo.name,
          statewideLocationId,
          countyAssignments,
        };
      })(),
    );
  }

  return stateCountyAssignmentPromises.get(stateAbbreviation);
};

export const getCountySelectionForFeature = (feature, assignmentData) => {
  const countyBasename =
    feature?.properties?.BASENAME || feature?.properties?.NAME;
  const countyDisplayName = feature?.properties?.NAME || countyBasename;
  const normalizedCountyName = normalizeCountyBasename(countyBasename);
  const matchedAssignment =
    assignmentData.countyAssignments[normalizedCountyName];

  if (matchedAssignment) {
    return {
      countyName: countyDisplayName,
      locationId: matchedAssignment.locationId,
      groupLabel: matchedAssignment.groupLabel,
      hsaId: matchedAssignment.hsaId,
      isStatewideFallback: false,
    };
  }

  return {
    countyName: countyDisplayName,
    locationId: assignmentData.statewideLocationId,
    groupLabel: `${assignmentData.stateName} statewide`,
    hsaId: "All",
    isStatewideFallback: true,
  };
};

export const getCountyDisplayLabel = (countyName = "") => {
  if (!countyName) {
    return "Selected county";
  }

  const normalized = countyName.trim().toLowerCase();
  if (normalized.includes("county")) {
    return countyName;
  }
  if (normalized === "district of columbia") {
    return "District of Columbia";
  }
  if (normalized.includes("parish")) {
    return countyName;
  }
  if (
    normalized.includes("borough") ||
    normalized.includes("census area") ||
    normalized.includes("municipality")
  ) {
    return countyName;
  }
  if (normalized.includes(NATIVE_HAWAIIAN)) {
    return countyName;
  }
  if (normalized.includes("city")) {
    return countyName;
  }
  return `${countyName} County`;
};
