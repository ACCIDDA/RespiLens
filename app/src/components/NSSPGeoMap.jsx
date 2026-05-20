import { useMemo } from "react";
import { geoAlbersUsa, geoMercator, geoPath } from "d3-geo";
import { NSSP_MAP_COLORS } from "../utils/nsspMap";

const MAP_WIDTH = 960;
const MAP_PADDING = {
  usa: 24,
  state: 72,
};

const NSSPGeoMap = ({
  featureCollection,
  height,
  projectionKind,
  onFeatureClick,
  isFeatureClickable,
  getFeatureKey,
  getFeatureLabel,
  getFeatureFill,
}) => {
  const pathGenerator = useMemo(() => {
    if (!featureCollection?.features?.length) {
      return null;
    }

    const projection =
      projectionKind === "usa" ? geoAlbersUsa() : geoMercator();
    const padding = MAP_PADDING[projectionKind] ?? MAP_PADDING.state;
    projection.fitExtent(
      [
        [padding, padding],
        [MAP_WIDTH - padding, height - padding],
      ],
      featureCollection,
    );

    return geoPath(projection);
  }, [featureCollection, height, projectionKind]);

  if (!featureCollection?.features?.length || !pathGenerator) {
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${MAP_WIDTH} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="Interactive geographic map"
    >
      {featureCollection.features.map((feature) => {
        const pathData = pathGenerator(feature);
        if (!pathData) {
          return null;
        }

        const label = getFeatureLabel(feature);
        const isClickable = isFeatureClickable
          ? isFeatureClickable(feature)
          : true;

        return (
          <path
            key={getFeatureKey(feature)}
            d={pathData}
            fill={getFeatureFill(feature)}
            stroke={NSSP_MAP_COLORS.outline}
            strokeWidth={0.8}
            style={{
              cursor: isClickable ? "pointer" : "not-allowed",
              transition: "fill 150ms ease",
            }}
            onClick={() => {
              if (isClickable) {
                onFeatureClick(feature);
              }
            }}
            onMouseEnter={(event) => {
              if (isClickable) {
                event.currentTarget.style.fill = NSSP_MAP_COLORS.hover;
              }
            }}
            onMouseLeave={(event) => {
              event.currentTarget.style.fill = getFeatureFill(feature);
            }}
          >
            <title>{label}</title>
          </path>
        );
      })}
    </svg>
  );
};

export default NSSPGeoMap;
