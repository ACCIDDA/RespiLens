import { useMemo, useState } from "react";
import { geoAlbersUsa, geoMercator, geoPath } from "d3-geo";
import { NSSP_MAP_COLORS } from "../utils/nsspMap";

const MAP_WIDTH = 960;
const MAP_PADDING = {
  usa: 24,
  state: 72,
};

// `displayHeight`: a CSS height to fit the map into (it keeps its aspect
// ratio, centred); by default it takes its natural height at full width
const NSSPGeoMap = ({
  featureCollection,
  height,
  displayHeight = "auto",
  projectionKind,
  onFeatureClick,
  isFeatureClickable,
  getFeatureKey,
  getFeatureLabel,
  getFeatureFill,
  getFeatureCallout,
}) => {
  const [activeCalloutKey, setActiveCalloutKey] = useState(null);
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

  const featureCallouts = useMemo(() => {
    if (!pathGenerator || !getFeatureCallout) {
      return [];
    }

    return featureCollection.features.flatMap((feature) => {
      const callout = getFeatureCallout(feature);
      if (!callout) {
        return [];
      }

      const [originX, originY] = pathGenerator.centroid(feature);
      if (!Number.isFinite(originX) || !Number.isFinite(originY)) {
        return [];
      }

      const [offsetX = 0, offsetY = 0] = callout.offset || [];
      return [
        {
          ...callout,
          feature,
          key: getFeatureKey(feature),
          originX,
          originY,
          x: originX + offsetX,
          y: originY + offsetY,
        },
      ];
    });
  }, [featureCollection, getFeatureCallout, getFeatureKey, pathGenerator]);

  if (!featureCollection?.features?.length || !pathGenerator) {
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${MAP_WIDTH} ${height}`}
      style={{ width: "100%", height: displayHeight, display: "block" }}
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
        const featureKey = getFeatureKey(feature);

        return (
          <path
            key={featureKey}
            d={pathData}
            fill={
              activeCalloutKey === featureKey
                ? NSSP_MAP_COLORS.hover
                : getFeatureFill(feature)
            }
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

      {featureCallouts.map(
        // `color`: an optional fill for the label (white text on it)
        ({ feature, key, label, color, originX, originY, x, y }) => {
          const featureLabel = getFeatureLabel(feature);
          const isClickable = isFeatureClickable
            ? isFeatureClickable(feature)
            : true;
          const isActive = activeCalloutKey === key;
          const fill = isActive
            ? NSSP_MAP_COLORS.hover
            : getFeatureFill(feature);

          const activate = () => {
            if (isClickable) {
              onFeatureClick(feature);
            }
          };

          return (
            <g
              key={`callout-${key}`}
              role={isClickable ? "button" : undefined}
              tabIndex={isClickable ? 0 : undefined}
              aria-label={featureLabel}
              aria-disabled={!isClickable || undefined}
              style={{ cursor: isClickable ? "pointer" : "not-allowed" }}
              onClick={activate}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  activate();
                }
              }}
              onMouseEnter={() => {
                if (isClickable) {
                  setActiveCalloutKey(key);
                }
              }}
              onMouseLeave={() => setActiveCalloutKey(null)}
              onFocus={() => {
                if (isClickable) {
                  setActiveCalloutKey(key);
                }
              }}
              onBlur={() => setActiveCalloutKey(null)}
            >
              <title>{featureLabel}</title>
              <line
                x1={originX}
                y1={originY}
                x2={x - 23}
                y2={y}
                stroke={NSSP_MAP_COLORS.outline}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
              <circle
                cx={originX}
                cy={originY}
                r={3.5}
                fill={fill}
                stroke={NSSP_MAP_COLORS.outline}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
              <rect
                x={x - 23}
                y={y - 16}
                width={46}
                height={32}
                rx={16}
                fill={color ?? fill}
                stroke={color ?? NSSP_MAP_COLORS.outline}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
                style={{ transition: "fill 150ms ease" }}
              />
              <text
                x={x}
                y={y}
                dy="0.35em"
                textAnchor="middle"
                fill={color ? "#ffffff" : "#172033"}
                fontSize={15}
                fontWeight={700}
                pointerEvents="none"
              >
                {label}
              </text>
            </g>
          );
        },
      )}
    </svg>
  );
};

export default NSSPGeoMap;
