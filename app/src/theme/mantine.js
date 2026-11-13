import { createTheme } from "@mantine/core";
import pinnedModelStyles from "./modelColors.json";

// Model colors - keep existing model color system intact
export const MODEL_COLORS = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#a0a0a0",
  "#bcbd22",
  "#17becf",
  "#3f51b5",
  "#b35900",
  "#0b6e4f",
  "#8e1b3a",
  "#5b2a86",
  "#5e3a1f",
  "#ad1f77",
  "#37717f",
  "#767600",
  "#00897b",
  "#d926a9",
  "#b48ede",
  "#7fb800",
  "#f26d5b",
  "#d4a017",
  "#a68a64",
];

// Each hub's main ensemble is the reference series rather than one model among
// many, so it gets its own dedicated colour instead of a palette slot.
export const ENSEMBLE_COLOR = "#0d8ae6";

// Likewise for each hub's baseline
const BASELINE_COLOR = "#a0a0a0";

// Blue and gray are reserved for the hub ensembles and baselines
const UNRESERVED_COLORS = MODEL_COLORS.slice(1).filter(
  (color) => color !== BASELINE_COLOR,
);

// For a model too new to be in modelColors.json: still depends on nothing but its name
const getColorFromName = (model) => {
  let hash = 0;
  for (const char of String(model)) {
    hash = (hash * 31 + char.codePointAt(0)) | 0;
  }
  return UNRESERVED_COLORS[Math.abs(hash) % UNRESERVED_COLORS.length];
};

// How a model is drawn on every chart, for everyone. Depends only on its name;
// assigned by scripts/assign_model_colors.py. Dashed: excluded from the hub's ensemble.
export const getModelStyle = (model) => {
  const pinned = pinnedModelStyles[model];
  return {
    color: pinned?.color ?? getColorFromName(model),
    marker: pinned?.marker ?? "circle",
    dashed: pinned?.dashed ?? false,
  };
};

export const getModelColor = (model) => getModelStyle(model).color;

// For series that aren't models, such as the NHSN and NSSP columns
export const getSeriesColor = (series, seriesOrder = []) => {
  const index = seriesOrder.indexOf(series);
  return index >= 0 ? MODEL_COLORS[index % MODEL_COLORS.length] : null;
};

// The platform's own UI font: San Francisco on Apple, Segoe UI on Windows,
// Roboto on Android. Platforms are named before `system-ui` because bare
// `system-ui` resolves to a CJK face on Chinese/Japanese Windows installs.
export const SYSTEM_SANS =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, "Helvetica Neue", Arial, sans-serif';

// Enhanced Mantine theme with overridden color palettes
export const theme = createTheme({
  primaryColor: "blue",
  primaryShade: { light: 6, dark: 5 },
  defaultRadius: "md",

  fontFamily: SYSTEM_SANS,
  fontFamilyMonospace:
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',

  headings: {
    fontFamily: SYSTEM_SANS,
    fontWeight: "600",
    sizes: {
      // Page titles everywhere match the dashboard's chart title (24px)
      h1: { fontSize: "1.5rem", lineHeight: "1.35", fontWeight: "600" },
      h2: { fontSize: "1.25rem", lineHeight: "1.35", fontWeight: "600" },
      h3: { fontSize: "1.1875rem", lineHeight: "1.35", fontWeight: "600" },
      h4: { fontSize: "1.0625rem", lineHeight: "1.4", fontWeight: "600" },
      h5: { fontSize: "0.9375rem", lineHeight: "1.45", fontWeight: "600" },
      h6: { fontSize: "0.875rem", lineHeight: "1.45", fontWeight: "600" },
    },
  },

  // Flat page: cards (which use xs/sm) are separated by hairline borders,
  // not shadows, matching the dashboard. Floating layers - dropdowns,
  // popovers, modals (md and up) - keep a soft elevation.
  shadows: {
    xs: "none",
    sm: "none",
    md: "0 2px 4px rgba(15, 23, 42, 0.06), 0 10px 24px rgba(15, 23, 42, 0.09)",
    lg: "0 4px 8px rgba(15, 23, 42, 0.06), 0 18px 36px rgba(15, 23, 42, 0.11)",
    xl: "0 8px 16px rgba(15, 23, 42, 0.07), 0 30px 60px rgba(15, 23, 42, 0.13)",
  },

  colors: {
    // Vivid azure at Carolina Blue's own hue (205 deg), held at full
    // saturation so the UI reads bright rather than muted.
    //
    // ACCESSIBILITY: saturation this high only clears 4.5:1 in a narrow
    // lightness band. Shade 6 (#0076d1, 4.65:1) is the lightest step that
    // carries white text; shades 4-5 are accent-only, and link text on white
    // uses shade 7 (6.34:1).
    blue: [
      "#f3fafe", // 0
      "#e0f1fd", // 1
      "#bce2fd", // 2
      "#90d0fd", // 3
      "#52b7ff", // 4
      "#179eff", // 5
      "#0076d1", // 6
      "#0062a8", // 7
      "#014d83", // 8
      "#023a61", // 9
    ],

    // Override red palette for error states
    red: [
      "#fef2f2", // 0 - lightest
      "#fee2e2", // 1
      "#fecaca", // 2
      "#fca5a5", // 3
      "#f87171", // 4
      "#ef4444", // 5
      "#dc2626", // 6 - primary red
      "#b91c1c", // 7
      "#991b1b", // 8
      "#7f1d1d", // 9 - darkest
    ],

    // Override green palette for success states
    green: [
      "#f0fdf4", // 0 - lightest
      "#dcfce7", // 1
      "#bbf7d0", // 2
      "#86efac", // 3
      "#4ade80", // 4
      "#22c55e", // 5
      "#16a34a", // 6 - primary green
      "#15803d", // 7
      "#166534", // 8
      "#14532d", // 9 - darkest
    ],

    // Override amber/yellow palette for warnings
    yellow: [
      "#fffbeb", // 0 - lightest
      "#fef3c7", // 1
      "#fde68a", // 2
      "#fcd34d", // 3
      "#fbbf24", // 4
      "#f59e0b", // 5
      "#d97706", // 6 - primary amber
      "#b45309", // 7
      "#92400e", // 8
      "#78350f", // 9 - darkest
    ],

    // Override cyan palette for info states
    cyan: [
      "#ecfeff", // 0 - lightest
      "#cffafe", // 1
      "#a5f3fc", // 2
      "#67e8f9", // 3
      "#22d3ee", // 4
      "#06b6d4", // 5
      "#0891b2", // 6 - primary cyan
      "#0e7490", // 7
      "#155e75", // 8
      "#164e63", // 9 - darkest
    ],

    // Override gray palette for neutral elements
    gray: [
      "#f9fafb", // 0 - lightest
      "#f3f4f6", // 1
      "#e5e7eb", // 2
      "#d1d5db", // 3
      "#9ca3af", // 4
      "#6b7280", // 5
      "#4b5563", // 6 - primary gray
      "#374151", // 7
      "#1f2937", // 8
      "#111827", // 9 - darkest
    ],
  },

  components: {
    Paper: {
      defaultProps: { radius: "md" },
    },
    Card: {
      defaultProps: { radius: "md" },
    },
    Button: {
      defaultProps: { radius: "md" },
      styles: {
        root: {
          fontWeight: 550,
          // Hover is a colour change only. Motion is reserved for cards,
          // which are large targets; on a toolbar button it reads as fidgety.
          transition:
            "background-color 120ms ease, border-color 120ms ease, color 120ms ease",
        },
      },
    },
    Title: {
      styles: { root: { letterSpacing: "-0.011em" } },
    },
    Modal: {
      defaultProps: { radius: "lg", shadow: "xl" },
    },
    Tooltip: {
      defaultProps: { radius: "sm" },
    },
    // Same look as the front-page Announcement: a neutral hairline box,
    // with the tone carried only by the icon (the stock `light` fill is
    // saturated, and its alpha is raised for buttons below).
    Alert: {
      defaultProps: { radius: "md" },
      styles: (_theme, { color = "blue" }) => ({
        root: {
          background: "var(--respilens-surface)",
          border: "1px solid var(--respilens-hairline-strong)",
          padding: "10px var(--mantine-spacing-md)",
        },
        icon: { color: `var(--mantine-color-${color}-7)` },
        title: { color: "var(--respilens-ink)" },
        message: {
          color: "var(--respilens-ink-soft)",
          fontSize: "var(--mantine-font-size-sm)",
          lineHeight: 1.55,
        },
      }),
    },
    Badge: {
      defaultProps: { radius: "sm" },
      styles: { label: { fontWeight: 600, letterSpacing: "0.01em" } },
    },
  },

  other: {
    // Keep model colors available for data visualizations
    modelColors: MODEL_COLORS,
    getModelColor: getModelColor,

    // Surface tokens, mirrored from styles/global.css
    canvas: "var(--respilens-canvas)",
    surface: "var(--respilens-surface)",
    hairline: "var(--respilens-hairline)",
  },
});

// MantineProvider injects its generated CSS variables at runtime, after any
// stylesheet, so a plain CSS override of these tokens is always overwritten.
// They have to go through the resolver to take effect.
//
// Mantine's `light` button variant hovers from 0.10 to 0.12 alpha, a step too
// small to perceive; these give the hover enough travel to read as feedback.
export const cssVariablesResolver = () => ({
  variables: {},
  light: {
    // Base tint raised from 0.10 so the control reads as a control, and the
    // hover given enough travel from it to register as feedback.
    "--mantine-color-blue-light": "rgba(0, 118, 209, 0.16)",
    "--mantine-color-blue-light-hover": "rgba(0, 118, 209, 0.28)",
    // Label sits on that tint, so it uses shade 7 rather than 6 (4.99:1).
    "--mantine-color-blue-light-color": "#0062a8",

    "--mantine-color-green-light": "rgba(22, 163, 74, 0.16)",
    "--mantine-color-green-light-hover": "rgba(22, 163, 74, 0.28)",
    "--mantine-color-red-light": "rgba(220, 38, 38, 0.14)",
    "--mantine-color-red-light-hover": "rgba(220, 38, 38, 0.26)",
    "--mantine-color-yellow-light": "rgba(217, 119, 6, 0.16)",
    "--mantine-color-yellow-light-hover": "rgba(217, 119, 6, 0.28)",
    "--mantine-color-gray-light": "rgba(75, 85, 99, 0.10)",
    "--mantine-color-gray-light-hover": "rgba(75, 85, 99, 0.18)",
  },
  dark: {},
});
