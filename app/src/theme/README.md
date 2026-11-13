# RespiLens Color System

This document outlines the centralized color system for the RespiLens application.

## Overview

The color system is centralized in `/src/theme/mantine.js` and provides:

- **Consistent branding** across all components
- **Semantic color meanings** for better UX
- **Model color preservation** for data visualization
- **Theme integration** with Mantine UI library

## Usage

### Accessing Colors in Components

```jsx
import { useMantineTheme } from "@mantine/core";

const MyComponent = () => {
  const theme = useMantineTheme();
  const colors = theme.other.colors;

  return <Text style={{ color: colors.primary }}>Primary colored text</Text>;
};
```

### Using Model Colors

```jsx
import { useMantineTheme } from "@mantine/core";

const MyComponent = ({ model }) => {
  const theme = useMantineTheme();
  const modelColor = theme.other.getModelColor(model);

  return <div style={{ backgroundColor: modelColor }}>Model visualization</div>;
};
```

## Color Categories

### Primary Brand Colors

- `colors.primary` - Main blue (#2563eb)
- `colors.primaryDark` - Darker blue (#1d4ed8)
- `colors.primaryLight` - Lighter blue (#3b82f6)

### Status Colors

- `colors.success` - Green (#10b981)
- `colors.warning` - Amber (#f59e0b)
- `colors.error` - Red (#ef4444)
- `colors.info` - Cyan (#06b6d4)

### Text Colors

- `colors.text.primary` - Dark gray (#111827)
- `colors.text.secondary` - Medium gray (#6b7280)
- `colors.text.muted` - Light gray (#9ca3af)
- `colors.text.inverse` - White (#ffffff)

### Background Colors

- `colors.background.primary` - White (#ffffff)
- `colors.background.secondary` - Very light gray (#f9fafb)
- `colors.background.tertiary` - Light gray (#f3f4f6)
- `colors.background.dark` - Dark mode background (#111827)

### Border Colors

- `colors.border.light` - Light border (#e5e7eb)
- `colors.border.medium` - Medium border (#d1d5db)
- `colors.border.dark` - Dark border (#6b7280)

### Forecast-Specific Colors

- `colors.forecast.line` - Main forecast line (#2563eb)
- `colors.forecast.confidence95` - 95% CI opacity ('10')
- `colors.forecast.confidence50` - 50% CI opacity ('30')
- `colors.forecast.groundTruth` - Ground truth data (#000000)

### Navigation Colors

- `colors.navigation.active` - Active nav item (#2563eb)
- `colors.navigation.hover` - Hover state (#3b82f6)
- `colors.navigation.text` - Nav text (#374151)

### Special Elements

- `colors.alpha` - Red alpha superscript (#ef4444)
- `colors.logo` - Logo text color (#2563eb)

## Model Colors

Every model is drawn the same way on every chart, for everyone, whatever else is selected:

- **Permanent assignment** in `modelColors.json`, kept up to date by `scripts/assign_model_colors.py`. Never edit an existing entry by hand: colors must not change.
- **One blue for each hub's main ensemble**, reserved for them
- **One gray for each hub's baseline**, reserved for them
- **Dashed lines** for models excluded from their hub's ensemble (`designated_model: false`)
- **Different markers** where a season has more models than colors (FluSight)
- **Opacity variations** for confidence intervals

### Model Color Usage

```jsx
// Get color for a specific model
const modelColor = getModelColor(modelName);

// Or all of it: { color, marker, dashed }
const modelStyle = getModelStyle(modelName);

// Use with opacity for confidence intervals
const ci95Color = `${modelColor}${colors.forecast.confidence95}`;
const ci50Color = `${modelColor}${colors.forecast.confidence50}`;
```

## Migration Guide

When updating existing components:

1. **Import theme hook**: Add `useMantineTheme` to imports
2. **Access colors**: Use `theme.other.colors` to access the color scheme
3. **Replace hardcoded colors**: Replace `'#2563eb'` with `colors.primary`
4. **Update semantic colors**: Use `colors.error` instead of `'red'`
5. **Model colors**: Continue using existing model color functions

### Before

```jsx
<Text style={{ color: "#ef4444" }}>Error text</Text>
```

### After

```jsx
const theme = useMantineTheme();
const colors = theme.other.colors;

<Text style={{ color: colors.error }}>Error text</Text>;
```

## Benefits

- **Consistency**: All colors defined in one place
- **Maintainability**: Easy to update brand colors across the app
- **Accessibility**: Colors chosen for good contrast ratios
- **Semantic meaning**: Colors convey meaning (error, success, etc.)
- **Theme compatibility**: Works with light/dark modes
- **Model preservation**: Existing model visualizations unchanged
