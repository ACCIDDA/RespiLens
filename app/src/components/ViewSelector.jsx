import { useEffect, useState } from "react";
import { Stack, Box, UnstyledButton } from "@mantine/core";
import {
  IconChevronDown,
  IconChevronRight,
  IconChartLine,
  IconActivityHeartbeat,
} from "@tabler/icons-react";
import { useView } from "../hooks/useView";
import { DATASETS } from "../config";

// Forecasts and surveillance data as two always-open groups, each with its
// own heading and icon. `showActive`: highlight the current view (off on pages
// that are not a chart, where the remembered view is not "where you are").
const ViewSelector = ({ showActive = true, onNavigate }) => {
  const { viewType, setViewType } = useView();
  const [isFluExpanded, setIsFluExpanded] = useState(false);

  const fluViews = [
    {
      label: "Standard view",
      value: DATASETS.flu.defaultView,
    },
    {
      label: "Peak forecasts",
      value: "flu_peak",
    },
    {
      label: "MetroCast Forecasts",
      value: DATASETS.metrocast.defaultView,
    },
  ];

  const forecastOptions = [
    {
      label: "Flu",
      value: DATASETS.flu.defaultView,
      children: fluViews,
    },
    {
      label: "COVID-19",
      value: DATASETS.covid.defaultView,
    },
    {
      label: "RSV",
      value: DATASETS.rsv.defaultView,
    },
  ];

  const surveillanceOptions = [
    {
      label: "NHSN",
      value: DATASETS.nhsn.defaultView,
    },
    {
      label: "NSSP",
      value: DATASETS.nssp.defaultView,
    },
  ];

  const fluViewValues = new Set(fluViews.map((view) => view.value));
  const isFluActive = fluViewValues.has(viewType) || viewType === "fludetailed";

  useEffect(() => {
    if (isFluActive) {
      setIsFluExpanded(true);
    }
  }, [isFluActive]);

  // `onNavigate` lets the mobile drawer close once a view is picked
  const handleViewSelect = (value) => {
    setViewType(value);
    onNavigate?.();
  };

  // Rows share the sidebar's nav-row style. `isActive` is the selected view;
  // `isParentActive` is the group holding it (Flu), which only gets the
  // active text colour so a single row is highlighted.
  const renderOptionButton = ({
    label,
    value,
    nested = false,
    rightSection = null,
    onClick,
    isActive = false,
    isParentActive = false,
  }) => (
    <UnstyledButton
      key={value || label}
      className="respilens-nav-row"
      data-indent={nested ? "2" : "1"}
      data-active={isActive || undefined}
      data-parent-active={isParentActive || undefined}
      aria-current={isActive ? "page" : undefined}
      onClick={onClick}
    >
      <span className="respilens-nav-row-label">{label}</span>
      {rightSection && (
        <span className="respilens-nav-row-end">{rightSection}</span>
      )}
    </UnstyledButton>
  );

  const renderSection = (title, Icon, options) => (
    <Stack gap={1}>
      <div className="respilens-nav-heading">
        <Icon size={17} stroke={1.75} />
        <span>{title}</span>
      </div>
      {options.map((option) => {
        if (!option.children) {
          return renderOptionButton({
            label: option.label,
            value: option.value,
            isActive: showActive && viewType === option.value,
            onClick: () => handleViewSelect(option.value),
          });
        }

        return (
          <Box key={option.label}>
            {renderOptionButton({
              label: option.label,
              value: option.value,
              isParentActive: showActive && isFluActive,
              rightSection: isFluExpanded ? (
                <IconChevronDown size={14} />
              ) : (
                <IconChevronRight size={14} />
              ),
              // Like COVID-19 and RSV, clicking Flu opens it. Only once a flu
              // view is showing does the row fall back to folding the list.
              onClick: () =>
                isFluActive
                  ? setIsFluExpanded((expanded) => !expanded)
                  : handleViewSelect(DATASETS.flu.defaultView),
            })}
            {isFluExpanded && (
              <Stack gap={1} mt={1}>
                {option.children.map((child) =>
                  renderOptionButton({
                    label: child.label,
                    value: child.value,
                    nested: true,
                    isActive: showActive && viewType === child.value,
                    onClick: () => handleViewSelect(child.value),
                  }),
                )}
              </Stack>
            )}
          </Box>
        );
      })}
    </Stack>
  );

  return (
    <Stack gap="xs">
      {renderSection("Forecasts", IconChartLine, forecastOptions)}
      {renderSection(
        "Surveillance data",
        IconActivityHeartbeat,
        surveillanceOptions,
      )}
    </Stack>
  );
};

export default ViewSelector;
