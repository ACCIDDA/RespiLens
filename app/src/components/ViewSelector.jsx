import { useEffect, useState } from "react";
import { Stack, Button, Paper, Text, Box } from "@mantine/core";
import { IconChevronDown, IconChevronRight } from "@tabler/icons-react";
import { useView } from "../hooks/useView";
import { DATASETS } from "../config";

const ViewSelector = () => {
  const { viewType, setViewType } = useView();
  const [isFluExpanded, setIsFluExpanded] = useState(false);

  const fluViews = [
    {
      label: "Forecasts",
      value: DATASETS.flu.defaultView,
    },
    {
      label: "Detailed view",
      value: "fludetailed",
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
  const isFluActive = fluViewValues.has(viewType);

  useEffect(() => {
    if (isFluActive) {
      setIsFluExpanded(true);
    }
  }, [isFluActive]);

  const handleViewSelect = (value) => {
    setViewType(value);
  };

  const renderOptionButton = ({
    label,
    value,
    isLast = false,
    nested = false,
    rightSection = null,
    onClick,
    isActive = false,
  }) => (
    <Button
      key={value || label}
      variant={isActive ? "light" : "subtle"}
      color={isActive ? "blue" : "gray"}
      size="sm"
      radius={0}
      fullWidth
      justify="space-between"
      rightSection={rightSection}
      onClick={onClick}
      styles={{
        root: {
          height: nested ? 34 : 36,
          paddingInline: nested ? 20 : 14,
          borderBottom: isLast
            ? "none"
            : "1px solid var(--mantine-color-gray-3)",
        },
        inner: {
          width: "100%",
          justifyContent: "space-between",
        },
        label: {
          width: "100%",
          textAlign: "left",
          fontWeight: nested ? 500 : 600,
        },
      }}
    >
      {label}
    </Button>
  );

  const renderSection = (title, options) => (
    <Paper
      shadow="sm"
      radius="md"
      withBorder
      style={{ display: "inline-block" }}
    >
      <Text
        size="xs"
        fw={700}
        c="dimmed"
        px="sm"
        pt="sm"
        pb={6}
        style={{ letterSpacing: "0.08em" }}
      >
        {title}
      </Text>
      <Stack
        gap={0}
        style={{ borderTop: "2px solid var(--mantine-color-gray-3)" }}
      >
        {options.map((option, index) => {
          const isLastTopLevel = index === options.length - 1;

          if (!option.children) {
            return renderOptionButton({
              label: option.label,
              value: option.value,
              isLast: isLastTopLevel,
              isActive: viewType === option.value,
              rightSection: <IconChevronRight size={14} />,
              onClick: () => handleViewSelect(option.value),
            });
          }

          return (
            <Box key={option.label}>
              {renderOptionButton({
                label: option.label,
                value: option.value,
                isLast: !isFluExpanded && isLastTopLevel,
                isActive: isFluActive,
                rightSection: <IconChevronDown size={14} />,
                onClick: () => setIsFluExpanded((expanded) => !expanded),
              })}
              {isFluExpanded && (
                <Stack gap={0}>
                  {option.children.map((child, childIndex) =>
                    renderOptionButton({
                      label: child.label,
                      value: child.value,
                      nested: true,
                      isLast: childIndex === option.children.length - 1,
                      isActive: viewType === child.value,
                      rightSection: <IconChevronRight size={14} />,
                      onClick: () => handleViewSelect(child.value),
                    }),
                  )}
                </Stack>
              )}
            </Box>
          );
        })}
      </Stack>
    </Paper>
  );

  return (
    <Stack gap="md">
      {renderSection("FORECASTS", forecastOptions)}
      {renderSection("SURVEILLANCE DATA", surveillanceOptions)}
    </Stack>
  );
};

export default ViewSelector;
