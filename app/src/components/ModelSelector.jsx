import { useRef, useState } from "react";
import {
  Group,
  Stack,
  Text,
  TextInput,
  SegmentedControl,
  Anchor,
  Checkbox,
  CloseButton,
  Tooltip,
} from "@mantine/core";
import { IconSearch } from "@tabler/icons-react";
import { getModelColor, getModelStyle } from "../config/datasets";
import { useKeyboardShortcut } from "../hooks/useKeyboardShortcut";
import ShortcutHint from "./ShortcutHint";

// Model picker under the chart: one filterable checklist, coloured like the
// chart's lines. "All" (the default, and while searching) lists every model
// so any of them can be toggled; "Selected" narrows it to what is plotted.
const ModelSelector = ({
  models = [],
  selectedModels = [],
  setSelectedModels,
  activeModels = null,
  // Map of model to a { label, title } superscript (MetroCast's ST / LT)
  modelTags = null,
  allowMultiple = true,
  disabled = false,
  modelColorFn = null,
  showDashedNote = true,
  // Only used to word the "not available" tooltip
  selectedDates = [],
  // The key that jumps to the filter box: true is M, or pass another key
  // (Shift+M is "M") for a page with a second list. Off by default.
  keyboardShortcut = false,
  title = "Models",
}) => {
  const [scope, setScope] = useState("all");
  const [search, setSearch] = useState("");
  const searchRef = useRef(null);
  const shortcutKey = keyboardShortcut === true ? "m" : keyboardShortcut;
  useKeyboardShortcut(
    shortcutKey,
    () => {
      searchRef.current?.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
      searchRef.current?.focus({ preventScroll: true });
    },
    Boolean(shortcutKey) && !disabled && models.length > 0,
  );

  const colorFor = (model) =>
    modelColorFn ? modelColorFn(model) : getModelColor(model);
  // The peak chart draws points, so nothing on it is dashed
  const hasDashedLine =
    showDashedNote &&
    !modelColorFn &&
    selectedModels.some((model) => getModelStyle(model).dashed);

  const isActive = (model) => !activeModels || activeModels.has(model);
  const isSelected = (model) => selectedModels.includes(model);

  const query = search.toLowerCase().trim();
  // Searching always looks through every model
  const pool = query || scope === "all" ? models : selectedModels;
  const visibleModels = query
    ? pool.filter((model) => model.toLowerCase().includes(query))
    : pool;

  const toggle = (model) => {
    if (disabled || !isActive(model)) return;
    if (isSelected(model)) {
      setSelectedModels(selectedModels.filter((m) => m !== model));
    } else if (allowMultiple) {
      setSelectedModels([...selectedModels, model]);
    } else {
      setSelectedModels([model]);
    }
  };

  // "Select all" adds every model with a forecast, or only the matches
  // while searching
  const toAdd = (query ? visibleModels : models).filter(
    (model) => isActive(model) && !isSelected(model),
  );
  const handleSelectAll = () =>
    setSelectedModels([...selectedModels, ...toAdd]);
  // An empty selection falls back to the dataset's default model
  const handleReset = () => setSelectedModels([]);

  const handleSearchKeyDown = (event) => {
    if (event.key === "Enter") {
      const first = visibleModels.find(
        (model) => isActive(model) && !isSelected(model),
      );
      if (first) toggle(first);
    } else if (event.key === "Escape") {
      setSearch("");
    }
  };

  if (!models.length) {
    return (
      <Text c="dimmed" fs="italic" size="sm">
        No models available
      </Text>
    );
  }

  return (
    <Stack
      gap="sm"
      mt="md"
      pt="md"
      style={{ borderTop: "1px solid var(--respilens-hairline)" }}
    >
      {/* One row: count, filter, scope, then the bulk actions beside them */}
      <Group gap="sm" wrap="wrap">
        <Text fw={600} size="sm">
          {title}{" "}
          <Text span c="dimmed" size="sm" fw={400}>
            {selectedModels.length} of {models.length}
          </Text>
        </Text>
        <Tooltip
          label={
            <ShortcutHint
              label="Filter models"
              shortcut={shortcutKey === "M" ? "⇧M" : shortcutKey}
            />
          }
          disabled={!shortcutKey}
          openDelay={400}
        >
          <TextInput
            ref={searchRef}
            w={240}
            size="xs"
            placeholder="Filter models…"
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
            onKeyDown={handleSearchKeyDown}
            leftSection={<IconSearch size={14} />}
            rightSection={
              search && (
                <CloseButton
                  size="sm"
                  onClick={() => setSearch("")}
                  aria-label="Clear search"
                />
              )
            }
            aria-label="Filter forecasting models"
            aria-keyshortcuts={
              shortcutKey ? shortcutKey.toUpperCase() : undefined
            }
            disabled={disabled}
          />
        </Tooltip>
        <SegmentedControl
          size="xs"
          value={query ? "all" : scope}
          onChange={setScope}
          data={[
            { value: "selected", label: "Selected" },
            { value: "all", label: "All" },
          ]}
          disabled={disabled || Boolean(query)}
          aria-label="Which models to list"
        />

        {allowMultiple && (
          <Group gap="md" ml="xs">
            <Tooltip
              label={
                query
                  ? `Add the ${toAdd.length} matching models`
                  : "Add every model with a forecast"
              }
              openDelay={300}
            >
              <Anchor
                component="button"
                size="sm"
                onClick={handleSelectAll}
                disabled={disabled || toAdd.length === 0}
              >
                Select all
              </Anchor>
            </Tooltip>
            <Tooltip label="Back to the default model" openDelay={300}>
              <Anchor
                component="button"
                size="sm"
                c="dimmed"
                onClick={handleReset}
                disabled={disabled || selectedModels.length === 0}
              >
                Reset
              </Anchor>
            </Tooltip>
          </Group>
        )}
      </Group>

      {visibleModels.length === 0 ? (
        <Text size="sm" c="dimmed">
          {query
            ? `No model matches “${search.trim()}”.`
            : "No models selected. Search above or switch to All."}
        </Text>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            columnGap: "var(--mantine-spacing-md)",
            rowGap: 2,
          }}
        >
          {visibleModels.map((model) => {
            const active = isActive(model);
            const checked = isSelected(model);
            // Colour only once plotted: unselected models have no line yet
            const color = checked ? colorFor(model) : undefined;
            const checkbox = (
              <Checkbox
                size="xs"
                radius="sm"
                checked={checked}
                onChange={() => toggle(model)}
                disabled={disabled || !active}
                color={color}
                styles={{
                  root: { padding: "3px 0", minWidth: 0 },
                  body: { alignItems: "center" },
                  labelWrapper: { minWidth: 0 },
                  label: {
                    fontSize: 13,
                    fontWeight: checked ? 500 : 400,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    cursor: active ? "pointer" : "not-allowed",
                  },
                }}
                label={
                  modelTags?.has(model) ? (
                    <>
                      {model}
                      <Text
                        component="sup"
                        span
                        c="dimmed"
                        fz={9}
                        fw={600}
                        ml={2}
                        title={modelTags.get(model).title}
                      >
                        {modelTags.get(model).label}
                      </Text>
                    </>
                  ) : (
                    model
                  )
                }
                title={active ? model : undefined}
              />
            );
            if (active) return <div key={model}>{checkbox}</div>;
            // Disabled inputs swallow hover events, so the tooltip sits on
            // a wrapper
            return (
              <Tooltip
                key={model}
                label={
                  selectedDates.length > 1
                    ? "Not available for these dates"
                    : "Not available for this date"
                }
                openDelay={200}
              >
                <div>{checkbox}</div>
              </Tooltip>
            );
          })}
        </div>
      )}
      {hasDashedLine && (
        <Text size="xs" c="dimmed">
          Dashed lines: models left out of the hub&apos;s ensemble.
        </Text>
      )}
    </Stack>
  );
};

export default ModelSelector;
