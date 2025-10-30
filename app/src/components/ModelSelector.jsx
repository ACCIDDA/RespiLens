import { useState } from 'react';
import { Stack, Group, Button, Text, Tooltip, Divider, Switch, Card, SimpleGrid, PillsInput, Pill, Combobox, useCombobox } from '@mantine/core';
import { IconCircleCheck, IconCircle, IconEye, IconEyeOff } from '@tabler/icons-react';
import { MODEL_COLORS } from '../config/datasets';

const ModelSelector = ({ 
  models = [],
  selectedModels = [], 
  setSelectedModels,
  allowMultiple = true,
  disabled = false
}) => {
  const [showAllAvailable, setShowAllAvailable] = useState(true);
  const [search, setSearch] = useState('');
  const combobox = useCombobox({
    onDropdownClose: () => combobox.resetSelectedOption(),
    onDropdownOpen: () => combobox.updateSelectedOptionIndex('active', 0),
  });

  const handleSelectAll = () => {
    setSelectedModels(models);
  };

  const handleSelectNone = () => {
    setSelectedModels([]);
  };

  const getModelColorByIndex = (model) => {
    // Only color selected models to avoid mismatched palette hints
    const index = selectedModels.indexOf(model);
    return index >= 0 ? MODEL_COLORS[index % MODEL_COLORS.length] : undefined;
  };

  const modelsToShow = showAllAvailable ? models : selectedModels;

  const handleValueSelect = (val) => {
    if (selectedModels.includes(val)) {
      setSelectedModels(selectedModels.filter(v => v !== val));
    } else if (allowMultiple) {
      setSelectedModels([...selectedModels, val]);
    } else {
      setSelectedModels([val]);
    }
  };

  const handleValueRemove = (val) => {
    setSelectedModels(selectedModels.filter(v => v !== val));
  };

  const filteredModels = models.filter(model =>
    model.toLowerCase().includes(search.toLowerCase().trim())
  );

  if (!models.length) {
    return (
      <Text c="dimmed" fs="italic" size="sm">
        No models available
      </Text>
    );
  }

  return (
    <Stack gap="md" mt="md">
      <Divider />
      
      <Text size="sm" fw={500} c="dimmed">
        Models ({selectedModels.length}/{models.length})
      </Text>

      {/* Custom MultiSelect with Colored Pills */}
      <Combobox
        store={combobox}
        onOptionSubmit={handleValueSelect}
        withinPortal
      >
        <Combobox.DropdownTarget>
          <PillsInput onClick={() => combobox.openDropdown()} size="sm">
            <Pill.Group>
              {selectedModels.map((model) => {
                const modelColor = getModelColorByIndex(model);
                return (
                  <Pill
                    key={model}
                    withRemoveButton
                    onRemove={() => handleValueRemove(model)}
                    style={{
                      backgroundColor: modelColor,
                      color: 'white',
                      padding: '2px 6px',
                      fontSize: '0.75rem'
                    }}
                  >
                    {model}
                  </Pill>
                );
              })}

              <Combobox.EventsTarget>
                <PillsInput.Field
                  onFocus={() => combobox.openDropdown()}
                  onBlur={() => combobox.closeDropdown()}
                  value={search}
                  placeholder="Quick search and select models..."
                  onChange={(event) => {
                    combobox.updateSelectedOptionIndex();
                    setSearch(event.currentTarget.value);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Backspace' && search.length === 0) {
                      event.preventDefault();
                      handleValueRemove(selectedModels[selectedModels.length - 1]);
                    }
                  }}
                />
              </Combobox.EventsTarget>
            </Pill.Group>
          </PillsInput>
        </Combobox.DropdownTarget>

        <Combobox.Dropdown>
          <Combobox.Options>
            {filteredModels.map((model) => {
              const modelColor = getModelColorByIndex(model);
              const isSelected = selectedModels.includes(model);
              return (
                <Combobox.Option value={model} key={model} style={{ padding: '4px 8px' }}>
                  <Group gap="xs" justify="space-between">
                    <Group gap="xs" align="center">
                      {isSelected ? (
                        <IconCircleCheck size={16} style={{ color: modelColor }} />
                      ) : (
                        <IconCircle size={16} style={{ color: 'var(--mantine-color-gray-5)' }} />
                      )}
                      <span
                        style={{
                          color: isSelected ? modelColor : 'inherit',
                          fontWeight: isSelected ? 600 : 400
                        }}
                      >
                        {model}
                      </span>
                    </Group>
                  </Group>
                </Combobox.Option>
              );
            })}
          </Combobox.Options>
        </Combobox.Dropdown>
      </Combobox>

      <Group gap="xs" justify="space-between" align="center" wrap="wrap">
        {allowMultiple && (
          <Group gap="xs" wrap="wrap">
            <Tooltip label="Select all available models">
              <Button
                variant="subtle"
                size="xs"
                onClick={handleSelectAll}
                disabled={disabled || selectedModels.length === models.length}
              >
                Select All
              </Button>
            </Tooltip>
            <Tooltip label="Clear all selected models">
              <Button
                variant="subtle"
                size="xs"
                onClick={handleSelectNone}
                disabled={disabled || selectedModels.length === 0}
              >
                Clear All
              </Button>
            </Tooltip>
          </Group>
        )}
        <Switch
          label="Show all available models"
          checked={showAllAvailable}
          onChange={(event) => setShowAllAvailable(event.currentTarget.checked)}
          size="sm"
          disabled={disabled}
          thumbIcon={
            showAllAvailable ? (
              <IconEye size={12} stroke={2.5} />
            ) : (
              <IconEyeOff size={12} stroke={2.5} />
            )
          }
        />
      </Group>

      {allowMultiple && (
        <Text size="xs" c="dimmed" hiddenFrom="xs">
          {selectedModels.length > 0 && `${selectedModels.length} selected`}
        </Text>
      )}

      {/* Model Grid Display */}
      {modelsToShow.length > 0 && (
        <SimpleGrid 
          cols={{ base: 1, xs: 2, sm: 3, md: 4, lg: 5 }}
          spacing="xs"
          verticalSpacing="xs"
        >
          {modelsToShow.map((model) => {
            const isSelected = selectedModels.includes(model);
            const modelColor = getModelColorByIndex(model);
            const inactiveColor = 'var(--mantine-color-gray-5)';
            
            return (
              <Card
                key={model}
                p="xs"
                radius="md"
                withBorder={!isSelected}
                variant={isSelected ? 'filled' : 'default'}
                style={{
                  cursor: disabled ? 'not-allowed' : 'pointer',
                  backgroundColor: isSelected ? modelColor : undefined,
                  borderColor: isSelected ? modelColor : undefined,
                  minWidth: 0
                }}
                opacity={disabled ? 0.5 : 1}
                onClick={() => {
                  if (disabled) return;
                  
                  if (isSelected) {
                    setSelectedModels(selectedModels.filter(m => m !== model));
                  } else {
                    if (allowMultiple) {
                      setSelectedModels([...selectedModels, model]);
                    } else {
                      setSelectedModels([model]);
                    }
                  }
                }}
              >
                <Group gap="xs" justify="space-between" align="center">
                  <Group gap="xs" align="center" flex={1}>
                    {isSelected ? (
                      <IconCircleCheck size={16} color="white" />
                    ) : (
                      <IconCircle size={16} color={inactiveColor} />
                    )}
                    <Text 
                      size="xs" 
                      fw={isSelected ? 600 : 400}
                      c={isSelected ? 'white' : 'inherit'}
                      style={{ 
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        flex: 1
                      }}
                      title={model}
                    >
                      {model}
                    </Text>
                  </Group>
                </Group>
              </Card>
            );
          })}
        </SimpleGrid>
      )}
    </Stack>
  );
};

export default ModelSelector;
