import { Button, Group, Stack, Text } from "@mantine/core";
import { MODEL_COLORS } from "../config/datasets";

const NSSPColumnSelector = ({
  availableColumns,
  selectedColumns,
  setSelectedColumns,
  columnLabelMap,
}) => {
  const toggleColumn = (column) => {
    if (selectedColumns.includes(column)) {
      setSelectedColumns(selectedColumns.filter((value) => value !== column));
      return;
    }

    setSelectedColumns([...selectedColumns, column]);
  };

  return (
    <Stack gap="sm">
      <div>
        <Text size="sm" fw={700}>
          Select a pathogen(s)
        </Text>
      </div>

      <Group gap="xs">
        {availableColumns.map((column, index) => {
          const isSelected = selectedColumns.includes(column);

          return (
            <Button
              key={column}
              onClick={() => toggleColumn(column)}
              variant={isSelected ? "filled" : "outline"}
              size="xs"
              style={
                isSelected
                  ? {
                      backgroundColor:
                        MODEL_COLORS[index % MODEL_COLORS.length],
                      color: "white",
                    }
                  : undefined
              }
            >
              {columnLabelMap[column] || column}
            </Button>
          );
        })}
      </Group>
    </Stack>
  );
};

export default NSSPColumnSelector;
