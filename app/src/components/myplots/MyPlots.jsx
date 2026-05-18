import { useState, useEffect } from "react";
import {
  Title,
  Text,
  Paper,
  Stack,
  ThemeIcon,
  Center,
  SimpleGrid,
  Box,
  Badge,
  Group,
  Button,
} from "@mantine/core";
import {
  IconChartScatter,
  IconExternalLink,
  IconTrash,
} from "@tabler/icons-react";
import { getSavedPlots, deletePlot } from "../../utils/plotStorage";
import { fetchNsspLocationLabel } from "../../utils/nsspGeo";
import MiniPlot from "./MiniPlot";

const MyPlots = () => {
  const [userSavedPlots, setUserSavedPlots] = useState([]);
  const [plotLocationLabels, setPlotLocationLabels] = useState({});

  useEffect(() => {
    const plots = getSavedPlots();
    setUserSavedPlots(plots);
  }, []);

  useEffect(() => {
    let isActive = true;

    const loadNsspLabels = async () => {
      const nsspPlots = userSavedPlots.filter(
        (plot) => plot.viewType === "nsspall" && plot.settings?.location,
      );

      if (nsspPlots.length === 0) {
        if (isActive) {
          setPlotLocationLabels({});
        }
        return;
      }

      const resolvedEntries = await Promise.all(
        nsspPlots.map(async (plot) => [
          plot.id,
          await fetchNsspLocationLabel(plot.settings.location),
        ]),
      );

      if (isActive) {
        setPlotLocationLabels(Object.fromEntries(resolvedEntries));
      }
    };

    loadNsspLabels();

    return () => {
      isActive = false;
    };
  }, [userSavedPlots]);

  const handleDelete = (id) => {
    if (deletePlot(id)) {
      setUserSavedPlots(getSavedPlots());
    }
  };

  const hasPlots = userSavedPlots.length > 0;

  const getPlotLocationLabel = (plot) => {
    if (plot.viewType === "nsspall") {
      return plotLocationLabels[plot.id] || plot.settings.location;
    }

    return plot.settings.location.toUpperCase();
  };

  const pageContainerStyle = {
    width: "100%",
    minHeight: "calc(100vh - 80px)",
    backgroundColor: "var(--mantine-color-body)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "40px",
  };

  return (
    <Box style={pageContainerStyle}>
      {!hasPlots ? (
        <Center style={{ flex: 1, width: "100%" }}>
          <Paper
            shadow="xl"
            p="xl"
            radius="lg"
            withBorder
            style={{
              width: "100%",
              maxWidth: "550px",
              backgroundColor: "var(--mantine-color-body)",
              border: "2px solid #2563eb",
            }}
          >
            <Stack align="center" gap="xl">
              <ThemeIcon size={80} variant="light" color="gray" radius="xl">
                <IconChartScatter size={40} />
              </ThemeIcon>

              <div style={{ textAlign: "center" }}>
                <Title order={2} mb="md">
                  No plots saved yet...
                </Title>
                <Text size="sm" c="dimmed">
                  You haven't added any visualizations to <b>My Plots</b> yet.
                  Click the "Add to My Plots" button on any plot to see it here
                  with any editorializations you choose. This feature is in its
                  alpha release; if you encounter bugs or have suggestions,
                  please report them
                  <a
                    href="https://github.com/ACCIDDA/RespiLens/issues/new?title=%E2%80%BC%EF%B8%8FMy%20Plots:%20user%20bug%20or%20suggestion%20%E2%80%BC%EF%B8%8F"
                    rel="noopener"
                    target="_blank"
                  >
                    {" "}
                    here.
                  </a>
                </Text>
              </div>

              <Text size="xs" fw={500} c="blue" style={{ opacity: 0.7 }}>
                Plots are stored locally in your browser.
              </Text>
            </Stack>
          </Paper>
        </Center>
      ) : (
        <Stack style={{ width: "100%", maxWidth: "1400px" }} gap="xl">
          <Paper p="md" radius="md" withBorder shadow="sm">
            <Group justify="space-between" align="center">
              <div>
                <Title order={2}>My Plots</Title>
                <Text size="sm" c="dimmed">
                  Your personalized library of saved visualizations.
                </Text>
                <Text size="sm" c="dimmed">
                  This feature is in its alpha release, and is still under
                  develoment. If you encounter a bug or have a suggestion,
                  please{" "}
                  <a
                    href="https://github.com/ACCIDDA/RespiLens/issues/new?title=%E2%80%BC%EF%B8%8FMy%20Plots:%20user%20bug%20or%20suggestion%20%E2%80%BC%EF%B8%8F"
                    rel="noopener"
                    target="_blank"
                  >
                    let us know.
                  </a>
                </Text>
              </div>
              <Badge variant="filled" size="lg" color="blue">
                {userSavedPlots.length} Saved
              </Badge>
            </Group>
          </Paper>

          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="xl">
            {userSavedPlots.map((plot) => (
              <Paper
                key={plot.id}
                p="lg"
                radius="md"
                withBorder
                shadow="md"
                style={{
                  backgroundColor: "var(--mantine-color-body)",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <Stack gap="sm" justify="space-between" h="100%">
                  <Box>
                    <Group justify="space-between" mb="xs" wrap="nowrap">
                      <Group
                        gap="xs"
                        wrap="nowrap"
                        style={{ flex: 1, minWidth: 0 }}
                      >
                        <Badge
                          color="gray"
                          variant="outline"
                          size="xs"
                          style={{ flexShrink: 0 }}
                        >
                          {plot.viewDisplayName.toUpperCase()}
                        </Badge>
                        <Text
                          fw={700}
                          size="sm"
                          c="blue.7"
                          truncate
                          style={{
                            flex: 1,
                            minWidth: 0,
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                          title={getPlotLocationLabel(plot)}
                        >
                          {getPlotLocationLabel(plot)}
                        </Text>
                      </Group>

                      <Button
                        variant="subtle"
                        color="red"
                        size="compact-xs"
                        leftSection={<IconTrash size={14} />}
                        onClick={() => handleDelete(plot.id)}
                        style={{ flexShrink: 0 }}
                      >
                        Remove
                      </Button>
                    </Group>
                    <Paper
                      withBorder
                      radius="sm"
                      bg="gray.0"
                      style={{ overflow: "hidden", height: 230 }}
                    >
                      <MiniPlot plot={plot} />
                    </Paper>
                  </Box>

                  <Button
                    component="a"
                    href={plot.fullUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    variant="light"
                    color="blue"
                    fullWidth
                    mt="xs"
                    leftSection={<IconExternalLink size={16} />}
                  >
                    Visit view
                  </Button>
                </Stack>
              </Paper>
            ))}
          </SimpleGrid>
        </Stack>
      )}
    </Box>
  );
};

export default MyPlots;
