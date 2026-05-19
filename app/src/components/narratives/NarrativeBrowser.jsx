import { useState, useEffect } from "react";
import {
  Container,
  Card,
  Title,
  Text,
  Group,
  Badge,
  Button,
  Stack,
  Loader,
  Center,
  ThemeIcon,
  Paper,
  SimpleGrid,
  TextInput,
  Select,
} from "@mantine/core";
import {
  IconBook,
  IconCalendar,
  IconUser,
  IconArrowRight,
  IconStar,
  IconClock,
  IconSearch,
} from "@tabler/icons-react";
import {
  narrativeRegistry,
  getAllTags,
  searchNarratives,
} from "../../data/narratives/index.js";
import Seo from "../Seo";

const NarrativeBrowser = ({ onNarrativeSelect }) => {
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterTag, setFilterTag] = useState("");
  const [sortBy, setSortBy] = useState("date");

  useEffect(() => {
    // Load narratives from registry
    console.log("Loading narratives from registry:", narrativeRegistry.length);
    setLoading(false);
  }, []);

  // Get filtered narratives based on search/filter criteria
  const filteredNarratives = searchNarratives(searchTerm, filterTag, sortBy);
  const featuredNarratives = filteredNarratives.filter((n) => n.featured);
  const regularNarratives = filteredNarratives.filter((n) => !n.featured);
  const allTags = getAllTags();

  if (loading) {
    return (
      <Center h="50vh">
        <Stack align="center" gap="md">
          <Loader size="lg" />
          <Text>Loading narratives...</Text>
        </Stack>
      </Center>
    );
  }

  const NarrativeCard = ({ narrative, featured = false }) => (
    <Card
      shadow={featured ? "md" : "sm"}
      padding="lg"
      radius="md"
      withBorder
      style={{
        cursor: "pointer",
        transition: "all 0.2s ease",
        border: featured
          ? "2px solid var(--mantine-primary-color-filled)"
          : undefined,
      }}
      onClick={() => onNarrativeSelect(narrative.id)}
    >
      <Group justify="space-between" align="flex-start" mb="md">
        <div style={{ flex: 1 }}>
          <Group gap="xs" mb="xs">
            <ThemeIcon
              size="sm"
              variant="light"
              color={featured ? "yellow" : "blue"}
            >
              {featured ? <IconStar size={16} /> : <IconBook size={16} />}
            </ThemeIcon>
            {featured && (
              <Badge size="xs" variant="light" color="yellow">
                Featured
              </Badge>
            )}
          </Group>
          <Title order={3} mb="xs" lineClamp={2}>
            {narrative.title}
          </Title>
        </div>
      </Group>

      <Text size="sm" c="dimmed" mb="md" lineClamp={3}>
        {narrative.description}
      </Text>

      <Group gap="xs" mb="md" wrap="wrap">
        {narrative.tags.map((tag) => (
          <Badge key={tag} size="xs" variant="light">
            {tag}
          </Badge>
        ))}
      </Group>

      <Group justify="space-between" align="center">
        <Stack gap={4}>
          <Group gap="xs">
            <IconUser size={14} />
            <Text size="xs" c="dimmed">
              {narrative.author}
            </Text>
          </Group>
          <Group gap="md">
            <Group gap="xs">
              <IconCalendar size={14} />
              <Text size="xs" c="dimmed">
                {narrative.date}
              </Text>
            </Group>
            <Group gap="xs">
              <IconClock size={14} />
              <Text size="xs" c="dimmed">
                {narrative.readTime}
              </Text>
            </Group>
            {narrative.slideCount && (
              <Group gap="xs">
                <IconBook size={14} />
                <Text size="xs" c="dimmed">
                  {narrative.slideCount} slides
                </Text>
              </Group>
            )}
          </Group>
        </Stack>

        <Button
          variant="light"
          size="xs"
          rightSection={<IconArrowRight size={14} />}
          onClick={(e) => {
            e.stopPropagation();
            onNarrativeSelect(narrative.id);
          }}
        >
          Read
        </Button>
      </Group>
    </Card>
  );

  return (
    <>
      <Seo
        title="RespiLens | Narratives"
        description="Explore interactive RespiLens narratives that combine respiratory disease data, forecasts, and visual explanations."
        canonicalPath="/narratives"
      />
      <Container size="xl" py="xl">
        {/* Header */}
        <Paper shadow="sm" p="lg" mb="xl">
          <Group align="center" mb="md">
            <ThemeIcon size="lg" variant="light">
              <IconBook size={24} />
            </ThemeIcon>
            <div>
              <Title order={1}>Data Narratives</Title>
              <Text c="dimmed">
                Explore interactive stories that bring respiratory disease data
                to life
              </Text>
            </div>
          </Group>

          {/* Search and Filters */}
          <Group grow>
            <TextInput
              placeholder="Search narratives..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              leftSection={<IconSearch size={16} />}
            />
            <Select
              placeholder="Filter by tag"
              value={filterTag}
              onChange={setFilterTag}
              data={[
                { value: "", label: "All tags" },
                ...allTags.map((tag) => ({ value: tag, label: tag })),
              ]}
              clearable
            />
            <Select
              placeholder="Sort by"
              value={sortBy}
              onChange={setSortBy}
              data={[
                { value: "date", label: "Date (newest first)" },
                { value: "title", label: "Title (A-Z)" },
                { value: "author", label: "Author (A-Z)" },
                { value: "readTime", label: "Read time (shortest first)" },
              ]}
            />
          </Group>
        </Paper>

        {/* Featured Narratives */}
        {featuredNarratives.length > 0 && (
          <>
            <Title order={2} mb="md">
              Featured Narratives
            </Title>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" mb="xl">
              {featuredNarratives.map((narrative) => (
                <NarrativeCard
                  key={narrative.id}
                  narrative={narrative}
                  featured={true}
                />
              ))}
            </SimpleGrid>
          </>
        )}

        {/* All Narratives */}
        <Title order={2} mb="md">
          {featuredNarratives.length > 0 ? "More Narratives" : "All Narratives"}
        </Title>

        {regularNarratives.length > 0 ? (
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
            {regularNarratives.map((narrative) => (
              <NarrativeCard key={narrative.id} narrative={narrative} />
            ))}
          </SimpleGrid>
        ) : (
          <Paper p="xl" style={{ textAlign: "center" }}>
            <Text c="dimmed">
              {searchTerm || filterTag
                ? "No narratives found matching your criteria."
                : "No additional narratives available."}
            </Text>
          </Paper>
        )}
      </Container>
    </>
  );
};

export default NarrativeBrowser;
