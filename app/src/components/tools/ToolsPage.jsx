import {
  Badge,
  Button,
  Card,
  Container,
  Group,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { IconClock, IconTools } from "@tabler/icons-react";
import { Link } from "react-router-dom";
import Seo from "../Seo";

const tools = [
  {
    title: "Reporting delay explorer",
    description:
      "Do you need to nowcast? What does your reporting delay distribution look like? Securely upload your reporting data to build your reporting triangle and answer these questions.",
    icon: IconClock,
    href: "/reporting-triangle",
    badge: "Nowcasting",
  },
];

const ToolsPage = () => {
  return (
    <>
      <Seo
        title="RespiLens | Toolbox"
        description="Browse RespiLens tools for respiratory disease data quality checks, reporting delay exploration, and nowcasting preparation."
        canonicalPath="/toolbox"
      />
      <Container size="xl" py="xl">
        <Stack gap="lg">
          <Group align="center">
            <IconTools size={28} />
            <Title order={1}>RespiLens Toolboox</Title>
          </Group>
          <Text c="dimmed" size="lg">
            Browse lightweight utilities for data QA and perhaps more at some
            point.
          </Text>

          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
            {tools.map((tool) => (
              <Card key={tool.title} withBorder radius="md" padding="lg">
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Group>
                      <tool.icon size={22} />
                      <Title order={3}>{tool.title}</Title>
                    </Group>
                    <Badge variant="light">{tool.badge}</Badge>
                  </Group>
                  <Text size="sm" c="dimmed">
                    {tool.description}
                  </Text>
                  <Button
                    component={Link}
                    to={tool.href}
                    size="sm"
                    w="fit-content"
                  >
                    Open tool
                  </Button>
                </Stack>
              </Card>
            ))}
          </SimpleGrid>
        </Stack>
      </Container>
    </>
  );
};

export default ToolsPage;
