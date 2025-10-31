import { Modal, Button, Group, Text, List, Alert, Anchor, Image, Title, Stack, Badge, ActionIcon } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconInfoCircle, IconBrandGithub, IconAlertTriangle, IconWorld } from '@tabler/icons-react';

const InfoOverlay = () => {
  const [opened, { open, close }] = useDisclosure(false);

  return (
    <>
      {/* Desktop - Full button with text */}
      <Button
        variant="subtle"
        color="red"
        size="sm"
        leftSection={<IconInfoCircle size={20} />}
        onClick={open}
        radius="xl"
        visibleFrom="sm"
      >
        Info
      </Button>

      {/* Mobile - Icon only */}
      <ActionIcon
        variant="subtle"
        color="red"
        size="lg"
        onClick={open}
        aria-label="Info"
        hiddenFrom="sm"
      >
        <IconInfoCircle size={20} />
      </ActionIcon>

      <Modal
        opened={opened}
        onClose={close}
        title={
          <Group gap="md">
            <Image src="respilens-logo.svg" alt="RespiLens logo" h={32} w={32} />
            <Title order={2} c="blue">RespiLens</Title>
          </Group>
        }
        size="lg"
        scrollAreaComponent={Modal.NativeScrollArea}
      >
        <Stack gap="md">
          <div>
            <Text size="sm" fw={500} mb="xs">Deployments</Text>
            <List spacing="xs" size="sm">
              <List.Item>
                <Group gap="xs" wrap="wrap">
                  <Badge size="xs" color="green" variant="light">Stable</Badge>
                  <Anchor
                    href="https://github.com/ACCIDDA/RespiLens"
                    target="_blank"
                    rel="noopener"
                  >
                    <Group gap={4}>
                      <IconBrandGithub size={14} />
                      <Text size="sm">ACCIDDA/RespiLens</Text>
                    </Group>
                  </Anchor>
                  <Text size="sm">deployed to</Text>
                  <Anchor
                    href="https://respilens.com"
                    target="_blank"
                    rel="noopener"
                  >
                    <Group gap={4}>
                      <IconWorld size={14} />
                      <Text size="sm">respilens.com</Text>
                    </Group>
                  </Anchor>
                </Group>
              </List.Item>
              <List.Item>
                <Group gap="xs" wrap="wrap">
                  <Badge size="xs" color="yellow" variant="light">Staging</Badge>
                  <Anchor
                    href="https://github.com/ACCIDDA/RespiLens-staging"
                    target="_blank"
                    rel="noopener"
                  >
                    <Group gap={4}>
                      <IconBrandGithub size={14} />
                      <Text size="sm">ACCIDDA/RespiLens-staging</Text>
                    </Group>
                  </Anchor>
                  <Text size="sm">deployed to</Text>
                  <Anchor
                    href="https://staging.respilens.com"
                    target="_blank"
                    rel="noopener"
                  >
                    <Group gap={4}>
                      <IconWorld size={14} />
                      <Text size="sm">staging.respilens.com</Text>
                    </Group>
                  </Anchor>
                </Group>
              </List.Item>
            </List>
          </div>

          <Text>
            RespiLens is a responsive web app to visualize respiratory disease forecasts in the US, focused on
            accessibility for state health departments and the general public. Key features include:
          </Text>

          <List spacing="xs" size="sm">
            <List.Item>URL-shareable views for specific forecasts</List.Item>
            <List.Item>Weekly automatic updates</List.Item>
            <List.Item>Multi-pathogen and multi-view</List.Item>
            <List.Item>Multi-date comparison capability</List.Item>
            <List.Item>Flexible model comparison</List.Item>
            <List.Item>Responsive and mobile friendly (for some views)</List.Item>
          </List>

          <div>
            <Title order={4} mb="xs">On the roadmap</Title>
            <List spacing="xs" size="sm">
              <List.Item>Scoring visualization and ability to select best models</List.Item>
              <List.Item>Model description on hover</List.Item>
            </List>
          </div>

          <Text size="sm">
            Made by Emily Przykucki (UNC Chapel Hill), {' '} 
            <Anchor href="https://josephlemaitre.com" target="_blank" rel="noopener">
              Joseph Lemaitre
            </Anchor>{' '}
            (UNC Chapel Hill) and others within ACCIDDA, the Atlantic Coast Center
            for Infectious Disease Dynamics and Analytics.
          </Text>

        </Stack>
      </Modal>
    </>
  );
};

export default InfoOverlay;
