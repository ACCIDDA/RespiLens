import {
  ActionIcon,
  Alert,
  Collapse,
  Anchor,
  Container,
  Group,
  List,
  Paper,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import {
  IconAlertCircle,
  IconCheck,
  IconChevronDown,
  IconChevronUp,
  IconFileDescription,
  IconFilter,
  IconUpload,
} from "@tabler/icons-react";
import Seo from "../Seo";

const SectionCard = ({ title, icon, children, defaultOpened = false }) => {
  const [opened, { toggle }] = useDisclosure(defaultOpened);

  return (
    <Paper withBorder radius="lg" p="lg">
      <Stack gap="md">
        <Stack
          gap="xs"
          onClick={toggle}
          style={{ cursor: "pointer", userSelect: "none" }}
        >
          <Group justify="flex-start" align="center" gap="xs">
            <ThemeIcon size={28} radius="xl" variant="light" color="blue">
              {icon}
            </ThemeIcon>
            <Title order={2}>{title}</Title>
            <ActionIcon variant="subtle" color="gray" aria-label={title}>
              {opened ? (
                <IconChevronUp size={18} />
              ) : (
                <IconChevronDown size={18} />
              )}
            </ActionIcon>
          </Group>
        </Stack>
        <Collapse in={opened}>
          <Stack gap="md">{children}</Stack>
        </Collapse>
      </Stack>
    </Paper>
  );
};

const Documentation = () => {
  return (
    <>
      <Seo
        title="Forecast Checker Documentation | RespiLens"
        description="Learn how to visualize your data with Forecast Checker, what user-uploaded forecast data must contain, what gets filtered out, and what causes upload failures."
        canonicalPath="/toolbox/forecast-checker/documentation"
      />
      <Container size="lg" pt="md" pb="xl">
        <Stack gap="lg">
          <Stack gap="sm">
            <Title order={1}>Forecast Checker Documentation</Title>
          </Stack>

          <SectionCard
            title="What is Forecast Checker?"
            icon={<IconUpload size={20} />}
            defaultOpened={true}
          >
            <Text>
              Forecast Checker allows users to quickly visualize their forecast
              data simply by dragging and dropping CSV file(s). From the user's
              forecast data (and, if using a listed hub, the ground truth data
              stored internally), a visualization dashboard will be built. All
              targets, locations, dates, and models found in the user's data
              will be available for selection. Additionally, a control panel is
              provided to modulate between linear, log base 10, log base 2, and
              square root y-axis scales, and to toggle which prediction
              intervals are visible on the plot.
            </Text>
            <Text>
              Before uploading, you will be prompted to select which hub your
              data "belongs" to.{" "}
              <b>
                You do not have to have a submitting model in order to visualize
                your data, but your data must use the same target data streams
                as the selected hub
              </b>{" "}
              (e.g., you are predicting "weekly incidence of influenza
              hospitalization" when you select FluSight). Your forecast data
              must also comply with the Forecast Checker validation
              requirements, which are listed below. If you want to view
              forecasts that do not belong to one of the listed hubs, you may do
              so by first selecting <b>Other Hub</b>, and then providing your
              own ground truth data. Once the ground truth data has been
              validated, you may proceed with uploading your forecast data.
            </Text>
            <Text>
              When you use Forecast Checker, the data does not leave your device
              (it is a private display). That is, if you navigate away from your
              visualization, you will have to re-upload your data to view it
              again.
            </Text>
          </SectionCard>

          <SectionCard
            title="What are the requirements for my forecast data?"
            icon={<IconFileDescription size={20} />}
          >
            <Text>
              Your uploaded forecast data must be a Hubverse-style forecast CSV.
              Forecast Checker can accept one file or multiple CSV files at
              once.
            </Text>
            <Text fw={600}>Required columns:</Text>
            <List spacing="sm">
              <List.Item>
                <code>location</code> (FIPS code encoded as a string)
              </List.Item>
              <List.Item>
                <code>reference_date</code>
              </List.Item>
              <List.Item>
                <code>target</code>
              </List.Item>
              <List.Item>
                <code>horizon</code>
              </List.Item>
              <List.Item>
                <code>output_type</code>
              </List.Item>
              <List.Item>
                <code>output_type_id</code>
              </List.Item>
              <List.Item>
                <code>value</code>
              </List.Item>
              <List.Item>
                <code>target_end_date</code>
              </List.Item>
            </List>
            <Text>
              <code>model_id</code> is optional. If it is missing, Forecast
              Checker assigns the fallback model name{" "}
              <code>user-uploaded-model</code> and assumes all data belongs to a
              single model.
            </Text>
            <Text fw={600}>Expected value patterns</Text>
            <List spacing="sm">
              <List.Item>
                <code>value</code> must be numeric.
              </List.Item>
              <List.Item>
                <code>target_end_date</code> must be parseable as a date.
              </List.Item>
              <List.Item>
                <code>horizon</code> must be parseable as an integer.
              </List.Item>
              <List.Item>
                The currently supported forecast outputs are quantitative
                forecast rows that the visualization layer can render.
              </List.Item>
            </List>
            <Text size="sm" c="dimmed">
              For more information on the Hubverse format, visit the{" "}
              <Anchor
                href="https://docs.hubverse.io/en/latest/user-guide/model-output.html"
                target="_blank"
                rel="noreferrer"
              >
                Hubverse model output guide
              </Anchor>
              .
            </Text>
          </SectionCard>

          <SectionCard
            title="What is filtered out of my forecast data?"
            icon={<IconFilter size={20} />}
          >
            <Text>
              Some stipulations of user-uploaded data are not enforced with
              fatal errors. Instead, Forecast Checker filters them out during
              preprocessing and continues with the remaining usable rows. A list
              of things that will be filtered out of your data, if found:
            </Text>
            <List spacing="sm">
              <List.Item>
                Duplicate rows, using the combined values from the{" "}
                <code>reference_date</code>, <code>target_end_date</code>,{" "}
                <code>location</code>, <code>horizon</code>, <code>target</code>
                , <code>output_type</code>, <code>output_type_id</code>, and
                optional <code>model_id</code> columns.
              </List.Item>
              <List.Item>
                Rows where the <code>output_type</code> column value is NOT{" "}
                <code>quantile</code>.
              </List.Item>
              <List.Item>
                Rows where the <code>output_type</code> column is{" "}
                <code>quantile</code> but the <code>output_type_id</code> column
                is not a valid numeric quantile between <code>0</code> and{" "}
                <code>1</code>.
              </List.Item>
              <List.Item>
                Quantile rows whose <code>output_type_id</code> values do not
                have the matching upper or lower partner needed to form a
                prediction interval. Unpaired quantiles are filtered out, while
                paired quantiles are kept and used to build interval shading in
                the dashboard.
              </List.Item>
              <List.Item>
                When Flu Metrocast Hub has been selected, rows where the{" "}
                <code>target_end_date</code> column is before{" "}
                <code>2025-11-22</code>.
              </List.Item>
              <List.Item>
                Rows where the <code>target</code> column is a flu peak target,
                which is currently excluded from this Forecast Checker workflow.
              </List.Item>
            </List>
            <Alert
              color="blue"
              variant="light"
              radius="lg"
              icon={<IconCheck size={16} />}
            >
              If enough usable rows remain after filtering, Forecast Checker
              will continue and build the dashboard.
            </Alert>
          </SectionCard>

          <SectionCard
            title="What will cause errors or failures?"
            icon={<IconAlertCircle size={20} />}
          >
            <Text>
              Forecast Checker will stop and return an error when the uploaded
              forecast data cannot be meaningfully processed. This can happen
              when:
            </Text>
            <List spacing="sm">
              <List.Item>
                One or more required forecast columns are missing.
              </List.Item>
              <List.Item>
                All rows are unusable after preprocessing and filtering.
              </List.Item>
              <List.Item>
                <code>target_end_date</code>, <code>value</code>, or{" "}
                <code>horizon</code> values cannot be parsed correctly.
              </List.Item>
              <List.Item>
                The frontend cannot load the hub reference files needed to pair
                your upload with ground truth and location metadata.
              </List.Item>
              <List.Item>
                Forecast rows reference locations that do not exist in the
                selected hub's <code>locations.csv</code>.
              </List.Item>
              <List.Item>
                No supported forecast rows remain after the app removes
                unsupported data.
              </List.Item>
            </List>
          </SectionCard>
        </Stack>
      </Container>
    </>
  );
};

export default Documentation;
