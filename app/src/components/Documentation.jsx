import {
  Anchor,
  Container,
  Stack,
  Code,
  Paper,
  Title,
  Text,
  Group,
  ThemeIcon,
  Table,
  ActionIcon,
  Tooltip,
  Box,
  Collapse,
  Timeline,
  List,
} from "@mantine/core";
import { useClipboard, useDisclosure } from "@mantine/hooks";
import {
  IconBinaryTree,
  IconClipboard,
  IconTransformFilled,
  IconCopy,
  IconCheck,
  IconFlagQuestion,
  IconBook,
  IconFileCheck,
  IconCloudUpload,
  IconChevronDown,
  IconChevronUp,
} from "@tabler/icons-react";
import JsonView from "@uiw/react-json-view";
import Seo from "./Seo";

const glossaryItems = [
  { term: "forecasts", definition: "Data that is predicted (future)." },
  {
    term: "ground_truth",
    definition: "Data that has actually been observed (past and present)",
  },
  {
    term: "horizon",
    definition:
      "Time horizon from original date, in weeks (e.g. horizon = 1 is 1 week past original date)",
  },
  {
    term: "hubverse_keys",
    definition: "Metadata for data that comes from a Hubverse hub.",
  },
  {
    term: "hubverse data",
    definition:
      ".csv data pulled from a Hubverse hub (namely, contains hubverse columns).",
  },
  {
    term: "model",
    definition:
      'Name of the model used for submitted data, e.g. "CovidHub-ensemble".',
  },
  {
    term: "projections",
    definition: "A style of RespiLens JSON data used for data forecasts.",
  },
  {
    term: "quantile",
    definition: "Confidence intervals that corresponding values represent.",
  },
  {
    term: "target/column",
    definition:
      'A particular value to be predicted/observed, e.g. "weekly incidence of RSV hospitalization".',
  },
  {
    term: "timeseries",
    definition: "A style of RespiLens JSON data used for observed values.",
  },
];

const GlossaryTable = () => {
  const rows = glossaryItems.map((item) => (
    <Table.Tr key={item.term}>
      <Table.Td>{item.term}</Table.Td>
      <Table.Td>{item.definition}</Table.Td>
    </Table.Tr>
  ));
  return (
    <>
      <Table withTableBorder withColumnBorders>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Term</Table.Th>
            <Table.Th>Definition</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>{rows}</Table.Tbody>
      </Table>
    </>
  );
};

const projectionsJsonData = {
  metadata: {
    location: "37",
    abbreviation: "NC",
    location_name: "North Carolina",
    population: 10488084,
    dataset: "pathogen forecast hub",
    series_type: "projection",
    hubverse_keys: {
      models: ["model_name", "another_model_name", "..."],
      targets: ["target_name_1", "target_name_2"],
      horizons: ["0", "1", "..."],
      output_types: ["quantile", "pmf"],
    },
  },
  ground_truth: {
    dates: ["YYYY-MM-DD1", "YYYY-MM-DD2", "..."],
    target_name_1: [1, 2, 3],
    target_name_2: [4, 5, 6],
  },
  forecasts: {
    "YYYY-MM-DD1": {
      target_name_1: {
        model_name: {
          type: "quantile",
          predictions: {
            0: {
              date: "YYYY-MM-DD1",
              quantiles: [0.025, 0.25, 0.5, 0.75, 0.975],
              values: [42.0, 12.3, 45.6, 78.9, 100],
            },
            1: {
              date: "YYYY-MM-DD2",
              quantiles: [0.025, 0.25, 0.5, 0.75, 0.975],
              values: [42.0, 12.3, 45.6, 78.9, 100],
            },
          },
        },
      },
    },
  },
};

const timeseriesJsonData = {
  metadata: {
    location: "37",
    abbreviation: "NC",
    location_name: "North Carolina",
    population: 10488084,
    dataset: "NHSN",
    series_type: "timeseries",
  },
  series: {
    dates: ["YYYY-MM-DD1", "YYYY-MM-DD2", "..."],
    column_1: [1.0, 2.0, 3.0],
    column_2: [4.0, 5.0, 6.0],
    "...": [0.1, 0.2, 0.3],
  },
};

const CopyableCodeBlock = ({ code }) => {
  const clipboard = useClipboard({ timeout: 1000 });

  return (
    <Box pos="relative">
      <Tooltip
        label={clipboard.copied ? "Copied" : "Copy"}
        withArrow
        position="left"
      >
        <ActionIcon
          variant="subtle"
          color={clipboard.copied ? "teal" : "gray"}
          onClick={() => clipboard.copy(code)}
          style={{ position: "absolute", top: 8, right: 8, zIndex: 1 }}
          aria-label={
            clipboard.copied
              ? "Code copied to clipboard"
              : "Copy code to clipboard"
          }
        >
          {clipboard.copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
        </ActionIcon>
      </Tooltip>
      <Code block>{code}</Code>
    </Box>
  );
};

const rScriptCommand = `Rscript external_to_projections.R \\
  --output-path <path/to/output-directory> \\
  --pathogen flu \\
  --data-path <path/to/hubverse-data.csv> \\
  --target-data-path <path/to/target-data.csv> \\
  --locations-data-path <path/to/locations.csv>`;

const pythonScriptCommand = `python external_to_projections.py \\
  --output-path <path/to/output-directory> \\
  --pathogen flu \\
  --data-path <path/to/hubverse-data.csv> \\
  --target-data-path <path/to/target-data.csv> \\
  --locations-data-path <path/to/locations.csv>`;

const CollapsiblePaper = ({
  id,
  title,
  icon,
  children,
  defaultOpened = false,
}) => {
  const [opened, { toggle }] = useDisclosure(defaultOpened);

  return (
    <Paper id={id} shadow="sm" p="lg" radius="md" withBorder>
      <Stack gap="md">
        <Group
          justify="space-between"
          onClick={toggle}
          style={{ cursor: "pointer", userSelect: "none" }}
          align="flex-start"
        >
          <Group gap="sm">
            <ThemeIcon size={36} radius="md" variant="light" color="blue">
              {icon}
            </ThemeIcon>
            <div>
              <Title order={2} style={{ fontSize: "1.25rem" }}>
                {title}
              </Title>
            </div>
          </Group>
          <ActionIcon variant="subtle" color="gray">
            {opened ? (
              <IconChevronUp size={20} />
            ) : (
              <IconChevronDown size={20} />
            )}
          </ActionIcon>
        </Group>

        <Collapse in={opened}>
          <Box pt="sm">
            <Stack gap="md">{children}</Stack>
          </Box>
        </Collapse>
      </Stack>
    </Paper>
  );
};

const Documentation = () => {
  return (
    <>
      <Seo
        title="MyRespiLens Documentation | RespiLens"
        description="Learn how to prepare, convert, and upload respiratory disease forecast data for use with MyRespiLens visualizations."
        canonicalPath="/documentation"
      />
      <Container size="xl" py="xl" style={{ maxWidth: "1100px" }}>
        <Stack gap="lg">
          <CollapsiblePaper
            title="MyRespiLens documentation"
            icon={<IconClipboard size={20} />}
            defaultOpened={true}
          >
            <Text size="sm">
              This page provides instructions for how to convert your data for
              use within the <strong>MyRespiLens</strong> feature. Within this
              documentation, you will find information regarding the{" "}
              <code>.csv</code> starting point that is expected for you data,
              scripts you can use to convert your <code>.csv</code> data to
              RespiLens projections <code>.json</code>, the standardized JSON
              models for the RespiLens platform, and a glossary for frequently
              used terms.
            </Text>

            <Stack gap="md" mt="md">
              <Timeline active={2} bulletSize={36} lineWidth={2}>
                <Timeline.Item
                  bullet={<IconFileCheck size={18} />}
                  title="1. Verify Data Format"
                >
                  <Text c="dimmed" size="sm">
                    Ensure your data is in the expected Hubverse .csv format.
                  </Text>
                </Timeline.Item>

                <Timeline.Item
                  bullet={<IconTransformFilled size={18} />}
                  title="2. Convert Data"
                >
                  <Text c="dimmed" size="sm">
                    Run the flexible conversion script to transform your .csv
                    into the RespiLens .json format.
                  </Text>
                </Timeline.Item>

                <Timeline.Item
                  bullet={<IconCloudUpload size={18} />}
                  title="3. Upload to MyRespiLens"
                  lineVariant="dashed"
                >
                  <Text c="dimmed" size="sm">
                    Drag and drop your generated JSON file into MyRespiLens to
                    visualize your projections!
                  </Text>
                </Timeline.Item>
              </Timeline>
            </Stack>
          </CollapsiblePaper>

          <CollapsiblePaper
            id="can-your-data-be-converted"
            title="can your data be converted?"
            icon={<IconFlagQuestion size={20} />}
          >
            <Text>
              To use a RespiLens conversion script for your data, you must first
              confirm that your data conforms to Hubverse structure. Hubverse
              structure (delineated in detail{" "}
              <Anchor
                href="https://docs.hubverse.io/en/latest/user-guide/model-output.html"
                target="_blank"
                rel="noopener noreferrer"
              >
                here
              </Anchor>
              ) refers to a tabular structure with a variety of requirements
              relating to column names and value types. Additionally, the script
              needs corresponding location and ground truth data in order to
              successfully build a RespiLens projections JSON file. The
              conversion pipeline's shortlist of requirements is below. Please
              note that if you pulled your data directly from a Hubverse hub{" "}
              <code>.csv</code> file, it likley already conforms to conversion
              requirements and you can skip to the next step!
            </Text>
            <Text>
              <strong style={{ color: "#2563eb" }}>Data Requirements</strong>
            </Text>
            <List withPadding spacing="xs" size="m">
              <List.Item>
                Must be a <code>.csv</code> file
              </List.Item>
              <List.Item>Must be for flu, RSV, or COVID-19</List.Item>
              <List.Item>
                Must contain columns <Code>location</Code>,{" "}
                <Code>reference_date</Code>, <Code>target</Code>,{" "}
                <Code>model_id</Code>, <Code>horizon</Code>,{" "}
                <Code>output_type</Code>, <Code>output_type_id</Code>,{" "}
                <Code>value</Code>, and <Code>target_end_date</Code>
              </List.Item>
              <List.Item>
                Note that MyRespiLens does not handle peak or 'pmf'
                visualization: only quantiles.
              </List.Item>
            </List>
            <Text>
              <strong style={{ color: "#2563eb" }}>
                Target/Ground Truth Data Requirements
              </strong>
            </Text>
            <List withPadding spacing="xs" size="m">
              <List.Item>
                Must be a <code>.csv</code> or <code>.parquet</code> file. Can
                be found on a hub GitHub repository in the{" "}
                <Code>target-data</Code> directory (
                <Anchor
                  href="https://github.com/cdcepi/FluSight-forecast-hub/tree/main/target-data"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  example
                </Anchor>
                )
              </List.Item>
              <List.Item>Must be for flu, RSV, or COVID-19</List.Item>
              <List.Item>
                If for flu, must contain columns <Code>as_of</Code>,{" "}
                <Code>target_end_date</Code>, <Code>location</Code>,{" "}
                <Code>target</Code>, and <Code>observation</Code>
              </List.Item>
              <List.Item>
                If for RSV or COVID-19, must contain columns <Code>as_of</Code>,{" "}
                <Code>date</Code>, <Code>location</Code>, <Code>target</Code>,
                and <Code>observation</Code>
              </List.Item>
            </List>
            <Text>
              <strong style={{ color: "#2563eb" }}>
                Location Data Requirements
              </strong>
            </Text>
            <List withPadding spacing="xs" size="m">
              <List.Item>
                Must be a <code>.csv</code> file. Can be found on a hub GitHub
                repository in the <Code>auxiliary-data</Code> directory, and is
                likely named <Code>locations.csv</Code> or similar (
                <Anchor
                  href="https://github.com/cdcepi/FluSight-forecast-hub/tree/main/auxiliary-data"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  example
                </Anchor>
                )
              </List.Item>
              <List.Item>
                Must have full coverage (keyed on location FIPS code) of all
                locations referenced in the input data
              </List.Item>
              <List.Item>
                Must contain columns <Code>location</Code>,{" "}
                <Code>abbreviation</Code>, <Code>location_name</Code>, and{" "}
                <Code>population</Code>
              </List.Item>
            </List>
            <Text>
              Again, note that if your data came directly from a hub, it is
              likely already in the correct format for conversion. The
              MyRespiLens conversion pipeline takes Hubverse-style{" "}
              <code>.csv</code> data and converts it to the RespiLens{" "}
              <strong>projections</strong> JSON structure, which is used for the
              flu, COVID-19, and RSV forecast views on the site. This is
              distinct from the RespiLens timeseries JSON structure, which is
              used for the CDC Respiratory Data view. Both data structures are
              shown via interactive JSON structure at the bottom of the page.
            </Text>
          </CollapsiblePaper>

          <CollapsiblePaper
            id="convert-data"
            title="convert your data"
            icon={<IconTransformFilled size={20} />}
          >
            <Text>
              The RespiLens backend framework provides both an <Code>R</Code>{" "}
              and <Code>python</Code> pipeline for converting Hubverse{" "}
              <Code>.csv</Code> data to the RespiLens projections JSON format.
              Once converted, this data can be used in the MyRespiLens feature.
              Running data conversion scripts assumes a local clone of the{" "}
              <Anchor
                href="https://github.com/ACCIDDA/RespiLens"
                target="_blank"
                rel="noopener noreferrer"
              >
                {" "}
                RespiLens
              </Anchor>{" "}
              GitHub repository. All scripts are located in the{" "}
              <Code>scripts/</Code> directory of <Code>RespiLens</Code>, and
              must be run in this directory via the command line. If you are
              unfamiliar with cloning GitHub repositories, running commands via
              command line, or experience unexpected issues during your data
              conversion process, consider checking out these resources:
              <List withPadding spacing="m" size="m">
                <List.Item>
                  <Anchor
                    href="https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    How to clone a GitHub repository
                  </Anchor>
                </List.Item>
                <List.Item>
                  <Anchor
                    href="https://www.twilio.com/docs/usage/tutorials/a-beginners-guide-to-the-command-line"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Command line basics
                  </Anchor>
                </List.Item>
                <List.Item>
                  <Anchor
                    href="https://github.com/ACCIDDA/RespiLens/issues/new"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Report an issue/ask a question via GitHub issue{" "}
                  </Anchor>
                </List.Item>
              </List>
            </Text>
            <Text>
              <strong>
                Using <Code>external_to_projections.R</Code>
              </strong>{" "}
              (run in <Code>scripts/</Code> directory):
            </Text>
            <CopyableCodeBlock code={rScriptCommand} />

            <Text>
              <strong>
                Using <Code>external_to_projections.py</Code>
              </strong>{" "}
              (run in <Code>scripts/</Code> directory):
            </Text>
            <CopyableCodeBlock code={pythonScriptCommand} />

            <Stack gap={2}>
              <Text>Where:</Text>
              <Text>
                <Code>--output-path</Code> is the absolute path to the directory
                where you would like converted data to be saved.
              </Text>
              <Text>
                <Code>--data-path</Code> is the absolute path to{" "}
                <Code>.csv</Code> data to be converted. You can use your own
                data that complies withe the Hubverse format, or filter/use
                files in a hub's <Code>model-output</Code> directory.
              </Text>
              <Text>
                <Code>--target-data-path</Code> is the absolute path to
                corresponding ground truth data. This can be found as the{" "}
                <Code>time-series</Code> file in a hub's{" "}
                <Code>target-data</Code> directory.
              </Text>
              <Text>
                <Code>--locations-data-path</Code> is the absolute path to
                corresponding location metadata (<Code>locations.csv</Code>).
                This can be found in the RespiLens <Code>scripts/</Code>{" "}
                directory, or in a hub's <Code>auxiliary-data</Code> directory.
              </Text>
              <Text>
                and <Code>--pathogen</Code> is the pathogen the hub data
                describes (flu, covid, or rsv).
              </Text>
            </Stack>

            <Text>
              Once converted, individual projections JSON files (found at the{" "}
              <Code>output-path</Code> you specified) can be drag-n-dropped into
              MyRespiLens.
            </Text>
            <Text>
              <strong style={{ color: "#2563eb" }}>
                {" "}
                NOTE: MyRespiLens does not work for RespiLens timeseries data.
              </strong>
            </Text>
          </CollapsiblePaper>

          <CollapsiblePaper
            id="projections"
            title="data format for projections"
            icon={<IconBinaryTree size={20} />}
          >
            <Text size="sm" c="dimmed">
              Used for MyRespiLens and flu, COVID-19, and RSV views.
            </Text>
            {projectionsJsonData &&
              Object.keys(projectionsJsonData).length > 0 && (
                <JsonView
                  value={projectionsJsonData}
                  displayDataTypes={false}
                  displayObjectSize={false}
                  collapsed={1}
                  enableClipboard={false}
                />
              )}
          </CollapsiblePaper>

          <CollapsiblePaper
            id="timeseries"
            title="data format for timeseries"
            icon={<IconBinaryTree size={20} />}
          >
            <Text size="sm" c="dimmed">
              Used for the NHSN view (and all the ground_truth keys of
              projections data!).
            </Text>
            <Text>
              <strong style={{ color: "#2563eb" }}>
                {" "}
                NOTE: MyRespiLens does not work for RespiLens timeseries data.
              </strong>
            </Text>
            {timeseriesJsonData &&
              Object.keys(timeseriesJsonData).length > 0 && (
                <JsonView
                  value={timeseriesJsonData}
                  displayDataTypes={false}
                  displayObjectSize={false}
                  collapsed={1}
                  enableClipboard={false}
                />
              )}
          </CollapsiblePaper>

          <CollapsiblePaper
            id="glossary"
            title="glossary"
            icon={<IconBook size={20} />}
          >
            <GlossaryTable />
          </CollapsiblePaper>
        </Stack>
      </Container>
    </>
  );
};

export default Documentation;
