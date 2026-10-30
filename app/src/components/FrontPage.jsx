import { SimpleGrid, Stack, Group, Text, Anchor } from "@mantine/core";
import { IconChartLine, IconActivityHeartbeat } from "@tabler/icons-react";
import PathogenOverviewGraph from "./PathogenOverviewGraph";
import NHSNOverviewGraph from "./NHSNOverviewGraph";
import NSSPOverviewGraph from "./NSSPOverviewGraph";
import Announcement from "./Announcement";
import { useView } from "../hooks/useView";
import { isFrontPageLocation } from "../utils/forecastRoutes";

// The tiles take a plain state code. A location carried over from another
// view can be scoped (NSSP's "CO_Denver County"); one no hub covers (an old
// ?location=ZZ link) falls back to the country rather than asking every
// tile for a state that does not exist.
const normalizeFrontPageLocation = (location) => {
  if (!location || location === "US_All") {
    return "US";
  }

  const code = location.includes("_") ? location.split("_")[0] : location;
  return isFrontPageLocation(code) ? code : "US";
};

const MyPlotsLink = () => {
  return (
    <span>
      Check out the new{" "}
      <Anchor
        href="/myplots"
        fw={700}
        c="blue.7"
        style={{ fontSize: "inherit", verticalAlign: "baseline" }}
      >
        My Plots
      </Anchor>{" "}
      feature, where you can assemble your own dashboard of saved plots.
    </span>
  );
};

const NsspViewLink = () => {
  return (
    <span>
      Check out our new{" "}
      <Anchor
        href="/surveillance/nssp"
        fw={700}
        c="blue.7"
        style={{ fontSize: "inherit", verticalAlign: "baseline" }}
      >
        NSSP view
      </Anchor>{" "}
      to visualize the CDC's National Syndromic Surveillance Program
      county-level data stream.
    </span>
  );
};

// Sections are separated by headings and whitespace rather than by a box
// drawn around each group. Headings match the sidebar's groups (same label
// and icon) so forecasts and surveillance data read as distinct.
const Section = ({ title, icon: Icon, children }) => (
  <Stack gap="xs">
    <Group gap={8} wrap="nowrap">
      <Icon size={16} stroke={2} className="respilens-eyebrow" />
      <Text size="sm" fw={700} className="respilens-eyebrow">
        {title}
      </Text>
    </Group>
    <SimpleGrid
      cols={{ base: 1, sm: 2, lg: 3 }}
      spacing="xl"
      verticalSpacing="xl"
    >
      {children}
    </SimpleGrid>
  </Stack>
);

// Green banner for the start of the 2026-27 FluSight season (shown today and
// tomorrow); also rendered on the FluSight views by DataVisualizationContainer
export const FluSightStartAnnouncement = () => (
  <Announcement
    id="flusight-starts-2026"
    startDate="2026-10-07T00:00:00"
    endDate="2026-10-08T23:59:59"
    announcementType="success"
    text="FluSight starts today! Influenza forecasts for the 2026–27 season are back."
  />
);

// Green banner for MetroCast's early 2026-27 start; also rendered on the
// MetroCast view by DataVisualizationContainer
export const MetroCastStartAnnouncement = () => (
  <Announcement
    id="metrocast-starts-2026"
    startDate="2026-10-23T00:00:00"
    endDate="2026-11-08T23:59:59"
    announcementType="success"
    text={
      <span>
        MetroCast is up early! Local influenza forecasts for the 2026–27 season
        are live, now with long-term forecasts through the end of the season.{" "}
        <Anchor
          href="/forecasts/flu-metrocast"
          fw={700}
          c="blue.7"
          style={{ fontSize: "inherit", verticalAlign: "baseline" }}
        >
          Check the forecasts
        </Anchor>
      </span>
    }
  />
);

// Site notices, rendered full width above the page title (outside the
// content container) by DataVisualizationContainer
export const FrontPageAnnouncements = () => (
  <>
    <FluSightStartAnnouncement />
    <MetroCastStartAnnouncement />
    <Announcement
      id="new-nssp-all-view"
      startDate="2026-05-20"
      endDate="2026-07-15"
      announcementType="update"
      text={<NsspViewLink />}
    />
    <Announcement
      id="new-myplots-feature"
      startDate="2026-04-06"
      endDate="2026-06-30"
      announcementType={"update"}
      text={<MyPlotsLink />}
    />
  </>
);

const FrontPage = () => {
  const { selectedLocation } = useView();
  const overviewLocation = normalizeFrontPageLocation(selectedLocation);

  return (
    <Stack gap="lg">
      <Section title="Forecasts" icon={IconChartLine}>
        <PathogenOverviewGraph
          viewType="covid_forecasts"
          title="COVID-19"
          location={overviewLocation}
        />
        <PathogenOverviewGraph
          viewType="flu_forecasts"
          title="Flu"
          location={overviewLocation}
        />
        <PathogenOverviewGraph
          viewType="rsv_forecasts"
          title="RSV"
          location={overviewLocation}
        />
      </Section>

      <Section title="Surveillance data" icon={IconActivityHeartbeat}>
        <NHSNOverviewGraph location={overviewLocation} />
        <NSSPOverviewGraph />
      </Section>
    </Stack>
  );
};

export default FrontPage;
