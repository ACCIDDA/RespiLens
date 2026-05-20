import { SimpleGrid, Stack, Title, Paper, Anchor } from "@mantine/core";
import PathogenOverviewGraph from "./PathogenOverviewGraph";
import NHSNOverviewGraph from "./NHSNOverviewGraph";
import NSSPOverviewGraph from "./NSSPOverviewGraph";
import Announcement from "./Announcement";
import { useView } from "../hooks/useView";

const normalizeFrontPageLocation = (location) => {
  if (!location || location === "US_All") {
    return "US";
  }

  return location.includes("_") ? location.split("_")[0] : location;
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
        href="/?view=nsspall"
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

const FrontPage = () => {
  const { selectedLocation } = useView();
  const overviewLocation = normalizeFrontPageLocation(selectedLocation);

  return (
    <Stack>
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
      <Announcement
        id={"hub-seasonal-warning"}
        startDate={"2026-06-01"}
        endDate={"2026-11-10"}
        announcementType={"alert"}
        text={
          "Forecast hubs are out of season. Forecasting will begin again in November."
        }
      />
      <Paper shadow="sm" p="lg" radius="md" withBorder>
        <Stack gap="md">
          <Title order={3}>Explore forecasts by pathogen</Title>
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
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
          </SimpleGrid>
        </Stack>
      </Paper>
      <Paper shadow="sm" p="lg" radius="md" withBorder>
        <Stack gap="md">
          <Title order={3}>Explore surveillance data by source</Title>
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
            <NHSNOverviewGraph location={overviewLocation} />
            <NSSPOverviewGraph />
          </SimpleGrid>
        </Stack>
      </Paper>
    </Stack>
  );
};

export default FrontPage;
