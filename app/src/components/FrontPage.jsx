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

const MetroCastLink = () => {
  const { setViewType } = useView();

  const handleClick = (e) => {
    e.preventDefault();
    setViewType("metrocast_forecasts");
  };

  return (
    <span>
      RespiLens now displays{" "}
      <Anchor
        component="button"
        onClick={handleClick}
        fw={700}
        c="blue.7"
        style={{ fontSize: "inherit", verticalAlign: "baseline" }}
      >
        flu MetroCast forecasts;
      </Anchor>{" "}
      metro area-level flu forecasts.
    </span>
  );
};

const FrontPage = () => {
  const { selectedLocation } = useView();
  const overviewLocation = normalizeFrontPageLocation(selectedLocation);

  return (
    <Stack>
      <Announcement
        id="new-metrocast-2026"
        startDate="2026-02-01"
        endDate="2026-04-15"
        announcementType={"update"}
        text={<MetroCastLink />}
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
        startDate={"2026-05-31"}
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
