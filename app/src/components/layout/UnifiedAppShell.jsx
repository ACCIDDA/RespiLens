import { useLocation, Link } from "react-router-dom";
import {
  AppShell,
  Center,
  Burger,
  Stack,
  Button,
  Divider,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import {
  IconChartLine,
  IconTarget,
  IconTrophy,
  IconDashboard,
  IconClipboard,
  IconChartScatter,
} from "@tabler/icons-react";
import { ENABLED_TOURNAMENTS } from "../../config";
import MainNavigation from "./MainNavigation";
import StateSelector from "../StateSelector";

const getShellConfig = (pathname) => {
  // For forecast view (main page), show navbar with StateSelector
  if (pathname === "/") {
    return {
      type: "forecast",
      header: { height: 60 },
      navbar: { width: 256, breakpoint: "sm" },
      padding: 0,
    };
  }

  // For all other pages, show navbar with just navigation
  return {
    type: "navigation",
    header: { height: 60 },
    navbar: { width: 256, breakpoint: "sm" },
    padding: 0,
  };
};

const UnifiedAppShell = ({ children, forecastProps = {} }) => {
  const location = useLocation();
  const config = getShellConfig(location.pathname);
  const [mobileOpened, { toggle: toggleMobile }] = useDisclosure();
  const [desktopOpened] = useDisclosure(true);

  const renderHeaderNavigation = () => {
    return (
      <Center
        h="100%"
        px="md"
        style={{ width: "100%", justifyContent: "space-between" }}
      >
        {/* Hamburger - always on mobile, left side */}
        <Burger
          opened={mobileOpened}
          onClick={toggleMobile}
          hiddenFrom="sm"
          size="sm"
          style={{ flexShrink: 0 }}
        />
        <MainNavigation />
      </Center>
    );
  };

  const navigationItems = [
    {
      href: "/",
      label: "Forecasts",
      icon: IconChartLine,
      active: location.pathname === "/",
    },
    {
      href: "/forecastle",
      label: "Forecastle",
      icon: IconTarget,
      active: location.pathname.startsWith("/forecastle"),
    },
    ...ENABLED_TOURNAMENTS.map((tournament) => ({
      href: tournament.path,
      label: tournament.navLabel,
      icon: IconTrophy,
      active: location.pathname.startsWith(tournament.path),
    })),
    {
      href: "/myrespilens",
      label: "MyRespiLens",
      icon: IconDashboard,
      active: location.pathname.startsWith("/myrespilens"),
    },
    {
      href: "/documentation",
      label: "Documentation",
      icon: IconClipboard,
      active: location.pathname.startsWith("/documentation"),
    },
    {
      href: "/myplots",
      label: "My Plots",
      icon: IconChartScatter,
      active: location.pathname.startsWith("/myplots"),
    },
  ];

  const renderNavbar = () => {
    if (config.type === "forecast") {
      return (
        <Stack gap="md">
          {/* Navigation Links - only visible on mobile */}
          <Stack gap="xs" hiddenFrom="sm">
            {navigationItems.map((item) => (
              <Button
                key={item.href}
                component={Link}
                to={item.href}
                variant={item.active ? "filled" : "subtle"}
                leftSection={<item.icon size={18} />}
                size="sm"
                fullWidth
                justify="start"
                onClick={toggleMobile}
              >
                {item.label}
              </Button>
            ))}
            <Divider />
          </Stack>

          {/* State Selector */}
          <StateSelector
            onStateSelect={forecastProps.onStateSelect}
            currentLocation={forecastProps.currentLocation}
            appShellMode={true}
          />
        </Stack>
      );
    }

    if (config.type === "navigation") {
      return (
        <Stack gap="xs">
          {/* Navigation Links for other pages */}
          {navigationItems.map((item) => (
            <Button
              key={item.href}
              component={Link}
              to={item.href}
              variant={item.active ? "filled" : "subtle"}
              leftSection={<item.icon size={18} />}
              size="sm"
              fullWidth
              justify="start"
              onClick={toggleMobile}
            >
              {item.label}
            </Button>
          ))}
        </Stack>
      );
    }

    return null;
  };

  return (
    <AppShell
      header={config.header}
      navbar={{
        ...config.navbar,
        ...(config.navbar && {
          collapsed: {
            mobile: !mobileOpened,
            // Only show sidebar on desktop for forecast page
            desktop: config.type === "forecast" ? !desktopOpened : true,
          },
        }),
      }}
      padding={config.padding}
    >
      <AppShell.Header p="md">{renderHeaderNavigation()}</AppShell.Header>

      {config.navbar && (
        <AppShell.Navbar
          p="md"
          style={{ overflow: "auto", display: "flex", flexDirection: "column" }}
        >
          {renderNavbar()}
        </AppShell.Navbar>
      )}

      <AppShell.Main>{children}</AppShell.Main>
    </AppShell>
  );
};

export default UnifiedAppShell;
