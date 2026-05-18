import { useLocation, Link } from "react-router-dom";
import { Group, Button, Image, Title, Anchor } from "@mantine/core";
import {
  IconChartLine,
  IconTarget,
  IconDashboard,
  IconChartScatter,
} from "@tabler/icons-react";
import InfoOverlay from "../InfoOverlay";
import { useView } from "../../hooks/useView";
import { APP_CONFIG } from "../../config/app";

const MainNavigation = () => {
  const location = useLocation();
  const { setViewAndLocation } = useView();

  const isActive = (path) => location.pathname.startsWith(path);
  const handleForecastsNavigation = () => {
    setViewAndLocation("frontpage", APP_CONFIG.defaultLocation);
  };

  const navigationItems = [
    {
      href: "/",
      label: "Forecasts",
      icon: IconChartLine,
      active: location.pathname === "/",
    },
    // { href: '/narratives', label: 'Narratives', icon: IconBook, active: isActive('/narratives') },   disable narratives for now
    {
      href: "/forecastle",
      label: "Forecastle",
      icon: IconTarget,
      active: isActive("/forecastle"),
    },
    {
      href: "/myrespilens",
      label: "MyRespiLens",
      icon: IconDashboard,
      active: isActive("/myrespilens"),
    },
    {
      href: "/myplots",
      label: "My Plots (α)",
      icon: IconChartScatter,
      active: isActive("/myplots"),
    },
    // { href: '/documentation', label: 'Documentation', icon: IconClipboard, active: isActive('/documentation')}
  ];

  return (
    <Group
      justify="space-between"
      align="center"
      w="100%"
      wrap="nowrap"
      gap="sm"
    >
      {/* Logo */}
      <Anchor
        component={Link}
        to="/"
        underline="never"
        c="inherit"
        onClick={handleForecastsNavigation}
      >
        <Group gap="sm" align="center" style={{ flexShrink: 0 }}>
          <Image
            src="respilens-logo.svg"
            alt="RespiLens Logo"
            h={28}
            w="auto"
            fit="contain"
          />
          <Title order={3} c="blue" visibleFrom="sm">
            RespiLens
            <sup
              style={{
                color: "var(--mantine-color-red-6)",
                fontSize: "0.75rem",
              }}
            ></sup>
          </Title>
        </Group>
      </Anchor>

      {/* Desktop Navigation - Full Buttons */}
      <Group gap="xs" visibleFrom="sm">
        {navigationItems.map((item) => (
          <Button
            key={item.href}
            component={Link}
            to={item.href}
            variant={item.active ? "filled" : "subtle"}
            leftSection={<item.icon size={16} />}
            size="sm"
            onClick={item.href === "/" ? handleForecastsNavigation : undefined}
          >
            {item.label}
          </Button>
        ))}
      </Group>

      {/* Actions */}
      <Group gap="xs" style={{ flexShrink: 0 }}>
        <InfoOverlay />
      </Group>
    </Group>
  );
};

export default MainNavigation;
