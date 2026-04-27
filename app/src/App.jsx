import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import { useEffect } from "react";
import { HelmetProvider } from "react-helmet-async";
import { ViewProvider } from "./contexts/ViewContext";
import { useView } from "./hooks/useView";
import DataVisualizationContainer from "./components/DataVisualizationContainer";
import MyPlots from "./components/myplots/MyPlots";
import NarrativeBrowser from "./components/narratives/NarrativeBrowser";
import SlideNarrativeViewer from "./components/narratives/SlideNarrativeViewer";
import ForecastleGame from "./components/forecastle/ForecastleGame";
import MyRespiLensDashboard from "./components/myrespi/MyRespiLensDashboard";
import TournamentDashboard from "./components/tournament/TournamentDashboard";
import UnifiedAppShell from "./components/layout/UnifiedAppShell";
import Documentation from "./components/Documentation";
import ReportingDelayPage from "./components/reporting/ReportingDelayPage";
import ToolsPage from "./components/tools/ToolsPage";
import { Center, Text } from "@mantine/core";
import { ENABLED_TOURNAMENTS } from "./config";

const ForecastApp = () => {
  // This component uses the view context, so it must be inside the provider.
  const { selectedLocation } = useView();

  if (!selectedLocation) {
    return (
      <Center h="100%">
        <Text c="dimmed" size="lg">
          Select a state to view forecasts
        </Text>
      </Center>
    );
  }
  return <DataVisualizationContainer />;
};

// We create this new component to hold our main layout.
// It can safely use hooks because it will be inside the Router and Provider.
const AppLayout = () => {
  const navigate = useNavigate(); // Safely used inside <Router>

  // <ShutdownBanner />  was below UnifiedAppShell, should we need it again
  return (
    <UnifiedAppShell>
      <Routes>
        <Route path="/" element={<ForecastApp />} />
        <Route
          path="/narratives"
          element={
            <NarrativeBrowser
              onNarrativeSelect={(id) => navigate(`/narratives/${id}`)}
            />
          }
        />
        <Route path="/narratives/:id" element={<SlideNarrativeViewer />} />
        <Route path="/forecastle" element={<ForecastleGame />} />
        {ENABLED_TOURNAMENTS.map((tournament) => (
          <Route
            key={tournament.id}
            path={tournament.path}
            element={<TournamentDashboard tournamentConfig={tournament} />}
          />
        ))}
        <Route path="/myplots" element={<MyPlots />} />
        <Route path="/myrespilens" element={<MyRespiLensDashboard />} />
        <Route path="/toolbox" element={<ToolsPage />} />
        <Route path="/reporting-triangle" element={<ReportingDelayPage />} />
        <Route path="/documentation" element={<Documentation />} />
      </Routes>
    </UnifiedAppShell>
  );
};

const App = () => {
  const AnalyticsTracker = () => {
    const location = useLocation();

    useEffect(() => {
      if (typeof window !== "undefined" && window.gtag) {
        window.gtag("config", import.meta.env.VITE_GA_MEASUREMENT_ID, {
          page_path: location.pathname + location.search,
        });
      }
    }, [location]);

    return null;
  };

  return (
    <HelmetProvider>
      <Router>
        <AnalyticsTracker />
        <ViewProvider>
          <AppLayout />
        </ViewProvider>
      </Router>
    </HelmetProvider>
  );
};

export default App;
