import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useLocation,
  useParams,
} from "react-router-dom";
import { HelmetProvider } from "react-helmet-async";
import { ViewProvider } from "./contexts/ViewContext";
import { useView } from "./hooks/useView";
import DataVisualizationContainer from "./components/DataVisualizationContainer";
import MyPlots from "./components/myplots/MyPlots";
import NarrativeBrowser from "./components/narratives/NarrativeBrowser";
import SlideNarrativeViewer from "./components/narratives/SlideNarrativeViewer";
import ForecastleGame from "./components/forecastle/ForecastleGame";
import ForecastCheckerDashboard from "./components/forecastchecker/ForecastCheckerDashboard";
import Documentation from "./components/forecastchecker/Documentation";
import TournamentDashboard from "./components/tournament/TournamentDashboard";
import AnalyticsTracker from "./components/AnalyticsTracker";
import UnifiedAppShell from "./components/layout/UnifiedAppShell";
import ReportingDelayPage from "./components/reporting/ReportingDelayPage";
import ToolsPage from "./components/tools/ToolsPage";
import AboutPage from "./components/AboutPage";
import NotFoundPage from "./components/NotFoundPage";
import MyPrivateHub from "./components/my-private-hub/MyPrivateHub";
import StateSummaryPage from "./components/summary/StateSummaryPage";
import { Center, Text } from "@mantine/core";
import { ENABLED_TOURNAMENTS } from "./config";
import {
  getForecastRouteError,
  isFrontPageLocation,
} from "./utils/forecastRoutes";

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

// /forecasts/... and /surveillance/...: a mistyped hub, view or state used
// to fall through to the front page under the address the reader typed.
const ForecastRoute = () => {
  const { pathname } = useLocation();
  const routeError = getForecastRouteError(pathname);

  return routeError ? <NotFoundPage {...routeError} /> : <ForecastApp />;
};

// /AR, /US: the front page for one location. A code no hub carries is not
// an address, so it 404s instead of quietly showing the US front page.
const FrontPageLocationRoute = () => {
  const { location } = useParams();

  return isFrontPageLocation(location) ? <ForecastApp /> : <NotFoundPage />;
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
        <Route path="/forecasts" element={<Navigate to="/" replace />} />
        <Route path="/forecasts/*" element={<ForecastRoute />} />
        <Route
          path="/surveillance"
          element={<Navigate to="/surveillance/nhsn" replace />}
        />
        <Route path="/surveillance/*" element={<ForecastRoute />} />
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
        <Route path="/about" element={<AboutPage />} />
        <Route path="/toolbox" element={<ToolsPage />} />
        <Route
          path="/toolbox/forecast-checker"
          element={<ForecastCheckerDashboard />}
        />
        <Route
          path="/toolbox/forecast-checker/documentation"
          element={<Documentation />}
        />
        <Route
          path="/toolbox/forecast-checker/:hub"
          element={<ForecastCheckerDashboard />}
        />
        <Route
          path="/toolbox/reporting-triangle"
          element={<ReportingDelayPage />}
        />
        <Route path="/toolbox/my-private-hub" element={<MyPrivateHub />} />
        {/* State summary (proof of concept: Utah only): /summary/UT/2026-10-24 */}
        <Route path="/summary/:location" element={<StateSummaryPage />} />
        <Route path="/summary/:location/:date" element={<StateSummaryPage />} />
        {/* Front page for a state: /AR (static routes rank first) */}
        <Route path="/:location" element={<FrontPageLocationRoute />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </UnifiedAppShell>
  );
};

const App = () => {
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
