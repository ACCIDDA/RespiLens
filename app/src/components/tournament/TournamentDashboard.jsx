import { useState, useEffect } from "react";
import { Alert, Container, Tabs } from "@mantine/core";
import { IconTrophy, IconChartLine } from "@tabler/icons-react";
import {
  getStoredParticipantId,
  getStoredParticipantName,
} from "../../utils/tournamentAPI";
import { TOURNAMENT_CONFIG } from "../../config";
import TournamentRegistration from "./TournamentRegistration";
import TournamentGame from "./TournamentGame";
import TournamentLeaderboard from "./TournamentLeaderboard";

const TournamentDashboard = ({ tournamentConfig = TOURNAMENT_CONFIG }) => {
  const [participantId, setParticipantId] = useState(null);
  const [participantName, setParticipantName] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("challenges");

  // Load participant data on mount
  useEffect(() => {
    const storedId = getStoredParticipantId(tournamentConfig);
    const storedName = getStoredParticipantName(tournamentConfig);

    setParticipantId(storedId || null);
    setParticipantName(storedName || null);

    setLoading(false);
  }, [tournamentConfig]);

  // Handle successful registration
  const handleRegistration = (id, name) => {
    setParticipantId(id);
    setParticipantName(name);
  };

  // Navigate to leaderboard
  const goToLeaderboard = () => {
    setActiveTab("leaderboard");
  };

  if (tournamentConfig.numChallenges === 0) {
    return (
      <Container size="md" py="xl">
        <Alert color="yellow" title="No tournament challenges enabled">
          Enable at least one challenge in the tournament registry.
        </Alert>
      </Container>
    );
  }

  // Show registration if not registered
  if (!participantId && !loading) {
    return (
      <TournamentRegistration
        tournamentConfig={tournamentConfig}
        onSuccess={handleRegistration}
      />
    );
  }

  if (loading) {
    return null;
  }

  return (
    <Container size="xl" py="md">
      <Tabs value={activeTab} onChange={setActiveTab}>
        <Tabs.List>
          <Tabs.Tab value="challenges" leftSection={<IconTrophy size={16} />}>
            Challenges
          </Tabs.Tab>
          <Tabs.Tab
            value="leaderboard"
            leftSection={<IconChartLine size={16} />}
          >
            Leaderboard
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="challenges" pt="md">
          <TournamentGame
            tournamentConfig={tournamentConfig}
            participantId={participantId}
            participantName={participantName}
            onAllCompleted={goToLeaderboard}
          />
        </Tabs.Panel>

        <Tabs.Panel value="leaderboard" pt="md">
          <TournamentLeaderboard
            tournamentConfig={tournamentConfig}
            participantId={participantId}
          />
        </Tabs.Panel>
      </Tabs>
    </Container>
  );
};

export default TournamentDashboard;
