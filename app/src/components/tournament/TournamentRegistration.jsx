import { useState } from "react";
import {
  Container,
  Title,
  Text,
  TextInput,
  Button,
  Card,
  Stack,
  Alert,
} from "@mantine/core";
import { IconUserPlus, IconAlertCircle } from "@tabler/icons-react";
import { registerParticipant } from "../../utils/tournamentAPI";
import { TOURNAMENT_CONFIG } from "../../config";

const TournamentRegistration = ({
  tournamentConfig = TOURNAMENT_CONFIG,
  onSuccess,
}) => {
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError("Please enter a recognizable name");
      return;
    }

    setLoading(true);

    try {
      const data = await registerParticipant(name, tournamentConfig);
      onSuccess(data.participantId, name);
    } catch (err) {
      setError(err.message || "Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container
      size="sm"
      py="xl"
      style={{ minHeight: "70vh", display: "flex", alignItems: "center" }}
    >
      <Card shadow="lg" p="xl" radius="md" withBorder style={{ width: "100%" }}>
        <Stack spacing="lg">
          <div style={{ textAlign: "center" }}>
            <Title order={1} mb="xs">
              Join the Tournament
            </Title>
            <Text color="dimmed" size="lg">
              {tournamentConfig.name}
            </Text>
            <Text color="dimmed" size="sm" mt="xs">
              {tournamentConfig.description}
            </Text>
          </div>

          {error && (
            <Alert
              icon={<IconAlertCircle size={16} />}
              title="Error"
              color="red"
            >
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <Stack spacing="md">
              <TextInput
                label="Your Name"
                placeholder="Enter a recognizable name (e.g., your name :) )"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                size="lg"
                description="Choose a name that others will recognize you by on the leaderboard"
              />

              <Button
                type="submit"
                size="lg"
                leftSection={<IconUserPlus size={20} />}
                loading={loading}
                fullWidth
                mt="md"
              >
                Register & Start
              </Button>
            </Stack>
          </form>

          <div
            style={{
              marginTop: 16,
              padding: 16,
              backgroundColor: "#f8f9fa",
              borderRadius: 8,
            }}
          >
            <Text size="sm" color="dimmed" style={{ textAlign: "center" }}>
              <strong>How it works:</strong>
            </Text>
            <Text size="sm" color="dimmed" mt="xs">
              • Complete {tournamentConfig.numChallenges} forecasting challenge
              {tournamentConfig.numChallenges === 1 ? "" : "s"}
              <br />
              • Predict hospitalization counts for different diseases and
              locations
              <br />
              • Get scored using Weighted Interval Score (lower is better)
              <br />• Climb the leaderboard and compete with others!
            </Text>
          </div>
        </Stack>
      </Card>
    </Container>
  );
};

export default TournamentRegistration;
