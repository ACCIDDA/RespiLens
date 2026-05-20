import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/",
  plugins: [
    react(),
    {
      name: "tournament-api-dev-proxy",
      configureServer(server) {
        server.middlewares.use("/__tournament_api__", async (req, res) => {
          if (req.method !== "POST") {
            res.statusCode = 405;
            res.setHeader("Content-Type", "application/json");
            res.end(
              JSON.stringify({ success: false, error: "Method not allowed" }),
            );
            return;
          }

          const requestUrl = new URL(req.url || "", "http://localhost");
          const target = requestUrl.searchParams.get("target");

          if (!target) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json");
            res.end(
              JSON.stringify({ success: false, error: "Missing target URL" }),
            );
            return;
          }

          const chunks = [];
          for await (const chunk of req) {
            chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
          }

          try {
            const upstreamResponse = await fetch(target, {
              method: "POST",
              headers: {
                "Content-Type": req.headers["content-type"] || "text/plain",
              },
              body: Buffer.concat(chunks).toString("utf8"),
            });

            const responseText = await upstreamResponse.text();
            res.statusCode = upstreamResponse.status;
            res.setHeader(
              "Content-Type",
              upstreamResponse.headers.get("content-type") ||
                "application/json",
            );
            res.end(responseText);
          } catch (error) {
            res.statusCode = 502;
            res.setHeader("Content-Type", "application/json");
            res.end(
              JSON.stringify({
                success: false,
                error:
                  error instanceof Error
                    ? error.message
                    : "Failed to reach tournament API",
              }),
            );
          }
        });
      },
    },
  ],
  server: {
    watch: { usePolling: true },
    publicDir: "public",
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
  },
});
