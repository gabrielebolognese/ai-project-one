/** `npm start`: serve the lab + API on http://127.0.0.1:5177 (override with --port / --host / --model). */
import { parseArgs } from "node:util";
import { startServer } from "./http.js";

const { values: a } = parseArgs({
  options: {
    port: { type: "string", default: process.env.PORT ?? "5177" },
    host: { type: "string", default: "127.0.0.1" },
    model: { type: "string" },
    web: { type: "string" },
    random: { type: "boolean", default: false },
  },
});

const running = await startServer({
  port: Number(a.port), host: a.host, modelDir: a.model, webDir: a.web, forceRandom: a.random,
});
console.log(`[server] open ${running.url} in a browser (Ctrl+C to stop)`);
const stop = () => { void running.close().then(() => process.exit(0)); };
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
