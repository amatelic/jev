import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    testTimeout: 120_000,
    hookTimeout: 120_000,
    // Exclude bench tests from normal runs (run separately: npx vitest run src/bench.test.ts)
    exclude: ["node_modules/**", "dist/**", "src/bench.test.ts"],
  },
});
