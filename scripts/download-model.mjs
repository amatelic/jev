/**
 * Pre-download models into the local cache without loading them for inference.
 * Retries with backoff — 429 rate limits need minutes, transient fetch
 * failures need seconds. Partial download progress is kept across retries
 * (transformers.js chunk cache).
 *
 * Usage: node scripts/download-model.mjs <model-alias-or-id> [model2 ...]
 */
async function main() {
  const models = process.argv.slice(2);
  if (models.length === 0) {
    console.error("usage: node scripts/download-model.mjs <model> [model2 ...]");
    process.exit(1);
  }

  const { OpenJev } = await import("open-jev");

  for (const model of models) {
    const maxAttempts = 12;
    let ok = false;
    for (let attempt = 1; attempt <= maxAttempts && !ok; attempt++) {
      const t0 = Date.now();
      console.log(`[download] ${model}: attempt ${attempt}/${maxAttempts}...`);
      try {
        const rt = await OpenJev.load({
          model,
          device: "cpu",
          dtype: "q4",
          onProgress: (p) => {
            if (p.status === "progress" && p.total > 0) {
              const pct = ((p.loaded / p.total) * 100).toFixed(1);
              process.stdout.write(`\r[download] ${model}: ${p.file ?? ""} ${pct}%   `);
            } else if (p.status && p.status !== "progress") {
              process.stdout.write(`\n[download] ${model}: ${p.status} ${p.file ?? ""}\n`);
            }
          },
        });
        await rt.dispose?.();
        console.log(`\n[download] ${model}: done in ${Math.round((Date.now() - t0) / 1000)}s`);
        ok = true;
      } catch (e) {
        const msg = String(e.message ?? e);
        const rateLimited = /429|Too Many Requests/i.test(msg);
        console.error(`\n[download] ${model}: attempt ${attempt} failed — ${msg.slice(0, 160)}`);
        if (attempt < maxAttempts) {
          const wait = rateLimited ? 90_000 : 15_000;
          console.log(`[download] waiting ${wait / 1000}s before retry (partial progress is kept)...`);
          await new Promise((r) => setTimeout(r, wait));
        } else {
          console.error(`[download] ${model}: FAILED after ${maxAttempts} attempts`);
          process.exitCode = 1;
        }
      }
    }
  }
}

main();
