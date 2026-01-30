import { test, expect } from '@playwright/test';

/**
 * Browser test for MiniMaxAI/MiniMax-M2.
 * Run with: npm run build && npx playwright test
 * Debug with: npx playwright test --debug
 * Or: npx playwright test --headed
 *
 * Use DevTools Network tab (or trace) to inspect whether fetches run in parallel.
 */
test('MiniMaxAI/MiniMax-M2 runs in browser and completes', async ({ page }) => {
  const requests: { url: string; start: number }[] = [];
  page.on('request', (req) => {
    const url = req.url();
    if (url.includes('huggingface.co') && (url.includes('resolve') || url.includes('api/models'))) {
      requests.push({ url: url.split('?')[0], start: Date.now() });
    }
  });

  await page.goto('/tests/browser/index.html');

  await expect(async () => {
    const result = await page.evaluate(() => window.__hfMemResult__);
    if (result?.status === 'error') throw new Error(result.message);
    if (result?.status !== 'ok') throw new Error('Still pending');
    return result;
  }).toPass({ timeout: 120_000 });

  const result = await page.evaluate(() => window.__hfMemResult__);
  expect(result.status).toBe('ok');
  expect(result.data.metadata.paramCount).toBeGreaterThan(0);

  // Log request order for debugging parallel vs sequential
  const ordered = requests.map((r, i) => `${i + 1}. ${r.url.slice(-60)}`);
  console.log('Request order:\n' + ordered.join('\n'));
});
