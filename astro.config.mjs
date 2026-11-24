import { defineConfig } from 'astro/config';
import vercel from '@astrojs/vercel';

// The site stays static. Only routes that set `export const prerender = false`
// (the /jobs review page and /api/jobs/*) run on demand as Vercel functions.
export default defineConfig({
  output: 'static',
  adapter: vercel(),
  site: 'https://reginalcampbell.com',
});
