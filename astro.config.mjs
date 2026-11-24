import { defineConfig } from 'astro/config';
import vercel from '@astrojs/vercel';

// The site stays static. Only routes that set `export const prerender = false`
// (the /jobs review page and /api/jobs/*) run on demand as Vercel functions.
export default defineConfig({
  output: 'static',
  adapter: vercel(),
  site: 'https://reginalcampbell.com',
  // Astro's built-in check compares Origin with the URL the function sees, which on
  // Vercel can carry a different host than the browser used. /api/jobs/* does its own
  // origin check against the real request hosts (src/lib/jobs/session.ts).
  security: { checkOrigin: false },
});
