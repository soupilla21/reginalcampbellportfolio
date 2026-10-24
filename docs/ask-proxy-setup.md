# Serving the portfolio assistant at reginalcampbell.com/ask

The assistant stays on Netlify. Vercel proxies `/ask/` to it, so visitors and
search engines see one domain. Nothing about the assistant's backend moves.

This branch ships in two phases so nothing breaks in between.

## Phase A: merge this branch as is

Safe to deploy immediately. It adds the `/evidence` page, the shared nav, and
the homepage entry points. Assistant links still go to
`briefing.reginalcampbell.com` (see `src/config/site.ts`), so no proxy is
needed yet.

## Phase B: switch the assistant to /ask/

### 1. Find the assistant's Netlify hostname

Netlify dashboard → the briefing site → Site configuration → Site details.
It looks like `something.netlify.app`. Use that, not the custom domain.

### 2. Check how the assistant loads its files

Open the assistant's built `index.html` and look at its `<script>`, `<link>`
and `fetch()` paths.

- **Relative paths** (`assets/app.js`, `./api/ask`): works behind the proxy as is.
- **Absolute paths** (`/assets/app.js`, `/api/ask`): these would request
  `reginalcampbell.com/assets/...`, which is your main site, and fail. Set
  the assistant's base path to `/ask/` (Vite: `base: "/ask/"`; Create React
  App: `"homepage": "/ask"`; plain HTML: make the paths relative) and
  redeploy it on Netlify before continuing.
- **Netlify Functions** called at `/.netlify/functions/...`: keep the second
  rewrite below. If the assistant doesn't use them, delete that rule.

### 3. Add the rewrite to `vercel.json`

Replace `YOUR-SITE.netlify.app` with the hostname from step 1:

```json
{
  "framework": "astro",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "redirects": [
    { "source": "/ask", "destination": "/ask/", "permanent": true }
  ],
  "rewrites": [
    { "source": "/ask/:path*", "destination": "https://YOUR-SITE.netlify.app/:path*" },
    { "source": "/.netlify/functions/:path*", "destination": "https://YOUR-SITE.netlify.app/.netlify/functions/:path*" }
  ]
}
```

The `/ask` → `/ask/` redirect matters: without the trailing slash, a relative
path like `assets/app.js` resolves to `/assets/app.js` on the main site.

Deploy, then open `https://reginalcampbell.com/ask/` and confirm the assistant
loads, answers a question, and shows no 404s in the browser console.

### 4. Point the site's links at /ask/

In `src/config/site.ts`, change:

```ts
export const ASK_URL = "/ask/";
```

### 5. Retire the subdomain with a 301

In the **assistant's** repo, add to its `_redirects` file (or create one in
its publish directory):

```
https://briefing.reginalcampbell.com/*  https://reginalcampbell.com/ask/:splat  301!
```

This only matches requests that arrive on the subdomain. Proxied requests
from Vercel arrive on the `netlify.app` hostname, so they are not redirected
and there is no loop.

Also set the assistant's `<link rel="canonical">` to
`https://reginalcampbell.com/ask/`.

Keep the `briefing` DNS record in Namecheap so old links keep redirecting.

### 6. Measure

GA4 events to watch for the 60 to 90 day keep-or-retire decision:

| Event | Fired when | Key parameters |
| --- | --- | --- |
| `evidence_hub_click` | A homepage link to /evidence is clicked | `source` |
| `evidence_link_click` | A visitor opens a PDF, artifact, section or code from /evidence | `evidence_item`, `link_kind` |
| `ask_click` | A visitor opens the assistant | `source`, `evidence_item`, `question` |

To report on the parameters, register them as custom dimensions in GA4
(Admin → Custom definitions).

If the assistant reads a `q` URL parameter, the suggested questions on
/evidence will pre-fill its input. If it doesn't, the links still open it.
