# Moving the portfolio assistant to reginalcampbell.com/ask/

The assistant (`portfolio-intelligence-bot` on Netlify) moves from
`briefing.reginalcampbell.com` to `reginalcampbell.com/ask/`. Vercel proxies the
pages; nothing about the assistant's backend, corpus, or caps moves.

The steps are ordered so the assistant is never down. Do them in order and
check each one before starting the next.

## How the pieces fit

| Request | Served by |
| --- | --- |
| `reginalcampbell.com/ask/`, `/ask/governance`, `/ask/assets/*` | Vercel rewrite to `portfolio-intelligence-bot.netlify.app` (the `/ask` prefix is stripped) |
| Questions (`POST /interrogate`) from reginalcampbell.com | **Directly** to `portfolio-intelligence-bot.netlify.app/interrogate`, not through Vercel |
| Questions from the subdomain or netlify.app | Same-origin `/interrogate`, as today |

Questions skip the proxy on purpose. Through Vercel, every visitor would reach
Netlify from Vercel's IP, so the per-visitor rate limits in `interrogate.js`
would become one shared limit for everyone. Calling Netlify directly keeps
real visitor IPs and keeps the streamed answer to a single hop. The function's
CORS check already allows `ALLOWED_ORIGIN`.

## Step 1: Netlify environment variable

Netlify → `portfolio-intelligence-bot` → Site configuration → Environment
variables. Set:

```
ALLOWED_ORIGIN = https://reginalcampbell.com
```

Exactly that: https, no `www`, no trailing slash. This does not affect the
subdomain, whose calls are same-origin and never need CORS.

## Step 2: Deploy the assistant (briefing repo patch)

Apply `portfolio-intelligence-bot-ask-subpath.patch` to the
`portfolio-intelligence-bot` repo, then build and deploy as you normally do
(`npx netlify deploy --prod`).

**Check:** `https://briefing.reginalcampbell.com/` still loads and answers a
question, and the "Read the full governance model" link opens
`/ask/governance`. The subdomain keeps working because `netlify.toml` maps
`/ask/*` back to the real files.

Note: the live site currently serves a `robots.txt` and `sitemap.xml` that are
not in the repo (they were added outside the build). A fresh deploy drops them.
That is harmless here, because the canonical tag now points to
`reginalcampbell.com/ask/` and the main site's sitemap takes over in step 4.

## Step 3: Deploy the main site (this branch)

Merge `rec2-evidence-ask`. It adds the `/ask` → `/ask/` redirect and the
`/ask/:path*` rewrite in `vercel.json`.

**Check, at `https://reginalcampbell.com/ask/`:**

- The assistant loads, with no 404s in the browser console.
- A question gets an answer, and it streams in progressively rather than all at once.
  In the Network tab, the POST goes to `portfolio-intelligence-bot.netlify.app/interrogate`.
- `/ask/governance` shows the governance page; "Back to the assistant" returns to `/ask/`.
- `/ask/?q=test` opens with "test" pre-filled and nothing sent.

If the question fails with a CORS error, step 1 isn't applied yet. Netlify
environment changes take effect on the next deploy, so redeploy the assistant.

## Step 4: Point the site's links at /ask/

In `src/config/site.ts` change:

```ts
export const ASK_URL = "/ask/";
```

Commit and deploy. Every "Ask the portfolio assistant" link and suggested
question on `/evidence` now opens `/ask/`, and the sitemap automatically adds
`/ask/` and `/ask/governance`.

## Step 5: Retire the subdomain

In the assistant repo's `netlify.toml`, uncomment the
`briefing.reginalcampbell.com/*` 301 block at the top, and redeploy.

**Check:** `https://briefing.reginalcampbell.com/governance` lands on
`https://reginalcampbell.com/ask/governance`, and `/ask/` on the main site
still answers questions. Vercel's requests arrive on the netlify.app hostname,
so the rule never redirects them and there is no loop.

Keep the `briefing` DNS record in Namecheap so old links keep redirecting.

## Rolling back

- After step 3: remove the `rewrites` and `redirects` from `vercel.json`. The subdomain is untouched.
- After step 4: set `ASK_URL` back to `"https://briefing.reginalcampbell.com"`.
- After step 5: comment the 301 block out again and redeploy the assistant.

## Measuring (60 to 90 day keep-or-retire decision)

Start the clock at step 4. Before the move the assistant had 57 questions in
about 14 weeks (usage logs are cumulative exports, so read only the latest file).

| GA4 event | Fired when | Parameters |
| --- | --- | --- |
| `evidence_hub_click` | A homepage link to /evidence is clicked | `source` |
| `evidence_link_click` | A PDF, artifact, section, resume, or code link on /evidence is opened | `evidence_item`, `link_kind` |
| `ask_click` | The assistant is opened from the site | `source`, `evidence_item`, `question` |

Register the parameters as custom dimensions in GA4 (Admin → Custom
definitions) to report on them. Pair these with the assistant's own usage logs
for questions actually asked.

## Labels

`/evidence` uses the assistant's governance labels, Verified, Illustrative
and In Development, with the same definitions. Each entry names its record in
`knowledge.json` (`corpusKeys`). When the corpus changes, update the matching
entry in `src/content/evidence/` so the two never disagree.

The homepage case study, "National Healthcare Data Fabric", combines the
Accenture and VA programs; the corpus treats that combined version as
illustrative. It is deliberately not listed on `/evidence` until the homepage
is revised.
