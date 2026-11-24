# /jobs review page

Private page where Reggie and Yahsh review the daily job matches from the
`yahsh-job-matcher` data repo and record feedback. Spec: `docs/matching_rubric_spec.md`
in the data repo (§2 trust boundaries, §11 tiers, §14 explanations, §15 feedback).

## How it runs

- The site stays static. `@astrojs/vercel` is added, and only these routes render on demand
  (`export const prerender = false`): `/jobs`, `/jobs/signin`, `/api/jobs/signin`,
  `/api/jobs/signout`, `/api/jobs/feedback`. Every other page is still prerendered.
- `src/middleware.ts` guards `/jobs` and `/api/jobs/*`. Any other path passes straight through.
- `npm run build` runs `scripts/merge-vercel-routes.mjs` after `astro build`. That step copies
  the redirects, rewrites and headers from `vercel.json` into `.vercel/output/config.json`
  with Vercel's own converter, so `/ask/`, the client-portal rewrites and the first-90-days
  noindex header keep working with the adapter.

## Vercel environment variables (project "portfolio")

| Name | Value |
|---|---|
| `JOBS_PASSWORD` | Shared password for Reggie and Yahsh. Changing it signs everyone out. |
| `GITHUB_DATA_TOKEN` | Fine-grained token, Contents read/write on `soupilla21/yahsh-job-matcher` only. |
| `DATA_REPO` | `soupilla21/yahsh-job-matcher` |
| `DATA_BRANCH` | Optional. Defaults to `main`. |

Set them for Production, and for Preview too if you test on a preview deployment.
Preview deployments then write to the same `data/feedback.json`.

## Sign-in

`/jobs/signin` asks for the password and who is reviewing (Reggie or Yahsh). It sets an
HttpOnly, Secure, SameSite=Lax cookie for 30 days, signed with HMAC-SHA256 under a key
derived from `JOBS_PASSWORD`. The reviewer saved with each feedback event comes from that
cookie, not from the browser's request body. "Switch / sign out" clears it.

POSTs must carry this site's Origin. Failed sign-ins wait 0.75 s. There is no lockout,
so use a long password.

## Data access (spec §2)

`src/lib/jobs/github.ts` refuses every path outside this allowlist before any request is made:

- read: `data/matches/latest.json`, `data/feedback.json`,
  `config/matching_profile_sanitized.json`, `data/matches/archive/YYYY-MM-DD.json`
- list: `data/matches/archive` (file names only, used for the day picker)
- write: `data/feedback.json`

Responses carry `Cache-Control: private, no-store`, `X-Robots-Tag: noindex, nofollow` and
`Referrer-Policy: same-origin`.

## Feedback

`POST /api/jobs/feedback` with `{posting_id, status, not_interested_reason?, reason_note?}`.
The route:

1. Checks that the posting is on the latest list.
2. Adds `event_id`, the reviewer and a server timestamp. The timestamp is kept strictly later
   than that posting's previous event, so the latest event always wins.
3. Validates the event and the whole file against `data/feedback_schema.json` 1.0.
4. Commits with the file's current SHA, retrying up to 3 times on a conflict.

If the existing `feedback.json` is invalid, the route refuses to write and says so. An
archived day is read-only on the page.

## Page controls

Matches are grouped by tier: Apply first, Worth reviewing, Stretch, then Tracking. Above the
list you can search by title or company, filter by status (the default hides Not interested),
sort by pipeline order, fit score, newest posted or company, and pick a day (Latest or an
archive date). The search, status and sort choices are kept in the URL, so you can share a
filtered view.

## Local testing

```bash
GITHUB_API_URL=http://127.0.0.1:8787 GITHUB_DATA_TOKEN=test DATA_REPO=owner/data \
JOBS_PASSWORD='test password' npx astro dev
```

`GITHUB_API_URL` only applies in `astro dev`. Point it at a mock of the GitHub Contents API.
Production builds always use api.github.com.

## Review period (added 2026-11-24)

While Reggie and Yahsh tune the method, `/jobs` collects two extra signals. All of it is
stored in `data/review.json` (schema `data/review_schema.json` in the data repo). The
pipeline never reads that file, so nothing here can fail a daily run.

- **Explainer.** A collapsible card at the top of `/jobs` (it remembers being collapsed on
  each device) and the full page `/jobs/about`, written from Reggie to Yahsh. The numbers on
  that page come from `src/lib/jobs/rules.ts`. When the latest run used a different
  `scoring_config_version`, the page says the numbers may be out of date. Update `rules.ts`
  when you retune.
- **Right kind / Wrong kind of job.** On every job, including archived days. "Wrong kind"
  needs a reason (wrong kind of work, wrong level, wrong industry, location or arrangement,
  other) and takes an optional note. The latest rating per job and reviewer counts. Shown
  only while `JOBS_REVIEW_MODE=on` in Vercel.
- **Progress and exit rule.** The panel under the header shows rated jobs out of 40, the day
  out of 14, and the share rated "right kind", overall and by tier. It says when either
  limit is reached. Values: `REVIEW_TARGET` and `REVIEW_DAYS` in `src/lib/jobs/review.ts`.
- **Passed on and Bring back.** Every Not interested click also saves a snapshot (title,
  company, link, tier, score, reason, note, who, when). The "Passed on" list shows them, and
  either reviewer can click **Bring back**. That records an `unreviewed` event in
  `data/feedback.json`, which the pipeline treats as back to default, so the job can return
  on the next run if the employer still has it open. This stays on after the review period.

Routes: `POST /api/jobs/review` with `{action: "rate", posting_id, verdict, wrong_reason?,
note?, date?}` or `{action: "restore", posting_id}`. Same sign-in, origin check, allowlist and
validate-then-commit rules as feedback. Notes never contain digits or `@`.

To end the review period, set `JOBS_REVIEW_MODE` to `off` (or delete it) and redeploy.
