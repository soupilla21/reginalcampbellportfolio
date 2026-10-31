# Advisory briefings (`/advisory/<slug>`)

Interactive executive briefings built from an advisory document package. The
first one is `/advisory/hrhs-ai` (HRHS Enterprise AI Advisory Package v2.1).

## How it is built

| Piece | Path | Role |
| --- | --- | --- |
| Source files | `advisory-packages/<slug>/<version>/files/` | Approved originals (PDF, PPTX). Outside `public/` on purpose. |
| Previews | `advisory-packages/<slug>/<version>/previews/` | First page of each PDF at 480px wide (WebP), plus the social card. |
| Types | `src/data/advisory/types.ts` | Manifest and evidence-tag types. |
| Manifest + content | `src/data/advisory/hrhs-ai.ts` | Document metadata, reading order, publication status, and every piece of narrative copy, each with source refs. |
| Registry + gate | `src/data/advisory/index.ts` | Which briefings exist and which may be built in this environment. |
| Page | `src/pages/advisory/[briefing].astro` | Layout only. Uses the shared `PageLayout` (site nav, footer, GA4). |
| Files route | `src/pages/advisory/[briefing]/files/[file].ts` | Emits approved files into the build, byte for byte. |
| Components | `src/components/advisory/` | Chapter header, bridge, takeaway, sources line, nav, the four chapter visuals, library + viewer. |
| Styles | `src/styles/advisory-briefing.css` | Scoped under `.hb`; imported by the briefing page only. |
| Behavior | `src/scripts/advisory-briefing.ts` | Progressive enhancement: tabs, scenarios, chapter marker, viewer, reveal. |

Shared files touched: `src/layouts/PageLayout.astro` (optional `ogImage` prop;
default output unchanged) and `src/pages/sitemap.xml.ts` (lists briefings only
when `status: "public"`). No other page's output changes.

## The publication gate

`meta.status` in the content module controls everything.

| Environment | `status: "review"` | `status: "public"` |
| --- | --- | --- |
| `npm run dev` / local `npm run build` | Built, noindex | Built, indexable |
| Vercel preview (`VERCEL_ENV=preview`) | Built, noindex | Built, indexable |
| Vercel production (`VERCEL_ENV=production`) | **Not built.** No page, no files. | Built, indexable, in sitemap |

This is a build-time exclusion, so merging to `main` while the status is
`review` publishes nothing. Reproduce the production result anywhere with
`ADVISORY_EXCLUDE_UNPUBLISHED=1 npm run build`.

`noindex` is not access control. Preview deployments are reachable by URL, so
before sharing one, confirm in Vercel: Project Settings, Deployment Protection,
Vercel Authentication is on for preview deployments (Standard Protection). For
an external reviewer without a Vercel account, use a shareable link or
password protection rather than relying on an unguessable URL.

## Adding or updating a package

1. Copy the approved files into `advisory-packages/<slug>/<version>/files/`.
   Never copy items the package marks private (interview prep, QA records).
2. Render previews: `pdftoppm -f 1 -l 1 -r 110 -png file.pdf out` and resize to
   480px wide WebP into `previews/`.
3. Update the manifest entries (version, file, preview, pages, status).
4. If the narrative changes, edit the content module. Every claim keeps a
   `refs` entry; figures must exist verbatim in the source documents.
5. `npm run build`, then review the page locally or on a preview deployment.

## Analytics

Uses the existing GA4 tag (`G-32FSF2EFEW`) and click tracker. No new property.
GA loads only on the production hostnames, so review builds send nothing.

| Event | When | Parameters |
| --- | --- | --- |
| `advisory_briefing_view` | page load | `briefing`, `status` |
| `briefing_entry` | hero buttons | `briefing`, `mode` (guided/direct) |
| `briefing_chapter_nav` | chapter bar | `briefing`, `chapter` |
| `briefing_path_select` | strategic path card | `path` (A/B/C) |
| `briefing_scenario_select` | sensitivity scenario | `scenario` |
| `advisory_document_open` | Read / preview / hero links | `briefing`, `document`, `method` |
| `advisory_document_download` | Download links | `briefing`, `document`, `format` |
| `briefing_contact_click` | author links | `briefing`, `target` |

No personal data is collected. The site has no consent banner today; these
events follow the site's existing behavior.

## Publication review checklist (HRHS v2.1)

Status changed to `public` on October 31, 2026 at the owner's direction. Open items below are follow-ups, not blockers.

The site serves the **public edition** of documents 00 to 08: re-exported from
the v2.1 Word sources with two changes only, verified by text diff and page
count. The running header no longer carries the evaluation-use marking, and the
classification line reads "Not a description of any real organization." The
overview (00A) and the deck never carried either and are the v2.1 originals.

- [x] Confidentiality marking removed from served documents (public edition).
- [x] Hiring-organization reference removed from served documents.
- [x] Private materials excluded: HRHS-AI-09 and QA1/QA2 are not in this
      repository. Keep it that way.
- [ ] **Scenario dates read as real.** The decision is dated December 9, 2026 and
      documents are dated November 2026. The page labels these as scenario
      dates; confirm that is sufficient once those dates pass.
- [ ] **Fictional organization name.** A search found no real organization named
      "Halcyon Ridge Health System". Re-check before public release.
- [ ] **Time-sensitive regulatory reference.** Document 07 cites HHS's projected
      July 2027 final action on the proposed HIPAA Security Rule update.
- [ ] **Minor source inconsistency.** The deck shows V-04 "Confirm by Day 30" and
      V-05 "Validate Days 15 to 20"; Document 03 lists V-04 at Jan 31, 2027 and
      V-05 "before Dec 9". The page does not repeat either date.
- [ ] **Disclosure wording.** Approve the short disclosure (hero) and the full
      disclosure (page foot).
- [ ] **Author line.** Only "Enterprise transformation leader", "PMP" and "U.S.
      Army veteran" are used, matching the live site. Approve.
- [ ] **Preview protection.** Vercel Authentication is on for preview
      deployments (checked October 31, 2026); keep it on.

This repository is public. Never commit package files that are not approved for
public release, even while a briefing's status is `review`: the gate keeps files
off the website, not out of GitHub.

When approved: set `status: "public"`, build, review the preview deployment,
then merge.
