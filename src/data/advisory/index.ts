// Registry of advisory briefings and the publication gate.
//
// THE GATE. A briefing whose status is "review" is built only where it cannot
// reach the public production site:
//   - local builds and `astro dev`               built
//   - Vercel preview deployments (VERCEL_ENV=preview)  built
//   - Vercel production (VERCEL_ENV=production)  NOT built: no page, no files
// The page and every document file come from getStaticPaths(), so on a
// production build there is nothing in dist/ to find, guess or leak. This is a
// build-time exclusion, not a hidden URL and not a client-side check.
//
// Preview deployments are only as private as the Vercel project's Deployment
// Protection setting. See docs/advisory-briefings.md before sharing one.
//
// Set ADVISORY_EXCLUDE_UNPUBLISHED=1 to reproduce the production behavior in
// any other environment (used by the verification script).

import * as hrhs from "./hrhs-ai";
import type { BriefingMeta, ManifestDocument } from "./types";

export interface Briefing {
  meta: BriefingMeta;
  documents: ManifestDocument[];
}

export const briefings: Briefing[] = [{ meta: hrhs.meta, documents: hrhs.documents }];

export function isBuilt(meta: BriefingMeta): boolean {
  if (meta.status === "public") return true;
  if (process.env.VERCEL_ENV === "production") return false;
  if (process.env.ADVISORY_EXCLUDE_UNPUBLISHED === "1") return false;
  return true;
}

export const builtBriefings = () => briefings.filter((b) => isBuilt(b.meta));
export const publicBriefings = () => briefings.filter((b) => b.meta.status === "public");

// Where a briefing's files live in the repo (outside public/ on purpose, so
// nothing is served unless the gate above lets the files route emit it).
export const packageDir = (meta: BriefingMeta) => `advisory-packages/${meta.slug}/${meta.packageVersion}`;

export const fileHref = (slug: string, name: string) => `/advisory/${slug}/files/${encodeURIComponent(name)}`;
