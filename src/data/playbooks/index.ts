// Publication gate for playbook pages (/advisory/first-90-days and any later
// ones). Builds on the advisory briefing rule (src/data/advisory/index.ts):
//   review    built locally and on Vercel previews, never on production
//   unlisted  built everywhere, but noindex/nofollow and out of the sitemap
//             (live at its URL for people who are given the link)
//   public    built everywhere, indexable, in the sitemap
import { isBuilt } from "../advisory";

export type PlaybookStatus = "review" | "unlisted" | "public";

export const isBuiltStatus = (status: PlaybookStatus) =>
  status === "review" ? isBuilt({ status: "review" } as Parameters<typeof isBuilt>[0]) : true;
