// Publication gate for playbook pages (/advisory/first-90-days and any later
// ones). Identical rule to the advisory briefings (src/data/advisory/index.ts):
// a "review" page is built locally and on Vercel previews, never on Vercel
// production, so merging to main publishes nothing until status is "public".
import { isBuilt } from "../advisory";
import type { BriefingStatus } from "../advisory/types";

export const isBuiltStatus = (status: BriefingStatus) =>
  isBuilt({ status } as Parameters<typeof isBuilt>[0]);
