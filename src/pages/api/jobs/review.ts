// POST /api/jobs/review: review-period actions, written to data/review.json.
//   {action: "rate", posting_id, verdict: "right"|"wrong", wrong_reason?, note?, date?}
//       Rates whether a job is the right kind for him. Works on archived days (date).
//       Only while JOBS_REVIEW_MODE is on.
//   {action: "restore", posting_id}
//       "Bring back" for a job marked Not interested: records an "unreviewed" feedback
//       event (the pipeline treats it as back to default) and a restore in review.json.
// The reviewer always comes from the signed session.
import type { APIRoute } from "astro";
import { findMatch, DATE_RE } from "../../../lib/jobs/data";
import { POSTING_ID } from "../../../lib/jobs/feedback";
import { appendFeedbackEvent } from "../../../lib/jobs/feedback-store";
import { ConfigError, PATHS, readJson } from "../../../lib/jobs/github";
import {
  activePasses, cleanNote, emptyReview, newId, nowIso, reviewFileErrors, reviewModeOn, WRONG_REASONS, type ReviewFile, type WrongReason,
} from "../../../lib/jobs/review";
import { sameOrigin } from "../../../lib/jobs/session";
import { appendWithRetry, ConflictError, InvalidFileError } from "../../../lib/jobs/store";

export const prerender = false;

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

export const POST: APIRoute = async ({ request, url, locals }) => {
  if (!sameOrigin(request, url)) return json(403, { error: "Forbidden." });
  const reviewer = locals.reviewer;
  if (!reviewer) return json(401, { error: "Signed out. Reload the page and sign in." });
  if (!(request.headers.get("content-type") ?? "").includes("application/json")) return json(415, { error: "Send JSON." });
  const raw = await request.text();
  if (raw.length > 4096) return json(413, { error: "Request too large." });
  let b: Record<string, unknown>;
  try { b = JSON.parse(raw); } catch { return json(400, { error: "Invalid JSON." }); }
  if (!b || typeof b !== "object") return json(400, { error: "Invalid request." });
  if (typeof b.posting_id !== "string" || !POSTING_ID.test(b.posting_id)) return json(400, { error: "Unknown posting." });
  const postingId = b.posting_id;

  try {
    if (b.action === "rate") {
      if (!reviewModeOn()) return json(403, { error: "The review period is switched off." });
      const verdict = b.verdict;
      if (verdict !== "right" && verdict !== "wrong") return json(400, { error: "Choose right or wrong kind." });
      let wrong_reason: WrongReason | undefined;
      if (verdict === "wrong") {
        if (typeof b.wrong_reason !== "string" || !(b.wrong_reason in WRONG_REASONS)) return json(400, { error: "Choose why it's the wrong kind." });
        wrong_reason = b.wrong_reason as WrongReason;
      }
      const { note, error } = cleanNote(b.note);
      if (error) return json(400, { error });
      if (wrong_reason === "other" && (note?.length ?? 0) < 3) return json(400, { error: "Add a short note for Other." });
      const date = typeof b.date === "string" && DATE_RE.test(b.date) ? b.date : null;
      const found = await findMatch(postingId, date);
      if (!found) return json(404, { error: "That posting isn't on this day's list. Reload the page." });
      const timestamp = nowIso();
      await appendWithRetry<ReviewFile>({
        path: PATHS.review, empty: emptyReview, validate: reviewFileErrors,
        apply: (f) => ({
          ...f, updated_at: timestamp,
          ratings: [...f.ratings, {
            event_id: newId(), posting_id: postingId, verdict,
            ...(wrong_reason ? { wrong_reason } : {}), ...(note ? { note } : {}),
            reviewer, timestamp,
            context: { run_date: found.runDate, tier: found.match.tier, fit_score: found.match.fit_score },
          }],
        }),
        message: `review: rated ${postingId} ${verdict}`,
      });
      return json(200, { ok: true, verdict, wrong_reason: wrong_reason ?? null, reviewer, at: timestamp });
    }

    if (b.action === "restore") {
      const review = await readJson<ReviewFile>(PATHS.review);
      if (!activePasses(review).some((p) => p.posting_id === postingId)) {
        return json(404, { error: "That job isn't on the Passed on list anymore. Reload the page." });
      }
      const event = await appendFeedbackEvent({ event_id: newId(), posting_id: postingId, status: "unreviewed", reviewer, timestamp: nowIso() });
      await appendWithRetry<ReviewFile>({
        path: PATHS.review, empty: emptyReview, validate: reviewFileErrors,
        apply: (f) => ({ ...f, updated_at: event.timestamp, restores: [...f.restores, { event_id: newId(), posting_id: postingId, reviewer, timestamp: event.timestamp }] }),
        message: `review: brought back ${postingId}`,
      });
      return json(200, { ok: true, reviewer, at: event.timestamp });
    }
    return json(400, { error: "Unknown action." });
  } catch (err) {
    if (err instanceof ConfigError) return json(500, { error: "The data repo isn't configured on the server." });
    if (err instanceof InvalidFileError) return json(500, { error: `${err.message} in the data repo is invalid. Fix it there, then try again.` });
    if (err instanceof ConflictError) return json(409, { error: "Someone else saved at the same moment. Try again." });
    console.error("review write failed:", err instanceof Error ? err.message : "unknown");
    return json(502, { error: "Couldn't reach the data repo. Try again." });
  }
};

export const ALL: APIRoute = () => json(405, { error: "Method Not Allowed" });
