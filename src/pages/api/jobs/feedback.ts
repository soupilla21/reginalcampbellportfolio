// POST /api/jobs/feedback (spec §15). Appends one event to data/feedback.json in the
// private data repo. The reviewer comes from the signed session, not the body.
// A Not interested click also saves a snapshot of the job to data/review.json, so the
// "Passed on" list can show it and either reviewer can bring it back later.
import type { APIRoute } from "astro";
import { buildEvent } from "../../../lib/jobs/feedback";
import { appendFeedbackEvent } from "../../../lib/jobs/feedback-store";
import { findMatch } from "../../../lib/jobs/data";
import { ConfigError, PATHS } from "../../../lib/jobs/github";
import { emptyReview, newId, reviewFileErrors, type ReviewFile } from "../../../lib/jobs/review";
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
  let body: unknown;
  try { body = JSON.parse(raw); } catch { return json(400, { error: "Invalid JSON." }); }

  const { event, error } = buildEvent(body, reviewer);
  if (!event) return json(400, { error });

  try {
    const found = await findMatch(event.posting_id);
    if (!found) return json(404, { error: "That posting isn't on the latest list. Reload the page." });
    await appendFeedbackEvent(event);

    let snapshotSaved: boolean | null = null;
    if (event.status === "not_interested") {
      const m = found.match;
      snapshotSaved = await appendWithRetry<ReviewFile>({
        path: PATHS.review,
        empty: emptyReview,
        validate: reviewFileErrors,
        apply: (f) => ({
          ...f,
          updated_at: event.timestamp,
          passes: [...f.passes, {
            event_id: newId(),
            posting_id: m.posting_id,
            title: m.posting.title.slice(0, 200),
            company: m.company.name.slice(0, 160),
            url: m.posting.url,
            tier: m.tier,
            fit_score: m.fit_score,
            run_date: found.runDate,
            reason: event.not_interested_reason!,
            ...(event.reason_note ? { note: event.reason_note } : {}),
            reviewer: event.reviewer,
            timestamp: event.timestamp,
          }],
        }),
        message: `review: passed ${m.posting_id}`,
      }).then(() => true, (err) => {
        console.error("pass snapshot failed:", err instanceof Error ? err.name : "unknown");
        return false;
      });
    }
    return json(200, {
      ok: true, status: event.status, reviewer: event.reviewer, at: event.timestamp,
      reason: event.not_interested_reason ?? null,
      ...(snapshotSaved === false ? { warning: "Saved, but it couldn't be added to the Passed on list." } : {}),
    });
  } catch (err) {
    if (err instanceof ConfigError) return json(500, { error: "The data repo isn't configured on the server." });
    if (err instanceof InvalidFileError) return json(500, { error: "data/feedback.json in the data repo is invalid. Fix it there, then try again." });
    if (err instanceof ConflictError) return json(409, { error: "Someone else saved at the same moment. Try again." });
    console.error("feedback write failed:", err instanceof Error ? err.message : "unknown");
    return json(502, { error: "Couldn't reach the data repo. Try again." });
  }
};

export const ALL: APIRoute = () => json(405, { error: "Method Not Allowed" });
