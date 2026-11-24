// POST /api/jobs/feedback (spec §15). Appends one event to data/feedback.json in the
// private data repo and commits it with the file's current SHA, retrying up to 3
// times on a conflict. The reviewer comes from the signed session, not the body.
import type { APIRoute } from "astro";
import { buildEvent, fileErrors, type FeedbackFile } from "../../../lib/jobs/feedback";
import { publishedIds } from "../../../lib/jobs/data";
import { ConfigError, PATHS, readJsonWithSha, writeJson } from "../../../lib/jobs/github";
import { sameOrigin } from "../../../lib/jobs/session";

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
    if (!(await publishedIds()).has(event.posting_id)) {
      return json(404, { error: "That posting isn't on the latest list. Reload the page." });
    }
    for (let attempt = 0; attempt < 4; attempt++) {
      const current = await readJsonWithSha<FeedbackFile>(PATHS.feedback);
      const file: FeedbackFile = current?.data ?? { schema_version: "1.0", updated_at: event.timestamp, events: [] };
      if (current && fileErrors(file).length) {
        console.error("feedback.json failed validation; refusing to write");
        return json(500, { error: "data/feedback.json in the data repo is invalid. Fix it there, then try again." });
      }
      // Keep the new event strictly after this posting's previous one, so "latest wins" is unambiguous
      // even when two clicks land in the same second.
      const prev = file.events.filter((e) => e.posting_id === event.posting_id).map((e) => e.timestamp).sort().pop();
      if (prev && Date.parse(prev) >= Date.parse(event.timestamp)) {
        event.timestamp = new Date(Date.parse(prev) + 1000).toISOString().replace(/\.\d{3}Z$/, "Z");
      }
      const next: FeedbackFile = { schema_version: "1.0", updated_at: event.timestamp, events: [...file.events, event] };
      if (fileErrors(next).length) return json(400, { error: "Invalid feedback." });
      const message = `feedback: ${event.posting_id} -> ${event.status}`;
      const result = await writeJson(PATHS.feedback, next, current?.sha ?? null, message);
      if (result === "ok") {
        return json(200, { ok: true, status: event.status, reviewer: event.reviewer, at: event.timestamp, reason: event.not_interested_reason ?? null });
      }
      await new Promise((r) => setTimeout(r, 200 + attempt * 300));
    }
    return json(409, { error: "Someone else saved at the same moment. Try again." });
  } catch (err) {
    if (err instanceof ConfigError) return json(500, { error: "The data repo isn't configured on the server." });
    console.error("feedback write failed:", err instanceof Error ? err.message : "unknown");
    return json(502, { error: "Couldn't reach the data repo. Try again." });
  }
};

export const ALL: APIRoute = () => json(405, { error: "Method Not Allowed" });
