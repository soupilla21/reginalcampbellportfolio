import type { APIRoute } from "astro";
import { isReviewer, jobsEnv } from "../../../lib/jobs/env";
import { INTRO_COOKIE, introLandingActive } from "../../../lib/jobs/review";
import { createSession, passwordMatches, safeNext, sameOrigin, SESSION_COOKIE, sessionCookieOptions } from "../../../lib/jobs/session";

export const prerender = false;

export const POST: APIRoute = async ({ request, url, cookies, redirect }) => {
  if (!sameOrigin(request, url)) return new Response("Forbidden", { status: 403 });
  const form = await request.formData().catch(() => null);
  const next = safeNext(form?.get("next"));
  const back = (code: string) => redirect(`/jobs/signin?error=${code}&next=${encodeURIComponent(next)}`, 303);

  const password = String(form?.get("password") ?? "");
  const reviewer = form?.get("reviewer");
  if (!isReviewer(reviewer)) return back("reviewer");
  if (!jobsEnv.passwordFor(reviewer)) return back("config");
  if (!passwordMatches(reviewer, password)) {
    await new Promise((r) => setTimeout(r, 750)); // slow down guessing
    return back("password");
  }
  cookies.set(SESSION_COOKIE, createSession(reviewer), sessionCookieOptions(url.protocol === "https:"));
  // During the review period, Yahsh lands on the explainer first, unless he followed a
  // link to a specific view (an archived day or a filtered list).
  cookies.delete(INTRO_COOKIE, { path: "/" });
  if (next === "/jobs" && (await introLandingActive(reviewer))) return redirect("/jobs/about", 303);
  return redirect(next, 303);
};

export const ALL: APIRoute = () => new Response("Method Not Allowed", { status: 405, headers: { Allow: "POST" } });
