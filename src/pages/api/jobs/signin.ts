import type { APIRoute } from "astro";
import { isReviewer, jobsEnv } from "../../../lib/jobs/env";
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
  return redirect(next, 303);
};

export const ALL: APIRoute = () => new Response("Method Not Allowed", { status: 405, headers: { Allow: "POST" } });
