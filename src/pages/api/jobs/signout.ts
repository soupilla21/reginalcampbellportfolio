import type { APIRoute } from "astro";
import { sameOrigin, SESSION_COOKIE } from "../../../lib/jobs/session";

export const prerender = false;

export const POST: APIRoute = ({ request, url, cookies, redirect }) => {
  if (!sameOrigin(request, url)) return new Response("Forbidden", { status: 403 });
  cookies.delete(SESSION_COOKIE, { path: "/" });
  return redirect("/jobs/signin", 303);
};

export const ALL: APIRoute = () => new Response("Method Not Allowed", { status: 405, headers: { Allow: "POST" } });
