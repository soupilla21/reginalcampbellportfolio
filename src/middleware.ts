// Guards /jobs and /api/jobs/* (PROJECT_CONTEXT: Access). Every other path passes
// straight through, so the rest of reginalcampbell.com is unaffected. Prerendered
// pages never reach the guarded branch.
import { defineMiddleware } from "astro:middleware";
import { readSession, SESSION_COOKIE } from "./lib/jobs/session";

const isGuarded = (p: string) => p === "/jobs" || p.startsWith("/jobs/") || p === "/api/jobs" || p.startsWith("/api/jobs/");
const isOpen = (p: string) => ["/jobs/signin", "/jobs/signin/", "/api/jobs/signin", "/api/jobs/signin/"].includes(p);

function harden(res: Response): Response {
  const h: Record<string, string> = {
    "Cache-Control": "private, no-store",
    "X-Robots-Tag": "noindex, nofollow",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
  };
  try {
    for (const [k, v] of Object.entries(h)) res.headers.set(k, v);
    return res;
  } catch {
    const copy = new Response(res.body, res);
    for (const [k, v] of Object.entries(h)) copy.headers.set(k, v);
    return copy;
  }
}

export const onRequest = defineMiddleware(async (ctx, next) => {
  const path = ctx.url.pathname;
  if (!isGuarded(path)) return next();

  if (!isOpen(path)) {
    const reviewer = readSession(ctx.cookies.get(SESSION_COOKIE)?.value);
    if (!reviewer) {
      if (path.startsWith("/api/")) {
        return harden(new Response(JSON.stringify({ error: "Signed out. Reload the page and sign in." }), {
          status: 401, headers: { "Content-Type": "application/json" },
        }));
      }
      const back = path + ctx.url.search;
      return harden(ctx.redirect(`/jobs/signin?next=${encodeURIComponent(back)}`, 302));
    }
    ctx.locals.reviewer = reviewer;
  }
  return harden(await next());
});
