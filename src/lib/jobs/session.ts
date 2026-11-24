// Signed session cookie for /jobs. The cookie holds the reviewer name and an expiry,
// signed with HMAC-SHA256 under a key derived from that reviewer's own password
// (JOBS_PASSWORD_REGGIE or JOBS_PASSWORD_YAHSH). Changing one person's password signs
// out only that person.
import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import { isReviewer, jobsEnv, type Reviewer } from "./env";

export const SESSION_COOKIE = "jobs_session";
export const SESSION_DAYS = 30;

function key(password: string): Buffer {
  return createHash("sha256").update("yahsh-jobs-session-v1:" + password).digest();
}
const b64u = (b: Buffer | string) => Buffer.from(b).toString("base64url");
const sign = (k: Buffer, data: string) => createHmac("sha256", k).update(data).digest();

/** Constant-time check of a password against the chosen reviewer's own password. */
export function passwordMatches(reviewer: Reviewer, input: string): boolean {
  const expected = jobsEnv.passwordFor(reviewer);
  if (!expected || typeof input !== "string" || input.length > 512) return false;
  const k = key(expected);
  return timingSafeEqual(sign(k, "pw:" + input), sign(k, "pw:" + expected));
}

export function createSession(reviewer: Reviewer): string {
  const password = jobsEnv.passwordFor(reviewer);
  if (!password) throw new Error(`No password set for ${reviewer}`);
  const payload = b64u(JSON.stringify({ v: 2, r: reviewer, exp: Date.now() + SESSION_DAYS * 864e5 }));
  return `${payload}.${b64u(sign(key(password), payload))}`;
}

export function readSession(value: string | undefined): Reviewer | null {
  if (!value || value.length > 512) return null;
  const [payload, sig] = value.split(".");
  if (!payload || !sig) return null;
  let data: { v?: unknown; r?: unknown; exp?: unknown };
  try { data = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")); } catch { return null; }
  // The reviewer named in the payload picks the key; the signature then proves it.
  if (data?.v !== 2 || !isReviewer(data.r) || typeof data.exp !== "number" || data.exp < Date.now()) return null;
  const password = jobsEnv.passwordFor(data.r);
  if (!password) return null;
  const expected = sign(key(password), payload);
  const given = Buffer.from(sig, "base64url");
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) return null;
  return data.r;
}

export const sessionCookieOptions = (secure: boolean) => ({
  httpOnly: true,
  secure,
  sameSite: "lax" as const,
  path: "/",
  maxAge: SESSION_DAYS * 86400,
});

/** POSTs must come from this site (defence in depth on top of SameSite=Lax).
 *  Accepts the hosts the request actually arrived on (Host, X-Forwarded-Host, the URL
 *  Astro sees) plus the site's own domains, so proxies and custom domains don't break it. */
const SITE_HOSTS = ["reginalcampbell.com", "www.reginalcampbell.com"];
export function sameOrigin(request: Request, url: URL): boolean {
  const raw = request.headers.get("origin") ?? (() => {
    const ref = request.headers.get("referer");
    try { return ref ? new URL(ref).origin : null; } catch { return null; }
  })();
  if (!raw || raw === "null") return false;
  let origin: URL;
  try { origin = new URL(raw); } catch { return false; }
  const local = origin.hostname === "localhost" || origin.hostname === "127.0.0.1";
  if (origin.protocol !== "https:" && !local) return false;
  const hosts = new Set<string>([url.host, ...SITE_HOSTS]);
  for (const h of [request.headers.get("host"), request.headers.get("x-forwarded-host")]) {
    if (h) h.split(",").forEach((v) => hosts.add(v.trim().toLowerCase()));
  }
  return hosts.has(origin.host.toLowerCase());
}

/** Only redirect back into /jobs. */
export function safeNext(next: unknown): string {
  if (typeof next !== "string" || !/^\/jobs(\/|\?|$)/.test(next) || next.startsWith("//")) return "/jobs";
  return next.slice(0, 500);
}
