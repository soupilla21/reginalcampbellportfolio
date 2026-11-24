// Server-side access to the private data repo through the GitHub Contents API.
// Spec §2: only the served files can be read, through an exact-path allowlist.
// Any other path is refused before a request is made. The only writable paths are
// data/feedback.json and data/review.json (review period, see docs/jobs-dashboard.md).
import { jobsEnv } from "./env";

export const PATHS = {
  latest: "data/matches/latest.json",
  feedback: "data/feedback.json",
  review: "data/review.json",
  profile: "config/matching_profile_sanitized.json",
  archiveDir: "data/matches/archive",
  archive: (date: string) => `data/matches/archive/${date}.json`,
} as const;

const READ_EXACT = new Set<string>([PATHS.latest, PATHS.feedback, PATHS.review, PATHS.profile]);
const READ_ARCHIVE = /^data\/matches\/archive\/\d{4}-\d{2}-\d{2}\.json$/;
const LIST_EXACT = new Set<string>([PATHS.archiveDir]); // file names only, filtered to dates
const WRITE_EXACT = new Set<string>([PATHS.feedback, PATHS.review]);

export class NotAllowedError extends Error {}
export class ConfigError extends Error {}
export class GitHubError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export const canRead = (path: string) => READ_EXACT.has(path) || READ_ARCHIVE.test(path);

function config() {
  const token = jobsEnv.dataToken();
  const repo = jobsEnv.dataRepo();
  if (!token || !repo) throw new ConfigError("GITHUB_DATA_TOKEN or DATA_REPO is not set");
  return { token, repo, branch: jobsEnv.dataBranch(), base: jobsEnv.apiBase() };
}

async function gh(path: string, init: RequestInit & { accept?: string } = {}) {
  const { token, repo, branch, base } = config();
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  const isRead = !init.method || init.method === "GET";
  const url = `${base}/repos/${repo}/contents/${encoded}${isRead ? `?ref=${encodeURIComponent(branch)}` : ""}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: init.accept ?? "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "reginalcampbell.com-jobs",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
    },
    signal: AbortSignal.timeout(10_000),
  });
  return res;
}

/** Read an allowlisted JSON file. Returns null when it does not exist. */
export async function readJson<T = unknown>(path: string): Promise<T | null> {
  if (!canRead(path)) throw new NotAllowedError(path);
  const res = await gh(path, { accept: "application/vnd.github.raw+json" });
  if (res.status === 404) return null;
  if (!res.ok) throw new GitHubError(res.status, `read failed (${res.status})`);
  return JSON.parse(await res.text()) as T;
}

/** Read an allowlisted JSON file with its blob SHA (needed to write it back). */
export async function readJsonWithSha<T = unknown>(path: string): Promise<{ data: T; sha: string } | null> {
  if (!canRead(path)) throw new NotAllowedError(path);
  const res = await gh(path);
  if (res.status === 404) return null;
  if (!res.ok) throw new GitHubError(res.status, `read failed (${res.status})`);
  const meta = (await res.json()) as { sha: string; content?: string; encoding?: string; type?: string };
  if (meta.type && meta.type !== "file") throw new GitHubError(500, "not a file");
  let text: string;
  if (meta.encoding === "base64" && meta.content) {
    text = Buffer.from(meta.content, "base64").toString("utf8");
  } else {
    // Files over 1 MB come back without inline content.
    const raw = await gh(path, { accept: "application/vnd.github.raw+json" });
    if (!raw.ok) throw new GitHubError(raw.status, `read failed (${raw.status})`);
    text = await raw.text();
  }
  return { data: JSON.parse(text) as T, sha: meta.sha };
}

/** Archive dates (YYYY-MM-DD), newest first. Only file names are read. */
export async function listArchiveDates(): Promise<string[]> {
  const path = PATHS.archiveDir;
  if (!LIST_EXACT.has(path)) throw new NotAllowedError(path);
  const res = await gh(path);
  if (res.status === 404) return [];
  if (!res.ok) throw new GitHubError(res.status, `list failed (${res.status})`);
  const items = (await res.json()) as { name?: string; type?: string }[];
  if (!Array.isArray(items)) return [];
  return items
    .filter((i) => i.type === "file" && typeof i.name === "string" && /^\d{4}-\d{2}-\d{2}\.json$/.test(i.name))
    .map((i) => i.name!.slice(0, 10))
    .sort()
    .reverse();
}

/** Write an allowlisted file. Returns "ok" or "conflict" (stale SHA). */
export async function writeJson(path: string, data: unknown, sha: string | null, message: string): Promise<"ok" | "conflict"> {
  if (!WRITE_EXACT.has(path)) throw new NotAllowedError(path);
  const { branch } = config();
  const body = JSON.stringify({
    message,
    content: Buffer.from(JSON.stringify(data, null, 2) + "\n", "utf8").toString("base64"),
    branch,
    ...(sha ? { sha } : {}),
  });
  const res = await gh(path, { method: "PUT", body });
  if (res.ok) return "ok";
  if (res.status === 409 || res.status === 422) return "conflict";
  throw new GitHubError(res.status, `write failed (${res.status})`);
}
