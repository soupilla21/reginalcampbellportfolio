// Review period (data/review.json, schema data/review_schema.json 1.0 in the data repo).
// Two kinds of records, both append-only:
//   ratings  - "right kind / wrong kind of job", per reviewer; the latest per (posting, reviewer) wins
//   passes   - a snapshot of each job marked Not interested, so it can be found and brought back
//   restores - "Bring back" clicks; a pass is active until a later restore for the same posting
// The pipeline never reads this file, so nothing here can fail a daily run.
import { randomUUID } from "node:crypto";
import { isReviewer, type Reviewer } from "./env";
import { POSTING_ID, REASONS, type Reason } from "./feedback";

export const REVIEW_TARGET = 40; // rated jobs
export const REVIEW_DAYS = 14;

export const WRONG_REASONS = {
  wrong_function: "Wrong kind of work",
  wrong_level: "Wrong level",
  wrong_industry: "Wrong industry",
  location_or_arrangement: "Location or work arrangement",
  other: "Other (add a note)",
} as const;
export type WrongReason = keyof typeof WRONG_REASONS;

type Tier = "apply_first" | "worth_reviewing" | "stretch" | "tracking";
const TIERS: Tier[] = ["apply_first", "worth_reviewing", "stretch", "tracking"];

export interface Rating {
  event_id: string;
  posting_id: string;
  verdict: "right" | "wrong";
  wrong_reason?: WrongReason;
  note?: string;
  reviewer: Reviewer;
  timestamp: string;
  context: { run_date: string; tier: Tier; fit_score: number };
}
export interface Pass {
  event_id: string;
  posting_id: string;
  title: string;
  company: string;
  url: string;
  tier: Tier;
  fit_score: number;
  run_date: string;
  reason: Reason;
  note?: string;
  reviewer: Reviewer;
  timestamp: string;
}
export interface Restore {
  event_id: string;
  posting_id: string;
  reviewer: Reviewer;
  timestamp: string;
}
export interface ReviewFile {
  schema_version: "1.0";
  updated_at: string;
  ratings: Rating[];
  passes: Pass[];
  restores: Restore[];
}

export const reviewModeOn = () => /^(on|true|1|yes)$/i.test((process.env.JOBS_REVIEW_MODE ?? (import.meta as any).env?.JOBS_REVIEW_MODE ?? "").trim());
export const emptyReview = (): ReviewFile => ({ schema_version: "1.0", updated_at: nowIso(), ratings: [], passes: [], restores: [] });
export const nowIso = () => new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
export const newId = () => randomUUID();

const UUID4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const str = (v: unknown, max: number, min = 1) => typeof v === "string" && v.length >= min && v.length <= max;
const noteOk = (v: unknown) => v === undefined || (typeof v === "string" && v.length <= 280 && !/[0-9@]/.test(v));
const keysOk = (o: Record<string, unknown>, allowed: string[]) => Object.keys(o).every((k) => allowed.includes(k));

function ratingErrors(r: any): string[] {
  const e: string[] = [];
  if (!r || typeof r !== "object") return ["not an object"];
  if (!keysOk(r, ["event_id", "posting_id", "verdict", "wrong_reason", "note", "reviewer", "timestamp", "context"])) e.push("unexpected field");
  if (!UUID4.test(r.event_id ?? "")) e.push("event_id");
  if (!POSTING_ID.test(r.posting_id ?? "")) e.push("posting_id");
  if (r.verdict !== "right" && r.verdict !== "wrong") e.push("verdict");
  if (r.verdict === "wrong" && !(r.wrong_reason in WRONG_REASONS)) e.push("wrong_reason");
  if (r.verdict === "right" && r.wrong_reason !== undefined) e.push("wrong_reason only with wrong");
  if (r.wrong_reason === "other" && !str(r.note, 280, 3)) e.push("note required for other");
  if (!noteOk(r.note)) e.push("note");
  if (!isReviewer(r.reviewer)) e.push("reviewer");
  if (!DATE_TIME.test(r.timestamp ?? "")) e.push("timestamp");
  const c = r.context;
  if (!c || typeof c !== "object" || !keysOk(c, ["run_date", "tier", "fit_score"]) || !DATE.test(c.run_date ?? "") ||
      !TIERS.includes(c.tier) || !Number.isInteger(c.fit_score) || c.fit_score < 0 || c.fit_score > 100) e.push("context");
  return e;
}

function passErrors(p: any): string[] {
  const e: string[] = [];
  if (!p || typeof p !== "object") return ["not an object"];
  if (!keysOk(p, ["event_id", "posting_id", "title", "company", "url", "tier", "fit_score", "run_date", "reason", "note", "reviewer", "timestamp"])) e.push("unexpected field");
  if (!UUID4.test(p.event_id ?? "")) e.push("event_id");
  if (!POSTING_ID.test(p.posting_id ?? "")) e.push("posting_id");
  if (!str(p.title, 200)) e.push("title");
  if (!str(p.company, 160)) e.push("company");
  if (typeof p.url !== "string" || !/^https:\/\//.test(p.url) || p.url.length > 2000) e.push("url");
  if (!TIERS.includes(p.tier)) e.push("tier");
  if (!Number.isInteger(p.fit_score) || p.fit_score < 0 || p.fit_score > 100) e.push("fit_score");
  if (!DATE.test(p.run_date ?? "")) e.push("run_date");
  if (!(p.reason in REASONS)) e.push("reason");
  if (!noteOk(p.note)) e.push("note");
  if (!isReviewer(p.reviewer)) e.push("reviewer");
  if (!DATE_TIME.test(p.timestamp ?? "")) e.push("timestamp");
  return e;
}

function restoreErrors(r: any): string[] {
  if (!r || typeof r !== "object") return ["not an object"];
  const e: string[] = [];
  if (!keysOk(r, ["event_id", "posting_id", "reviewer", "timestamp"])) e.push("unexpected field");
  if (!UUID4.test(r.event_id ?? "")) e.push("event_id");
  if (!POSTING_ID.test(r.posting_id ?? "")) e.push("posting_id");
  if (!isReviewer(r.reviewer)) e.push("reviewer");
  if (!DATE_TIME.test(r.timestamp ?? "")) e.push("timestamp");
  return e;
}

export function reviewFileErrors(f: unknown): string[] {
  if (!f || typeof f !== "object" || Array.isArray(f)) return ["file is not an object"];
  const file = f as Record<string, unknown>;
  const errs: string[] = [];
  if (!keysOk(file, ["schema_version", "updated_at", "ratings", "passes", "restores"])) errs.push("unexpected field");
  if (file.schema_version !== "1.0") errs.push("schema_version");
  if (!DATE_TIME.test(String(file.updated_at ?? ""))) errs.push("updated_at");
  for (const [key, check] of [["ratings", ratingErrors], ["passes", passErrors], ["restores", restoreErrors]] as const) {
    const list = file[key];
    if (!Array.isArray(list)) errs.push(key);
    else list.forEach((item, i) => check(item).forEach((m) => errs.push(`${key}[${i}].${m}`)));
  }
  return errs;
}

export const cleanNote = (v: unknown): { note?: string; error?: string } => {
  const note = typeof v === "string" ? v.trim().replace(/\s+/g, " ") : "";
  if (/[0-9@]/.test(note)) return { error: "Notes can't contain numbers or @." };
  if (note.length > 280) return { error: "Notes are limited to 280 characters." };
  return note ? { note } : {};
};

/** Latest rating per posting and reviewer. */
export function latestRatings(file: ReviewFile | null): Map<string, Partial<Record<Reviewer, Rating>>> {
  const map = new Map<string, Partial<Record<Reviewer, Rating>>>();
  for (const r of file?.ratings ?? []) {
    const entry = map.get(r.posting_id) ?? {};
    const cur = entry[r.reviewer];
    if (!cur || r.timestamp >= cur.timestamp) entry[r.reviewer] = r;
    map.set(r.posting_id, entry);
  }
  return map;
}

/** Passes that haven't been brought back, newest first. */
export function activePasses(file: ReviewFile | null): Pass[] {
  const lastRestore = new Map<string, string>();
  for (const r of file?.restores ?? []) {
    if ((lastRestore.get(r.posting_id) ?? "") < r.timestamp) lastRestore.set(r.posting_id, r.timestamp);
  }
  const latestPass = new Map<string, Pass>();
  for (const p of file?.passes ?? []) {
    const cur = latestPass.get(p.posting_id);
    if (!cur || p.timestamp >= cur.timestamp) latestPass.set(p.posting_id, p);
  }
  return [...latestPass.values()]
    .filter((p) => !(lastRestore.get(p.posting_id) && lastRestore.get(p.posting_id)! >= p.timestamp))
    .sort((a, b) => b.timestamp.localeCompare(a.timestamp));
}

export interface Progress {
  rated: number;
  target: number;
  day: number;
  days: number;
  done: boolean;
  doneReason: "target" | "time" | null;
  rightPct: number | null;
  byTier: { tier: Tier; right: number; total: number }[];
  started: string | null;
}

/** Progress toward the exit rule: REVIEW_TARGET rated jobs or REVIEW_DAYS days, whichever comes first. */
export function reviewProgress(file: ReviewFile | null, now = Date.now()): Progress {
  const latest = latestRatings(file);
  const all = [...latest.values()].flatMap((e) => Object.values(e) as Rating[]);
  const started = (file?.ratings ?? []).map((r) => r.timestamp).sort()[0] ?? null;
  const day = started ? Math.floor((now - Date.parse(started)) / 864e5) + 1 : 0;
  const rated = latest.size;
  const right = all.filter((r) => r.verdict === "right").length;
  const byTier = TIERS.map((tier) => {
    const inTier = all.filter((r) => r.context.tier === tier);
    return { tier, right: inTier.filter((r) => r.verdict === "right").length, total: inTier.length };
  }).filter((t) => t.total > 0);
  const doneReason = rated >= REVIEW_TARGET ? "target" : day > REVIEW_DAYS ? "time" : null;
  return {
    rated, target: REVIEW_TARGET, day: Math.min(day, REVIEW_DAYS), days: REVIEW_DAYS,
    done: doneReason !== null, doneReason,
    rightPct: all.length ? Math.round((right / all.length) * 100) : null,
    byTier, started,
  };
}

