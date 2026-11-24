// Feedback events (data/feedback_schema.json 1.0), validated here before every write.
import { randomUUID } from "node:crypto";
import { isReviewer, type Reviewer } from "./env";

export const STATUSES = ["interested", "not_interested", "applied", "interview", "unreviewed"] as const;
export type Status = (typeof STATUSES)[number];

export const REASONS = {
  wrong_role_type: "Wrong type of role",
  too_junior: "Too junior",
  too_senior: "Too senior",
  location_or_arrangement: "Location or work arrangement",
  travel: "Travel",
  pay_too_low: "Pay too low",
  company_not_interested: "Not interested in this company",
  industry_not_interested: "Not interested in this industry",
  requirements_not_met: "Requirements not met",
  posting_closed: "Posting closed",
  duplicate: "Duplicate",
  other: "Other (add a note)",
} as const;
export type Reason = keyof typeof REASONS;

export interface FeedbackEvent {
  event_id: string;
  posting_id: string;
  status: Status;
  not_interested_reason?: Reason;
  reason_note?: string;
  reviewer: Reviewer;
  timestamp: string;
}
export interface FeedbackFile {
  schema_version: "1.0";
  updated_at: string;
  events: FeedbackEvent[];
}

const UUID4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
export const POSTING_ID = /^p_[0-9a-f]{16}$/;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const EVENT_KEYS = new Set(["event_id", "posting_id", "status", "not_interested_reason", "reason_note", "reviewer", "timestamp"]);

/** Errors for one event against the schema, including the conditional rules. */
export function eventErrors(e: unknown): string[] {
  const errs: string[] = [];
  if (!e || typeof e !== "object" || Array.isArray(e)) return ["event is not an object"];
  const ev = e as Record<string, unknown>;
  for (const k of Object.keys(ev)) if (!EVENT_KEYS.has(k)) errs.push(`unexpected field ${k}`);
  if (typeof ev.event_id !== "string" || !UUID4.test(ev.event_id)) errs.push("event_id");
  if (typeof ev.posting_id !== "string" || !POSTING_ID.test(ev.posting_id)) errs.push("posting_id");
  if (!STATUSES.includes(ev.status as Status)) errs.push("status");
  if (!isReviewer(ev.reviewer)) errs.push("reviewer");
  if (typeof ev.timestamp !== "string" || !DATE_TIME.test(ev.timestamp)) errs.push("timestamp");
  const hasReason = ev.not_interested_reason !== undefined;
  if (ev.status === "not_interested") {
    if (!hasReason || !(ev.not_interested_reason as string in REASONS)) errs.push("not_interested_reason");
  } else if (hasReason) {
    errs.push("not_interested_reason is only allowed with not_interested");
  }
  if (ev.reason_note !== undefined) {
    const n = ev.reason_note;
    if (typeof n !== "string" || n.length > 280 || /[0-9@]/.test(n)) errs.push("reason_note");
  }
  if (ev.not_interested_reason === "other" && (typeof ev.reason_note !== "string" || ev.reason_note.length < 3)) {
    errs.push("reason_note is required for other");
  }
  return errs;
}

export function fileErrors(f: unknown): string[] {
  if (!f || typeof f !== "object" || Array.isArray(f)) return ["file is not an object"];
  const file = f as Record<string, unknown>;
  const errs: string[] = [];
  for (const k of Object.keys(file)) if (!["schema_version", "updated_at", "events"].includes(k)) errs.push(`unexpected field ${k}`);
  if (file.schema_version !== "1.0") errs.push("schema_version");
  if (typeof file.updated_at !== "string" || !DATE_TIME.test(file.updated_at)) errs.push("updated_at");
  if (!Array.isArray(file.events)) errs.push("events");
  else file.events.forEach((e, i) => eventErrors(e).forEach((m) => errs.push(`events[${i}].${m}`)));
  return errs;
}

/** Builds a server-side event from the page's request body. */
export function buildEvent(body: unknown, reviewer: Reviewer): { event?: FeedbackEvent; error?: string } {
  if (!body || typeof body !== "object") return { error: "Invalid request." };
  const b = body as Record<string, unknown>;
  const status = b.status as Status;
  if (!STATUSES.includes(status)) return { error: "Unknown status." };
  if (typeof b.posting_id !== "string" || !POSTING_ID.test(b.posting_id)) return { error: "Unknown posting." };
  const note = typeof b.reason_note === "string" ? b.reason_note.trim().replace(/\s+/g, " ") : "";
  if (/[0-9@]/.test(note)) return { error: "Notes can't contain numbers or @." };
  if (note.length > 280) return { error: "Notes are limited to 280 characters." };
  const event: FeedbackEvent = {
    event_id: randomUUID(),
    posting_id: b.posting_id,
    status,
    reviewer,
    timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
  };
  if (status === "not_interested") {
    if (typeof b.not_interested_reason !== "string" || !(b.not_interested_reason in REASONS)) return { error: "Choose a reason." };
    event.not_interested_reason = b.not_interested_reason as Reason;
    if (event.not_interested_reason === "other" && note.length < 3) return { error: "Add a short note for Other." };
  }
  if (note) event.reason_note = note;
  // Key order matches the schema for readable diffs.
  const ordered = {
    event_id: event.event_id,
    posting_id: event.posting_id,
    status: event.status,
    ...(event.not_interested_reason ? { not_interested_reason: event.not_interested_reason } : {}),
    ...(event.reason_note ? { reason_note: event.reason_note } : {}),
    reviewer: event.reviewer,
    timestamp: event.timestamp,
  } as FeedbackEvent;
  const errs = eventErrors(ordered);
  return errs.length ? { error: "Invalid feedback." } : { event: ordered };
}

/** Latest event per posting_id (events are append-only; later timestamps win). */
export function latestByPosting(events: FeedbackEvent[]): Map<string, FeedbackEvent> {
  const map = new Map<string, FeedbackEvent>();
  for (const e of events) {
    const cur = map.get(e.posting_id);
    if (!cur || e.timestamp >= cur.timestamp) map.set(e.posting_id, e);
  }
  return map;
}
