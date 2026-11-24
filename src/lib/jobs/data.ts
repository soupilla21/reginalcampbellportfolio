// Loads one day of matches for the /jobs page and overlays live feedback (spec §15).
import { latestByPosting, type FeedbackFile, type Reason, type Status } from "./feedback";
import { listArchiveDates, PATHS, readJson } from "./github";

// Types cover only the fields the page reads (data/daily_matches_schema.json 1.2).
export interface Match {
  posting_id: string;
  run_status: "new" | "carried_over" | "updated" | "released_from_hold";
  tier: "apply_first" | "worth_reviewing" | "stretch" | "tracking";
  rank: number;
  fit_score: number;
  score_breakdown: Record<"role_direction" | "requirements_evidence" | "seniority" | "preferences" | "sector_bonus" | "penalties" | "raw_total", number>;
  evidence_confidence: { level: "high" | "medium" | "low"; must_have_coverage: number; mean_claim_weight: number };
  tier_adjustments: string[];
  posting: {
    title: string; url: string; apply_url?: string | null; location_text: string | null;
    work_arrangement: string; employment_type: string; travel: { stated: boolean; percent_max: number | null };
    seniority: string; role_direction: string;
    pay: { status: string; floor_check: string; display_text: string | null };
    source: { type: string; ref: string }; posted_at: string | null; first_seen: string; last_checked: string; still_open: boolean;
  };
  company: { name: string; sector_tag: string; sector_tag_source: string; sector_band: string };
  explanation: {
    summary: string;
    strengths: { requirement: string; match_strength: "strong" | "partial"; claim_ids: string[]; note: string }[];
    gaps: { requirement: string; kind: string; blocking: boolean; blocking_kind?: string | null; note: string }[];
    open_questions: { oq_id: string; label: string }[];
    flags: string[];
  };
  cited_claim_ids: string[];
  feedback_snapshot: { status: Status; reviewer: string | null; updated_at: string | null };
}
export interface MatchesFile {
  schema_version: string;
  run: {
    run_id: string; run_date: string; completed_at: string; scoring_config_version: string;
    profile: { approval_status: string };
    model: { model: string; prompt_version: string };
    counts: Record<string, any>;
  };
  matches: Match[];
  tracking: Match[];
  removed: { posting_id: string; reason: string }[];
}

export interface CurrentFeedback {
  status: Status;
  reviewer: string | null;
  at: string | null;
  reason?: Reason;
  note?: string;
}

export const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

/** Claim ID → claim text from the sanitized profile, whatever its nesting. */
export function claimTexts(profile: unknown): Map<string, string> {
  const out = new Map<string, string>();
  const ID = /^(C-[A-Z]+-\d{2}|ED-\d{2}|CERT-\d{2}|LIC-\d{2}|EXP-\d{2})$/;
  const TEXT_KEYS = ["text", "claim", "claim_text", "statement", "summary", "label", "description"];
  const walk = (node: unknown, depth: number) => {
    if (depth > 8 || !node || typeof node !== "object") return;
    if (Array.isArray(node)) return node.forEach((n) => walk(n, depth + 1));
    const o = node as Record<string, unknown>;
    const id = [o.id, o.claim_id].find((v) => typeof v === "string" && ID.test(v)) as string | undefined;
    if (id) {
      const text = TEXT_KEYS.map((k) => o[k]).find((v) => typeof v === "string" && v.trim());
      if (typeof text === "string") out.set(id, text.trim());
    }
    for (const [k, v] of Object.entries(o)) {
      if (ID.test(k) && typeof v === "string") out.set(k, v.trim());
      else walk(v, depth + 1);
    }
  };
  walk(profile, 0);
  return out;
}

export async function loadDay(date: string | null) {
  const path = date ? PATHS.archive(date) : PATHS.latest;
  const [day, feedback, profile, archiveDates] = await Promise.all([
    readJson<MatchesFile>(path),
    readJson<FeedbackFile>(PATHS.feedback),
    readJson(PATHS.profile).catch(() => null), // claim text is a nice-to-have
    listArchiveDates().catch(() => [] as string[]),
  ]);
  const live = latestByPosting(Array.isArray(feedback?.events) ? feedback!.events : []);
  const current = new Map<string, CurrentFeedback>();
  if (day) {
    for (const m of [...(day.matches ?? []), ...(day.tracking ?? [])]) {
      const snap = m.feedback_snapshot;
      const ev = live.get(m.posting_id);
      if (ev && (!snap?.updated_at || ev.timestamp >= snap.updated_at)) {
        current.set(m.posting_id, { status: ev.status, reviewer: ev.reviewer, at: ev.timestamp, reason: ev.not_interested_reason, note: ev.reason_note });
      } else {
        current.set(m.posting_id, { status: snap?.status ?? "unreviewed", reviewer: snap?.reviewer ?? null, at: snap?.updated_at ?? null });
      }
    }
  }
  return { day, current, claims: claimTexts(profile), archiveDates };
}

/** posting_ids the pipeline has published on the latest list (feedback targets). */
export async function publishedIds(): Promise<Set<string>> {
  const latest = await readJson<MatchesFile>(PATHS.latest);
  const ids = new Set<string>();
  for (const m of [...(latest?.matches ?? []), ...(latest?.tracking ?? [])]) ids.add(m.posting_id);
  return ids;
}
