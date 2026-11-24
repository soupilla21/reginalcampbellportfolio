// Display labels for the /jobs page. Safe to import from client code (no secrets).
export const TIERS = {
  apply_first: { label: "Apply first", blurb: "Fit 80+ with medium or better evidence." },
  worth_reviewing: { label: "Worth reviewing", blurb: "Fit 65–79, or capped by a gap or low evidence." },
  stretch: { label: "Stretch", blurb: "Fit 50–64." },
  tracking: { label: "Tracking", blurb: "Marked Applied or Interview." },
} as const;
export type Tier = keyof typeof TIERS;

export const STATUS_LABELS: Record<string, string> = {
  unreviewed: "Not reviewed",
  interested: "Interested",
  not_interested: "Not interested",
  applied: "Applied",
  interview: "Interview",
};

export const DIRECTIONS: Record<string, string> = {
  airline_sales_account: "Airline sales / accounts",
  bd_partnerships: "BD & partnerships",
  multichannel_channel_sales: "Channel / multichannel sales",
  re_acquisitions: "Real estate acquisitions",
  other_sales_account: "Other sales / accounts",
  other_customer_facing: "Other customer-facing",
  non_target: "Outside target directions",
};

export const ARRANGEMENTS: Record<string, string> = {
  onsite: "On-site",
  hybrid: "Hybrid",
  remote: "Remote",
  field_territory: "Field / territory",
  not_stated: "Arrangement not stated",
};

export const SENIORITY: Record<string, string> = {
  entry: "Entry level",
  associate: "Associate",
  mid: "Mid level",
  senior_individual_contributor: "Senior IC",
  manager: "Manager",
  senior_manager: "Senior manager",
  director: "Director",
  vp_or_above: "VP or above",
  not_stated: "Seniority not stated",
};

export const FLAGS: Record<string, string> = {
  salary_not_listed: "Pay not listed",
  salary_unparseable: "Pay text unclear",
  floor_within_range: "Floor inside the pay range",
  ote_only: "OTE only",
  non_usd_pay: "Pay not in USD",
  employment_type_not_stated: "Full-time not stated",
  location_not_stated: "Location not stated",
  arrangement_not_stated: "Arrangement not stated",
  travel_not_stated: "Travel not stated",
  seniority_not_stated: "Seniority not stated",
  sector_ai_provisional: "Sector assigned by AI",
  formerly_licensed_only: "Formerly licensed only",
  supporting_claim_used: "Uses supporting-only evidence",
  client_attested_metric_cited: "Cites client-attested figures",
  requirements_not_extracted: "Requirements not extracted",
  profile_pending_approval: "Profile pending approval",
};

export const RUN_STATUS: Record<string, string> = {
  new: "New",
  updated: "Updated",
  released_from_hold: "From hold",
  carried_over: "",
};

export const ADJUSTMENTS: Record<string, string> = {
  confidence_gate_downgrade: "Moved down: low evidence confidence",
  blocking_gap_cap: "Capped: blocking gap",
};

export const BREAKDOWN: [string, string, number][] = [
  ["role_direction", "Role direction", 35],
  ["requirements_evidence", "Requirements evidence", 35],
  ["seniority", "Seniority", 10],
  ["preferences", "Preferences", 10],
  ["sector_bonus", "Sector bonus", 10],
  ["penalties", "Penalties", -20],
];

export const humanize = (s: string) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
