// Content for /advisory/first-90-days: "90 Days to a Decision-Ready
// Technology Portfolio".
//
// A 90-day plan for technical program governance, set in an illustrative
// healthcare supply chain and spend portfolio (programs A, B and C). Every
// artifact uses one recurring conflict, so the numbers agree everywhere:
//
//   Integration engineering: 6 people, Weeks 10-15, 1,440 hours available.
//   Program B needs 960, Program C needs 720: a 240-hour shortfall.
//   Option 2 (phase scope): B 660 + C 600 = 1,260, a 180-hour buffer.
//   Contractor option: about 480 hours, about $72K, 12% of C's $600K budget.
//   Thresholds (Tier 1): 1 business day, 2 weeks, 10%, 24 hours.
//   B launches Week 16. C's audit fieldwork starts Week 18. B's deferred
//   scope ships by Week 20.
//
// Edit copy here; the page file is layout only. Keep the figures above in
// sync if anything changes (scripts/check-first-90-days.ts verifies them).
// Wording rules for this audience: status colors are green / yellow / red
// (never "RAG" or "amber"), and no em dashes.

export type Status = "review" | "unlisted" | "public";
export type Rag = "g" | "a" | "r";
export type RoleKey = "sponsor" | "delivery" | "resource" | "control" | "consumer";

export const meta = {
  slug: "first-90-days",
  // "review": local and Vercel previews only, never production.
  // "unlisted": live on production at its URL, but noindex/nofollow and out
  //   of the sitemap. Shared by link (here: with an interview panel).
  // "public": live, indexable, in the sitemap.
  status: "unlisted" as Status,
  title: "90 Days to a Decision-Ready Technology Portfolio | Reginal Campbell",
  description:
    "A 90-day plan for technical program governance in healthcare supply chain and spend technology: clear ownership, one material trade-off resolved, and an executive delivery baseline. Illustrative scenario.",
  date: "November 2026",
  preparedFor: "Prepared for the Technical Program Management, Director, Advisory panel",
};

export const files = {
  summary: "first-decision-by-day-90-summary.pdf",
  kit: "governance-starter-kit.pdf",
};

export const documents = [
  { key: "summary", file: files.summary, preview: "summary-preview.webp", title: "One-page summary", meta: "PDF · 1 page", text: "The plan on a single page: the 90-day outcomes, the Day 60 decision, the delivery path and the working agreement." },
  { key: "kit", file: files.kit, preview: "kit-preview.webp", title: "Governance Starter Kit", meta: "PDF · 7 pages", text: "The six core artifacts, filled in: decision rights, stoplight status card, RAID log, steering pre-read, executive one-pager and decision log." },
];

// ── Opening ──
export const hero = {
  eyebrow: "90-day plan",
  title: "90 Days to a Decision-Ready Technology Portfolio",
  subtitle: "From competing engineering priorities to accountable decisions, credible roadmaps and measurable delivery.",
  outcomes: [
    { day: 30, word: "Clarity", text: "Portfolio, owners and dependencies mapped", href: "#plan" },
    { day: 60, word: "Decisions", text: "A material trade-off resolved", href: "#decision" },
    { day: 90, word: "Confidence", text: "Executive delivery baseline established", href: "#success" },
  ],
  decisionTeaser: {
    title: "The decision I'd put in front of leadership",
    stats: [
      { value: "240 hrs", label: "Illustrative capacity shortfall" },
      { value: "180 hrs", label: "Remaining buffer after the recommended trade-off" },
    ],
  },
  scenarioNote: "Illustrative scenario set in healthcare supply chain and spend technology. Informed by public information about the industry; not actual Vizient initiatives.",
};

export const routes = [
  { href: "#decision", label: "Executive view", time: "About 3 minutes", text: "The Day 60 decision, the 90-day plan, the delivery path, what success looks like, and evidence from my work." },
  { href: "#operating-model", label: "Explore the operating model", time: "Optional deep dive", text: "Decision rights, stoplight status, RAID log, steering pre-read, decision log, all twelve artifacts and the PDFs." },
];

export const principles = [
  { label: "Decisions over reporting", text: "Every forum exists to decide something. If it doesn't, it gets cut." },
  { label: "Objective status", text: "Colors come from thresholds, adjusted for risk tier, not from how the week felt." },
  { label: "Escalate at the first credible warning", text: "Conflicts reach the people who can resolve them while there are still options." },
  { label: "Adoption is the finish line", text: "A capability is done when people use it, not when it ships." },
];

// ── The illustrative portfolio ──
export const programs = [
  { key: "A", name: "AI-assisted sourcing pilot", note: "Recommends savings opportunities to sourcing analysts. Adoption is the test.", budget: "$1.2M", tier: "Tier 2", sponsor: "VP, Sourcing Solutions", stage: "Pilot wave 2 of 3", rag: "a" as Rag },
  { key: "B", name: "Spend analytics release", note: "Launch date committed to customers: Week 16.", budget: "$900K", tier: "Tier 1", sponsor: "VP, Analytics Products", stage: "Build, launch Week 16", rag: "g" as Rag },
  { key: "C", name: "Supplier and contract data platform", note: "Feeds A and B. Audit fieldwork starts Week 18.", budget: "$600K", tier: "Tier 1", sponsor: "VP, Data Platform", stage: "Migration and controls", rag: "a" as Rag },
];
export const portfolioNote = "One six-person integration engineering team serves B and C. That's where the trouble starts.";

// ── Roles (moved near the top) ──
export const roles: { key: RoleKey; name: string; examples: string; give: string; get: string; decide: string }[] = [
  { key: "sponsor", name: "Sponsor", examples: "Product, portfolio or business leaders", give: "Priorities and risk acceptance", get: "The executive one-pager and decision-ready options", decide: "Steering and the executive readout" },
  { key: "delivery", name: "Delivery owner", examples: "Program and engineering leads", give: "Plans and honest forecasts", get: "Clear thresholds and fast escalation paths", decide: "Inside your tolerances" },
  { key: "resource", name: "Shared-resource owner", examples: "Engineering, data platform, architecture", give: "Capacity commitments", get: "Early warning of conflicts, and no surprise demands", decide: "Steering, when a conflict is escalated" },
  { key: "control", name: "Control owner", examples: "Security, privacy, compliance, finance", give: "Criteria and sign-off", get: "Evidence before release, not after", decide: "Release gates and risk acceptance" },
  { key: "consumer", name: "Business owner", examples: "The leader accountable for adoption", give: "Adoption and feedback", get: "The readiness checklist and the adoption scorecard", decide: "Go-live sign-off" },
];

// ── The Day 60 decision (interactive) ──
export const steering = {
  question: "Two programs. One integration team. Capacity for 1,440 hours against 1,680 hours of demand. What would you approve?",
  decision: "Choose how to resolve the integration engineering conflict between Program B and Program C in Weeks 10–15.",
  situation: [
    "The 6-person integration team is committed to Program B's launch (960 hours) and Program C's controls and migration work (720 hours) in the same six weeks.",
    "Available capacity is 1,440 hours. The shortfall is 240 hours.",
    "B's Week 16 launch date is committed to customers. C's access controls and data lineage for contract-pricing data must be in place before audit fieldwork starts in Week 18.",
  ],
  cause: "Both plans assumed the team's full availability. Neither could see the other's peak. The shared RAID log is what surfaced it, on Day 47.",
  capacity: { available: 1440, b: 960, c: 720, phasedB: 660, phasedC: 600, contract: 480 },
  options: [
    {
      n: 1, key: "sequence", name: "Sequence delivery", tagline: "Protect B, accept C's delay",
      short: "Finish B first, then C.",
      bars: { b: 960, c: 0, capacity: 1440 },
      hours: "B 960; C deferred", schedule: "B holds. C slips about 6 weeks.", cost: "None added",
      risk: "Audit finding likely: C's controls miss Week 18. A's pilot also waits on C's data.",
      approvers: "Control owner must accept the audit exposure in writing.",
      fallback: "None that recovers the audit date. Once fieldwork starts, the finding stands.",
      reversible: "Yes, but the audit consequence is not.",
      consequence: "Fast and cheap, but it trades a customer commitment for a compliance one. Few control owners would sign it.",
    },
    {
      n: 2, key: "phase", name: "Phase scope", tagline: "Preserve both critical commitments",
      short: "B launches with core scope; two non-critical features (scheduled exports, benchmark drill-down) follow by Week 20. C does its audit-critical work first: access controls and lineage for contract-pricing data.",
      bars: { b: 660, c: 600, capacity: 1440 },
      hours: "B 660 + C 600 = 1,260 (180-hour buffer)", schedule: "B's date holds. C's audit-critical work holds. The rest of C follows after Week 16.", cost: "None added",
      risk: "The business owner launches with less scope. Customers get two features four weeks later.",
      approvers: "B's business owner accepts the reduced launch scope. The control owner confirms the audit-critical scope matches what auditors will test.",
      fallback: "If the buffer falls below 80 hours, steering decides on contractor capacity by Day 74.",
      reversible: "Yes. Deferred scope ships by Week 20.",
      consequence: "Both external commitments hold at no added cost. The price is a smaller launch, decided by the person who owns the launch.",
      recommended: true,
    },
    {
      n: 3, key: "capacity", name: "Add capacity", tagline: "Authorize contractor spending",
      short: "Two contract integration engineers for six weeks.",
      bars: { b: 960, c: 720, capacity: 1920 },
      hours: "1,440 + about 480 contract hours", schedule: "Both hold, if onboarding takes under 2 weeks.", cost: "About $72K: 12% of C's budget, over the 10% threshold",
      risk: "Onboarding uses a third of the window. Knowledge of the contract-data model leaves with the contractors.",
      approvers: "Steering, because the spend is over the 10% threshold.",
      fallback: "If onboarding runs past 2 weeks, fall back to phasing scope.",
      reversible: "Partly. Spend is committed.",
      consequence: "Keeps every feature on schedule on paper, but pays $72K for hours that arrive late in the window.",
    },
  ],
  assumptions: [
    "Hour estimates are within plus or minus 10%.",
    "Contractor onboarding takes about two weeks.",
    "The control owner confirms which controls the auditors will test.",
  ],
  recommendation: "Option 2. It protects both external commitments at no added cost and leaves a 180-hour buffer. I'd get contractor pricing now as a fallback, to decide by Day 74 if the buffer is used up.",
  asks: [
    { who: "Business Owner, B", ask: "Accept the reduced launch scope and the Week 20 follow-up." },
    { who: "Control Owner", ask: "Confirm the audit-critical scope matches what the auditors will test." },
    { who: "Steering", ask: "Approve Option 2 and the fallback trigger." },
  ],
  noDecision: "The team starts Week 10 double-booked. Work goes to whoever asks loudest, and both commitments are at risk.",
  after: [
    "Decision Log #005 recorded and sent within 24 hours.",
    "B's two deferred features get a Week 20 date and an owner.",
    "C re-baselined: audit-critical scope first. Its schedule goes to yellow with a recovery date.",
    "I-007 closed in the RAID log. Contractor fallback added to the Day 74 agenda.",
  ],
};

// ── The 90-day plan (summary in the executive view, detail in the deep dive) ──
export const phases = [
  {
    id: "day-30", days: "Days 1–30", band: "Clarity", title: "Map the portfolio and set decision rights",
    outcome: "Everyone knows who owns the yes and the no.",
    input: [
      "A 30-minute intake with each stakeholder: What decision is stuck? Which status report do you no longer trust? What would make governance worth your time?",
      "Current roadmaps, architecture and release plans, status reports and standing meetings.",
      "Dependencies between programs, shared engineering and data teams, and vendors.",
    ],
    mechanism: [
      "A portfolio inventory with sponsor, stage, budget exposure and risk tier.",
      "A dependency map from data to release, with an owner on every node.",
      "A decision rights matrix for scope, budget, schedule, risk acceptance and go-live.",
    ],
    output: "Steering 1 approves the charter and decision rights. Decision Log #001.",
    gives: "One 30-minute conversation, then a one-page charter to react to. Not a template to fill in.",
  },
  {
    id: "day-60", days: "Days 31–60", band: "Decisions", title: "Establish the operating rhythm",
    outcome: "One version of the truth, and status nobody has to argue about.",
    input: [
      "Every program's risks, issues and dependencies, moved into one shared log.",
      "The existing status formats, consolidated into one.",
    ],
    mechanism: [
      "Stoplight status (green, yellow, red) set by thresholds and adjusted for risk tier.",
      "Escalation thresholds published before anyone needs them.",
      "An issue-routing ladder, so each problem lands at the lowest level that can solve it.",
    ],
    output: "Steering 2 makes the first cross-program trade-off. Decision Log #005.",
    gives: "You know what's coming to your desk before it arrives, in the same format every time.",
  },
  {
    id: "day-90", days: "Days 61–90", band: "Confidence", title: "Executive readout and delivery baseline",
    outcome: "Leadership makes the call with evidence, not optimism.",
    input: ["Gate evidence, release readiness, engineering delivery indicators and the first adoption numbers."],
    mechanism: [
      "A one-page executive view that opens with the decision needed.",
      "A scorecard that keeps delivery, adoption and operations separate.",
      "Release readiness criteria, signed by the business owner, not IT.",
    ],
    output: "The Executive Readout makes one call per program: proceed, pivot, slow or stop. Decision Log #007.",
    gives: "A decision record, a baseline to measure against, and a governance model trimmed by a retro rather than expanded.",
  },
];

// ── Technical delivery path ──
export const deliveryPath = [
  {
    key: "data", name: "Data contracts and ingestion quality", owner: "Data platform lead (Program C)",
    dependsOn: "Source systems for supplier, contract and invoice data",
    done: "Every source feed has a data contract; 99% of records pass validation; lineage is recorded for contract-pricing data.",
    slips: "Analytics and AI validation can't start on trusted data. The slip moves every node downstream, so it goes on the critical-path report the same day.",
  },
  {
    key: "validate", name: "Analytics and AI capability validation", owner: "Analytics product lead (B) and sourcing product lead (A)",
    dependsOn: "Data contracts and ingestion quality",
    done: "Spend figures reconcile to source contracts within tolerance; AI sourcing recommendations meet the evaluation threshold on held-out data and pass analyst review.",
    slips: "Release dates hold only by cutting scope, never by skipping validation. The trade-off goes to the business owner.",
  },
  {
    key: "gates", name: "Architecture, security and responsible AI gates", owner: "Architecture lead and control owner (security, privacy, compliance)",
    dependsOn: "Validation evidence",
    done: "Architecture review closed; access controls tested; for AI: model evaluation, data provenance, human review of recommended actions, monitoring after release, and a release approval sized to the risk.",
    slips: "No release without the gate. A control failure goes red immediately and to the control owner and sponsor within 24 hours.",
    ai: true,
  },
  {
    key: "release", name: "Product release and adoption measurement", owner: "Business owner, with the program lead",
    dependsOn: "Gates passed; support runbook and rollback ready",
    done: "Readiness checklist signed by the business owner; adoption target set before launch and measured from week one.",
    slips: "Launch with core scope, or slow the rollout. Adoption below target triggers a Slow call at the readout, not a celebration.",
  },
];

export const indicators = [
  { name: "Critical-path exposure", value: "2 of 5", text: "Wave 1 milestones on the critical path depend on integration engineering." },
  { name: "Dependency aging", value: "11 days", text: "D-004 has been open 11 days against a 10-day limit, so it's on this week's agenda." },
  { name: "Release predictability", value: "9 of 10", text: "Milestones delivered within 5 business days of forecast." },
  { name: "Platform reliability", value: "99.9%", text: "Data refresh jobs completed on time over the last 30 days." },
  { name: "Escalation to decision", value: "9 bus. days", text: "I-007: escalated Day 48, decided Day 60. Within the Tier 1 target of 10 business days." },
];

// ── Artifact 1: decision rights matrix ──
export const raciRoles = ["Exec Sponsor", "Steering", "TPM Director", "Program Lead", "Shared-Resource Owner", "Control Owner", "Business Owner"];
export const raci: { decision: string; cells: string[]; highlight?: boolean }[] = [
  { decision: "Scope change within one program", cells: ["I", "I", "C", "A/R", "C", "C", "C"] },
  { decision: "Scope change affecting another program", cells: ["I", "A", "R", "C", "C", "C", "C"] },
  { decision: "Budget change within tolerance", cells: ["I", "I", "A", "R", "", "", "C"] },
  { decision: "Budget change over tolerance", cells: ["C", "A", "R", "C", "", "C", "I"] },
  { decision: "Schedule slip within tolerance", cells: ["", "I", "A", "R", "C", "", "I"] },
  { decision: "Schedule slip over tolerance", cells: ["C", "A", "R", "C", "C", "", "C"] },
  { decision: "Shared resource conflict between programs", cells: ["I", "A", "R", "C", "C", "", "I"] },
  { decision: "Risk acceptance (security, privacy, compliance)", cells: ["C", "I", "R", "C", "", "A", "I"] },
  { decision: "Go-live approval", cells: ["I", "C", "R", "R", "", "C", "A"], highlight: true },
  { decision: "Pause or stop a program", cells: ["A", "R", "R", "C", "I", "C", "C"] },
];
export const raciNote = "Go-live belongs to the business owner, not IT. Delivery proves readiness; the business decides to launch.";

// ── Artifact 2: stoplight status card ──
export const tierNote = "Thresholds shown are for Tier 1 programs. Tier 2 tolerances are twice as wide. Security, privacy and compliance issues have no tolerance at any tier.";
export const ragRows: { dim: string; g: string; a: string; r: string }[] = [
  { dim: "Schedule", g: "Milestones within 5 business days of plan", a: "Forecast slip of 6 days to 2 weeks", r: "Forecast slip over 2 weeks" },
  { dim: "Budget", g: "Within 5% of plan", a: "5% to 10% variance", r: "Over 10% variance" },
  { dim: "Scope", g: "Baseline unchanged, or changes approved", a: "A change is pending a decision inside the program", r: "A change affects another program or the business outcome" },
  { dim: "Resources", g: "Committed people available in their window", a: "A gap the program can cover itself", r: "A gap that needs people from another program" },
  { dim: "Risk", g: "No open high-severity risks", a: "A high-severity risk with an owner and a mitigation", r: "No viable mitigation, or a control failure" },
];
export const ragMeaning = {
  g: "Proceed.",
  a: "Recoverable inside the program. A named owner has a recovery date.",
  r: "Recovery needs a decision at steering.",
};
export const ragRules = [
  "No surprise reds: material risks are escalated at the earliest credible warning.",
  "Critical control failures and security incidents go red immediately, whether or not they were yellow first.",
  "Every yellow has an owner and a recovery date. A yellow without both is reported as red.",
  "Status describes the forecast, not just today. Anyone can challenge a color by pointing to the criteria.",
];

// ── Escalation thresholds (Tier 1) ──
export const thresholds = [
  { trigger: "A shared team is unavailable in its committed window", limit: "Any", to: "TPM Director, then Steering", within: "1 business day" },
  { trigger: "A milestone is at risk", limit: "Over 2 weeks", to: "Steering", within: "Next steering, or called within 5 business days" },
  { trigger: "Budget variance", limit: "Over 10%", to: "Steering", within: "Next steering" },
  { trigger: "Security, privacy or compliance issue", limit: "Any", to: "Control Owner and Exec Sponsor", within: "24 hours" },
];
export const routing = [
  { level: "Team", decides: "Task-level fixes inside the plan" },
  { level: "Program lead", decides: "Anything inside the program's tolerances" },
  { level: "TPM Director", decides: "Slips and budget changes within tolerance; shared-calendar moves" },
  { level: "Steering", decides: "Anything outside tolerance or across programs" },
];

// ── Artifact 3: RAID log excerpt (as of Day 52) ──
export const raid: { id: string; type: string; prog: string; text: string; owner: string; impact: string; rag: Rag; next: string; key?: boolean }[] = [
  { id: "R-011", type: "Risk", prog: "A", text: "Sourcing analysts not completing training before pilot wave 2", owner: "Business Owner, A", impact: "Adoption below target", rag: "a", next: "Add side-by-side support sessions, Day 55" },
  { id: "D-004", type: "Dependency", prog: "A → C", text: "A's sourcing recommendations need C's normalized supplier data before pilot wave 2", owner: "Program Lead, C", impact: "A's wave 2 slips if C's data slips", rag: "a", next: "Confirm data readiness date, Day 58" },
  { id: "A-006", type: "Assumption", prog: "C", text: "Contract feeds from the two legacy source systems map cleanly to the new data contracts", owner: "Program Lead, C", impact: "3 weeks of rework if wrong", rag: "a", next: "Mapping proof of concept, Day 57" },
  { id: "I-007", type: "Issue", prog: "B / C", text: "Integration engineering is committed to B (960 hrs) and C (720 hrs) in Weeks 10–15. Capacity is 1,440 hrs: a 240-hour shortfall.", owner: "TPM Director", impact: "B's launch date or C's audit readiness", rag: "r", next: "Steering 2 decision, Day 60", key: true },
  { id: "R-013", type: "Risk", prog: "C", text: "Audit fieldwork in Week 18 finds contract-pricing data without complete access controls or lineage", owner: "Control Owner", impact: "Audit finding", rag: "a", next: "Tied to the outcome of I-007" },
  { id: "R-009", type: "Risk", prog: "B", text: "Reconciling spend figures to source contracts takes longer than planned", owner: "Program Lead, B", impact: "1 to 2 week launch slip", rag: "g", next: "Reconciliation underway; track weekly" },
];
export const raidNote = "I-007 was logged as risk R-012 on Day 47, became an issue on Day 48 under the 1-business-day rule, and went to steering because the fix needs people from another program.";

// ── Artifact 5: executive one-pager (Day 90) ──
export const exec = {
  decision: "Approve Proceed for Program B's Week 16 launch with core scope. Approve Slow for Program A's pilot wave 3 until adoption reaches target.",
  programs: [
    { key: "A", rag: "a" as Rag, confidence: "Medium", trend: "→", line: "Delivered on time; analyst adoption is short of target." },
    { key: "B", rag: "g" as Rag, confidence: "High", trend: "↑", line: "Release readiness criteria met. Deferred features on track for Week 20." },
    { key: "C", rag: "g" as Rag, confidence: "Medium", trend: "↑", line: "Audit-critical controls 85% complete. On track for Week 18." },
  ],
  valueAtRisk: "If A's adoption stalls, the sourcing savings the pilot identified won't be captured at scale for about a quarter.",
  risks: [
    { text: "A's adoption: 70% of wave 1–2 analysts active, against an 80% target.", owner: "Business Owner, A" },
    { text: "C's legacy feed mapping passed for one of two source systems; the second needs a workaround.", owner: "Program Lead, C" },
    { text: "B's spend reconciliation is on track; the fallback date is held.", owner: "Program Lead, B" },
  ],
  changed: "Option 2 was carried out. The contractor fallback wasn't needed and was closed on Day 74. C moved from yellow to green.",
  scorecard: [
    { area: "Delivery", value: "9 of 10", text: "milestones on time; decision turnaround averages 3 days" },
    { area: "Adoption", value: "70%", text: "of A's analysts active, against an 80% target" },
    { area: "Operational", value: "99.9%", text: "data refreshes on time; support tickets trending down" },
  ],
};

// ── Artifact 6: decision log ──
export const decisions = [
  { n: "001", day: 30, forum: "Steering 1", decision: "Approved the governance charter, decision rights and cadence.", by: "Steering", basis: "Stakeholder intake; portfolio inventory", revisit: "Governance retro at Day 90" },
  { n: "002", day: 30, forum: "Steering 1", decision: "Adopted the stoplight status criteria, tiered by risk, and escalation thresholds.", by: "Steering", basis: "Inconsistent status formats found in intake", revisit: "Two or more color disputes in a month" },
  { n: "003", day: 41, forum: "Program review", decision: "Moved A's pilot wave 2 by one week to finish analyst training (within tolerance).", by: "TPM Director", basis: "R-011; slip under 2 weeks", revisit: "Training completion below 90%" },
  { n: "004", day: 48, forum: "TPM escalation", decision: "Paused new requests to integration engineering until Steering 2.", by: "TPM Director", basis: "I-007; 1-business-day rule", revisit: "Steering 2 decision" },
  { n: "005", day: 60, forum: "Steering 2", decision: "Approved Option 2: phased scope for B, audit-critical scope first for C.", by: "Steering, with B's business owner", basis: "Pre-read options analysis", revisit: "Buffer falls below 80 hours", key: true },
  { n: "006", day: 74, forum: "Steering", decision: "Contractor fallback not needed; closed.", by: "Steering", basis: "Buffer at 150 hours", revisit: "None" },
  { n: "007", day: 90, forum: "Executive Readout", decision: "Proceed: B launches Week 16. Slow: A's pilot wave 3 waits until adoption reaches 80%.", by: "Exec Sponsor and business owners", basis: "Readiness criteria; scorecard", revisit: "Adoption at 80% for two weeks running" },
];

// ── Lighter artifacts ──
export const forums = [
  { name: "Working team", cadence: "Weekly", decides: "Tasks and fixes inside the plan", members: "Program lead and team" },
  { name: "Program review", cadence: "Every two weeks", decides: "Changes inside tolerance; what goes to steering", members: "TPM Director, program leads, shared-resource owners" },
  { name: "Steering committee", cadence: "Monthly, or within 5 business days of a red", decides: "Cross-program trade-offs; anything outside tolerance", members: "Sponsors, control owners, business owners; named delegates" },
  { name: "Executive readout", cadence: "Quarterly (first one at Day 90)", decides: "Proceed, pivot, slow or stop; funding", members: "Exec sponsor and business owners" },
];

export const weeklyStatus = {
  program: "C. Supplier and contract data platform",
  asOf: "Week 10 (Day 66)",
  overall: "a" as Rag,
  dims: [
    { dim: "Schedule", rag: "a" as Rag, note: "Re-baselined at Steering 2. Audit-critical controls due Week 17." },
    { dim: "Budget", rag: "g" as Rag, note: "2% under plan." },
    { dim: "Scope", rag: "g" as Rag, note: "Change approved at Steering 2 (Decision #005): non-audit feed migrations move after Week 16." },
    { dim: "Resources", rag: "g" as Rag, note: "Integration team allocated per Decision #005." },
    { dim: "Risk", rag: "a" as Rag, note: "R-013 (audit) and A-006 (feed mapping) open, both owned." },
  ],
  recovery: "Owner: Program Lead, C. Back to green when contract-pricing access controls and lineage pass testing, target Week 12.",
  done: "Supplier data contracts published. Source system 1 mapped and validated.",
  next: "Source system 2 mapping proof of concept. Access-control testing on contract-pricing data.",
  needs: "None this week.",
};

export const scorecard = [
  { group: "Delivery", rows: [["Milestones on time", "9 of 10"], ["Average decision turnaround", "3 days"], ["Open reds", "0"]] },
  { group: "Adoption", rows: [["A: analysts active (waves 1–2)", "70% (target 80%)"], ["A: training completed", "88%"], ["B: customer adoption", "Target set: 40% of eligible users within 60 days of launch"]] },
  { group: "Operational", rows: [["Data refreshes on time", "99.9%"], ["A: support tickets per 100 users", "2.1, trending down"]] },
];

export const readiness = {
  program: "B. Spend analytics release",
  asOf: "Day 88",
  items: [
    { item: "Acceptance criteria signed by the business owner", done: true },
    { item: "Spend figures reconciled to source contracts", done: true },
    { item: "Support runbook published", done: true },
    { item: "Rollback rehearsed end to end", done: true },
    { item: "Customer success team trained", done: true },
    { item: "Security and privacy review closed", done: true },
    { item: "Business owner go-live sign-off", done: false, note: "Requested at the Day 90 Executive Readout" },
  ],
};

// ── The gallery: all twelve artifacts ──
export const artifacts: { n: number; id: string; name: string; who: string; when: string; supports: string; roles: RoleKey[] }[] = [
  { n: 1, id: "art-raci", name: "Decision rights matrix", who: "Everyone in governance", when: "Set at Day 30; reviewed at the retro", supports: "Who owns each yes and no", roles: ["sponsor", "delivery", "resource", "control", "consumer"] },
  { n: 2, id: "art-rag", name: "Stoplight status card", who: "Everyone who reports or reads status", when: "Set at Day 30", supports: "What every color means", roles: ["sponsor", "delivery", "resource", "control", "consumer"] },
  { n: 3, id: "art-raid", name: "RAID log", who: "Program leads, shared-resource and control owners", when: "Live; reviewed every two weeks", supports: "What needs escalating, and when", roles: ["delivery", "resource", "control"] },
  { n: 4, id: "art-preread", name: "Steering pre-read", who: "Steering members", when: "Three business days before each steering", supports: "The one decision on the agenda", roles: ["sponsor", "resource", "control", "consumer"] },
  { n: 5, id: "art-exec", name: "Executive one-pager", who: "Exec sponsor and business owners", when: "Each executive readout", supports: "Proceed, pivot, slow or stop", roles: ["sponsor", "consumer"] },
  { n: 6, id: "art-log", name: "Decision log", who: "Everyone in governance", when: "Updated within 24 hours of each decision", supports: "What was decided, on what evidence, and when to revisit it", roles: ["sponsor", "delivery", "resource", "control", "consumer"] },
  { n: 7, id: "art-inventory", name: "Portfolio inventory", who: "Sponsors and shared-resource owners", when: "Built by Day 30; refreshed monthly", supports: "Where money, risk and people are committed", roles: ["sponsor", "resource"] },
  { n: 8, id: "art-forums", name: "Forum map", who: "Everyone in governance", when: "Set at Day 30", supports: "Which meeting decides what", roles: ["sponsor", "delivery", "resource", "control", "consumer"] },
  { n: 9, id: "art-thresholds", name: "Escalation thresholds", who: "Program leads, shared-resource and control owners", when: "Published by Day 45", supports: "When an issue must move up, and how fast", roles: ["delivery", "resource", "control"] },
  { n: 10, id: "art-status", name: "Weekly status report", who: "Program leads, shared-resource owners, business owners", when: "Weekly, same format for every program", supports: "What changed, and what's needed", roles: ["delivery", "resource", "consumer"] },
  { n: 11, id: "art-scorecard", name: "Delivery and adoption scorecard", who: "Sponsors and business owners", when: "Monthly; baselined by Day 90", supports: "Whether the investment is being used", roles: ["sponsor", "consumer"] },
  { n: 12, id: "art-readiness", name: "Release readiness checklist", who: "Business owners, control owners, program leads", when: "Two weeks before each release", supports: "Whether to launch", roles: ["delivery", "control", "consumer"] },
];

// ── Evidence from my work (facts only from the published case study) ──
export const evidence = {
  title: "National healthcare data exchange",
  org: "Accenture Federal Services · 2024 to 2026",
  role: "Product Manager and Technical Program Lead",
  challenge: "A national healthcare data exchange platform processing more than 2 billion transactions a month. At that volume, the hard problems weren't in any single feature. They were in keeping engineering, cloud, security, compliance and product teams moving in the same direction while the platform kept growing.",
  did: [
    "Client-facing delivery lead, responsible for executive decision support, operating cadence and portfolio-level planning.",
    "Aligned quarterly planning, investment prioritization and feature delivery across engineering, cloud, security, compliance and product teams.",
  ],
  results: [
    { value: "2B+", label: "transactions a month on the platform" },
    { value: "+30%", label: "user satisfaction, through standardized data definitions and governance" },
    { value: "~7%", label: "month-over-month expansion, sustained" },
  ],
  link: "This page is the same operating model, applied to a new portfolio: shared definitions treated as controls, a cadence that produces decisions, and planning that keeps cross-functional teams pointed the same way.",
  href: "/case-studies/national-healthcare-data-exchange",
};

export const agreement = {
  commitments: [
    "Risks are raised at the earliest credible warning. Control failures are reported immediately.",
    "Decisions are logged and sent within 24 hours.",
    "No meeting without a decision on the agenda.",
    "One status format and one set of color definitions for every program.",
    "Forecasts I can defend: I'll challenge a delivery date before I report it.",
  ],
  proposals: [
    "Decision turnaround times by severity, for example 48 hours on critical-path items.",
    "A named delegate who can decide when you can't attend.",
    "Pre-reads read before the meeting, so the meeting is spent deciding.",
  ],
  close: "A recommendation should change when the evidence changes. Accountability for delivery should not.",
  invitation: "This is the approach I'd bring. The first thing I'd want to validate with the team is where the actual constraints sit.",
};

export const disclosure =
  "All programs, people, figures and decisions in the scenario are illustrative. It is set in healthcare supply chain and spend technology and informed by public information about the industry; it does not describe actual Vizient initiatives, systems or plans. The plan is a starting hypothesis I would test against the real portfolio in the first 30 days. The evidence section describes my own work and is drawn from my published case study.";
