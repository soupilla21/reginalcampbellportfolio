// Content for /advisory/first-90-days: "First Decision by Day 90".
//
// A 90-day governance plan for technology enablement, shown through filled-in
// sample artifacts. Every artifact uses one illustrative portfolio (programs
// A, B and C) and one recurring conflict, so the numbers agree everywhere:
//
//   Integration engineering: 6 people, Weeks 10-15, 1,440 hours available.
//   Program B needs 960, Program C needs 720: a 240-hour shortfall.
//   Option 2 (phase scope): B 660 + C 600 = 1,260, a 180-hour buffer.
//   Contractor fallback: about $72K, 12% of C's $600K budget (over 10%).
//   Thresholds: 1 business day, 2 weeks, 10%, 24 hours.
//   B launches Week 16. C's audit fieldwork starts Week 18. B's deferred
//   scope ships by Week 20.
//
// Edit copy here; the page file is layout only. Keep the figures above in
// sync if anything changes (scripts/check-first-90-days.ts verifies them).

export type Status = "review" | "public";
export type Rag = "g" | "a" | "r";
export type RoleKey = "sponsor" | "delivery" | "resource" | "control" | "consumer";

export const meta = {
  slug: "first-90-days",
  // "review": built locally and on Vercel previews, never on production.
  // Flip to "public" to publish (see docs/advisory-briefings.md, same gate).
  status: "public" as Status,
  title: "First Decision by Day 90 | How I'd Run Technology Enablement Governance | Reginal Campbell",
  description:
    "A 90-day plan for governing technology enablement across functions: decision rights, an operating rhythm, and the first executive decision, shown through illustrative sample artifacts.",
  date: "November 2026",
};

export const files = {
  summary: "first-decision-by-day-90-summary.pdf",
  kit: "governance-starter-kit.pdf",
};

export const hero = {
  eyebrow: "90-day governance plan",
  title: "First Decision by Day 90",
  subtitle: "How I'd run technology enablement governance",
  hook: "Governance fails when it becomes a reporting factory. It works when it forces clear, accountable decisions. Here's how I'd get to the first one by Day 90.",
  lede: "A plan for governing technology enablement across functions, and the artifacts you'd receive from me along the way. Every example is filled in, so you can judge the work, not a description of it.",
};

export const principles = [
  { label: "Decisions over reporting", text: "Every forum exists to decide something. If it doesn't, it gets cut." },
  { label: "Objective status", text: "Colors come from thresholds, not from how the week felt." },
  { label: "Escalate before the slip", text: "Conflicts reach the people who can resolve them while there are still options." },
  { label: "Adoption is the finish line", text: "A system is done when people use it, not when it goes live." },
];

export const checkpoints = [
  { day: 30, label: "Steering 1", text: "Decision rights approved", href: "#day-30" },
  { day: 60, label: "Steering 2", text: "First cross-program trade-off", href: "#day-60" },
  { day: 90, label: "Executive Readout", text: "Proceed, pivot, slow or stop", href: "#day-90" },
];

export const programs = [
  { key: "A", name: "Collaboration platform migration", note: "Touches every department.", budget: "$1.2M", tier: "Tier 2", sponsor: "VP, Workplace Technology", stage: "Wave 2 of 3", rag: "a" as Rag },
  { key: "B", name: "Customer portal release", note: "Fixed public launch date: Week 16.", budget: "$900K", tier: "Tier 1", sponsor: "VP, Digital Experience", stage: "Build, launch Week 16", rag: "g" as Rag },
  { key: "C", name: "Identity and access consolidation", note: "Audit fieldwork starts Week 18.", budget: "$600K", tier: "Tier 1", sponsor: "Chief Information Security Officer", stage: "Cutover planning", rag: "a" as Rag },
];

// ── Phases ──
export const phases = [
  {
    id: "day-30",
    days: "Days 1–30",
    band: "Diagnose",
    title: "Diagnose and set decision rights",
    outcome: "Everyone knows who owns the yes and the no.",
    input: [
      "A 30-minute intake with each stakeholder, built on three questions: What decision is stuck? Which status report do you no longer trust? What would make governance worth your time?",
      "Current roadmaps, charters, status reports and standing meetings.",
      "Dependencies between programs, shared teams and vendor commitments.",
    ],
    mechanism: [
      "A portfolio inventory with sponsor, stage, budget exposure and risk tier.",
      "A decision rights matrix for scope, budget, schedule, risk acceptance and go-live.",
      "A forum design: which body decides what, who attends, and how often.",
    ],
    output: "Steering 1 approves the charter and decision rights. Decision Log #001.",
    gives: "One 30-minute conversation, then a one-page charter to react to. Not a template to fill in.",
  },
  {
    id: "day-60",
    days: "Days 31–60",
    band: "Operate",
    title: "Establish the operating rhythm",
    outcome: "One version of the truth, and status nobody has to argue about.",
    input: [
      "Every program's risks, issues and dependencies, moved into one shared log.",
      "The existing status formats, consolidated into one.",
    ],
    mechanism: [
      "RAG criteria set by thresholds, not judgment calls.",
      "Escalation thresholds published before anyone needs them.",
      "An issue-routing ladder, so each problem lands at the lowest level that can solve it.",
    ],
    output: "Steering 2 makes the first cross-program trade-off. Decision Log #005.",
    gives: "You know what's coming to your desk before it arrives, in the same format every time.",
  },
  {
    id: "day-90",
    days: "Days 61–90",
    band: "Decide",
    title: "Executive readout and adoption baseline",
    outcome: "Leadership makes the call with evidence, not optimism.",
    input: ["Gate evidence, go-live readiness and the first adoption numbers."],
    mechanism: [
      "A one-page executive view that opens with the decision needed.",
      "An enablement scorecard that keeps delivery, adoption and operations separate.",
      "Go-live readiness criteria, signed by the business owner, not IT.",
    ],
    output: "The Executive Readout makes one call per program: proceed, pivot, slow or stop. Decision Log #007.",
    gives: "A decision record, a baseline to measure against, and a governance model trimmed by a retro rather than expanded.",
  },
];

// ── Artifact 1: decision rights matrix ──
export const raciRoles = ["Exec Sponsor", "Steering", "TPM Director", "Program Lead", "Shared-Resource Owner", "Control Owner", "Business Owner"];
export const raci: { decision: string; cells: string[]; highlight?: boolean }[] = [
  { decision: "Scope change within one program", cells: ["I", "I", "C", "A/R", "C", "C", "C"] },
  { decision: "Scope change affecting another program", cells: ["I", "A", "R", "C", "C", "C", "C"] },
  { decision: "Budget change up to 10%", cells: ["I", "I", "A", "R", "", "", "C"] },
  { decision: "Budget change over 10%", cells: ["C", "A", "R", "C", "", "C", "I"] },
  { decision: "Schedule slip up to 2 weeks", cells: ["", "I", "A", "R", "C", "", "I"] },
  { decision: "Schedule slip over 2 weeks", cells: ["C", "A", "R", "C", "C", "", "C"] },
  { decision: "Shared resource conflict between programs", cells: ["I", "A", "R", "C", "C", "", "I"] },
  { decision: "Risk acceptance (security or compliance)", cells: ["C", "I", "R", "C", "", "A", "I"] },
  { decision: "Go-live approval", cells: ["I", "C", "R", "R", "", "C", "A"], highlight: true },
  { decision: "Pause or stop a program", cells: ["A", "R", "R", "C", "I", "C", "C"] },
];
export const raciNote = "Go-live belongs to the business owner, not IT. Delivery proves readiness; the business decides to launch.";

// ── Artifact 2: RAG definition card ──
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
  "Nothing goes red without going amber first.",
  "Every amber has an owner and a recovery date. An amber without both is reported as red.",
  "Status describes the forecast, not just today.",
  "The program lead reports the color; the criteria decide it. Anyone can challenge a color by pointing to the criteria.",
];

// ── Artifact 9: escalation thresholds ──
export const thresholds = [
  { trigger: "A shared team is unavailable in its committed window", limit: "Any", to: "TPM Director, then Steering", within: "1 business day" },
  { trigger: "A milestone is at risk", limit: "Over 2 weeks", to: "Steering", within: "Next steering, or called within 5 business days" },
  { trigger: "Budget variance", limit: "Over 10%", to: "Steering", within: "Next steering" },
  { trigger: "Security or compliance issue", limit: "Any", to: "Control Owner and Exec Sponsor", within: "24 hours" },
];
export const routing = [
  { level: "Team", decides: "Task-level fixes inside the plan" },
  { level: "Program lead", decides: "Anything inside the program's tolerances" },
  { level: "TPM Director", decides: "Slips up to 2 weeks, budget up to 10%, shared-calendar moves" },
  { level: "Steering", decides: "Anything outside tolerance or across programs" },
];

// ── Artifact 3: RAID log excerpt (as of Day 52) ──
export const raid: { id: string; type: string; prog: string; text: string; owner: string; impact: string; rag: Rag; next: string; key?: boolean }[] = [
  { id: "R-011", type: "Risk", prog: "A", text: "Pilot users not completing training before Wave 2 cutover", owner: "Business Owner, A", impact: "Adoption below target", rag: "a", next: "Add floor-walker support, Day 55" },
  { id: "D-004", type: "Dependency", prog: "A → C", text: "The collaboration suite needs C's single sign-on before Wave 2", owner: "Program Lead, C", impact: "A's Wave 2 slips if SSO slips", rag: "g", next: "Confirm SSO readiness date, Day 58" },
  { id: "A-006", type: "Assumption", prog: "C", text: "The SSO vendor's connector supports the two legacy finance apps", owner: "Program Lead, C", impact: "3 weeks of rework if wrong", rag: "a", next: "Vendor proof of concept, Day 57" },
  { id: "I-007", type: "Issue", prog: "B / C", text: "Integration engineering is committed to B (960 hrs) and C (720 hrs) in Weeks 10–15. Capacity is 1,440 hrs: a 240-hour shortfall.", owner: "TPM Director", impact: "B's launch date or C's audit readiness", rag: "r", next: "Steering 2 decision, Day 60", key: true },
  { id: "R-013", type: "Risk", prog: "C", text: "Audit fieldwork in Week 18 finds privileged accounts not yet cleaned up", owner: "Control Owner", impact: "Audit finding", rag: "a", next: "Tied to the outcome of I-007" },
  { id: "R-009", type: "Risk", prog: "B", text: "Payment provider certification takes longer than planned", owner: "Program Lead, B", impact: "1 to 2 week launch slip", rag: "g", next: "Certification submitted; track weekly" },
];
export const raidNote = "I-007 was logged as risk R-012 on Day 47, became an issue on Day 48 under the 1-business-day rule, and went to steering because the fix needs people from another program.";

// ── Artifact 4: Steering 2 pre-read ──
export const steering = {
  decision: "Choose how to resolve the integration engineering conflict between Program B and Program C in Weeks 10–15.",
  situation: [
    "The 6-person integration team is committed to Program B's launch (960 hours) and Program C's cutover (720 hours) in the same six weeks.",
    "Available capacity is 1,440 hours. The shortfall is 240 hours.",
    "B's Week 16 launch date has been announced to customers. C's privileged-account cleanup must finish before audit fieldwork starts in Week 18.",
  ],
  cause: "Both plans assumed the team's full availability. Neither could see the other's peak. (The shared RAID log is what surfaced it.)",
  capacity: { available: 1440, b: 960, c: 720, phasedB: 660, phasedC: 600 },
  options: [
    {
      n: 1, name: "Sequence", short: "Finish B first, then C.",
      hours: "B 960; C deferred", schedule: "B holds. C slips about 6 weeks.", cost: "None added",
      risk: "Audit finding likely: cleanup misses Week 18.", reversible: "Yes",
    },
    {
      n: 2, name: "Phase scope", short: "B launches with core scope; two non-critical integrations (notification preferences, analytics export) follow by Week 20. C does its audit-critical work first: privileged accounts and finance apps.",
      hours: "B 660 + C 600 = 1,260 (180-hour buffer)", schedule: "B's date holds. C's audit-critical work holds. The rest of C follows after Week 16.", cost: "None added",
      risk: "The business owner accepts a reduced launch scope.", reversible: "Yes. Deferred scope ships by Week 20.", recommended: true,
    },
    {
      n: 3, name: "Add capacity", short: "Two contract integration engineers for six weeks.",
      hours: "1,440 + about 480 contract hours", schedule: "Both hold, if onboarding takes under 2 weeks.", cost: "About $72K: 12% of C's budget, over the 10% threshold",
      risk: "Onboarding uses a third of the window. Knowledge leaves with the contractors.", reversible: "Partly. Spend is committed.",
    },
  ],
  recommendation: "Option 2. It protects both external commitments at no added cost and leaves a 180-hour buffer. Get contractor pricing now as a fallback, to decide by Day 74 if the buffer is used up.",
  asks: [
    { who: "Business Owner, B", ask: "Accept the reduced launch scope and the Week 20 follow-up." },
    { who: "Control Owner", ask: "Confirm the audit-critical scope matches what the auditors will test." },
    { who: "Steering", ask: "Approve Option 2 and the fallback trigger." },
  ],
  noDecision: "The team starts Week 10 double-booked. Work goes to whoever asks loudest, and both commitments are at risk.",
  meeting: [
    { min: "5 min", what: "Confirm the facts. No re-presenting the pre-read." },
    { min: "15 min", what: "Discuss the options and the trade-off each one accepts." },
    { min: "5 min", what: "Decide, name owners, set the revisit trigger." },
  ],
  after: [
    "Decision Log #005 recorded and sent within 24 hours.",
    "B's two deferred integrations get a Week 20 date and an owner.",
    "C re-baselined: audit-critical scope first. Status moves to amber with a recovery date.",
    "I-007 closed in the RAID log. Contractor fallback added to the Day 74 agenda.",
    "Intake for integration engineering reopens.",
  ],
};

// ── Artifact 5: executive one-pager (Day 90) ──
export const exec = {
  decision: "Approve Proceed for Program B's Week 16 launch with core scope. Approve Slow for Program A's Wave 3 until adoption reaches target.",
  programs: [
    { key: "A", rag: "a" as Rag, confidence: "Medium", trend: "→", line: "Delivered on time; adoption is short of target." },
    { key: "B", rag: "g" as Rag, confidence: "High", trend: "↑", line: "Go-live readiness criteria met. Deferred scope on track for Week 20." },
    { key: "C", rag: "g" as Rag, confidence: "Medium", trend: "↑", line: "Audit-critical scope 85% complete. On track for Week 18." },
  ],
  valueAtRisk: "If A's adoption stalls, the license consolidation savings planned for Wave 3 slip by about a quarter.",
  risks: [
    { text: "A's adoption: 70% of Wave 1–2 users active, against an 80% target.", owner: "Business Owner, A" },
    { text: "C's legacy connector proof of concept passed for one of two finance apps; the second needs a workaround.", owner: "Program Lead, C" },
    { text: "B's payment certification is on track; the fallback date is held.", owner: "Program Lead, B" },
  ],
  changed: "Option 2 was carried out. The contractor fallback wasn't needed and was closed on Day 74. C moved from amber to green.",
  scorecard: [
    { area: "Delivery", value: "9 of 10", text: "milestones on time; decision turnaround averages 3 days" },
    { area: "Adoption", value: "70%", text: "of A's users active, against an 80% target" },
    { area: "Operational", value: "2.1", text: "support tickets per 100 users on A, trending down" },
  ],
};

// ── Artifact 6: decision log ──
export const decisions = [
  { n: "001", day: 30, forum: "Steering 1", decision: "Approved the governance charter, decision rights and cadence.", by: "Steering", basis: "Stakeholder intake; portfolio inventory", revisit: "Governance retro at Day 90" },
  { n: "002", day: 30, forum: "Steering 1", decision: "Adopted the RAG criteria and escalation thresholds.", by: "Steering", basis: "Inconsistent status formats found in intake", revisit: "Two or more color disputes in a month" },
  { n: "003", day: 41, forum: "Program review", decision: "Moved A's Wave 2 by one week to finish training (within tolerance).", by: "TPM Director", basis: "R-011; slip under 2 weeks", revisit: "Training completion below 90%" },
  { n: "004", day: 48, forum: "TPM escalation", decision: "Paused new requests to integration engineering until Steering 2.", by: "TPM Director", basis: "I-007; 1-business-day rule", revisit: "Steering 2 decision" },
  { n: "005", day: 60, forum: "Steering 2", decision: "Approved Option 2: phased scope for B, audit-critical scope first for C.", by: "Steering, with B's business owner", basis: "Pre-read options analysis", revisit: "Buffer falls below 80 hours", key: true },
  { n: "006", day: 74, forum: "Steering", decision: "Contractor fallback not needed; closed.", by: "Steering", basis: "Buffer at 150 hours", revisit: "None" },
  { n: "007", day: 90, forum: "Executive Readout", decision: "Proceed: B launches Week 16. Slow: A's Wave 3 waits until adoption reaches 80%.", by: "Exec Sponsor and business owners", basis: "Readiness criteria; scorecard", revisit: "Adoption at 80% for two weeks running" },
];

// ── Lighter artifacts ──
export const forums = [
  { name: "Working team", cadence: "Weekly", decides: "Tasks and fixes inside the plan", members: "Program lead and team" },
  { name: "Program review", cadence: "Every two weeks", decides: "Changes inside tolerance; what goes to steering", members: "TPM Director, program leads, shared-resource owners" },
  { name: "Steering committee", cadence: "Monthly, or within 5 business days of a red", decides: "Cross-program trade-offs; anything outside tolerance", members: "Sponsors, control owners, business owners; named delegates" },
  { name: "Executive readout", cadence: "Quarterly (first one at Day 90)", decides: "Proceed, pivot, slow or stop; funding", members: "Exec sponsor and business owners" },
];

export const weeklyStatus = {
  program: "C. Identity and access consolidation",
  asOf: "Week 10 (Day 66)",
  overall: "a" as Rag,
  dims: [
    { dim: "Schedule", rag: "a" as Rag, note: "Re-baselined at Steering 2. Audit-critical scope due Week 17." },
    { dim: "Budget", rag: "g" as Rag, note: "2% under plan." },
    { dim: "Scope", rag: "a" as Rag, note: "Approved change: non-audit apps move after Week 16." },
    { dim: "Resources", rag: "g" as Rag, note: "Integration team allocated per Decision #005." },
    { dim: "Risk", rag: "a" as Rag, note: "R-013 (audit) and A-006 (connector) open, both owned." },
  ],
  recovery: "Owner: Program Lead, C. Back to green when privileged-account cleanup passes test, target Week 12.",
  done: "Privileged-account inventory complete. Finance app 1 connected to SSO.",
  next: "Finance app 2 connector proof of concept. Cleanup of the first 40% of privileged accounts.",
  needs: "None this week.",
};

export const scorecard = [
  { group: "Delivery", rows: [["Milestones on time", "9 of 10"], ["Average decision turnaround", "3 days"], ["Open reds", "0"]] },
  { group: "Adoption", rows: [["A: users active (Waves 1–2)", "70% (target 80%)"], ["A: training completed", "88%"], ["B: customer registrations", "Target set: 40% of eligible within 60 days of launch"]] },
  { group: "Operational", rows: [["A: tickets per 100 users", "2.1, trending down"], ["A: time to resolve priority tickets", "1.5 days"]] },
];

export const readiness = {
  program: "B. Customer portal release",
  asOf: "Day 88",
  items: [
    { item: "Acceptance criteria signed by the business owner", done: true },
    { item: "Payment provider certification passed", done: true },
    { item: "Support runbook published", done: true },
    { item: "Rollback rehearsed end to end", done: true },
    { item: "Support agents trained", done: true },
    { item: "Security review closed", done: true },
    { item: "Business owner go-live sign-off", done: false, note: "Requested at the Day 90 Executive Readout" },
  ],
};

// ── The gallery: all twelve artifacts ──
export const artifacts: {
  n: number; id: string; name: string; who: string; when: string; supports: string; roles: RoleKey[]; full: boolean; shownIn?: string;
}[] = [
  { n: 1, id: "art-raci", name: "Decision rights matrix", who: "Everyone in governance", when: "Set at Day 30; reviewed at the retro", supports: "Who owns each yes and no", roles: ["sponsor", "delivery", "resource", "control", "consumer"], full: true, shownIn: "#day-30" },
  { n: 2, id: "art-rag", name: "RAG definition card", who: "Everyone who reports or reads status", when: "Set at Day 30", supports: "What every color means", roles: ["sponsor", "delivery", "resource", "control", "consumer"], full: true, shownIn: "#day-60" },
  { n: 3, id: "art-raid", name: "RAID log", who: "Program leads, shared-resource and control owners", when: "Live; reviewed every two weeks", supports: "What needs escalating, and when", roles: ["delivery", "resource", "control"], full: true, shownIn: "#steering-2" },
  { n: 4, id: "art-preread", name: "Steering pre-read", who: "Steering members", when: "Three business days before each steering", supports: "The one decision on the agenda", roles: ["sponsor", "resource", "control", "consumer"], full: true, shownIn: "#steering-2" },
  { n: 5, id: "art-exec", name: "Executive one-pager", who: "Exec sponsor and business owners", when: "Each executive readout", supports: "Proceed, pivot, slow or stop", roles: ["sponsor", "consumer"], full: true, shownIn: "#day-90" },
  { n: 6, id: "art-log", name: "Decision log", who: "Everyone in governance", when: "Updated within 24 hours of each decision", supports: "What was decided, on what evidence, and when to revisit it", roles: ["sponsor", "delivery", "resource", "control", "consumer"], full: true, shownIn: "#day-90" },
  { n: 7, id: "art-inventory", name: "Portfolio inventory", who: "Sponsors and shared-resource owners", when: "Built by Day 30; refreshed monthly", supports: "Where money, risk and people are committed", roles: ["sponsor", "resource"], full: false },
  { n: 8, id: "art-forums", name: "Forum map", who: "Everyone in governance", when: "Set at Day 30", supports: "Which meeting decides what", roles: ["sponsor", "delivery", "resource", "control", "consumer"], full: false },
  { n: 9, id: "art-thresholds", name: "Escalation thresholds", who: "Program leads, shared-resource and control owners", when: "Published by Day 45", supports: "When an issue must move up, and how fast", roles: ["delivery", "resource", "control"], full: false },
  { n: 10, id: "art-status", name: "Weekly status report", who: "Program leads, shared-resource owners, business owners", when: "Weekly, same format for every program", supports: "What changed, and what's needed", roles: ["delivery", "resource", "consumer"], full: false },
  { n: 11, id: "art-scorecard", name: "Enablement scorecard", who: "Sponsors and business owners", when: "Monthly; baselined by Day 90", supports: "Whether the investment is being used", roles: ["sponsor", "consumer"], full: false },
  { n: 12, id: "art-readiness", name: "Go-live readiness checklist", who: "Business owners, control owners, program leads", when: "Two weeks before each go-live", supports: "Whether to launch", roles: ["delivery", "control", "consumer"], full: false },
];

// ── Your seat at the table: by governance role ──
export const roles: { key: RoleKey; name: string; give: string; get: string; decide: string }[] = [
  { key: "sponsor", name: "Sponsor", give: "Priorities and risk acceptance", get: "The executive one-pager and decision-ready options", decide: "Steering and the executive readout" },
  { key: "delivery", name: "Delivery owner", give: "Plans and honest status", get: "Clear thresholds and fast escalation paths", decide: "Inside your tolerances" },
  { key: "resource", name: "Shared-resource owner", give: "Capacity commitments", get: "Early warning of conflicts, and no surprise demands", decide: "Steering, when a conflict is escalated" },
  { key: "control", name: "Control owner", give: "Criteria and sign-off (security, compliance, finance)", get: "Evidence before go-live, not after", decide: "Readiness gates and risk acceptance" },
  { key: "consumer", name: "Business owner", give: "Adoption and feedback", get: "The readiness checklist and the adoption scorecard", decide: "Go-live sign-off" },
];

export const agreement = {
  commitments: [
    "Nothing goes red without going amber first.",
    "Risks reach the sponsor the day they cross a threshold.",
    "Decisions are logged and sent within 24 hours.",
    "No meeting without a decision on the agenda.",
    "One status format, with one RAG definition, for every program.",
  ],
  proposals: [
    "Decision turnaround times by severity, for example 48 hours on critical-path items.",
    "A named delegate who can decide when you can't attend.",
    "Pre-reads read before the meeting, so the meeting is spent deciding.",
  ],
  close: "A recommendation should change when the evidence changes. Accountability for delivery should not.",
};

export const disclosure =
  "All programs, people, figures and decisions on this page are illustrative. They show a method, not a description of any real organization or engagement. The plan is a starting hypothesis I would test against the real portfolio in the first 30 days.";
