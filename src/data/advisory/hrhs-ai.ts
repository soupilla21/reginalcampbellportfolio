// HRHS Enterprise AI advisory briefing: manifest and narrative content.
//
// SOURCE OF TRUTH: HRHS_AI_Advisory_Package_v2.1 (November 2026). Every
// figure below is copied from the package, which labels it simulated [S],
// estimated [E], assumed [A] or judgment [J]. Nothing here is a real
// engagement result. Do not add figures that are not in the package.
//
// Copy rules for this file: no em dashes; "scenario" or "case" wording
// whenever a fictional fact is stated; recommendations stay conditional.

import type { BriefingMeta, EvidenceTag, ManifestDocument, SourceRef } from "./types";

export const meta: BriefingMeta = {
  slug: "hrhs-ai",
  packageVersion: "v2.1",
  // Change to "public" only after the publication review in
  // docs/advisory-briefings.md is signed off. See that file.
  status: "public",
  title: "From AI Ambition to Engineering Decisions | HRHS Advisory Case Study | Reginal Campbell",
  description:
    "An executive advisory case study on a fictional health system: how a conditional AI strategy recommendation becomes owned engineering decisions, executable plans and lean governance.",
  socialImage: "hrhs-ai-social-card.png",
};

// ─────────────────────────────────────────────────────────────────────────
// DOCUMENT MANIFEST
// Reading order is the Reader's Guide (HRHS-AI-00) v2.1 order.
// Documents 00 to 08 are the PUBLIC EDITION: re-exported from the v2.1 Word
// sources with only the evaluation-use header marking and the hiring-
// organization reference removed (verified by text diff; same page counts).
// 00A and the deck never carried either and are the v2.1 originals.
// The Reader's Guide marks 09 (interview preparation) and the QA records as
// private and outside the review set; they are listed here as `private` so the
// omission is explicit, and their files are not in this repository.
// ─────────────────────────────────────────────────────────────────────────
const P = { width: 480, height: 621 };

export const documents: ManifestDocument[] = [
  {
    id: "00A", code: "HRHS-AI-00A", title: "Case Study Overview",
    description: "The premise, the approach (assess, advise, execute) and how to navigate the package, on one page.",
    purpose: "Understand the purpose of the case in 90 seconds.",
    version: "2.1", date: "November 2026", pages: 1, group: "start", order: 1,
    file: "HRHS-AI-00A_Case_Study_Overview_v2.1.pdf", preview: "HRHS-AI-00A_Case_Study_Overview_v2.1.webp",
    previewSize: P, status: "approved", featured: true,
  },
  {
    id: "DECK", code: "Executive Case Study", title: "Five-Slide Executive Case Study",
    description: "The dilemma, the evidence, three paths, the engineering reality and how leadership will know execution is working.",
    purpose: "Discuss the case in 6 to 8 minutes.",
    version: "2.1", date: "November 2026", pages: 5, group: "start", order: 2,
    file: "HRHS-AI_Executive_Case_Study_v2.1.pdf", preview: "HRHS-AI_Executive_Case_Study_v2.1.webp",
    previewSize: { width: 480, height: 270 },
    original: { file: "HRHS-AI_Executive_Case_Study_v2.1.pptx", label: "PowerPoint original" },
    status: "approved", featured: true,
  },
  {
    id: "01", code: "HRHS-AI-01", title: "Executive Issue Brief",
    description: "AI demand is outpacing the organization's ability to govern, fund and staff it. The bottom line, the options and the five decisions requested.",
    purpose: "Grasp the decision in five minutes.",
    version: "2.1", date: "November 2026", pages: 2, group: "reasoning", order: 3,
    file: "HRHS-AI-01_Executive_Issue_Brief_v2.1.pdf", preview: "HRHS-AI-01_Executive_Issue_Brief_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "03", code: "HRHS-AI-03", title: "Strategic Options & Advisory Analysis",
    description: "Three legitimate approaches scored on eleven weighted criteria, with costs, benefits, sensitivity, downside cases and the conditions that would change the recommendation.",
    purpose: "Compare options, financials and sensitivity.",
    version: "2.1", date: "November 2026", pages: 5, group: "reasoning", order: 4,
    file: "HRHS-AI-03_Strategic_Options_Advisory_Analysis_v2.1.pdf", preview: "HRHS-AI-03_Strategic_Options_Advisory_Analysis_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "08", code: "HRHS-AI-08", title: "Engineering Portfolio Decision Brief",
    description: "A specialist capacity conflict surfaces on Day 18. How it becomes a bounded, owned decision: constraints, three recovery alternatives, decision rights and a critical-path plan.",
    purpose: "See a capacity conflict resolved.",
    version: "2.1", date: "November 2026", pages: 4, group: "execution", order: 5,
    file: "HRHS-AI-08_Engineering_Portfolio_Decision_Brief_v2.1.pdf", preview: "HRHS-AI-08_Engineering_Portfolio_Decision_Brief_v2.1.webp",
    previewSize: P, status: "approved", featured: true,
  },
  {
    id: "02", code: "HRHS-AI-02", title: "AI Readiness & Current-State Assessment",
    description: "Eight readiness dimensions rated 1 to 3 of 5, the findings behind them, critical gaps, assumptions and what the evidence implies for each option.",
    purpose: "Review the simulated evidence and gaps.",
    version: "2.1", date: "November 2026", pages: 4, group: "reasoning", order: 6,
    file: "HRHS-AI-02_AI_Readiness_Current_State_Assessment_v2.1.pdf", preview: "HRHS-AI-02_AI_Readiness_Current_State_Assessment_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "04", code: "HRHS-AI-04", title: "Executive Decision & Authorization Record",
    description: "What must be decided, what must be committed for the decision to be executable, mandatory controls and who owns each decision. The decision record is intentionally blank.",
    purpose: "See decisions and capacity commitments.",
    version: "2.1", date: "November 2026", pages: 3, group: "framework", order: 7,
    file: "HRHS-AI-04_Executive_Decision_Authorization_Record_v2.1.pdf", preview: "HRHS-AI-04_Executive_Decision_Authorization_Record_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "05", code: "HRHS-AI-05", title: "Decision-to-Delivery Execution Playbook",
    description: "A credible delivery pathway for each option under one set of controls, and the engineering coordination model that routes issues to the lowest level able to resolve them.",
    purpose: "See delivery and engineering coordination.",
    version: "2.1", date: "November 2026", pages: 7, group: "execution", order: 8,
    file: "HRHS-AI-05_Decision_to_Delivery_Execution_Playbook_v2.1.pdf", preview: "HRHS-AI-05_Decision_to_Delivery_Execution_Playbook_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "06", code: "HRHS-AI-06", title: "90-Day Mobilization & Delivery Plan",
    description: "Capacity first, governance and baselines next, a controlled pilot only if gates are met, and an evidence-based decision at Day 90.",
    purpose: "Review the critical path, capacity and gates.",
    version: "2.1", date: "November 2026", pages: 4, group: "execution", order: 9,
    file: "HRHS-AI-06_90_Day_Mobilization_Delivery_Plan_v2.1.pdf", preview: "HRHS-AI-06_90_Day_Mobilization_Delivery_Plan_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "07", code: "HRHS-AI-07", title: "Governance, Risk & Value Realization Framework",
    description: "Lean oversight proportional to risk, decision authorities, escalation thresholds, value rules Finance can verify and the executive scorecard.",
    purpose: "See lean oversight and value validation.",
    version: "2.1", date: "November 2026", pages: 4, group: "framework", order: 10,
    file: "HRHS-AI-07_Governance_Risk_Value_Realization_Framework_v2.1.pdf", preview: "HRHS-AI-07_Governance_Risk_Value_Realization_Framework_v2.1.webp",
    previewSize: P, status: "approved",
  },
  {
    id: "00", code: "HRHS-AI-00", title: "Executive Reader's Guide",
    description: "The scenario, the decision, the recommended reading order and how to read the evidence tags.",
    purpose: "Orient to the package and its evidence conventions.",
    version: "2.1", date: "November 2026", pages: 1, group: "framework", order: 11,
    file: "HRHS-AI-00_Executive_Readers_Guide_v2.1.pdf", preview: "HRHS-AI-00_Executive_Readers_Guide_v2.1.webp",
    previewSize: P, status: "approved",
  },
];

// Documented exclusions (not in this repository, never served).
export const excludedDocuments = [
  { code: "HRHS-AI-09", title: "Interview Discussion and Challenge Guide", reason: "Marked candidate preparation, not for distribution to the hiring team." },
  { code: "HRHS-AI-QA1", title: "Version 2.1 Change Log", reason: "Quality assurance record; the Reader's Guide keeps QA outside the review set." },
  { code: "HRHS-AI-QA2", title: "Consistency Report and Open Items", reason: "Quality assurance record; refers to the hiring process." },
];

export const groups: Record<ManifestDocument["group"], { label: string; note: string }> = {
  start: { label: "Start here", note: "Ninety seconds, then six to eight minutes." },
  reasoning: { label: "The reasoning", note: "Evidence, options and the conditional recommendation." },
  execution: { label: "Engineering execution", note: "Capacity, delivery pathways and the first 90 days." },
  framework: { label: "Decision and governance framework", note: "Authorization, controls, oversight and the reader's guide." },
};

export const docById = (id: string) => documents.find((d) => d.id === id);

// ─────────────────────────────────────────────────────────────────────────
// NARRATIVE CONTENT
// ─────────────────────────────────────────────────────────────────────────
type Cited = { tags?: EvidenceTag[]; refs: SourceRef[] };

export const hero = {
  eyebrow: "Executive advisory case study",
  title: "Approving an AI strategy does not create the capacity to deliver it.",
  lede:
    "A fictional regional health system has eleven proposed AI use cases, two pilots running without enterprise review, and 92% of next year's discretionary IT capacity already committed. Leadership has to decide how to approach enterprise AI, and what it can actually commit to in FY2027.",
  positioning: "A case study in conditional recommendations, engineering portfolio trade-offs and delivery accountability.",
  thesis: {
    text: "A defensible recommendation is only valuable when leaders understand its tradeoffs and the organization can execute the resulting decision.",
    refs: [{ doc: "00", section: "The case" }],
  },
  // Opening visual: FY2027 discretionary IT hours (02, F-08; 01).
  capacity: {
    total: 85000,
    committed: 78200,
    uncommitted: 6800,
    needed: 14000,
    specialist: 4600,
    refs: [{ doc: "02", section: "F-08" }, { doc: "01" }],
  },
  facts: [
    { value: "11 + 2", label: "proposed AI use cases, plus two pilots running without enterprise review", tags: ["S"], refs: [{ doc: "DECK", section: "Slide 1" }] },
    { value: "92%", label: "of FY2027 discretionary IT capacity already committed to ERP, EHR and network programs", tags: ["S"], refs: [{ doc: "00" }] },
    { value: "Dec 9", label: "scenario date of the executive decision; budgets close in December", tags: ["S"], refs: [{ doc: "01" }] },
  ] as ({ value: string; label: string } & Cited)[],
};

// Chapter 1: the tensions. Each one is a documented trade-off, not a label.
export const tensions: ({
  id: string;
  title: string;
  pull: { label: string; text: string };
  constraint: { label: string; text: string };
  consideration: string;
} & Cited)[] = [
  {
    id: "value-readiness",
    title: "Value versus readiness",
    pull: {
      label: "The opportunity",
      text: "$640M in annual supply spend with an estimated 71% on contract against an 85% target, and $38M in annual denial write-offs, 30% to 50% assumed preventable.",
    },
    constraint: {
      label: "The constraint",
      text: "Readiness rates 1 to 3 of 5 across eight dimensions. AI governance rates 1. In supply chain, 18% of the 112,000-record item master is incomplete or duplicate; in revenue cycle, 23% of denials are coded \"other\". Clinical data is reachable only through the EHR vendor.",
    },
    consideration:
      "Start where data is usable with remediation. Supply chain is a strong first candidate; revenue cycle follows after the denial pilot is evaluated; the clinical pilot is governed, not expanded.",
    tags: ["S", "A", "J"],
    refs: [{ doc: "02", section: "Readiness matrix; F-02; F-05" }, { doc: "DECK", section: "Slide 2" }],
  },
  {
    id: "demand-capacity",
    title: "Demand versus engineering capacity",
    pull: {
      label: "The demand",
      text: "Eleven use cases from seven sponsors through four intake routes. Only 3 of 11 state a measurable business objective, and two proposals overlap without either sponsor knowing.",
    },
    constraint: {
      label: "The constraint",
      text: "About 6,800 of 85,000 discretionary IT hours are uncommitted, mostly generalist. A governed first phase needs about 14,000, including 4,600 specialist hours held by the ERP and EHR programs.",
    },
    consideration:
      "Funding contracted hours does not make the plan executable. The specialists know this organization's item master, integration patterns and security architecture; they must be released, and their replacements need four to six weeks to ramp up.",
    tags: ["S", "E", "A", "J"],
    refs: [{ doc: "02", section: "F-01; F-08" }, { doc: "01" }],
  },
  {
    id: "speed-control",
    title: "Speed versus consistency of control",
    pull: {
      label: "The case for speed",
      text: "A vendor-led approach reaches first value in four to six months at the lowest three-year cost.",
    },
    constraint: {
      label: "The case for control",
      text: "Every option must meet the same mandatory privacy, security and contractual controls. Vendor-led applies them product by product, relies more on vendor evidence and adds vendor surfaces.",
    },
    consideration:
      "The vendor-led option overtakes the governed option once about three weighting points move from safety to speed. That is a leadership choice about risk posture, not an analytical error, so the recommendation is stated as conditional on it.",
    tags: ["E", "J"],
    refs: [{ doc: "03", section: "Scoring; Sensitivity" }],
  },
  {
    id: "urgency-risk",
    title: "Urgency versus unresolved risk",
    pull: {
      label: "Why act this cycle",
      text: "FY2027 budgets close in December, and a provisional AI envelope of up to $4.5M will be reallocated if not justified.",
    },
    constraint: {
      label: "What is unresolved",
      text: "An ambient documentation pilot processes patient information under vendor data-use and retention terms Privacy has not reviewed, and the contract auto-renews January 31 with an expansion option. No disclosure or violation has been identified. A denial-prediction pilot has run five months without a success measure.",
    },
    consideration:
      "Two decisions reduce unresolved risk at little cost and should proceed even if the strategy decision is deferred: charter lean governance, and bring both pilots under it within 30 days, replacing the auto-renewal with a 90-day extension.",
    tags: ["S", "A", "R"],
    refs: [{ doc: "01", section: "Why act; D-03; D-04" }, { doc: "02", section: "F-04; F-05" }],
  },
  {
    id: "investment-return",
    title: "Investment versus near-term return",
    pull: {
      label: "What the governed option adds",
      text: "Reusable controls, consistent risk evaluation and shared technical capabilities that lower the cost and risk of each later use case.",
    },
    constraint: {
      label: "What it costs",
      text: "About $1.4M more than vendor-led over three years ($7.75M against $6.35M). No option repays its cost within three years in the low case.",
    },
    consideration:
      "Those advantages are worth $1.4M only if there is a real pipeline beyond the first wave. If the second wave is never authorized, the shared foundations are overbuilt and the governed option is financially the weakest choice.",
    tags: ["E", "J"],
    refs: [{ doc: "03", section: "Summary; Three-year view" }],
  },
];

// Chapter 2: the decision framework (03, 01, 04).
export const objectives = {
  text: "Leadership asked for an approach that",
  items: [
    "produces measurable operating value in FY2027",
    "avoids unmanaged patient-data, safety or compliance exposure",
    "fits constrained IT capacity and margin",
    "avoids lock-in to an architecture the organization must later unwind",
  ],
  note: "These objectives conflict. The analysis makes the trade-offs explicit rather than pretending one option wins on all four.",
  refs: [{ doc: "03", section: "Decision objectives" }],
};

export const options = [
  {
    key: "A", name: "Targeted, Vendor-Led",
    idea: "Buy AI embedded in qualified vendor products; govern each deployment.",
    plus: "Fastest value (4 to 6 months); lowest three-year cost.",
    minus: "Controls applied product by product; most vendor dependency.",
    cash: "$2.55M", threeYear: "$6.35M", hours: "9,000", specialist: "~1,200", envelope: "Fits",
    position: "Preferred if AI scope stays at one or two use cases, or capacity cannot be released.",
  },
  {
    key: "B", name: "Governed, Phased",
    idea: "Minimum shared governance and technical controls; prioritized use cases in gated waves.",
    plus: "Reusable controls and shared capabilities lower the cost of later use cases.",
    minus: "~$1.4M more than A over three years; needs specialist release.",
    cash: "$3.75M", threeYear: "$7.75M", hours: "14,000", specialist: "4,600", envelope: "Fits, with $0.75M contingency",
    position: "Provisionally recommended, conditional on validated specialist capacity and a pipeline beyond Wave 1.",
    recommended: true,
  },
  {
    key: "C", name: "Platform-First",
    idea: "Centralized AI platform, data foundations and operating model before broad deployment.",
    plus: "Strongest long-term foundation.",
    minus: "Value deferred to 2028; exceeds the envelope and FY2027 capacity.",
    cash: "$7.9M", threeYear: "$12.3M", hours: "30,000", specialist: "~11,000", envelope: "No, ~$3.4M above",
    position: "Only if capacity is freed and a broad portfolio is committed.",
  },
];
export const optionRefs: SourceRef[] = [{ doc: "03", section: "The three options; Costs and capacity" }, { doc: "04" }, { doc: "DECK", section: "Slide 3" }];

// Sensitivity scenarios exactly as published in 03.
export const scenarios = [
  { id: "base", label: "Base weights", scores: { A: 3.43, B: 3.52, C: 2.51 },
    note: "B leads by 0.09, which is within judgment error. The conditions decide, not the score." },
  { id: "equal", label: "Equal weights", scores: { A: 3.36, B: 3.55, C: 2.55 },
    note: "With every criterion weighted equally, B's lead widens slightly." },
  { id: "speed", label: "Speed first", scores: { A: 3.59, B: 3.44, C: 2.27 },
    note: "Moving 8 points from safety to time-to-value puts A ahead. A overtakes B at about 3 points." },
  { id: "single", label: "Single wave only", scores: { A: 3.43, B: 3.30, C: 2.51 },
    note: "If there are no use cases beyond Wave 1, B's value and scalability ratings fall and A is preferred outright." },
] as const;
export const scenarioRefs: SourceRef[] = [{ doc: "03", section: "Sensitivity and downside scenarios" }];

// Eleven weighted criteria from 03 (ratings are judgments [J]).
export const criteria = [
  { name: "Expected business value", weight: 15, A: 3, B: 4, C: 3, basis: "B lowers the marginal cost of later use cases; value depends on a pipeline existing." },
  { name: "Time to initial value", weight: 10, A: 5, B: 3, C: 1, basis: "A: 4 to 6 months. B: pilot results ~Day 90. C: 9 to 12 months." },
  { name: "Total cost and investment", weight: 10, A: 4, B: 3, C: 2, basis: "Three-year cost: A $6.35M; B $7.75M; C $12.3M." },
  { name: "Readiness fit", weight: 10, A: 4, B: 3, C: 1, basis: "C needs capabilities currently rated 1 to 2." },
  { name: "Implementation complexity", weight: 8, A: 4, B: 3, C: 2, basis: "C builds platform, data and operating model at once." },
  { name: "Security, privacy, compliance, safety", weight: 12, A: 3, B: 4, C: 4, basis: "All options meet mandatory controls; A applies them product by product with more vendor surfaces." },
  { name: "Workforce and change impact", weight: 5, A: 3, B: 4, C: 3, basis: "B paces adoption with training per wave." },
  { name: "Vendor dependency and flexibility", weight: 8, A: 2, B: 4, C: 4, basis: "A concentrates dependency on vendor roadmaps." },
  { name: "Delivery capacity and execution risk", weight: 10, A: 4, B: 3, C: 1, basis: "B needs 4,600 specialist hours; C needs ~30,000 total hours." },
  { name: "Scalability and sustainability", weight: 7, A: 2, B: 4, C: 5, basis: "A creates little reusable capability." },
  { name: "Reversibility and exit", weight: 5, A: 3, B: 4, C: 2, basis: "B commits in tranches; C front-loads irreversible spend." },
];

export const recommendationChanges = [
  { prefer: "Option A", when: "Leadership does not intend more than one or two AI use cases in three years; or speed outweighs control in its risk posture; or specialist capacity cannot be validated and recovery options fail." },
  { prefer: "Reduced B", when: "Specialist capacity is partly available: sequence Wave 1, spend classification first." },
  { prefer: "Option C", when: "More than 15 use cases are committed in three years, ERP or EHR work completes early and frees ~20,000 hours, ~$3.4M above the envelope is available, and leadership accepts value deferral into 2028." },
  { prefer: "Pause the clinical pilot", when: "The vendor refuses amended terms and Privacy identifies a reportable issue, or a major security event diverts IT capacity." },
];

export const recommendation = {
  label: "Conditional recommendation",
  text: "Provisionally adopt Option B, authorizing only a 90-day mobilization and Tranche 1 ($1.5M). Proceed with Wave 1 only if named specialists are provisionally allocated by Day 15 and validated as available in the delivery window by Day 20. Return to leadership at Day 90 with measured evidence and a restated financial case.",
  refs: [{ doc: "03", section: "Recommendation" }],
};

export const decisionsRequested = [
  { id: "D-01", decision: "Select the approach (provisionally: Option B).", approver: "CEO, on ELT advice" },
  { id: "D-02", decision: "Authorize up to $3.75M in two tranches: $1.5M now, $2.25M at Day 90 on evidence. $0.75M contingency held by the CFO.", approver: "CFO" },
  { id: "D-03", decision: "Charter lean AI governance; adopt interim policy and controls MC-01 to MC-10.", approver: "CEO" },
  { id: "D-04", decision: "Bring both pilots under governance within 30 days; replace the auto-renewal with a 90-day extension.", approver: "CMO and CFO" },
  { id: "D-05", decision: "Provisional named specialist allocations by Day 15, validated against ERP and EHR release calendars by Day 20.", approver: "CIO/CTO with ERP and EHR sponsors" },
];

// Chapter 3: from decision to execution (04, 06, 08, 05).
export const capacityStates = [
  { id: "planned", label: "Planned", day: "Estimate", text: "Hours in the estimate by skill: 14,000 for the governed first phase.", owner: "Program Office" },
  { id: "provisional", label: "Provisionally committed", day: "Day 15", text: "Engineering managers submit named people and hours.", owner: "Engineering managers" },
  { id: "validated", label: "Validated", day: "Day 20", text: "Confirmed against ERP and EHR release calendars and cutover plans.", owner: "TPM Director with program directors" },
  { id: "available", label: "Available in window", day: "Days 1 to 120", text: "Validated hours that fall inside the window Wave 1 actually needs them.", owner: "TPM Director, in the capacity ledger" },
];
export const capacityRefs: SourceRef[] = [{ doc: "04", section: "Authorization dependencies" }, { doc: "06", section: "Capacity plan" }];

export const gates = [
  { id: "G0", day: "Day 0", name: "Authorize", owner: "ELT", decision: "Authorize Option B and Tranche 1.", evidence: "Envelope confirmed; owners named." },
  { id: "CAP", day: "Days 15 / 20", name: "Capacity allocated, then validated", owner: "CIO/CTO and TPM Director", decision: "Provisional allocations submitted; validated or conflict logged.", evidence: "Named allocations; capacity ledger reconciled against release calendars." },
  { id: "G1", day: "Day 30", name: "Governance baseline", owner: "Steering Committee", decision: "Accept the governance baseline, or take a recovery decision on capacity.", evidence: "Charters approved; inventory v1 covers all departments; both pilots registered; on-contract baseline validated." },
  { id: "G2", day: "Day 60", name: "Design and readiness", owner: "Steering Committee with CISO", decision: "Accept design and readiness.", evidence: "Acceptance criteria approved; remediation at agreed threshold; inventory complete; support model defined." },
  { id: "G3", day: "~Day 65", name: "Pilot go / no-go", owner: "Risk Review Board and product owner", decision: "Start the controlled pilot, or not.", evidence: "Controls MC-01 to MC-06 evidenced; release readiness check passed. A no-go is valid and triggers remediation, not schedule pressure." },
  { id: "G4", day: "Day 90", name: "Scale, revise or stop", owner: "CFO and ELT", decision: "Scale, revise or stop Wave 1; release Tranche 2 ($2.25M) or not.", evidence: "Pilot results against acceptance criteria; Finance-confirmed baselines; restated financial case." },
];
export const gateRefs: SourceRef[] = [{ doc: "06", section: "Decision checkpoints" }, { doc: "DECK", section: "Slide 5" }];

export const criticalPath = {
  text: "The Wave 1 critical path runs through specialist capacity and data, not technology.",
  steps: ["Allocation (Day 15)", "Validation (Day 20)", "Replacement contractors productive (Day 45)", "Item-master remediation (Days 31 to 58)", "Acceptance criteria (Day 60)", "G3 (~Day 65)", "Pilot (Days 66 to 88)", "G4 (Day 90)"],
  note: "The AI gateway is off the Wave 1 critical path because Wave 1 uses no patient data. It must be ready before any use of patient data (target month 6). Any slip in specialist release moves G3 day for day.",
  refs: [{ doc: "06", section: "Critical path" }],
};

// The Day 18 conflict (08). Explicitly a hypothetical scenario in the source.
export const conflict = {
  timeline: [
    { day: "Day 15", event: "Engineering managers submit named allocations covering all 4,600 specialist hours.", state: "Provisionally committed" },
    { day: "Day 17", event: "ERP issues an updated conversion plan: the item-master specialist is needed until go-live (~Day 120). EHR confirms the security architect through month 5.", state: "Under validation" },
    { day: "Day 18", event: "The TPM Director reconciles the plans in the capacity ledger. Conflict logged; Executive Sponsor and CIO/CTO notified the same day; Wave 1 spend beyond governance held.", state: "Unavailable in window" },
    { day: "Days 20 to 30", event: "Shortfall quantified by skill; recovery options prepared; decision at G1.", state: "900 hours validated" },
  ],
  skills: [
    { skill: "ERP item-master data", needed: 1400, releasable: 400 },
    { skill: "Integration engineering", needed: 600, releasable: 300 },
    { skill: "Security architecture", needed: 300, releasable: 0 },
    { skill: "Enterprise architecture", needed: 300, releasable: 200 },
  ],
  summary: "The total hours still reconcile. The skills and the timing do not.",
  refs: [{ doc: "08", section: "A. Decision required" }, { doc: "DECK", section: "Slide 4" }],
};

export const alternatives = {
  dimensions: [
    { id: "change", label: "What changes" },
    { id: "delivery", label: "Delivery" },
    { id: "value", label: "Business value" },
    { id: "cost", label: "Cost" },
    { id: "risk", label: "Main risks" },
    { id: "reversibility", label: "Reversibility" },
    { id: "owner", label: "Decision-maker" },
  ],
  items: [
    {
      key: "1", name: "Sequence Wave 1", recommended: true,
      change: "Wave 1a: spend classification pilot by Day 90. Wave 1b: contract matching starts after ERP go-live; pilot ~Day 180.",
      delivery: "Specialist need in Days 1 to 120 falls to ~900 hours: it fits.",
      value: "Contract-matching savings arrive ~4 months later: about $0.1M to $0.25M less benefit in FY2027.",
      cost: "Neutral; ~$0.2M of FY2027 spend shifts later.",
      risk: "Thinner Day 90 evidence; momentum loss in supply chain.",
      reversibility: "High.",
      owner: "Steering Committee, within wave scope.",
    },
    {
      key: "2", name: "Reallocate engineering capacity",
      change: "Defer the ERP phase 2 reporting module one quarter to release the item-master specialist now.",
      delivery: "Original Wave 1 schedule held, if ERP confirms no go-live impact.",
      value: "No AI value delay; ERP reporting benefits delayed one quarter.",
      cost: "Neutral to the AI budget; possible ERP contractor extension.",
      risk: "ERP go-live slip; cross-program precedent.",
      reversibility: "Medium.",
      owner: "ERP executive sponsor with CIO/CTO; ELT if a go-live date moves.",
    },
    {
      key: "3", name: "Purchase or source externally",
      change: "Managed item-master cleansing plus spend analytics SaaS with embedded classification.",
      delivery: "Reduces specialist need ~1,100 hours; earliest start ~Day 75; pilot ~Day 140.",
      value: "Contract matching delayed ~6 weeks against the original plan.",
      cost: "+$0.35M one-time, +$0.12M per year; fundable from contingency.",
      risk: "Procurement delay; vendor terms; dependency. Shifts toward vendor-led traits and needs an exit plan.",
      reversibility: "Medium (one-year term).",
      owner: "CFO (contingency) with VP Supply Chain and CIO/CTO.",
    },
  ],
  decision: "Sequence Wave 1 now. Obtain vendor pricing for a Day 60 decision. Reallocate ERP capacity only after written confirmation of acceptable impact.",
  consequence: "Sequencing weakens the Day 90 evidence base, so the G4 decision weighs accuracy, the denial pilot evaluation and a restated case. If no alternative restores a credible path by Day 60, the documented conditions for switching to Option A are met, and the program office brings that recommendation forward.",
  refs: [{ doc: "08", section: "C. Recovery alternatives; D. Conditional recommendation" }],
};

export const routing = {
  intro: "The TPM Director does not manage engineers; engineering managers do. The role owns the integrated plan, makes conflicts visible early, routes each issue to the lowest level that can resolve it, and brings executives only the decisions that need their authority.",
  example: "Worked example from the case: gateway logging must feed the security monitoring platform, but the engineer who owns that onboarding is on EHR cutover testing through month 5. The same week, contract-matching accuracy is 8 points below target on two categories.",
  rows: [
    { type: "Risk", meaning: "may happen", item: "EHR cutover could slip and extend the engineer's assignment.", resolvedBy: "Program Office: log with trigger date, monitor weekly." },
    { type: "Dependency", meaning: "must be resolved", item: "Gateway needs monitoring onboarding by Day 75, before any patient data.", resolvedBy: "Security engineering manager: owner and need-by date in the plan." },
    { type: "Impediment", meaning: "blocking now", item: "No onboarding slot in Days 45 to 75.", resolvedBy: "Engineering managers, same week: trade a slot or use the vendor's standard connector." },
    { type: "Executive decision", meaning: "needs authority", item: "Only resolvable by delaying an EHR cutover test cycle.", resolvedBy: "EHR sponsor and CIO/CTO, from an options memo." },
    { type: "Team-level issue", meaning: "reported as a metric", item: "Accuracy gap on two categories.", resolvedBy: "Delivery team fixes matching rules; not escalated." },
  ],
  resolution: "Because the gateway is not on the Wave 1 critical path, the impediment is resolved with the vendor connector and monitoring onboarding moves to month 6. No executive time is needed. Recognizing that is the coordination work.",
  refs: [{ doc: "05", section: "Engineering coordination model" }],
};

// Chapter 4: governance and accountability (07, 04, 08).
export const tiers = [
  { tier: "Tier 1", scope: "Administrative, no patient data", path: "Program Office and a privacy analyst", sla: "5 business days" },
  { tier: "Tier 2", scope: "Patient data or financial impact", path: "AI Risk Review Board", sla: "10 business days" },
  { tier: "Tier 3", scope: "Clinical or patient-impacting", path: "Risk Review Board with CMIO; CMO approves", sla: "10 business days" },
];

export const cadence = [
  { body: "ELT", rhythm: "G0, G4, quarterly", remit: "Option, tranches, high-risk acceptance, cross-program priorities", chair: "CEO" },
  { body: "AI Steering Committee", rhythm: "Monthly, 60 min", remit: "Gates G1 and G2, escalations, portfolio priority, value review", chair: "COO" },
  { body: "AI Risk Review Board", rhythm: "Biweekly, 45 min", remit: "Tier 2 and 3 approvals, control exceptions, incidents", chair: "CPO and CISO" },
  { body: "Dependency review", rhythm: "Weekly, 30 min", remit: "Cross-team dependencies and impediments", chair: "TPM Director" },
];

export const decisionRights = [
  { decision: "Select or change the option", authority: "ELT (CEO final)", notBy: "" },
  { decision: "Release a tranche; budget change above $250K", authority: "CFO with ELT", notBy: "Steering Committee" },
  { decision: "Release committed ERP/EHR staff; move ERP/EHR milestones", authority: "ERP or EHR executive sponsor with CIO/CTO; ELT if a go-live moves", notBy: "AI program" },
  { decision: "Scope or sequencing within a wave", authority: "Steering Committee, on product owner recommendation", notBy: "Individual teams" },
  { decision: "Add a use case to the portfolio", authority: "Steering Committee, after tiering", notBy: "Departments" },
  { decision: "Tier 3 deployment or expansion", authority: "CMO on CMIO recommendation", notBy: "Business sponsors" },
  { decision: "Halt for privacy or security nonconformance", authority: "CPO or CISO", notBy: "Risk acceptance" },
  { decision: "Design within the reference pattern, tooling, estimates", authority: "Engineering managers and teams", notBy: "Program Office" },
  { decision: "Accept residual risk", authority: "Risk owner up to Medium; ELT for High", notBy: "Cannot accept a bypass of MC-01 to MC-10" },
];
export const programDirectorRole = "The AI Program Director surfaces conflicts, quantifies options, coordinates inputs, drafts decisions and escalates on triggers. No approval authority.";

export const escalations = [
  { trigger: "Validated capacity unavailable in the delivery window", to: "Executive Sponsor and CIO/CTO", timing: "1 business day" },
  { trigger: "Gate date at risk by more than 2 weeks", to: "Steering Committee", timing: "Next meeting or ad hoc" },
  { trigger: "Workstream variance above 10% or $100K", to: "Steering Committee", timing: "Next meeting" },
  { trigger: "Variance above $250K, or wave scope change", to: "CFO and ELT", timing: "10 business days" },
  { trigger: "Suspected patient-data privacy incident", to: "CPO, breach procedure", timing: "24 hours" },
  { trigger: "Tier 3 safety event or near miss", to: "CMIO; use paused", timing: "Immediate" },
];

export const outcomes = [
  { id: "continue", label: "Continue", text: "Acceptance criteria met; controls conform; trend toward benefit target." },
  { id: "revise", label: "Revise", text: "Partially met with a bounded fix; one revision cycle before stop." },
  { id: "slow", label: "Slow", text: "Capacity or dependency shortfall: sequence scope rather than compress quality gates." },
  { id: "expand", label: "Expand", text: "Continue criteria met for two periods; Finance-validated benefits; capacity committed; re-tiered." },
  { id: "stop", label: "Stop", text: "Criteria unmet after revision, unresolved nonconformance or safety event, or benefits not plausible at full cost. Execute the exit plan." },
];

export const scorecard = {
  leading: [
    { id: "L1", measure: "Inventory coverage", owner: "AI Program Director" },
    { id: "L2", measure: "Capacity delivery: validated vs actual hours by skill", owner: "CIO/CTO" },
    { id: "L3", measure: "Data readiness of target item records", owner: "Data Governance Lead" },
    { id: "L4", measure: "Output acceptance without material change", owner: "Product owner" },
    { id: "L5", measure: "Control conformance, MC-01 to MC-10", owner: "CISO" },
    { id: "L6", measure: "Intake cycle time to tier decision", owner: "CPO" },
  ],
  lagging: [
    { id: "O1", measure: "On-contract spend rate (from est. 71%)", owner: "VP Supply Chain" },
    { id: "O2", measure: "Preventable denial write-offs, net recovery collected", owner: "VP Revenue Cycle" },
    { id: "O3", measure: "Documentation time per encounter, only if the pilot continues", owner: "CMIO" },
    { id: "O4", measure: "AI incidents by severity", owner: "CISO and CMIO" },
    { id: "O5", measure: "Program spend vs authorized baseline", owner: "CFO delegate" },
    { id: "O6", measure: "Finance-validated net benefits vs case range", owner: "CFO" },
  ],
  valueRules: [
    "Cost savings, revenue recovery, cost avoidance and risk reduction are never summed into one figure.",
    "Gross and net are reported separately.",
    "Revenue recovery is reported when collected, not when a denial is prevented.",
    "Finance validates baselines before a pilot and benefits before reporting.",
  ],
};

export const leanNote = {
  text: "In version 2.0 the package removed biweekly steering, separate workstream status meetings and duplicate reporting. Gates now fall on the monthly Steering Committee at Days 30, 60 and 90.",
  refs: [{ doc: "07", section: "Design principle" }],
};

// Closing: leadership perspective. Quoted lines are the author's own words
// from the package (00A, 05). Credentials limited to what the site states.
export const perspective = {
  quote: "A recommendation should change when the evidence changes. Accountability for delivery should not.",
  quoteRef: { doc: "00A", section: "Leadership philosophy" } as SourceRef,
  paragraphs: [
    "The point of this case is not to prove that Option B is right. Its lead is 0.09 on a five-point scale, inside judgment error, and the package says so on the first page. The point is to show the work an executive needs before committing money and people: what is known, what is assumed, what each path costs, and the specific conditions under which the answer flips.",
    "Before a decision, the advisor owes leadership an independent recommendation. After it, the obligation is to deliver the option leadership chose, with the same controls and honest reporting, and to come back when the evidence says the chosen path is no longer working. That is why every pathway carries its own pivot conditions: changing course should be a planned governance decision, not a failure narrative.",
  ],
  paragraphRefs: [{ doc: "03" }, { doc: "05", section: "Advisory to delivery" }] as SourceRef[],
  principle: "AI informs. Humans decide. Leaders own.",
  author: {
    name: "Reginal Campbell, PMP",
    // Lines below are taken from the live site's own footer and pages.
    line: "Enterprise transformation leader · U.S. Army veteran",
    focus: ["AI strategy execution", "Engineering and technology portfolio governance", "Complex program delivery", "Data and analytics transformation", "Operating models and executive accountability"],
  },
};

export const disclaimer = {
  short: "Fictional case study. Halcyon Ridge Health System does not exist; every figure, finding, date and decision is simulated.",
  full: "Halcyon Ridge Health System is fictional. All organizational information, stakeholder interviews, findings, financial figures and decisions in this briefing and its documents are simulated scenario data, estimates or labeled assumptions. The analysis was prepared independently by Reginal Campbell to demonstrate an advisory and technical program leadership methodology. It is not a completed client engagement, was not commissioned or endorsed by any organization, and describes no real organization. No recommendation shown here has been approved or implemented, and no business outcome is claimed. Scenario dates (for example, the December 9, 2026 decision) are part of the fiction.",
};
