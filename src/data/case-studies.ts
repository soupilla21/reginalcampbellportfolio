// Case studies in the six-part format: business problem, my role,
// constraints, decisions, measurable results, lessons learned.
//
// Every part is optional. A part left undefined is not rendered (used when a
// part is held back on purpose). A part set to a Placeholder renders as a
// visible "Placeholder" box so a missing fact can't go live unnoticed.
//
// Every claim used here must come from the site, /evidence, the project
// documents, or Reggie directly. Sources are listed per claim in `sources`
// so the page can be reviewed for confidentiality before publishing.

export type Placeholder = { placeholder: string };
export type Text = string | Placeholder;

export interface Metric {
  label: string;
  value: Text;
  detail: Text;
}

export interface CaseStudy {
  slug: string;
  status: "published" | "draft"; // drafts build but are noindex and kept out of the sitemap
  eyebrow: string;
  title: string;
  summary: string;
  role: string;
  problem?: Text[];
  myRole?: Text[];
  constraints?: Text[];
  decisions?: Text[];
  results?: { metrics?: Metric[]; text?: Text[]; quote?: string };
  lessons?: Text[];
  links?: { label: string; href: string }[];
  cta: { label: string; href: string; secondary?: { label: string; href: string } };
  sources: { claim: string; source: string }[];
}

export const ph = (placeholder: string): Placeholder => ({ placeholder });
export const isPlaceholder = (t: unknown): t is Placeholder =>
  typeof t === "object" && t !== null && "placeholder" in t;

export const caseStudies: CaseStudy[] = [
  {
    slug: "national-healthcare-data-exchange",
    status: "published",
    eyebrow: "Enterprise Delivery · Accenture Federal Services · 2024 to 2026",
    title: "National Healthcare Data Exchange",
    role: "Product Manager and Technical Program Lead",
    summary:
      "Product Manager and Technical Program Lead for a national healthcare data exchange platform processing more than 2 billion transactions a month.",
    problem: [
      "A national healthcare data exchange platform moving more than 2 billion transactions every month. At that volume, the hard problems aren't in any single feature. They're in keeping engineering, cloud, security, compliance, and product teams moving in the same direction while the platform keeps growing.",
      "At 2B+ monthly transactions, a governance gap isn't a documentation issue; it's a liability. The standard for federal healthcare data isn't best effort. It's auditability, traceability, and demonstrated control at every layer of the stack.",
    ],
    myRole: [
      "I was the client-facing delivery lead, responsible for executive decision support, operating cadence, and portfolio-level planning.",
      "I aligned quarterly planning, investment prioritization, and feature delivery across cross-functional engineering, cloud, security, compliance, and product teams.",
    ],
    // constraints, decisions, lessons: held back for now (Reggie, 30 Oct 2026).
    results: {
      metrics: [
        { label: "Scale", value: "2B+", detail: "Transactions per month on the national healthcare data exchange platform" },
        { label: "User satisfaction", value: "+30%", detail: "Through standardized data definitions and governance" },
        { label: "Growth", value: "~7%", detail: "Month-over-month expansion, sustained" },
      ],
      text: [
        "Standardized data definitions and governance raised user satisfaction by 30%, and the platform sustained roughly 7% month-over-month expansion.",
      ],
      quote:
        "At this volume, a shared data definition is a control, not a convenience. When every team means the same thing by the same field, prioritization gets faster and the platform can grow without losing the trust of the people who depend on it.",
    },
    links: [
      { label: "Evidence and questions", href: "/evidence#accenture-national-healthcare-data-exchange" },
      { label: "Earlier federal work, including the $40M VA platform", href: "/portfolio#experience-foundation" },
    ],
    cta: {
      label: "Discuss an engagement",
      href: "/advisory",
      secondary: { label: "Contact me about a role", href: "/?type=role#contact" },
    },
    sources: [
      { claim: "Accenture Federal Services, 2024 to 2026", source: "Homepage #case-study eyebrow; /evidence entry (period)" },
      { claim: "Role title: Product Manager and Technical Program Lead", source: "Homepage #case-study; /evidence entry (subtitle)" },
      { claim: "National healthcare data exchange platform, more than 2 billion transactions a month (2B+)", source: "Homepage #case-study and hero subhead; /evidence claims; Week 1 record" },
      { claim: "Problem paragraph 1 (keeping engineering, cloud, security, compliance and product teams moving together)", source: "Homepage #case-study, 'The Platform' (verbatim)" },
      { claim: "Problem paragraph 2 (governance gap is a liability; auditability, traceability, control)", source: "Homepage #case-study, 'Why it matters at this scale' (verbatim)" },
      { claim: "Client-facing delivery lead; executive decision support, operating cadence, portfolio-level planning", source: "Homepage #case-study 'My Role'; /evidence summary" },
      { claim: "Aligned quarterly planning, investment prioritization and feature delivery across cross-functional teams", source: "Homepage #case-study 'My Role'; /evidence claims" },
      { claim: "+30% user satisfaction through standardized data definitions and governance", source: "Homepage #case-study metric card and 'What Changed'" },
      { claim: "~7% month-over-month expansion, sustained", source: "Homepage #case-study metric card and 'What Changed'" },
      { claim: "Pull quote on shared data definitions", source: "Homepage #case-study callout (verbatim)" },
      { claim: "$40M VA platform (link text only)", source: "Homepage #case-study link; /portfolio" },
    ],
  },
];
