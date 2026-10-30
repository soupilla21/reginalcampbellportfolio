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
  // For client work: "{client}" in any text is replaced at build time. The
  // client's real name is NOT stored in this repo until the client approves
  // naming; until then `name` stays undefined and the anonymized label is used.
  client?: { anonymized: string; name?: string };
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

// Replace "{client}" / "{Client}" tokens throughout a study.
export function resolveClient(study: CaseStudy): CaseStudy {
  if (!study.client) return study;
  const label = study.client.name ?? study.client.anonymized;
  const cap = label.charAt(0).toUpperCase() + label.slice(1);
  const walk = (v: unknown): unknown =>
    typeof v === "string" ? v.replaceAll("{client}", label).replaceAll("{Client}", cap)
    : Array.isArray(v) ? v.map(walk)
    : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, walk(x)]))
    : v;
  return walk(study) as CaseStudy;
}

const rawCaseStudies: CaseStudy[] = [
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
    // constraints, decisions, lessons: left off permanently (Reggie, 30 Oct 2026).
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
  {
    // Home inspection client engagement. Anonymized until the client approves naming (Reggie, 30 Oct 2026).
    slug: "home-inspection-website",
    status: "published",
    client: { anonymized: "a regional home inspection company" },
    eyebrow: "Small-Business Digital Services · InnovatePM Solutions · 2026",
    title: "Website Launch for a Regional Home Inspection Company",
    role: "Consultant: proposal, build and launch",
    summary:
      "Website proposal, build and production launch for {client} that had no website, with Google Analytics 4 and Microsoft Clarity installed.",
    problem: [
      "{Client} had no website.",
    ],
    myRole: [
      "I wrote the website proposal, built the site and took it through production launch. The work was delivered through InnovatePM Solutions.",
      "Scope also covered domain and hosting setup, the site content, the Google Business Profile and listings, and a handoff with training for the owner.",
    ],
    constraints: [
      "The work had to fit a small-business budget.",
      "The business has no in-house technical staff, so the owner needed to be able to run the site without a developer.",
    ],
    decisions: [
      "Mobile-first layout, built for customers searching on their phones.",
      "The Google Business Profile and listings were set up alongside the site, so search finds the business in both places.",
      "A clear path to call or book an inspection from every page.",
      "Google Analytics 4 and Microsoft Clarity went in before launch, so results are measurable from day one.",
    ],
    results: {
      metrics: [
        { label: "Launched", value: "2026", detail: "Production launch" },
        { label: "Analytics", value: "GA4 + Clarity", detail: "Installed at launch" },
      ],
      text: [
        "The site launched to production in 2026 with Google Analytics 4 and Microsoft Clarity installed, so {client} can see visitor traffic and on-page behavior from the start.",
      ],
    },
    lessons: [
      "Measure from day one. Analytics installed after launch loses the baseline you need to show results.",
      "For a local service business, listings matter as much as the site. The Google profile is often found before the website.",
    ],
    cta: {
      label: "Discuss a project",
      href: "/?type=small-business#contact",
      secondary: { label: "Small-business services", href: "/digital-services" },
    },
    sources: [
      { claim: "InnovatePM engagement: website proposal, build and production launch", source: "Reggie's Week 2 brief; project memory" },
      { claim: "Google Analytics 4 and Microsoft Clarity installed before launch", source: "Reggie's Week 2 brief; Reggie, closeout answers (30 Oct 2026)" },
      { claim: "No website before; scope (domain and hosting, content, Google Business Profile and listings, handoff and training); launched 2026; no result figure", source: "Reggie, closeout answers (30 Oct 2026)" },
      { claim: "Constraints, decisions and lessons as written", source: "Reggie, closeout answers (30 Oct 2026), chosen from offered options" },
      { claim: "Anonymized as 'a regional home inspection company'", source: "Reggie, decisions D5 and closeout (30 Oct 2026)" },
    ],
  },
];

export const caseStudies: CaseStudy[] = rawCaseStudies.map(resolveClient);
