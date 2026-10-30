// Where "Ask the assistant" links point.
//
// Phase A (safe to merge now): keep pointing at the existing subdomain.
// Phase B: once the /ask/ rewrite in vercel.json is live and
// https://reginalcampbell.com/ask/ loads the assistant and answers a
// question, change this to "/ask/" (with the trailing slash). Every entry
// point on the site, and the sitemap, reads this one value.
export const ASK_URL = "/ask/";

// Builds an assistant link, pre-filling a suggested question as ?q=.
// The assistant only pre-fills the box if it reads the q parameter; if it
// does not, the link still opens the assistant normally.
export function askHref(question?: string): string {
  if (!question) return ASK_URL;
  const sep = ASK_URL.includes("?") ? "&" : "?";
  return `${ASK_URL}${sep}q=${encodeURIComponent(question)}`;
}

// The hosted resume. Every resume link on the site reads this one value.
export const RESUME_URL = "/reginal-campbell-resume.pdf";

// Contact form inquiry types. The key is what a ?type= link carries
// (for example /?type=advisory#contact); the label is what the visitor
// sees and what Formspree receives as inquiry_type.
export const INQUIRY_TYPES = [
  { key: "advisory", label: "Enterprise advisory" },
  { key: "small-business", label: "Small-business project" },
  { key: "role", label: "Executive role" },
  { key: "other", label: "Other" },
] as const;
export type InquiryKey = (typeof INQUIRY_TYPES)[number]["key"];

// Link to the homepage contact form with an inquiry type pre-selected.
export function contactHref(type: InquiryKey): string {
  return `/?type=${type}#contact`;
}
