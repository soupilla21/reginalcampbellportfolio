// Where "Ask the assistant" links point.
//
// Phase A (safe to merge now): keep pointing at the existing subdomain.
// Phase B: once the /ask/ rewrite in vercel.json is live and
// https://reginalcampbell.com/ask/ loads the assistant and answers a
// question, change this to "/ask/" (with the trailing slash). Every entry
// point on the site, and the sitemap, reads this one value.
export const ASK_URL = "https://briefing.reginalcampbell.com";

// Builds an assistant link, pre-filling a suggested question as ?q=.
// The assistant only pre-fills the box if it reads the q parameter; if it
// does not, the link still opens the assistant normally.
export function askHref(question?: string): string {
  if (!question) return ASK_URL;
  const sep = ASK_URL.includes("?") ? "&" : "?";
  return `${ASK_URL}${sep}q=${encodeURIComponent(question)}`;
}
