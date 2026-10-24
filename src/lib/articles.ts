import { getCollection, type CollectionEntry } from "astro:content";

/**
 * Every published article. This is the only place the draft filter lives:
 * a `draft: true` entry must never reach a route, listing, feed, sitemap,
 * related-reading block, or search index. Order is not guaranteed; callers sort.
 */
export async function getPublishedArticles(): Promise<CollectionEntry<"articles">[]> {
  return getCollection("articles", ({ data }) => !data.draft);
}

// Marks where the argument ends and the apparatus begins. Anchored on the headings
// themselves, not on the `---` rules that precede them, because a horizontal rule is
// ambiguous inside a body while these two lines are house style. Both patterns must
// match a whole line, so a mid-article heading like "## Sources of Truth" cannot
// truncate the count. Whichever appears first wins; if neither appears, the whole
// body is counted.
const BODY_END = /^[ \t]*(?:#{1,6}[ \t]*Sources[ \t]*|\*\*About the author\*\*[ \t]*)$/m;

export function calculateReadingTime(rawMarkdown: string): number {
  // Count the argument only. The Sources list and author bio are reference apparatus,
  // not reading, and including them was overstating every article by 2 to 4 minutes.
  const boundary = rawMarkdown.match(BODY_END);
  const body = boundary ? rawMarkdown.slice(0, boundary.index) : rawMarkdown;

  // Strip HTML tags and collapse markdown links to their visible text before counting,
  // so inline footnote markup and citation links don't inflate the word count.
  const text = body
    .replace(/^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$/gm, " ")
    .replace(/<[^>]+>/g, " ")
    // Tag stripping leaves the visible marker behind: <sup><a ...>[1]</a></sup>
    // reduces to a bare "[1]", which the whitespace split then counts as a word.
    // Drop those markers (including reuse ids like [1b]) so citations cost nothing.
    .replace(/\[\d+[a-z]?\]/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
  const words = text.split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 225));
}
