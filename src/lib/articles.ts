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

type Article = CollectionEntry<"articles">;
export type Category = Article["data"]["category"];

/** Display labels for the category enum in src/content.config.ts. */
export const CATEGORY_LABELS: Record<Category, string> = {
  leadership: "Leadership",
  "ai-technology": "AI & Technology",
  "hiring-careers": "Hiring & Careers",
  "life-society": "Life & Society",
};

const RELATED_COUNT = 3;
const SAME_CATEGORY_SLOTS = 2;

const newestFirst = (a: Article, b: Article) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf();

/**
 * Up to three articles to show after the author bio.
 *
 * - If the article sets `related` in frontmatter, those slugs are used, in order.
 *   An unknown, draft, or self slug fails the build rather than rendering a 404 link.
 * - Otherwise: up to two from the same category (most shared tags first, then newest),
 *   then the newest articles from other categories to fill the remaining slots.
 *   The cross-category slot keeps each category from becoming a closed loop.
 *
 * `published` must already be draft-filtered (getPublishedArticles), so drafts can
 * never be selected.
 */
export function getRelatedArticles(current: Article, published: Article[]): Article[] {
  const others = published.filter((a) => a.id !== current.id);

  const override = current.data.related;
  if (override && override.length > 0) {
    return override.map((slug) => {
      const match = others.find((a) => a.id === slug);
      if (!match) {
        throw new Error(
          `Article "${current.id}" lists related slug "${slug}", which is not a published article other than itself.`
        );
      }
      return match;
    });
  }

  const tags = new Set(current.data.tags ?? []);
  const sharedTags = (a: Article) => (a.data.tags ?? []).filter((t) => tags.has(t)).length;

  const sameCategory = others
    .filter((a) => a.data.category === current.data.category)
    .sort((a, b) => sharedTags(b) - sharedTags(a) || newestFirst(a, b))
    .slice(0, SAME_CATEGORY_SLOTS);

  const otherCategories = others
    .filter((a) => a.data.category !== current.data.category)
    .sort(newestFirst)
    .slice(0, RELATED_COUNT - sameCategory.length);

  return [...sameCategory, ...otherCategories];
}

/** Filter order on /articles. */
export const CATEGORY_ORDER: Category[] = ["leadership", "ai-technology", "hiring-careers", "life-society"];

/**
 * The "Author's picks" articles on /articles, in `featured` order. Zod validates one entry at a time,
 * so two articles claiming the same position would pass the schema; this fails
 * the build instead of silently showing one of them.
 */
export function getFeaturedArticles(published: Article[]): Article[] {
  const featured = published.filter((a) => a.data.featured !== undefined);
  const seen = new Map<number, string>();
  for (const a of featured) {
    const pos = a.data.featured as number;
    if (seen.has(pos)) {
      throw new Error(
        `Articles "${seen.get(pos)}" and "${a.id}" both set featured: ${pos}. Each Author's picks position must be unique.`
      );
    }
    seen.set(pos, a.id);
  }
  return featured.sort((a, b) => (a.data.featured as number) - (b.data.featured as number));
}
