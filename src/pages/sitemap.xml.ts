import type { APIRoute } from "astro";
import { getCollection } from "astro:content";

export const GET: APIRoute = async ({ site }) => {
  const base = site?.toString().replace(/\/$/, "") ?? "https://reginalcampbell.com";
  const articles = await getCollection("articles", ({ data }) => !data.draft);
  const artifacts = await getCollection("artifacts", ({ data }) => !data.draft);

  // Artifacts sort by `order` ascending, then alphabetically by title for entries
  // without one — matching the /artifacts index sort so the sitemap reflects the
  // intended reading sequence rather than filesystem order.
  const sortedArtifacts = artifacts.sort((a, b) => {
    const ao = a.data.order;
    const bo = b.data.order;
    if (ao !== undefined && bo !== undefined) return ao - bo;
    if (ao !== undefined) return -1;
    if (bo !== undefined) return 1;
    return a.data.title.localeCompare(b.data.title);
  });

  const urls = [
    { loc: `${base}/`, changefreq: "monthly", priority: "1.0" },
    { loc: `${base}/articles`, changefreq: "monthly", priority: "0.8" },
    ...articles.map((article) => ({
      loc: `${base}/articles/${article.id}`,
      changefreq: "monthly",
      priority: "0.7",
      lastmod: (article.data.updatedDate ?? article.data.pubDate).toISOString().split("T")[0],
    })),
    { loc: `${base}/artifacts`, changefreq: "monthly", priority: "0.8" },
    ...sortedArtifacts.map((artifact) => ({
      loc: `${base}/artifacts/${artifact.id}`,
      changefreq: "monthly",
      priority: "0.7",
      lastmod: artifact.data.lastReviewed.toISOString().split("T")[0],
    })),
  ];

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map(
    (u) => `  <url>
    <loc>${u.loc}</loc>${u.lastmod ? `\n    <lastmod>${u.lastmod}</lastmod>` : ""}
    <changefreq>${u.changefreq}</changefreq>
    <priority>${u.priority}</priority>
  </url>`
  )
  .join("\n")}
</urlset>
`;

  return new Response(body, {
    headers: { "Content-Type": "application/xml; charset=utf-8" },
  });
};
